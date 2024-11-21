import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from sae import TopKSparseAutoencoder
from utils.cuda_utils import set_up_cuda


class TopKSparseAutoencoderDataset(Dataset):
    def __init__(self, data_dir: Path):
        """"""
        # List and sort data files to ensure consistent order across processes
        self.data_files = list(data_dir.glob("*.pt"))
        self.data_files.sort()

    def __len__(self) -> int:
        """"""
        return len(self.data_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """"""
        data = torch.load(self.data_files[idx], weights_only=True)
        return data


def train_epoch(
    epoch: int,
    num_epochs: int,
    model: DistributedDataParallel,
    dataloader: DataLoader,
    criterion: nn.MSELoss,
    optimizer: optim.AdamW,
    k_aux: int,
    aux_loss_coeff: float,
    latent_last_nonzero: torch.Tensor,
    dead_steps_threshold: int,
    logs_per_epoch: int,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
) -> None:
    """"""
    # Set the model to train mode
    model.train()

    # Initialize epoch log variables and helpers
    loss_acc = torch.tensor(0.0, device=device)
    aux_loss_acc = torch.tensor(0.0, device=device)
    total_loss_acc = torch.tensor(0.0, device=device)
    log_interval = len(dataloader) // logs_per_epoch
    accumulated_loss_count = dist.get_world_size() * log_interval

    # Create epoch progress bar on main process
    if rank == 0:
        progress_bar = tqdm(
            total=len(dataloader),
            desc=f"Training | Epoch {epoch + 1}/{num_epochs}",
        )

    for batch_idx, batch in enumerate(dataloader):
        # batch shape is [1, preprocessed_batch_size, d_model]. Squeeze to [preprocessed_batch_size, d_model] and
        # cast to model dtype and device, then preprocess to normalize.
        batch = batch.squeeze(0).to(dtype).to(device)
        batch_normalized, mean, norm = model.module.preprocess_input(batch)

        # Zero the gradients and perform forward pass
        optimizer.zero_grad()
        reconstructed, h, h_sparse = model.forward_1d_normalized(batch_normalized)

        # Compute main loss in normalized space
        loss = criterion(reconstructed, batch_normalized)

        # If enough latents haven't been activated in more than dead_steps_threshold training steps then calculate an
        # auxiliary loss to help reactivate the latents.
        dead_mask = latent_last_nonzero > dead_steps_threshold
        dead_latents = dead_mask.sum().item()
        if dead_latents >= k_aux:
            # Calculate an auxiliary reconstruction with only dead latents and an additionaly amount (k_aux) of TopK
            # filtered latents.
            h_masked = h * dead_mask
            reconstructed_aux, _ = model.module.decode_latent(h=h_masked, k=k_aux)

            # Compute auxiliary loss as MSE between residual and the aux reconstruction to make dead latents explain
            # what the main latents could not and thereby activate them again and make them useful again.
            residual = batch_normalized - reconstructed.detach()
            aux_loss = criterion(reconstructed_aux, residual)
        else:
            # If there are not enough dead latents to activate, set auxiliary loss to 0.
            aux_loss = torch.tensor(0.0, device=device)

        # Compute total loss with auxiliary loss coefficient
        total_loss = loss + aux_loss_coeff * aux_loss

        # Perform backward pass, project out gradient info as recommended by OpenAI paper, then step the optimizer
        # and normalize the decoder weights again.
        total_loss.backward()
        model.module.project_decoder_grads()
        optimizer.step()
        model.module.normalize_decoder_weights()

        # Accumulate losses
        loss_acc += loss.detach()
        aux_loss_acc += aux_loss.detach()
        total_loss_acc += total_loss.detach()

        # Update the progress bar on main process
        if rank == 0:
            progress_bar.update(1)

        # Update and blocking sync latent_last_zero at end of batch to not create a barrier mid batch processing.
        # Take minimum in case a latent was activated in another process.
        latent_last_nonzero *= (h_sparse == 0).all(dim=0).long()
        latent_last_nonzero += 1
        dist.all_reduce(latent_last_nonzero, op=dist.ReduceOp.MIN)

        # Gather and average losses across processes, reset them for next interval, determine dead latents and then
        # log to wandb and tqdm
        if (batch_idx + 1) % log_interval == 0:
            dist.all_reduce(loss_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(aux_loss_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_loss_acc, op=dist.ReduceOp.SUM)
            avg_loss = loss_acc.item() / accumulated_loss_count
            avg_aux_loss = aux_loss_acc.item() / accumulated_loss_count
            avg_total_loss = total_loss_acc.item() / accumulated_loss_count

            # Reset log variables for next interval
            loss_acc = torch.tensor(0.0, device=device)
            aux_loss_acc = torch.tensor(0.0, device=device)
            total_loss_acc = torch.tensor(0.0, device=device)

            # Determine dead latent debug statistics
            dead_latents_ratio = dead_latents / dead_mask.numel()
            max_dead_latent = latent_last_nonzero.max().item()
            max_dead_latent_count = (latent_last_nonzero == max_dead_latent).sum().item()

            # Log to wandb and tqdm
            if rank == 0:
                wandb.log(
                    data={
                        "train/loss": avg_loss,
                        "train/aux_loss": avg_aux_loss,
                        "train/total_loss": avg_total_loss,
                        "debug/dead_latents_ratio": dead_latents_ratio,
                        "debug/max_dead_latent": max_dead_latent,
                        "debug/max_dead_latent_count": max_dead_latent_count,
                    },
                    step=epoch * len(dataloader) + batch_idx + 1,
                )
                progress_bar.set_postfix(
                    loss=f"{avg_loss:.6f}",
                    aux_loss=f"{avg_aux_loss:.6f}",
                    total_loss=f"{avg_total_loss:.6f}",
                )

    # Close the progress bar on main process
    if rank == 0:
        progress_bar.close()


def validate_epoch(
    epoch: int,
    num_epochs: int,
    model: DistributedDataParallel,
    dataloader: DataLoader,
    criterion: nn.MSELoss,
    k_aux: int,
    aux_loss_coeff: float,
    latent_last_nonzero: torch.Tensor,
    dead_steps_threshold: int,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
) -> tuple[float, float, float]:
    """"""
    # Set the model to eval mode
    model.eval()

    # Initialize epoch log variables
    loss_acc = torch.tensor(0.0, device=device)
    aux_loss_acc = torch.tensor(0.0, device=device)
    total_loss_acc = torch.tensor(0.0, device=device)

    # Create epoch progress bar on main process
    if rank == 0:
        progress_bar = tqdm(
            total=len(dataloader),
            desc=f"Validation | Epoch {epoch + 1}/{num_epochs}",
        )

    with torch.no_grad():
        for batch in dataloader:
            # batch shape is [1, preprocessed_batch_size, d_model]. Squeeze to [preprocessed_batch_size, d_model] and
            # cast to model dtype and device, then preprocess to normalize.
            batch = batch.squeeze(0).to(dtype).to(device)
            batch_normalized, mean, norm = model.module.preprocess_input(batch)

            # Perform forward pass
            reconstructed, h, h_sparse = model.forward_1d_normalized(batch_normalized)

            # Compute main loss in normalized space
            loss = criterion(reconstructed, batch_normalized)

            # Compute auxiliary loss if necessary
            dead_mask = latent_last_nonzero > dead_steps_threshold
            dead_latents = dead_mask.sum().item()
            if dead_latents >= k_aux:
                h_masked = h * dead_mask
                reconstructed_aux, _ = model.module.decode_latent(h=h_masked, k=k_aux)
                residual = batch_normalized - reconstructed.detach()
                aux_loss = criterion(reconstructed_aux, residual)
            else:
                aux_loss = torch.tensor(0.0, device=device)

            # Compute total loss with auxiliary loss coefficient
            total_loss = loss + aux_loss_coeff * aux_loss

            # Accumulate losses
            loss_acc += loss.detach()
            aux_loss_acc += aux_loss.detach()
            total_loss_acc += total_loss.detach()

            # Update the progress bar on main process
            if rank == 0:
                progress_bar.update(1)

    # Close the progress bar on main process
    if rank == 0:
        progress_bar.close()

    # Gather and average losses across processes
    dist.all_reduce(loss_acc, op=dist.ReduceOp.SUM)
    dist.all_reduce(aux_loss_acc, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_loss_acc, op=dist.ReduceOp.SUM)
    avg_loss = loss_acc.item() / (dist.get_world_size() * len(dataloader))
    avg_aux_loss = aux_loss_acc.item() / (dist.get_world_size() * len(dataloader))
    avg_total_loss = total_loss_acc.item() / (dist.get_world_size() * len(dataloader))

    return avg_loss, avg_aux_loss, avg_total_loss


def train_autoencoder(
    model: TopKSparseAutoencoder,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int,
    early_stopping_patience: int,
    learning_rate: float,
    learning_rate_min: float,
    optimizer_betas: tuple[float, float],
    optimizer_eps: float,
    k_aux: int,
    aux_loss_coeff: float,
    dead_steps_threshold: int,
    logs_per_epoch: int,
    checkpoint_dir: Path,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
) -> TopKSparseAutoencoder:
    """"""
    logging.info("Sending model to device and wrapping in DistributedDataParallel...")
    model = model.to(device)
    model = DistributedDataParallel(model)

    logging.info("Setting up optimizer, scheduler and loss function...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=optimizer_betas,
        eps=optimizer_eps,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate_min)
    criterion = nn.MSELoss()
    if rank == 0:
        wandb.log(data={"learning_rate": learning_rate}, step=0)

    # Initialize latent_last_nonzero tensor to keep track of the last non-zero step for each latent
    latent_last_nonzero = torch.zeros(model.module.n_latents, dtype=torch.long, device=device)

    # Early stopping variables
    best_val_avg_total_loss = float("inf")
    patience_counter = 0

    logging.info("Starting training loop...")
    for epoch in range(num_epochs):
        # Wait for all processes to synchronize
        dist.barrier()

        # Set the epoch for the samplers
        train_dataloader.sampler.set_epoch(epoch)
        val_dataloader.sampler.set_epoch(epoch)

        # Train an epoch
        train_epoch(
            epoch=epoch,
            num_epochs=num_epochs,
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            k_aux=k_aux,
            aux_loss_coeff=aux_loss_coeff,
            latent_last_nonzero=latent_last_nonzero,
            dead_steps_threshold=dead_steps_threshold,
            logs_per_epoch=logs_per_epoch,
            dtype=dtype,
            device=device,
            rank=rank,
        )

        # Validate an epoch
        val_avg_loss, val_avg_aux_loss, val_avg_total_loss = validate_epoch(
            epoch=epoch,
            num_epochs=num_epochs,
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            k_aux=k_aux,
            aux_loss_coeff=aux_loss_coeff,
            latent_last_nonzero=latent_last_nonzero,
            dead_steps_threshold=dead_steps_threshold,
            dtype=dtype,
            device=device,
            rank=rank,
        )

        # Step the scheduler
        scheduler.step()
        updated_lr = scheduler.get_last_lr()[0]

        # Log metrics in wandb and console and save checkpoint
        if rank == 0:
            wandb.log(
                data={
                    "val/loss": val_avg_loss,
                    "val/aux_loss": val_avg_aux_loss,
                    "val/total_loss": val_avg_total_loss,
                    "learning_rate": updated_lr,
                },
                step=(epoch + 1) * len(train_dataloader),
            )
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Updated LR: {updated_lr:.2e}")
            logging.info(
                f"val/loss: {val_avg_loss:.6f} "
                f"| val/aux_loss: {val_avg_aux_loss:.6f} "
                f"| val/total_loss: {val_avg_total_loss:.6f}",
            )

            checkpoint_path = checkpoint_dir / f"model_checkpoint_epoch-{epoch + 1}.pth"
            torch.save(model.module.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved to: {checkpoint_path}")

        # Early stopping check
        if val_avg_total_loss < best_val_avg_total_loss:
            best_val_avg_total_loss = val_avg_total_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Check if early stopping criteria is met
        if patience_counter >= early_stopping_patience:
            logging.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

    return model.module


def parse_arguments() -> argparse.Namespace:
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--b_pre_path", type=Path, required=True)
    parser.add_argument("--model_save_path", type=Path, required=True)
    parser.add_argument("--model_load_path", type=Path, default=None)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("sae_checkpoints"))
    return parser.parse_args()


def main() -> None:
    """"""
    # Initialize distributed process group
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(rank)
    set_up_cuda()

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s] [Rank {rank}] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Parse arguments and set up paths
    args = parse_arguments()
    args.data_dir = args.data_dir.resolve()
    args.b_pre_path = args.b_pre_path.resolve()
    args.model_save_path = args.model_save_path.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    if args.model_load_path:
        args.model_load_path = args.model_load_path.resolve()

    # Set up configuration
    d_model = 3072
    n_latents = 2**16  # 65536
    k = 64
    k_aux = 2048
    aux_loss_coeff = 1 / 32
    dead_steps_threshold = 80_000  # ~1 epoch in training steps
    sae_normalization_eps = 1e-6
    batch_size = 1024
    num_epochs = 10
    early_stopping_patience = 10  # disabled
    learning_rate = 5e-5
    learning_rate_min = learning_rate / 5
    optimizer_betas = (0.85, 0.9999)
    optimizer_eps = 6.25e-10
    dtype = torch.float32
    dataloader_num_workers = 8
    logs_per_epoch = 1000
    train_val_split = 0.95

    if rank == 0:
        logging.info("Logging into and initializing wandb...")
        wandb.login()
        wandb.init(
            project="llama3_interpretability_sae",
            config={
                "d_model": d_model,
                "n_latents": n_latents,
                "k": k,
                "k_aux": k_aux,
                "aux_loss_coeff": aux_loss_coeff,
                "dead_steps_threshold": dead_steps_threshold,
                "sae_normalization_eps": sae_normalization_eps,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "early_stopping_patience": early_stopping_patience,
                "learning_rate": learning_rate,
                "learning_rate_min": learning_rate_min,
                "optimizer_betas": optimizer_betas,
                "optimizer_eps": optimizer_eps,
                "dtype": dtype,
                "dataloader_num_workers": dataloader_num_workers,
                "logs_per_epoch": logs_per_epoch,
                "train_val_split": train_val_split,
                "world_size": world_size,
            },
        )

        logging.info("#### Starting SAE training script.")
        logging.info("#### Arguments:")
        logging.info(f"# data_dir={args.data_dir}")
        logging.info(f"# b_pre_path={args.b_pre_path}")
        logging.info(f"# model_save_path={args.model_save_path}")
        logging.info(f"# model_load_path={args.model_load_path}")
        logging.info(f"# checkpoint_dir={args.checkpoint_dir}")
        logging.info("#### Distributed Configuration:")
        logging.info(f"# world_size={world_size}")
        logging.info(f"# rank={rank}")
        logging.info(f"# device={device}")
        logging.info("#### Configuration:")
        logging.info(f"# d_model={d_model}")
        logging.info(f"# n_latents={n_latents}")
        logging.info(f"# k={k}")
        logging.info(f"# k_aux={k_aux}")
        logging.info(f"# aux_loss_coeff={aux_loss_coeff}")
        logging.info(f"# dead_steps_threshold={dead_steps_threshold}")
        logging.info(f"# sae_normalization_eps={sae_normalization_eps}")
        logging.info(f"# batch_size={batch_size}")
        logging.info(f"# num_epochs={num_epochs}")
        logging.info(f"# early_stopping_patience={early_stopping_patience}")
        logging.info(f"# learning_rate={learning_rate}")
        logging.info(f"# learning_rate_min={learning_rate_min}")
        logging.info(f"# optimizer_betas={optimizer_betas}")
        logging.info(f"# optimizer_eps={optimizer_eps}")
        logging.info(f"# dtype={dtype}")
        logging.info(f"# dataloader_num_workers={dataloader_num_workers}")
        logging.info(f"# logs_per_epoch={logs_per_epoch}")
        logging.info(f"# train_val_split={train_val_split}")

        # Create a new directory for the checkpoints
        run_name = datetime.now(tz=UTC).strftime("run_%Y-%m-%d_%H-%M-%S")
        args.checkpoint_dir = args.checkpoint_dir / run_name
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Checkpoints will be saved to: {args.checkpoint_dir}")
    dist.barrier()

    logging.info(
        "Loading pre-computed b_pre, being the mean activation value of the training data...",
    )
    b_pre = torch.load(args.b_pre_path, weights_only=True)
    assert b_pre.shape == (d_model,), \
        f"b_pre shape mismatch. Expected {(d_model,)}, got {b_pre.shape}"

    # Initialize the model
    logging.info("Initializing Sparse Autoencoder model...")
    model = TopKSparseAutoencoder(
        d_model=d_model,
        n_latents=n_latents,
        k=k,
        b_pre=b_pre,
        dtype=dtype,
        normalize_eps=sae_normalization_eps,
    )
    if args.model_load_path:
        logging.info("Loading model weights from checkpoint...")
        model_weights = torch.load(
            args.model_load_path,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        model.load_state_dict(model_weights)
        model.to(dtype=dtype)
        del model_weights
        logging.info(f"Model weights loaded from {args.model_load_path}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable parameters: {trainable_params}")
    logging.info(f"Total parameters: {total_params}")
    dist.barrier()

    logging.info("Creating dataset...")
    dataset = TopKSparseAutoencoderDataset(args.data_dir)
    assert (batch_size, d_model) == dataset[0].shape, \
        f"Dataset shape mismatch. Expected {(batch_size, d_model)}, got {dataset[0].shape}"
    train_val_index = int(len(dataset) * train_val_split)
    train_dataset = Subset(dataset, indices=range(train_val_index))
    val_dataset = Subset(dataset, indices=range(train_val_index, len(dataset)))
    dist.barrier()

    logging.info("Creating distributed sampler and dataloader...")
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=42,
    )
    # Set batch_size to 1 in dataloader since data is already batched in preprocessing
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=train_sampler,
        num_workers=dataloader_num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        sampler=val_sampler,
        num_workers=dataloader_num_workers,
        pin_memory=True,
    )
    logging.info(f"Train dataloader created with {len(train_dataloader)} batches.")
    logging.info(f"Validation dataloader created with {len(val_dataloader)} batches.")
    dist.barrier()

    logging.info("Starting training of Sparse Autoencoder...")
    trained_model = train_autoencoder(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        learning_rate=learning_rate,
        learning_rate_min=learning_rate_min,
        optimizer_betas=optimizer_betas,
        optimizer_eps=optimizer_eps,
        k_aux=k_aux,
        aux_loss_coeff=aux_loss_coeff,
        dead_steps_threshold=dead_steps_threshold,
        logs_per_epoch=logs_per_epoch,
        checkpoint_dir=args.checkpoint_dir,
        dtype=dtype,
        device=device,
        rank=rank,
    )

    # Save the model only on the main process
    if rank == 0:
        torch.save(trained_model.state_dict(), args.model_save_path)
        logging.info(f"Trained model saved to {args.model_save_path}")
        logging.info("CUDA Memory Summary:")
        logging.info(torch.cuda.memory_summary())
        logging.info("Finishing wandb run and saving trained model...")
        wandb.finish()

    # Clean up the process group
    dist.destroy_process_group()
    logging.info("FIN.")


if __name__ == "__main__":
    main()
