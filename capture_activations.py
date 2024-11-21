import argparse
import json
import logging
import time
from multiprocessing import Process, Queue
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from llama_3.args import ModelArgs
from llama_3.model_text_only import Transformer
from llama_3.tokenizer import Tokenizer
from openwebtext_sentences_dataset import OpenWebTextSentencesDataset
from utils.cuda_utils import set_up_cuda


def load_model(
    model_path: Path,
    model_args: ModelArgs,
    store_layer_activ: list[int],
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> Transformer:
    """Load and prepare the model."""
    # Initialize the model on CPU, usually with bfloat16 data type
    logging.info("Initializing model on CPU...")
    torch.set_default_dtype(dtype)
    model = Transformer(model_args, store_layer_activ=store_layer_activ)

    logging.info("Loading model weights into CPU memory...")
    model_weights = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )

    logging.info("Loading model weights into model...")
    model.load_state_dict(model_weights)
    del model_weights

    logging.info(f"Model weights loaded successfully. Moving model to device {device}...")
    model.to(device)

    logging.info("Setting model to eval mode...")
    model.eval()

    logging.info("Model created successfully.")
    return model


def capture_activations(
    model: Transformer,
    dataloader: DataLoader,
    save_activation_queue: Queue,
    device: torch.device,
) -> None:
    """Process batches and store activations."""
    # Create an update log every 0.5% through the process assigned batches
    total_batches = len(dataloader)
    update_interval = max(1, total_batches // 200)

    for batch_idx, (batch, indices, seq_lens) in enumerate(dataloader):
        # Move batch to device and set activation states to input
        batch = batch.to(device)
        with torch.no_grad():
            model(batch, start_pos=0)

        # Get activation states for each layer
        layer_activations = model.get_layer_residual_activs()

        # Trim activations based on actual sequence lengths and clone them for later disk storage
        trimmed_activations = {}
        for layer, activations in layer_activations.items():
            trimmed_activations[layer] = [
                act[:seq_len].clone() for act, seq_len in zip(activations, seq_lens, strict=True)
            ]

        # Check if the queue is too large, if so, wait
        while save_activation_queue.qsize() > 10:
            logging.warning(
                f"Queue size is too large ({save_activation_queue.qsize()}), waiting...",
            )
            time.sleep(0.1)

        # Put activations in the queue for saving
        save_activation_queue.put((trimmed_activations, indices))

        # Update progress bar every 0.5% of the process assigned batches
        if (batch_idx + 1) % update_interval == 0:
            progress = (batch_idx + 1) / total_batches
            logging.info(f"Progress: {progress:.1%} ({batch_idx + 1}/{total_batches})")

    # Signal the saving process to stop
    logging.info("Activation capture complete. Sending stop signal to saving process.")
    save_activation_queue.put(None)


def save_activations_process(queue: Queue, activation_out_dir: Path) -> None:
    """Process for saving activations asynchronously."""
    while True:
        # Wait until next item from the queue, if it is None, then stop
        item = queue.get()
        if item is None:
            logging.info("Received stop signal. Finishing activation saving process.")
            break
        layer_activations, indices = item

        # Store all activations for each layer
        for layer, activations in layer_activations.items():
            layer_dir = activation_out_dir / f"layer_{layer}"

            # Store all sequence activations in each batch of sentences as separate files
            for i, activ in enumerate(activations):
                dataset_idx = indices[i].item()
                filename = f"activations_l{layer}_idx{dataset_idx}.pt"
                file_path = layer_dir / filename
                torch.save(activ, file_path)


def setup_output_dir(output_dir: Path, store_layer_activ: list[int]) -> None:
    """"""
    logging.info(f"Setting up output directories in {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    for layer in store_layer_activ:
        (output_dir / f"layer_{layer}").mkdir(parents=True, exist_ok=True)


def parse_arguments() -> argparse.Namespace:
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("activation_outs/"))
    parser.add_argument("--num_samples", type=int, default=None)
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
    args.model_dir = args.model_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    tokenizer_path = args.model_dir / "tokenizer.model"
    params_path = args.model_dir / "params.json"
    model_path = args.model_dir / "consolidated.00.pth"

    # Set up configuration
    store_layer_activ = [22]
    batch_size = 32
    dataloader_num_workers = 8
    dtype = torch.bfloat16
    max_token_length = 192
    add_bos_token = False
    dataset_shuffle = True

    if rank == 0:
        logging.info("#### Starting activation capture script.")
        logging.info("#### Arguments:")
        logging.info(f"# model_dir={args.model_dir}")
        logging.info(f"# output_dir={args.output_dir}")
        logging.info(f"# num_samples={args.num_samples}")
        logging.info("#### Distributed Configuration:")
        logging.info(f"# world_size={world_size}")
        logging.info(f"# rank={rank}")
        logging.info(f"# device={device}")
        logging.info("#### Configuration:")
        logging.info(f"# store_layer_activ={store_layer_activ}")
        logging.info(f"# batch_size={batch_size}")
        logging.info(f"# dataloader_num_workers={dataloader_num_workers}")
        logging.info(f"# dtype={dtype}")
        logging.info(f"# max_token_length={max_token_length}")
        logging.info(f"# add_bos_token={add_bos_token}")
        logging.info(f"# dataset_shuffle={dataset_shuffle}")

        setup_output_dir(
            output_dir=args.output_dir,
            store_layer_activ=store_layer_activ,
        )
    dist.barrier()

    logging.info("Loading tokenizer...")
    tokenizer = Tokenizer(str(tokenizer_path))
    dist.barrier()

    logging.info("Creating dataset, sampler and dataloader...")
    dataset = OpenWebTextSentencesDataset(
        tokenizer=tokenizer,
        max_token_length=max_token_length,
        num_samples=args.num_samples,
        shuffle=dataset_shuffle,
        add_bos_token=add_bos_token,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=42,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        num_workers=dataloader_num_workers,
        pin_memory=True,
    )
    logging.info(f"Dataloader created with {len(dataloader)} batches.")
    dist.barrier()

    logging.info(f"Loading model parameters from {params_path}...")
    with params_path.open("r") as f:
        model_params = json.load(f)
    model_args = ModelArgs(**model_params)
    model = load_model(
        model_path=model_path,
        model_args=model_args,
        store_layer_activ=store_layer_activ,
        device=device,
        dtype=dtype,
    )
    dist.barrier()

    logging.info(
        "Setting up asynchronous saving process for activations to enable continuous GPU usage...",
    )
    save_activation_queue = Queue()
    save_process = Process(
        target=save_activations_process,
        args=(save_activation_queue, args.output_dir),
    )
    save_process.start()
    dist.barrier()

    logging.info("Starting capture of activations...")
    capture_activations(
        model=model,
        dataloader=dataloader,
        save_activation_queue=save_activation_queue,
        device=device,
    )

    # Wait for the saving process to finish
    logging.info("Waiting for the activation saving process to finish...")
    save_process.join()

    logging.info(f"Process with rank {rank} finished.")

    if rank == 0:
        logging.info("CUDA Memory Summary:")
        logging.info(torch.cuda.memory_summary())

    dist.destroy_process_group()
    logging.info("FIN.")


if __name__ == "__main__":
    main()
