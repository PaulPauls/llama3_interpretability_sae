import argparse
import logging
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sae import TopKSparseAutoencoder, load_sae_model
from utils.cuda_utils import set_up_cuda


class SequenceActivationDataset(Dataset):
    def __init__(self, data_dir: Path, filename_prefix: str):
        self.data_files = list(data_dir.rglob("*.pt"))
        self.data_files.sort()
        self.filename_prefix = filename_prefix

        # assert that data indices are continuous and starting at 0
        assert self.data_files[0].stem[len(self.filename_prefix) :] == "0"
        assert self.data_files[-1].stem[len(self.filename_prefix) :] == str(
            len(self.data_files) - 1,
        )

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        file_path = self.data_files[idx]
        filename_idx = int(file_path.stem[len(self.filename_prefix) :])
        assert filename_idx == idx

        data = torch.load(file_path, weights_only=True)
        return data, idx

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor, int]],
    ) -> tuple[torch.Tensor, list[int], list[int]]:
        """"""
        sequences, indices = zip(*batch, strict=False)

        # Calculate cumulative sequence lengths for boundaries
        seq_lengths = [seq.size(0) for seq in sequences]
        boundaries = [0]
        current_pos = 0
        for length in seq_lengths:
            current_pos += length
            boundaries.append(current_pos)

        # Stack all sequences along seq_len dimension
        stacked_sequences = torch.cat(sequences, dim=0)

        return stacked_sequences, boundaries, list(indices)


def update_top_sentences_dict(
    top_sentences_dict: dict,
    top_latent_val: float,
    top_latent_idx: int,
    file_idx: int,
    top_n_sentences: int,
) -> None:
    """"""
    highest_vals = top_sentences_dict[top_latent_idx]
    # If the list is not full, add the new value automatically and then sort
    if len(highest_vals) < top_n_sentences:
        highest_vals.append([top_latent_val, file_idx])
        highest_vals.sort(key=lambda x: x[0])
        top_sentences_dict[top_latent_idx] = highest_vals
    # If the list is full, check if the new value is higher than the lowest value in the list. Then add and sort.
    elif top_latent_val > highest_vals[0][0]:
        highest_vals[0] = [top_latent_val, file_idx]
        highest_vals.sort(key=lambda x: x[0])
        top_sentences_dict[top_latent_idx] = highest_vals


def capture_top_activating_sentences(
    model: TopKSparseAutoencoder,
    dataloader: DataLoader,
    top_n_sentences: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict, dict, dict]:
    """"""
    # Initialize top sentences dict that will store the top activating sentences for each latent and aggregation method
    top_sentences_mean = defaultdict(list)
    top_sentences_max = defaultdict(list)
    top_sentences_last = defaultdict(list)

    # Start inference loop for top activating sentence capture
    for batch, boundaries, indices in tqdm(dataloader):
        batch = batch.to(dtype).to(device)

        # Forward pass through the model
        batch_normalized, mean, norm = model.preprocess_input(batch)
        with torch.no_grad():
            _, _, h_sparse = model.forward_1d_normalized(batch_normalized)

        # Unbatch the h_sparse tensor into sequences using the predetermined boundaries
        sequence_h_sparse = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            sequence_h_sparse.append(h_sparse[start:end])

        # Perform aggregation to aggregated_latent of shape (n_latents,) for each sequence
        for seq_h_sparse, file_idx in zip(sequence_h_sparse, indices, strict=True):
            # Determine aggregated (mean/max/last) latent activation for the sequence
            aggregated_mean_latent = seq_h_sparse.mean(dim=0)
            aggregated_max_latent = seq_h_sparse.max(dim=0).values
            aggregated_last_latent = seq_h_sparse[-1]

            # Determine top activating latent and its index for each aggregation method
            top_mean_latent, top_mean_latent_idx = torch.max(aggregated_mean_latent, dim=0)
            top_max_latent, top_max_latent_idx = torch.max(aggregated_max_latent, dim=0)
            top_last_latent, top_last_latent_idx = torch.max(aggregated_last_latent, dim=0)

            # Update top sentences dict with top activating sentences
            update_top_sentences_dict(
                top_sentences_dict=top_sentences_mean,
                top_latent_val=float(top_mean_latent.item()),
                top_latent_idx=int(top_mean_latent_idx.item()),
                file_idx=file_idx,
                top_n_sentences=top_n_sentences,
            )
            update_top_sentences_dict(
                top_sentences_dict=top_sentences_max,
                top_latent_val=float(top_max_latent.item()),
                top_latent_idx=int(top_max_latent_idx.item()),
                file_idx=file_idx,
                top_n_sentences=top_n_sentences,
            )
            update_top_sentences_dict(
                top_sentences_dict=top_sentences_last,
                top_latent_val=float(top_last_latent.item()),
                top_latent_idx=int(top_last_latent_idx.item()),
                file_idx=file_idx,
                top_n_sentences=top_n_sentences,
            )

    logging.info("Finished capturing top activating sentences.")
    return (
        dict(top_sentences_mean),
        dict(top_sentences_max),
        dict(top_sentences_last),
    )


def parse_arguments():
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--captured_data_output_dir", type=Path, required=True)
    return parser.parse_args()


def main():
    """"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Parse arguments and base setup
    args = parse_arguments()
    args.data_dir = args.data_dir.resolve()
    args.model_path = args.model_path.resolve()
    args.captured_data_output_dir = args.captured_data_output_dir.resolve()
    set_up_cuda()

    # Set up configuration
    top_n_sentences = 100
    sae_top_k = 64
    sae_normalization_eps = 1e-6
    batch_size = 128
    layer = 22
    filename_prefix = f"activations_l{layer}_idx"
    dtype = torch.float32
    device = torch.device("cuda")
    dataloader_num_workers = 8

    logging.info("#### Starting script to capture top activating sentences")
    logging.info("#### Arguments:")
    logging.info(f"# data_dir={args.data_dir}")
    logging.info(f"# model_path={args.model_path}")
    logging.info(f"# captured_data_output_dir={args.captured_data_output_dir}")
    logging.info("#### Configuration:")
    logging.info(f"# top_n_sentences: {top_n_sentences}")
    logging.info(f"# sae_top_k: {sae_top_k}")
    logging.info(f"# sae_normalization_eps: {sae_normalization_eps}")
    logging.info(f"# batch_size: {batch_size}")
    logging.info(f"# layer: {layer}")
    logging.info(f"# filename_prefix: {filename_prefix}")
    logging.info(f"# dtype: {dtype}")
    logging.info(f"# device: {device}")
    logging.info(f"# dataloader_num_workers: {dataloader_num_workers}")

    logging.info("Initializing and loading model...")
    model = load_sae_model(
        model_path=args.model_path,
        sae_top_k=sae_top_k,
        sae_normalization_eps=sae_normalization_eps,
        device=device,
        dtype=dtype,
    )

    logging.info("Creating SequenceActivation Dataset and Dataloader...")
    dataset = SequenceActivationDataset(
        data_dir=args.data_dir,
        filename_prefix=filename_prefix,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataloader_num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )
    logging.info(f"Dataset size: {len(dataset)}")
    logging.info(f"Batches in Dataloader: {len(dataloader)}")

    logging.info("Capturing top activating sentences...")
    (
        top_sentences_mean,
        top_sentences_max,
        top_sentences_last,
    ) = capture_top_activating_sentences(
        model=model,
        dataloader=dataloader,
        top_n_sentences=top_n_sentences,
        dtype=dtype,
        device=device,
    )

    logging.info("Saving top activating sentences...")
    args.captured_data_output_dir.mkdir(parents=True, exist_ok=True)
    top_sentences_mean_filepath = args.captured_data_output_dir / "top_sentences_mean.yaml"
    top_sentences_max_filepath = args.captured_data_output_dir / "top_sentences_max.yaml"
    top_sentences_last_filepath = args.captured_data_output_dir / "top_sentences_last.yaml"

    with top_sentences_mean_filepath.open("w") as f:
        yaml.dump(top_sentences_mean, f)
    with top_sentences_max_filepath.open("w") as f:
        yaml.dump(top_sentences_max, f)
    with top_sentences_last_filepath.open("w") as f:
        yaml.dump(top_sentences_last, f)
    logging.info(f"Saved top activating sentences to: {args.captured_data_output_dir}")

    logging.info("FIN.")


if __name__ == "__main__":
    main()
