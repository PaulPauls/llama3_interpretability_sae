import argparse
import logging
import math
import multiprocessing as mp
import os
from functools import partial
from pathlib import Path

import torch


class WelfordAccumulator:
    def __init__(self, shape: int, dtype: torch.dtype = torch.float64):
        """"""
        self.count = 0
        self.mean = torch.zeros(shape, dtype=dtype)

    def update(self, new_value: torch.Tensor) -> None:
        """
        :param new_value: tensor of shape (seq_len, d_model)
        """
        self.count += new_value.shape[0]
        new_value = new_value.to(self.mean.dtype)
        delta = new_value - self.mean
        self.mean += delta.sum(dim=0) / self.count

    def merge(self, other: "WelfordAccumulator") -> None:
        """Merge another accumulator into this one."""
        total_count = self.count + other.count
        delta = other.mean - self.mean
        self.mean += delta * (other.count / total_count)
        self.count = total_count


def create_batches(
    tensor: torch.Tensor,
    batch_size: int,
) -> tuple[tuple[torch.Tensor], torch.Tensor]:
    """
    :param tensor: tensor of shape (seq_len, d_model)
    :param batch_size: desired batch size
    :return: tuple of batches in desired batch_size and carry over tensor. Batches are of shape (batch_size, d_model)
        and carry over tensor is of shape (seq_len % batch_size, d_model)
    """
    seq_len, d_model = tensor.shape
    num_full_batches = seq_len // batch_size

    # Split the tensor into batches and carry over
    split_tensor = tensor.split(batch_size)
    batches = split_tensor[:num_full_batches]
    carry_over = (
        split_tensor[num_full_batches:][0]
        if len(split_tensor) > num_full_batches
        else torch.tensor([])
    )

    return batches, carry_over


def process_tensors(
    input_files: list[Path],
    output_dir: Path,
    batch_size: int,
) -> tuple[WelfordAccumulator, torch.Tensor, int]:
    """"""
    # Get the current process ID for logging and storing batches
    pid = os.getpid()
    logging.info(f"[PID {pid}] Processing {len(input_files)} tensor files")

    # Initialize variables
    carry_over = torch.tensor([])
    output_count = 0
    update_interval = len(input_files) // 200

    # Initialize Welford accumulator
    welford_acc = None

    for i, filepath in enumerate(input_files):
        # Log progress
        if i % update_interval == 0:
            progress = i / len(input_files)
            logging.info(f"[PID {pid}] Progress: {progress:.1%} ({i}/{len(input_files)})")

        # Load the tensor
        tensor = torch.load(filepath, weights_only=True)

        # Initialize accumulator if not done yet
        if welford_acc is None:
            welford_acc = WelfordAccumulator(tensor.shape[1])

        # Update the accumulator with entire tensor
        welford_acc.update(tensor)

        # Combine with carry_over from previous iteration
        if carry_over.numel() > 0:
            tensor = torch.cat([carry_over, tensor], dim=0)

        # Process the tensor
        batches, carry_over = create_batches(tensor, batch_size)

        # Save the batches
        for batch in batches:
            output_file = output_dir / f"batch_{pid}_{output_count}.pt"
            torch.save(batch.clone(), output_file)
            del batch
            output_count += 1

        del tensor

    # Return the welford accumulator, the last carry over with less than batch_size elements, and the number of batches
    # processed
    logging.info(f"[PID {pid}] Finished processing all tensor files")
    return welford_acc, carry_over, output_count


def parse_arguments() -> argparse.Namespace:
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--mean_filepath", type=Path, default=None)
    parser.add_argument("--num_processes", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    """"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Parse arguments and set up paths
    args = parse_arguments()
    args.input_dir = args.input_dir.resolve()
    if args.output_dir is None:
        args.output_dir = args.input_dir.parent / (args.input_dir.name + "_batched")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mean_filepath is None:
        args.mean_filepath = args.input_dir.parent / (args.input_dir.name + "_mean.pt")
    args.mean_filepath = args.mean_filepath.resolve()

    logging.info("[MAIN] #### Starting SAE preprocessing script")
    logging.info("[MAIN] #### Configuration:")
    logging.info(f"[MAIN] # Input directory: {args.input_dir}")
    logging.info(f"[MAIN] # Output directory: {args.output_dir}")
    logging.info(f"[MAIN] # Mean tensor filepath: {args.mean_filepath}")
    logging.info(f"[MAIN] # Number of processes: {args.num_processes}")
    logging.info(f"[MAIN] # Batch size: {args.batch_size}")

    # Get all tensor files from input directory
    input_files = list(args.input_dir.rglob("*.pt"))
    logging.info(f"[MAIN] Found {len(input_files)} tensor files in input directory")

    # Split input files into chunks for each process
    chunk_size = math.ceil(len(input_files) / args.num_processes)
    chunks = [input_files[i : i + chunk_size] for i in range(0, len(input_files), chunk_size)]
    logging.info(
        f"[MAIN] Split input files into {len(chunks)} chunks with sizes: {[len(chunk) for chunk in chunks]}",
    )

    # Create a partial function of `process_tensors` because pool.map only accepts functions with one argument
    process_tensors_partial = partial(
        process_tensors,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )

    # Use multiprocessing to process chunks in parallel
    with mp.Pool(processes=args.num_processes) as pool:
        results = pool.map(process_tensors_partial, chunks)
    logging.info("[MAIN] Finished processing all chunks")

    logging.info("[MAIN] Combining results from all processes...")
    total_acc = WelfordAccumulator(results[0][0].mean.shape)
    carry_over_sum = 0
    total_batches = 0

    for acc, carry_over, num_batches in results:
        total_acc.merge(acc)
        carry_over_sum += carry_over.shape[0]
        total_batches += num_batches

    logging.info("[MAIN] #### Welford accumulator statistics:")
    logging.info(f"[MAIN] # Total number of activations processed: {total_acc.count}")
    logging.info(
        f"[MAIN] # Average sequence length of inputs: {total_acc.count / len(input_files):.1f}",
    )
    logging.info(f"[MAIN] # Total number of activations in carry-over discarded: {carry_over_sum}")
    logging.info(f"[MAIN] # Total number of batches created: {total_batches}")
    logging.info(f"[MAIN] # Mean tensor mean: {total_acc.mean.mean().item():.6f}")

    # Save the mean tensor
    torch.save(total_acc.mean, args.mean_filepath)
    logging.info(f"[MAIN] Mean tensor saved to {args.mean_filepath}")
    logging.info("[MAIN] FIN")


if __name__ == "__main__":
    main()
