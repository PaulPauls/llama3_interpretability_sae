import argparse
import logging
import os
from pathlib import Path

import anthropic
import yaml
from anthropic.types.beta.messages.beta_message_batch_individual_response import (
    BetaMessageBatchIndividualResponse,
)
from tqdm import tqdm


def serialize_msg_batch(msg_batch_response: list[BetaMessageBatchIndividualResponse]) -> dict:
    """"""
    response_dict = dict()
    for response in msg_batch_response:
        if response.result.type == "canceled":
            response_dict[response.custom_id] = {
                "canceled": True,
            }
        else:
            response_dict[response.custom_id] = {
                "content": response.result.message.content[0].text,
                "model": response.result.message.model,
                "input_tokens": response.result.message.usage.input_tokens,
                "output_tokens": response.result.message.usage.output_tokens,
            }
    return response_dict


def parse_arguments() -> argparse.Namespace:
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_ids_filepath", type=Path, required=True)
    parser.add_argument("--response_output_dir", type=Path, required=True)
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
    args.response_ids_filepath = args.response_ids_filepath.resolve()
    args.response_output_dir = args.response_output_dir.resolve()
    args.response_output_dir.mkdir(parents=True, exist_ok=True)

    assert os.getenv("ANTHROPIC_API_KEY"), "Please set ANTHROPIC_API_KEY environment variable"

    logging.info("#### Starting script to retrieve Anthropic batches")
    logging.info("#### Arguments:")
    logging.info(f"# response_ids_filepath: {args.response_ids_filepath}")
    logging.info(f"# response_output_dir: {args.response_output_dir}")

    logging.info("Initializing Claude API...")
    client = anthropic.Anthropic()

    logging.info("Loading response IDs to check for...")
    with args.response_ids_filepath.open("r") as f:
        response_ids = yaml.safe_load(f)

    logging.info(f"Checking {len(response_ids)} response IDs...")
    for response_id in tqdm(response_ids, desc="Checking response IDs"):
        response_filepath = args.response_output_dir / f"{response_id}.yaml"
        if response_filepath.exists():
            logging.info(f"Skipping batch {response_id} as it already exists locally")
            continue

        retrieval = client.beta.messages.batches.retrieve(response_id)
        if retrieval.processing_status == "in_progress":
            logging.info(f"Batch {response_id} is still in progress: {retrieval.request_counts}")
            continue

        if retrieval.processing_status == "canceling":
            logging.info(f"Batch {response_id} is being canceled")
            continue

        logging.info(f"Batch {response_id} ended, retrieving results...")
        msg_batch_response = [r for r in client.beta.messages.batches.results(response_id)]
        results = serialize_msg_batch(msg_batch_response)
        with (args.response_output_dir / f"{response_id}.yaml").open("w") as f:
            yaml.dump(results, f)
        logging.info(f"Results for batch {response_id} saved to {response_filepath}")

    logging.info("FIN")


if __name__ == "__main__":
    main()
