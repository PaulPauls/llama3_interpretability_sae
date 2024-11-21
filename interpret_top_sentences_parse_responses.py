import argparse
import json
import logging
from pathlib import Path

import yaml
from tqdm import tqdm


def parse_content_response(content: str) -> dict[str, str | float]:
    """"""
    dict_start_idx = content.rfind("{")
    dict_end_idx = content.rfind("}")
    dict_string = content[dict_start_idx : dict_end_idx + 1]

    response_dict = json.loads(dict_string)
    assert len(response_dict) == 2
    assert "common_semantic" in response_dict
    assert "certainty" in response_dict
    assert isinstance(response_dict["common_semantic"], str)
    assert len(response_dict["common_semantic"]) > 0
    assert isinstance(response_dict["certainty"], float)
    assert 0 <= response_dict["certainty"] <= 1

    return response_dict


def parse_arguments() -> argparse.Namespace:
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieved_responses_dir", type=Path, required=True)
    parser.add_argument("--parsed_responses_output_filepath", type=Path, required=True)
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
    args.retrieved_responses_dir = args.retrieved_responses_dir.resolve()
    args.parsed_responses_output_filepath = args.parsed_responses_output_filepath.resolve()

    logging.info("#### Starting script to parse Anthropic responses")
    logging.info("#### Arguments:")
    logging.info(f"# retrieved_responses_dir: {args.retrieved_responses_dir}")
    logging.info(f"# parsed_responses_output_filepath: {args.parsed_responses_output_filepath}")

    logging.info("Parsing retrieved responses...")
    parsed_response_dict = dict()
    retrieved_responses = list(args.retrieved_responses_dir.glob("*.yaml"))
    for response_file in tqdm(retrieved_responses):
        with open(response_file) as f:
            response_dict = yaml.safe_load(f)
        for custom_id, response in response_dict.items():
            latent_idx = int(custom_id)
            try:
                parsed_response_dict[latent_idx] = parse_content_response(response["content"])
            except Exception as e:
                logging.exception(
                    f"Error parsing response for file {response_file} and custom ID {custom_id}: {e}",
                )

    logging.info("Sorting parsed responses by certainty of the semantic commonality...")
    sorted_parsed_response_dict = dict(
        sorted(parsed_response_dict.items(), key=lambda x: x[1]["certainty"], reverse=True),
    )

    logging.info("Saving parsed responses...")
    with args.parsed_responses_output_filepath.open("w") as f:
        yaml.safe_dump(sorted_parsed_response_dict, f)


if __name__ == "__main__":
    main()
