import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


def download_hf_hub_file(model_id: str, filename: str, token: str, download_dir: Path) -> None:
    """"""
    file_path = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        token=token,
        local_dir=download_dir,
    )
    print(f"{filename} downloaded to: {file_path}")


def parse_arguments() -> argparse.Namespace:
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["llama_3.1-8B", "llama_3.2-3B"])
    return parser.parse_args()


def main() -> None:
    """"""
    args = parse_arguments()

    if args.model == "llama_3.1-8B":
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        download_dir = Path("llama_3.1-8B_model/")
    elif args.model == "llama_3.2-3B":
        model_id = "meta-llama/Llama-3.2-3B-Instruct"
        download_dir = Path("llama_3.2-3B_model/")
    else:
        raise ValueError("Invalid model choice")
    download_dir.mkdir(parents=True, exist_ok=True)

    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Please set HF_TOKEN environment variable")

    filename_llama_model = "original/consolidated.00.pth"
    filename_llama_params = "original/params.json"
    filename_tokenizer = "original/tokenizer.model"

    download_hf_hub_file(model_id, filename_llama_model, token, download_dir)
    download_hf_hub_file(model_id, filename_llama_params, token, download_dir)
    download_hf_hub_file(model_id, filename_tokenizer, token, download_dir)


if __name__ == "__main__":
    main()
