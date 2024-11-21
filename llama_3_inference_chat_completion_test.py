import argparse
import logging
import os
from pathlib import Path

import torch

from llama_3.datatypes import SystemMessage, UserMessage
from llama_3_inference import Llama3Inference
from sae import load_sae_model
from utils.cuda_utils import set_torch_seed_for_inference


def parse_arguments() -> argparse.Namespace:
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_model_dir", type=Path, required=True)
    parser.add_argument("--sae_model_path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Parse arguments and set up paths
    args = parse_arguments()
    args.llama_model_dir = args.llama_model_dir.resolve()
    llama_tokenizer_path = args.llama_model_dir / "tokenizer.model"
    llama_params_path = args.llama_model_dir / "params.json"
    llama_model_path = args.llama_model_dir / "consolidated.00.pth"
    if args.sae_model_path is not None:
        args.sae_model_path = args.sae_model_path.resolve()

    # Set up configuration
    max_new_tokens = 128
    temperature = 0.7
    top_p = 0.9
    seed = 42
    sae_layer_idx = None
    sae_h_bias = None
    sae_top_k = 64
    sae_normalization_eps = 1e-6
    sae_dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("#### Starting sample Llama3 Chat Completion")
    logging.info("#### Arguments:")
    logging.info(f"# llama_model_dir={args.llama_model_dir}")
    logging.info(f"# sae_model_path={args.sae_model_path}")
    logging.info("#### Configuration:")
    logging.info(f"# max_new_tokens={max_new_tokens}")
    logging.info(f"# temperature={temperature}")
    logging.info(f"# top_p={top_p}")
    logging.info(f"# seed={seed}")
    logging.info(f"# sae_layer_idx={sae_layer_idx}")
    logging.info(f"# sae_h_bias={sae_h_bias}")
    logging.info(f"# sae_top_k={sae_top_k}")
    logging.info(f"# sae_normalization_eps={sae_normalization_eps}")
    logging.info(f"# sae_dtype={sae_dtype}")
    logging.info(f"# device={device}")

    # Set up CUDA and seed for inference
    set_torch_seed_for_inference(seed)

    # Load the SAE model if provided and set up the forward fn for the specified sae_layer_idx
    sae_layer_forward_fn = None
    if args.sae_model_path is not None:
        assert sae_layer_idx is not None
        sae_model = load_sae_model(
            model_path=args.sae_model_path,
            sae_top_k=sae_top_k,
            sae_normalization_eps=sae_normalization_eps,
            device=device,
            dtype=sae_dtype,
        )
        sae_layer_forward_fn = {sae_layer_idx: sae_model.forward}

        if sae_h_bias is not None:
            logging.info("Setting SAE h_bias...")
            h_bias = torch.zeros(sae_model.n_latents)
            h_bias[sae_h_bias[0]] = sae_h_bias[1]
            h_bias = h_bias.to(sae_dtype).to(device)
            sae_model.set_latent_bias(h_bias)

    # Initialize the Llama3Inferenence generator
    llama_inference = Llama3Inference(
        tokenizer_path=llama_tokenizer_path,
        params_path=llama_params_path,
        model_path=llama_model_path,
        device=device,
        sae_layer_forward_fn=sae_layer_forward_fn,
    )

    # Prepare batch for chat completion
    logging.info("Generating sample chat completions...")
    system_message = SystemMessage(
        content="You are a pirate chatbot who always responds in pirate speak!",
    )
    user_message_1 = UserMessage(content="Who are you?")
    user_message_2 = UserMessage(content="What is your purpose?")
    user_message_3 = UserMessage(content="Where are you from?")
    user_message_4 = UserMessage(content="What is your favorite color?")
    message_sequences = [
        [system_message, user_message_1],
        [system_message, user_message_2],
        [system_message, user_message_3],
        [system_message, user_message_4],
    ]

    # Generate chat completions and print results iteratively
    for next_messages in llama_inference.generate_chat_completions(
        message_sequences=message_sequences,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    ):
        # Clear the console for a more 'commercial LLM web UI' feel
        os.system("clear")

        # Update each completion with the new message content and print
        for i, message in enumerate(next_messages):
            print(f"#### Chat Completion {i + 1}: ".ljust(80, "#"))
            print(message.content)
            print("#" * 80)

    logging.info("#### FIN!")


if __name__ == "__main__":
    main()
