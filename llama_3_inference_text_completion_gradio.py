import argparse
import logging
import random
from pathlib import Path

import gradio as gr
import torch

from llama_3_inference import Llama3Inference
from sae import load_sae_model
from utils.cuda_utils import set_torch_seed_for_inference


class Llama3GradioInterface:
    def __init__(
        self,
        tokenizer_path: Path,
        params_path: Path,
        model_path: Path,
        sae_model_path: Path = None,
        sae_layer_idx: int = None,
    ):
        """"""
        # Load the SAE model if provided and set up the forward fn for the specified sae_layer_idx
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sae_layer_forward_fn = None
        self.sae_model = None
        if sae_model_path is not None:
            assert sae_layer_idx is not None
            self.sae_model = load_sae_model(
                model_path=sae_model_path,
                sae_top_k=64,
                sae_normalization_eps=1e-6,
                device=self.device,
                dtype=torch.float32,
            )
            sae_layer_forward_fn = {sae_layer_idx: self.sae_model.forward}

        # Initialize the Llama3Inferenence generator
        self.llama_inference = Llama3Inference(
            tokenizer_path=tokenizer_path,
            params_path=params_path,
            model_path=model_path,
            device=self.device,
            sae_layer_forward_fn=sae_layer_forward_fn,
        )
        logging.info(f"Model initialized on device: {self.device}")

    def generate_completion(
        self,
        text: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        sae_h_bias_index: int | None = None,
        sae_h_bias_value: float | None = None,
    ) -> str:
        """"""
        # Input validation
        text = text.strip()
        if not text:
            return "Please enter some text to complete."

        # Log input parameters
        logging.info("Received `generate_completion` request with parameters:")
        logging.info(f"# text={text}")
        logging.info(f"# max_new_tokens={max_new_tokens}")
        logging.info(f"# temperature={temperature}")
        logging.info(f"# top_p={top_p}")
        logging.info(f"# seed={seed}")
        logging.info(f"# sae_h_bias_index={sae_h_bias_index}")
        logging.info(f"# sae_h_bias_value={sae_h_bias_value}")

        # Set random seed if seed set to 0:
        if seed == 0:
            seed = random.randint(1, 2**16 - 1)
        set_torch_seed_for_inference(seed)
        logging.info(f"Set generation seed to: {seed}")

        # Set SAE h_bias if provided
        if self.sae_model:
            if sae_h_bias_index >= 0 and sae_h_bias_value:
                logging.info("Setting SAE h_bias...")
                h_bias = torch.zeros(self.sae_model.n_latents)
                h_bias[sae_h_bias_index] = sae_h_bias_value
                h_bias = h_bias.to(torch.float32).to(self.device)
                self.sae_model.set_latent_bias(h_bias)
            else:
                self.sae_model.unset_latent_bias()

        # Generate text completions and print results iteratively
        text_prompts = text.split("\n")
        text_completions = [
            f"#### Text Completion {i + 1}: ####\n" for i in range(len(text_prompts))
        ]
        for next_tokens_text in self.llama_inference.generate_text_completions(
            prompts=text_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            # Update each completion with the new tokens (or initial minimal sequence) and print
            for i, new_token in enumerate(next_tokens_text):
                text_completions[i] += new_token
            output_str = "\n\n".join(text_completions)
            yield output_str

    def create_interface(self):
        """"""
        inputs = [
            gr.Textbox(
                label="Enter 1 prompt per line for parallel text completion",
                placeholder="Once upon a time, in a land far, far away\n"
                            "The quick brown fox jumps over\n"
                            "In the year 2050, technology had advanced to the point where\n"
                            "The secret to happiness is\n",
                lines=3,
            ),
            gr.Slider(
                minimum=1,
                maximum=512,
                value=128,
                step=1,
                label="Maximum new tokens",
            ),
            gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature",
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.1,
                label="Top-p",
            ),
            gr.Number(
                label="Seed (0 is random)",
                value=0,
                precision=0,
            ),
        ]
        if self.sae_model is not None:
            inputs.append(
                gr.Number(
                    label="SAE h_bias index",
                    value=0,
                    precision=0,
                ),
            )
            inputs.append(
                gr.Number(
                    label="SAE h_bias value",
                    value=0,
                ),
            )

        interface = gr.Interface(
            title="Llama 3 Text Completion",
            fn=self.generate_completion,
            inputs=inputs,
            outputs=gr.Textbox(
                label="Generated Text Completions",
                lines=30,
            ),
            flagging_mode="never",
        )
        return interface


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_model_dir", type=Path, required=True)
    parser.add_argument("--sae_model_path", type=Path, default=None)
    parser.add_argument("--sae_layer_idx", type=int, default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main():
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
        assert args.sae_layer_idx is not None, "sae_layer_idx must be specified when using SAE"

    logging.info("Initializing Llama3 Gradio Interface...")
    llama_interface = Llama3GradioInterface(
        tokenizer_path=llama_tokenizer_path,
        params_path=llama_params_path,
        model_path=llama_model_path,
        sae_model_path=args.sae_model_path,
        sae_layer_idx=args.sae_layer_idx,
    )
    interface = llama_interface.create_interface()

    logging.info("Launching Gradio interface...")
    interface.queue().launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0",
    )


if __name__ == "__main__":
    main()
