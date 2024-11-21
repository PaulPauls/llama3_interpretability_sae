import json
import logging
from collections.abc import Generator
from pathlib import Path

import torch

from llama_3.args import ModelArgs
from llama_3.chat_format import ChatFormat
from llama_3.datatypes import Message, StopReason
from llama_3.model_text_only import Transformer
from llama_3.tokenizer import Tokenizer


class Llama3Inference:
    def __init__(
        self,
        tokenizer_path: Path,
        params_path: Path,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        sae_layer_forward_fn: dict[int, callable] | None = None,
    ):
        """"""
        self.device = device

        # Load the tokenizer and LLama 3.1 Chat formatter
        self.tokenizer = Tokenizer(str(tokenizer_path))
        self.chat_format = ChatFormat(self.tokenizer)

        # Load the stop tokens
        self.stop_tokens = torch.tensor(self.tokenizer.stop_tokens, device=self.device)

        # Load the model parameters
        with open(params_path) as f:
            model_params = json.load(f)

        # Initialize model arguments
        model_args = ModelArgs(**model_params)
        assert (
            model_args.vocab_size == self.tokenizer.n_words
        ), "Model and tokenizer vocab sizes do not match"

        # Initialize the model on CPU, usually with bfloat16 data type
        logging.info("Initializing Llama model...")
        torch.set_default_dtype(dtype)
        self.model = Transformer(model_args, sae_layer_forward_fn=sae_layer_forward_fn)

        # Load the model weights into CPU memory
        logging.info("Loading Llama model weights into memory...")
        model_weights = torch.load(
            model_path,
            map_location=torch.device("cpu"),
            weights_only=True,
        )

        # Load the model weights into the model
        logging.info("Loading Llama model weights into model...")
        self.model.load_state_dict(model_weights)
        del model_weights

        # Move the model to the appropriate GPU
        logging.info(f"Llama Model weights loaded successfully. Moving model to {self.device}...")
        self.model.to(self.device)

        logging.info("Setting Llama model to eval mode...")
        self.model.eval()

        logging.info("Llama Model created successfully.")

    def generate_chat_completions(
        self,
        message_sequences: list[list[Message]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Generator[list[Message], None, None]:
        """"""
        # Prepare the input tokens with the chat format
        input_tokens = [
            self.chat_format.encode_dialog_prompt(messages).tokens for messages in message_sequences
        ]

        # Get the batch size and min/max/total sequence lengths
        batch_size = len(input_tokens)
        min_input_len = min(len(x) for x in input_tokens)
        max_input_len = max(len(x) for x in input_tokens)
        total_seq_len = max_input_len + max_new_tokens
        if total_seq_len > self.model.params.max_seq_len:
            raise ValueError(f"Total sequence length {total_seq_len} exceeds model's max_seq_len")

        # Initialize the tokens tensor to operate on with padding tokens and move input tokens to this state
        tokens = torch.full(
            (batch_size, total_seq_len),
            self.tokenizer.pad_id,
            dtype=torch.long,
            device=self.device,
        )
        for batch_i, in_t in enumerate(input_tokens):
            tokens[batch_i, : len(in_t)] = torch.tensor(in_t, dtype=torch.long, device=self.device)

        # Initialize variables relevant to generation loop and progress
        prev_pos = 0
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        stop_reasons = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        input_tokens_mask = tokens != self.tokenizer.pad_id

        with torch.no_grad():
            for cur_pos in range(min_input_len, total_seq_len):
                # Get the model outputs and logits for the next token
                outputs = self.model(tokens[:, prev_pos:cur_pos], start_pos=prev_pos)
                logits = outputs[:, -1, :]

                if temperature > 0:
                    # If temperature is given, apply temperature, then softmax and then top-p sampling
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_tokens = self.sample_top_p(probs, top_p).squeeze(-1)
                else:
                    # If temperature is not given, just get the most likely token
                    next_tokens = torch.argmax(logits, dim=-1)

                # Determine which tokens to update due to input mask or finished sequences, then update them
                next_tokens = torch.where(
                    (input_tokens_mask[:, cur_pos] | finished),
                    tokens[:, cur_pos],
                    next_tokens,
                )
                tokens[:, cur_pos] = next_tokens

                # Check for stop tokens for chat decoding and adapt finished state
                next_stop_tokens = torch.isin(next_tokens, self.stop_tokens)
                if next_stop_tokens.any():
                    stop_reasons[next_stop_tokens] = next_tokens[next_stop_tokens]
                    finished[next_stop_tokens] = True

                # Decode and yield the current state of each chat completion
                current_completions = []
                for i, seq in enumerate(tokens):
                    seq = seq[len(input_tokens[i]) : cur_pos + 1]
                    seq = seq[seq != self.tokenizer.pad_id]
                    stop_token = stop_reasons[i].item()
                    if stop_token == self.tokenizer.special_tokens["<|eot_id|>"]:
                        stop_reason = StopReason.end_of_turn
                    elif stop_token == self.tokenizer.special_tokens["<|eom_id|>"]:
                        stop_reason = StopReason.end_of_message
                    else:
                        stop_reason = StopReason.out_of_tokens
                    completion = self.chat_format.decode_assistant_message(
                        seq.tolist(),
                        stop_reason,
                    )
                    current_completions.append(completion)
                yield current_completions

                # Break if all sequences are finished
                if finished.all():
                    break

                # Move generation loop forward
                prev_pos = cur_pos

    def generate_text_completions(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Generator[list[str], None, None]:
        """"""
        # Prepare the input tokens with the tokenizer only
        input_tokens = [self.tokenizer.encode(prompt, bos=True, eos=False) for prompt in prompts]

        # Get the batch size and min/max/total sequence lengths
        batch_size = len(input_tokens)
        min_input_len = min(len(x) for x in input_tokens)
        max_input_len = max(len(x) for x in input_tokens)
        total_seq_len = max_input_len + max_new_tokens
        if total_seq_len > self.model.params.max_seq_len:
            raise ValueError(f"Total sequence length {total_seq_len} exceeds model's max_seq_len")

        # Initialize the tokens tensor to operate on with padding tokens and move input tokens to this state
        tokens = torch.full(
            (batch_size, total_seq_len),
            self.tokenizer.pad_id,
            dtype=torch.long,
            device=self.device,
        )
        for batch_i, in_t in enumerate(input_tokens):
            tokens[batch_i, : len(in_t)] = torch.tensor(in_t, dtype=torch.long, device=self.device)

        # Yield initial sequences up to min_input_len that'll be expanded upon
        initial_sequences = []
        for seq in tokens[:, 1:min_input_len]:
            text = self.tokenizer.decode(seq.tolist())
            initial_sequences.append(text)
        yield initial_sequences

        # Initialize variables relevant to generation loop and progress
        prev_pos = 0
        input_tokens_mask = tokens != self.tokenizer.pad_id

        with torch.no_grad():
            for cur_pos in range(min_input_len, total_seq_len):
                # Get the model outputs and logits for the next token
                outputs = self.model(tokens[:, prev_pos:cur_pos], start_pos=prev_pos)
                logits = outputs[:, -1, :]

                if temperature > 0:
                    # If temperature is given, apply temperature, then softmax and then top-p sampling
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_tokens = self.sample_top_p(probs, top_p).squeeze(-1)
                else:
                    # If temperature is not given, just get the most likely token
                    next_tokens = torch.argmax(logits, dim=-1)

                # Determine which tokens to update due to input mask and then update them
                next_tokens = torch.where(
                    (input_tokens_mask[:, cur_pos]),
                    tokens[:, cur_pos],
                    next_tokens,
                )
                tokens[:, cur_pos] = next_tokens

                # Decode the new generated next tokens and yield them
                next_tokens_text = []
                for token in next_tokens:
                    text = self.tokenizer.decode([token.item()])
                    next_tokens_text.append(text)
                yield next_tokens_text

                # Move generation loop forward
                prev_pos = cur_pos

    @staticmethod
    def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
        """Perform top-p (nucleus) sampling on a probability distribution."""
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token
