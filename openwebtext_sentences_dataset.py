import logging
from unittest.mock import Mock

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from llama_3.tokenizer import Tokenizer


class OpenWebTextSentencesDataset(Dataset):
    """"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_token_length: int,
        num_samples: int | None = None,
        shuffle: bool = False,
        add_bos_token: bool = False,
        seed: int = 42,
    ):
        """"""
        logging.info(
            "Initializing OpenWebText-Sentences Dataset with max_token_length=%s...",
            max_token_length,
        )
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.add_bos_token = add_bos_token

        # Load the OpenWebText dataset
        self.dataset = load_dataset("paulpauls/openwebtext-sentences", split="train")
        if shuffle:
            logging.info("Shuffling the dataset...")
            self.dataset = self.dataset.shuffle(seed=seed)

        # Slice the dataset if num_samples is specified
        if num_samples is not None:
            num_samples = min(num_samples, len(self.dataset))
            self.dataset = self.dataset.select(range(num_samples))
        logging.info(f"Dataset size: {len(self)} samples")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[list[int], int, int]:
        """"""
        # Get a single sentence
        sentence = self.dataset[idx]["text"]

        # Encode the sentence
        sentence_tokens = self.tokenizer.encode(sentence, bos=self.add_bos_token, eos=False)
        sentence_tokens = sentence_tokens[: self.max_token_length]

        # Calculate the actual sequence length
        seq_len = len(sentence_tokens)

        return sentence_tokens, idx, seq_len

    def collate_fn(
        self,
        batch: list[tuple[list[int], int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return type:
        torch.Tensor shape (batch_size, max_token_length)
        torch.Tensor shape (batch_size,)
        torch.Tensor shape (batch_size,)
        """
        max_seq_len = max([seq_len for _, _, seq_len in batch])

        padded_sentence_tokens = [
            sentence_tokens + [self.tokenizer.pad_id] * (max_seq_len - seq_len)
            for sentence_tokens, _, seq_len in batch
        ]

        collated_batch = (
            torch.tensor(padded_sentence_tokens, dtype=torch.long),
            torch.tensor([idx for _, idx, _ in batch]),
            torch.tensor([seq_len for _, _, seq_len in batch]),
        )

        return collated_batch


def mock_instantiate_dataset() -> None:
    """"""
    mock_tokenizer = Mock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    mock_tokenizer.pad_id = 0
    OpenWebTextSentencesDataset(
        tokenizer=mock_tokenizer,
        max_token_length=123,
        num_samples=None,
        add_bos_token=False,
    )


if __name__ == "__main__":
    mock_instantiate_dataset()
