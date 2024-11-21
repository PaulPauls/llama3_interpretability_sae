import argparse
import logging
import os
from pathlib import Path

import anthropic
import yaml
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request
from datasets import load_dataset
from tqdm import tqdm


class ClaudeSemanticFinder:
    def __init__(self, model: str, max_tokens: int, batch_size: int):
        """"""
        self.client = anthropic.Anthropic()
        self.prompt_template = self.get_claude_prompt_template()
        self.model = model
        self.max_tokens = max_tokens
        self.batch_size = batch_size

        self.requests = []
        self.requests_tokens = 0

    def add_request(self, custom_id: str, sentences: list[str]) -> None:
        """"""
        concatenated_sentences = ""
        for s_idx, s in enumerate(sentences):
            s = s.replace("\n", " ")
            concatenated_sentences += f"{s_idx + 1}. {s}\n"
        content = self.prompt_template.format(concatenated_sentences)
        messages = [
            {
                "role": "user",
                "content": content,
            },
        ]
        self.requests_tokens += self.calculate_messages_tokens(messages)
        self.requests.append(
            Request(
                custom_id=custom_id,
                params=MessageCreateParamsNonStreaming(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=messages,
                ),
            ),
        )

    def calculate_messages_tokens(self, messages: list[dict[str, str]]) -> int:
        """"""
        token_count = 0
        for msg in messages:
            continous_string = ""
            for k, v in msg.items():
                continous_string += v + "\n" + k + "\n"
            token_count += self.client.count_tokens(continous_string)
        return token_count

    def send_batches(self) -> tuple[list[str], int]:
        """"""
        batched_requests = [
            self.requests[i : i + self.batch_size]
            for i in range(0, len(self.requests), self.batch_size)
        ]
        msg_batch_response_ids = []
        for batch in tqdm(batched_requests, desc="Sending batches to Claude API"):
            msg_batch_response = self.client.beta.messages.batches.create(requests=batch)
            msg_batch_response_ids.append(msg_batch_response.id)

        request_tokens_sent = self.requests_tokens
        self.requests = 0
        self.requests_tokens = 0

        return msg_batch_response_ids, request_tokens_sent

    @staticmethod
    def get_claude_prompt_template() -> str:
        """"""
        prompt = """
You are a semantic analysis expert tasked with identifying common elements across a set of sentences. Your goal is to find a shared semantic topic or peculiarity and assess your certainty about this commonality.

Here is the list of sentences you need to analyze:

<sentences>
{}</sentences>

Please follow these steps to complete your analysis:

1. Carefully read and analyze all the provided sentences.

2. In your semantic analysis, follow these sub-steps:
   a. List key words or phrases from each sentence
   b. Group these elements into potential themes
   c. Consider and list any potential exceptions or outliers

3. Identify a common semantic topic, theme, or peculiarity shared among all or most of the sentences. This could relate to subject matter, writing style, grammatical structure, or any other linguistic or thematic element.

4. Assess your level of certainty about the common element you've identified. Consider:
   - How many sentences exhibit this commonality
   - How strong or obvious the connection is
   - Whether there are any outliers or exceptions

5. Calculate your certainty score:
   - Count the number of sentences that fit the common element
   - Assess the strength of the connection (weak, moderate, strong)
   - Convert this to a score between 0 and 1

6. Use <semantic_analysis> tags to show your thought process. Break down your observations, considerations, and reasoning.

7. After your analysis, prepare a JSON response with two fields:
   - "common_semantic": A string describing the common semantic element you've identified.
   - "certainty": A float between 0 and 1, where 0 represents complete uncertainty and 1 represents absolute certainty.

8. Output your response as a JSON object only, without any additional text or tags.

Here's an example of how your output should be structured (note that this is a generic example and your actual output should reflect your analysis of the provided sentences):

{{
  "common_semantic": "Description of the identified common element",
  "certainty": 0.75
}}

Remember:
- Consider all sentences in your analysis.
- The common element doesn't need to apply to every single sentence if there are clear outliers.
- Your certainty score should reflect the strength of the commonality and any exceptions.
- Provide a concise yet informative description in the "common_semantic" field.

Begin your analysis now:
        """
        return prompt.strip()


def parse_arguments() -> argparse.Namespace:
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_sentences_dict_filepath", type=Path, required=True)
    parser.add_argument("--response_ids_filepath", type=Path, required=True)
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
    args.top_sentences_dict_filepath = args.top_sentences_dict_filepath.resolve()
    args.response_ids_filepath = args.response_ids_filepath.resolve()

    # Set up configuration
    model = "claude-3-5-sonnet-20241022"
    max_tokens = 4096
    batch_size = 500
    samples_to_submit = 50
    dataset_shuffle = True
    shuffle_seed = 42
    assert os.getenv("ANTHROPIC_API_KEY"), "Please set ANTHROPIC_API_KEY environment variable"

    logging.info("#### Starting script to interpret top activating sentences in batches")
    logging.info("#### Arguments:")
    logging.info(f"# top_sentences_dict_filepath: {args.top_sentences_dict_filepath}")
    logging.info(f"# response_ids_filepath: {args.response_ids_filepath}")
    logging.info("#### Configuration:")
    logging.info(f"# model: {model}")
    logging.info(f"# max_tokens: {max_tokens}")
    logging.info(f"# batch_size: {batch_size}")
    logging.info(f"# samples_to_submit: {samples_to_submit}")
    logging.info(f"# dataset_shuffle: {dataset_shuffle}")
    logging.info(f"# shuffle_seed: {shuffle_seed}")

    logging.info(f"Initializing Claude Semantic Finder for model {model}...")
    semantic_finder = ClaudeSemanticFinder(
        model=model,
        max_tokens=max_tokens,
        batch_size=batch_size,
    )

    logging.info("Loading OpenWebText-Sentences dataset...")
    dataset = load_dataset("paulpauls/openwebtext-sentences", split="train")
    if dataset_shuffle:
        logging.info("Shuffling the dataset...")
        dataset = dataset.shuffle(seed=shuffle_seed)

    logging.info("Loading top activating sentences...")
    with args.top_sentences_dict_filepath.open("r") as f:
        top_sentences_dict = yaml.safe_load(f)

    logging.info("Processing top activating sentences and building requests...")
    for latent_idx, top_sentence_tuples in tqdm(top_sentences_dict.items()):
        # Skip if there are not enough samples to submit for interpretation
        if len(top_sentence_tuples) < samples_to_submit:
            continue
        # Select only the highest activated samples
        top_sentence_tuples = top_sentence_tuples[-samples_to_submit:]

        # Get the sentences for the top activating samples and add a request
        top_sentences = [
            dataset[int(dataset_idx)]["text"] for _, dataset_idx in top_sentence_tuples
        ]
        semantic_finder.add_request(
            custom_id=str(latent_idx),
            sentences=top_sentences,
        )
    logging.info(
        f"Built {len(semantic_finder.requests)} requests with {semantic_finder.requests_tokens} tokens",
    )

    logging.info("Sending batches to Claude API...")
    msg_batch_response_ids, _ = semantic_finder.send_batches()
    logging.info(f"Sent {len(msg_batch_response_ids)} batches to Claude API")

    logging.info("Saving response IDs to file for future retrieval...")
    with args.response_ids_filepath.open("w") as f:
        yaml.dump(msg_batch_response_ids, f)
    logging.info(f"Saved response IDs to {args.response_ids_filepath}")
    logging.info("FIN")


if __name__ == "__main__":
    main()
