from transformers import BertTokenizerFast
import datasets 
from functools import partial


TOKENIZER_CHECKPOINT = "bert-base-uncased"
MAX_LENGTH = 128


def get_dataset(filepath: str) -> tuple[datasets.Dataset, int]:
    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_CHECKPOINT)
    wikipedia = datasets.load_dataset("text", data_files=filepath, split="train")

    tokenize = lambda x, tokenizer: tokenizer(x["text"], return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    tokenize_partial: partial = partial(tokenize, tokenizer=tokenizer)
    wikipedia_encoded = wikipedia.map(tokenize_partial, batched=True, num_proc=16)
    return wikipedia_encoded, len(tokenizer)