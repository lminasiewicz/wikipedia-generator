from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
import datasets 
from functools import partial


TOKENIZER_CHECKPOINT = "bert-base-uncased"


def get_dataloader(filepath: str, batch_size: int = 32) -> DataLoader:
    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_CHECKPOINT)
    wikipedia = datasets.load_dataset("text", data_files=filepath, split="train")

    tokenize = lambda x, tokenizer: tokenizer(x["text"], return_tensors="pt", padding=True, truncation=True)
    tokenize_partial: partial = partial(tokenize, tokenizer=tokenizer)
    wikipedia_encoded = wikipedia.map(tokenize_partial, batched=True, num_proc=16)

    wikipedia_encoded.set_format(type='torch', columns=['token_type_ids'])
    dataloader = DataLoader(wikipedia_encoded, batch_size=32)

    print(f"Dataset {filepath} loaded successfully.")
    return dataloader