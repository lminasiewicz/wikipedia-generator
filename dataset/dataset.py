import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import numpy as np
import io


class WikipediaDataset(Dataset):
    def __init__(self, filepath: str = "./wikipedia.txt") -> None:
        with io.open(filepath, "r", encoding="utf-8") as wikipedia:
            raw_text = wikipedia.read()

        self.tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.articles = np.array(
            [np.array([self.tokenizer.tokenize(sub.replace("\n\n", "")) for sub in article.split("\n\n", 1)], dtype=object)
            for article in raw_text.split("\n" * 5)], dtype=np.ndarray
        )
        self.vocab_size = len(self.tokenizer)
        self.words = self.load_words()

    
    def load_words(self) -> None:
        pass




with io.open("./wikipedia.txt", "r", encoding="utf-8") as wikipedia:
    raw_text = wikipedia.read()

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
print(len(tokenizer))
