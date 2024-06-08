import torch
import torch.nn as nn
import numpy as np
from ..dataset.dataset import WikipediaDataset


class LSTM_Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        