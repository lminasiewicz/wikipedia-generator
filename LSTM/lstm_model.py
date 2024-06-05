import torch
import torch.nn as nn
import numpy as np
from ..dataset.dataset import WikipediaDataset


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LSTM_Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        