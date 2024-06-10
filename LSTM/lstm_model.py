import torch
import torch.nn as nn
import numpy as np


class LSTM_Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        