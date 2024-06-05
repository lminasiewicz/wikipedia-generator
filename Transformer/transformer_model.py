import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable



class WikipediaGeneratorModel(nn.Transformer):

    def __init__(self, d_model: int = 512, head_count: int = 8, num_encoder_layers: int = 6, 
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1) -> None:
        
        super().__init__(d_model=d_model, nhead=head_count, num_encoder_layers=num_encoder_layers, 
                         num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        
        

        
