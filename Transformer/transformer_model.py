import torch
import torch.nn as nn
from postitional_encoding import PositionalEncoder



class WikipediaGeneratorModel(nn.Module):

    def __init__(self, token_count: int, d_model: int = 512, head_count: int = 8, num_decoder_layers: int = 6, 
                 dropout: float = 0.1, max_len: int = 256) -> None:
        
        super().__init__()
        self.dimensions = d_model
        self.max_len = max_len

        # Modules
        self.positional_encoder = PositionalEncoder(d_model=d_model, dropout=dropout, max_len=max_len)
        self.embedder = nn.Embedding(token_count, d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=head_count)
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=num_decoder_layers)
        self.out = nn.Linear(d_model, token_count)
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, data) -> torch.Tensor:
        if data.size(0) > self.max_len - 1:
            data = data[:self.max_len - 1]
        mask_size = data.size(0)

        # input embedding
        data = self.embedder(data)

        # generate masked attention mask
        mask = self.generate_mask(mask_size).to(data.device)

        # encode positional embedding, decode with transformer decoder, and final linear transformation
        data = self.positional_encoder(data)
        data = self.decoder(data, memory=data, tgt_mask=mask, memory_mask=mask)
        data = self.out(self.dropout(data))
        return data
    

    def generate_mask(self, size: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(size, size)) == 0).transpose(0, 1)
        mask = mask.float().masked_fill(mask, float("-inf"))
        return mask
        

        


