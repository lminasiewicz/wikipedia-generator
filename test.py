import torch
from dataset.dataset import WikipediaDataset


x = torch.tensor([[1, 2, 3, 4], 
                  [5, 6, 7, 8], 
                  [9, 10, 11, 12]])

# y = torch.tensor(torch.arange(0, 50, dtype=torch.float))

z = (torch.triu(torch.ones(5, 5)) == 0).transpose(0, 1)
z = z.float().masked_fill(z, float("-inf"))

# print(x.unsqueeze(1))
# print(y.unsqueeze(2))
# print(y.unsqueeze(-3))

# print(torch.softmax(z, 0))

