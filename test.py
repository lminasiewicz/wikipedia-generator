import torch


x = torch.tensor([[1, 2, 3, 4], 
                  [5, 6, 7, 8], 
                  [9, 10, 11, 12]])

y = torch.tensor(torch.arange(0, 50, dtype=torch.float))

print(x.unsqueeze(1))
# print(y.unsqueeze(2))
# print(y.unsqueeze(-3))