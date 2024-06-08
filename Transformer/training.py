import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {DEVICE}\n")