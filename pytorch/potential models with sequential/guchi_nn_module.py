import torch
from torch import nn

class Guchi(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input +1
        return output

guchi = Guchi()
input = torch.tensor(1.0)
output = guchi(input)

print(output)