from torch import nn
from torch.nn import Conv2d, MaxPool2d,Flatten,Linear,Sequential
import torch

class Guchi(nn.Module):
    def __init__(self):
        super(Guchi, self).__init__()
        self.model1 = nn.Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(64 * 4 * 4, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

if __name__ == '__main__':
    guchi = Guchi()
    input = torch.ones((64,3,32,32))
    output = guchi(input)
    print(output.shape)