import torchvision
import torch
from torch import nn
from torch.nn import Conv2d, Flatten
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset  = torchvision.datasets.CIFAR10("anlidata", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class Guchi(nn.Module):
    def __init__(self):
        super(Guchi, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32,32,5, padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear = Linear(64*4*4, 64)
        self.linear2 = Linear(64,10)
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

guchi = Guchi()
print(guchi)
writer = SummaryWriter("anli")
step = 0
input = torch.ones((64,3,32,32))
output= guchi(input)
print(output.shape)
print(type(output))

for data in dataloader:
    imgs, targets = data
    writer.add_graph(guchi, imgs)

writer.close()