import torch
from torch import nn
from torch.nn import ReLU
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Sigmoid
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("datarelu", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

input = torch.tensor([[1,-0.5],[-1,3]])
output = torch.reshape(input,(-1,1,2,2))
print(output.shape)

class Guchi(nn.Module):
    def __init__(self):
        super(Guchi,self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1= Sigmoid()

    def forward(self, input):
        output = self.relu1(input)
        return output

guchi = Guchi()
output = guchi(input)
print(output.shape)
#f非线性， 模型放大能力， 不同非线性公式

writer = SummaryWriter("logs_relu")
step = 0
for data in dataloader:
    imgs, dataset= data
    writer.add_images("input", imgs, step)
    outputs = guchi(imgs)
    writer.add_images("output", outputs,step)
    step+= 1
writer.close()