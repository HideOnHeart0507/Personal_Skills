import torchvision
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import Linear

dataset = torchvision.datasets.CIFAR10("datarelu", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class Guchi(nn.Module):
    def __init__(self):
        super(Guchi,self).__init__()
        self.linear1= Linear(196608,10,False)

    def forward(self, input):
        output = self.linear1(input)
        return output

guchi = Guchi()
step = 0
writer = SummaryWriter("logs_linear")

for data in dataloader:
    imgs, targets= data
    writer.add_images("input", imgs, step)
    print(imgs.shape)
    output = torch.reshape(imgs,(1,1,1,-1))
    output = guchi(output)
    print(output.shape)
    writer.add_images("output", output, step)
    step +=1

writer.close()