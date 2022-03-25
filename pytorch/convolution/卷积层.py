import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataCIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download= True)

dataloader = DataLoader(dataset, batch_size=64)

class Guchi(nn.Module):
    def __init__(self):
        super(Guchi,self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride = 1, padding=0)

    def forward(self,input):
        output = self.conv1(input)
        return output

guchi = Guchi()
writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = guchi(imgs)
    print(imgs.shape)
    print(output.shape)
    writer.add_images("input", imgs, step)
    #torchsize([64,6,30,30]) -> ([xxx,3,30,30])
    output = torch.reshape(output, (-1,3,30,30))
    writer.add_images("output", output, step)
    step+=1
writer.close()