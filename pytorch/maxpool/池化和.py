import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.utils.tensorboard import SummaryWriter
dataset = torchvision.datasets.CIFAR10("datachi", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

'''input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]], dtype=torch.float32
                     )
input = torch.reshape(input, (-1,1,5,5))
print(input.shape)
'''


class Guchi(nn.Module):
    def __init__(self):
        super(Guchi,self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self,input):
        output = self.maxpool1(input)
        return output


guchi = Guchi()
step = 0
writer = SummaryWriter("logs_maxpool")

for data in dataloader:
    imgs, targets = data
    writer.add_images("input",imgs,step)
    output = guchi(imgs)
    writer.add_images("output", output, step)
    step += 1
#最大池化 保留大数， 减少数据量， 训练的更快
writer.close()