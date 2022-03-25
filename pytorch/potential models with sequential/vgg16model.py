
import torch
from torch import nn
from torch.nn import Conv2d, Flatten
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.utils.tensorboard import SummaryWriter
from torch.nn import ReLU

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.model1 = nn.Sequential(
            Conv2d(3, 64, 3, padding=1),
            MaxPool2d(2),
            Conv2d(64, 128, 3, padding=1),
            Conv2d(128, 128, 3, padding=1),
            MaxPool2d(2),
            Conv2d(128, 256, 3, padding=1),
            Conv2d(256, 256, 3, padding=1),
            Conv2d(256, 256, 3, padding=1),
            MaxPool2d(2),
            Conv2d(256, 512, 3, padding=1),
            Conv2d(512, 512, 3, padding=1),
            Conv2d(512, 512, 3, padding=1),
            MaxPool2d(2),
            Conv2d(512, 512, 3, padding=1),
            Conv2d(512, 512, 3, padding=1),
            Conv2d(512, 512, 3, padding=1),
            MaxPool2d(2),
            Flatten(),
            Linear(25088, 4096),
            Linear(4096, 4096),
            Linear(4096, 1000)
        )

    def forward(self, input):
        output = self.model1(input)
        return output

# writer = SummaryWriter("vgg16")
guchi = VGG16()
print(guchi)
input = torch.ones((64, 3, 224, 224))
output = guchi(input)
print(output.shape)
print(type(output))

output = torch.reshape(output,(64,1,1,-1))
# writer.add_graph(guchi, input)
# writer.close()
