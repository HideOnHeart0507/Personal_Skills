from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import models
import torchvision
from torch import nn
#
# traindata = datasets.ImageNet('data_imagenet', split="train", download=True, transform=torchvision.transforms.ToTensor())
vgg16 = models.vgg16(pretrained=False)
print("ok")
vgg16_t= models.vgg16(pretrained=True)
print("ok")

dataset = torchvision.datasets.CIFAR10("../dataCIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download= True)

vgg16_t.classifier.add_module('add_linear', nn.Linear(1000,10))
dataloader = DataLoader(dataset, batch_size=64)
vgg16.classifier[6] = nn.Linear(4096,10)