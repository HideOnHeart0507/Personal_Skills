from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import models
import torchvision
from torch import nn
import torch
#
# traindata = datasets.ImageNet('data_imagenet', split="train", download=True, transform=torchvision.transforms.ToTensor())
vgg16 = models.vgg16(pretrained=False)
print("ok")
torch.save(vgg16,'vgg16_method1.pth')

torch.save(vgg16.state_dict(), "vgg16_method2.pth")
# 官方推荐
