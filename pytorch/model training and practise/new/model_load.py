from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import models
import torchvision
from torch import nn
import torch
#
# traindata = datasets.ImageNet('data_imagenet', split="train", download=True, transform=torchvision.transforms.ToTensor())
torch.load("vgg16_method1.pth")

vgg16= torchvision.models.vgg16(pretained = False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))

