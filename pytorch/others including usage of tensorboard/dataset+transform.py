import torchvision
from torch.utils.tensorboard import SummaryWriter
from PIL import Image


dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
trainset =torchvision.datasets.CIFAR10(root="./dataset", train = True, transform=dataset_transform, download= True)
testset =torchvision.datasets.CIFAR10(root="./dataset", train = False, transform=dataset_transform, download= True)


writer = SummaryWriter("logs")
for i in range(10):
    img, target = testset[i]
    writer.add_image("dataset", img, i)
print(testset[0])
writer.close()