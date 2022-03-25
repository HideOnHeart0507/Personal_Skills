import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor()) # test

test_loader = DataLoader(dataset= test_data, batch_size=64, shuffle=True, num_workers=0, drop_last= False)



#test dataset[0] and target
img,target = test_data[0]
print(img.shape)
print(target)

step = 0
writer = SummaryWriter("dataloader")
for epoch in range(2):
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("epoch :{}".format(epoch), imgs, step)
        step += 1

writer.close()