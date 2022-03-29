import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *

train_data = torchvision.datasets.CIFAR10(root='data', train=True, transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10(root='data', train=False, transform=torchvision.transforms.ToTensor(),download=True)
train_data_size=len(train_data)
test_data_size=len(test_data)
print("length:{}".format(train_data_size))

dataloader_train = DataLoader(train_data, batch_size=64)
dataloader_test = DataLoader(test_data, batch_size=64)

class Guchi(nn.Module):
    def __init__(self):
        super(Guchi, self).__init__()
        self.model1 = nn.Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(64 * 4 * 4, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

# 创建网络模型
guchi = Guchi()
if torch.cuda.is_available():
    guchi = guchi.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
# 优化器
learningrate= 1e-2
optimizer = torch.optim.SGD(guchi.parameters(),lr=learningrate)

# 设置训练的参数
total_train_Step = 0
total_test_Step = 0
epoch = 10
writer = SummaryWriter("logs_train")

for i in range(epoch):
    print("第{}轮训练开始".format(i+1))
    guchi.train()
    for data in dataloader_train:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        output = guchi(imgs)
        loss = loss_fn(output, targets)
        # 优化器调用优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_Step+=1
        if total_train_Step %100 == 0:
            print("training rounds: {}, loss ={}".format(total_train_Step, loss.item()))
            writer.add_scalar("train_loss",loss.item(), total_train_Step)

    # 测试步骤
    guchi.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in dataloader_test:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            output = guchi(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss
            accuracy = (output.argmax(1)==targets).sum()
            total_accuracy += accuracy
    print("total lost: {}".format(total_test_loss))
    print("total accuracy: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_Step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_Step)
    total_test_Step+=1
    torch.save(guchi, "guchi_{}.pth".format(i))

writer.close()