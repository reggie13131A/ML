import torch
from torch import nn
from torch.nn import functional as F
import torchvision

# 定义一个inception类，有5参数，in_channel, c1,c2,c3,c4这些都是元组，为了本卷积层输入与另一个卷积层输出相同
class Inception(nn.Module):
    def __init__(self, in_channel, c1, c2, c3, c4, **kwargs):
        # PyTorch 要求在定义子模块之前，先调用 super().__init__() 来完成父类 nn.Module 的初始化。
        super(Inception, self).__init__()
        # 第一条路径是1*1的卷积核,c1对应着的是1*1卷积核的输出通道数
        self.path1 = nn.Conv2d(in_channel, c1, kernel_size=1)
        # 第二条路径是1*1 + 3*3，c2[0]对应着1*1卷积核的输出通道数，c2[1]对应着3*3卷积核的输出通道数
        self.path2_1 = nn.Conv2d(in_channel, c2[0], kernel_size=1)
        self.path2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 第三条路径是1*1， 5*5
        self.path3_1 = nn.Conv2d(in_channel, c3[0], kernel_size=1)
        self.path3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 第四条路径是一个3*3的maxPooling，再加上一个1*1的卷积核，c4[0]对应着1*1卷积核的输出通道数，c4[1]对应着3*3卷积核的输出通道数
        # 因为池化层是不会改变输出的通道数的，输入多少就输出多少，所以这里不要指定输出通道数
        self.path4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.path4_2 = nn.Conv2d(in_channel, c4, kernel_size=1)

    def forward(self, x):
        # 第一层的输出：
        p1 = F.relu(self.path1(x))
        # 第二层，x输入进路径2，加一个relu，再输入进2_2，再加一个relu
        p2 = F.relu(self.path2_2(F.relu(self.path2_1(x))))
        p3 = F.relu(self.path3_2(F.relu(self.path3_1(x))))
        p4 = F.relu(self.path4_2(self.path4_1(x)))
        # 在输出通道数的那个维度，把四个concat起来，输出通道数的维度是1，所以dim=1
        return torch.cat((p1, p2, p3, p4), dim=1)

# 定义每一个GoogleNet的block
b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # 输入为 RGB 图像 (3 通道)
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 192, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b3 = nn.Sequential(
        Inception(192, 64, (96, 128), (16, 32), 32),
        Inception(256, 128, (128, 192), (32, 96), 64),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b4 = nn.Sequential(
        Inception(480, 192, (96, 208), (16, 48), 64),
        Inception(512, 160, (112, 224), (24, 64), 64),
        Inception(512, 128, (128, 256), (24, 64), 64),
        Inception(512, 112, (144, 288), (32, 64), 64),
        Inception(528, 256, (160, 320), (32, 128), 128),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b5 = nn.Sequential(
        Inception(832, 256, (160, 320), (32, 128), 128),
        Inception(832, 384, (192, 384), (48, 128), 128),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten())
net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

x = torch.randn(1, 1, 96, 96)
for layer in net:
    x = layer(x)
    print(layer.__class__.__name__, 'output shape:\t', x.shape)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize((0.5,), (0.5,))
])

# 定义batch_size
batch_size = 64
# 使用torch下的dataloader来进行数据的加载
train_set = torch.utils.data.DataLoader(torchvision.datasets.FashionMNIST(root='./mnlist', train=True, transform=transform), batch_size=batch_size, shuffle=True)
test_set = torch.utils.data.DataLoader(torchvision.datasets.FashionMNIST(root='./mnlist', train=False, transform=transform), batch_size=batch_size, shuffle=True)

def evaluate_accuracy_gpu(net, data_iter, device):
    net.eval()  # 设置为评估模式
    correct, total = 0, 0
    with torch.no_grad():  # 禁用梯度计算
        # x是图片，y是标签
        for X, y in data_iter:
            # 将X，y都挪到device里面
            X, y = X.to(device), y.to(device)
            # y_hat是图片X输入到网络里面的输出
            y_hat = net(X)
            _, predicted = torch.max(y_hat, 1)  # 获取预测类别
            correct += (predicted == y).sum().item()  # 统计正确预测数量
            total += y.size(0)  # 统计总样本数量
    net.train()  # 恢复为训练模式
    return correct / total  # 返回准确率

# 一个计数类
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __getitem__(self, idx):
        return self.data[idx]

def train(net, train_set, test_set, epoch, lr, device):
    def init_weight(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weight)
    print('training on', device)
    # 将模型移动到device中去
    net.to(device)
    # 定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # 使用损失函数
    loss = nn.CrossEntropyLoss()
    # 在epoch中使用训练
    for i in range(epoch):
        # 在每一轮训练中，需要将模型设置为训练模式
        net.train()
        # 定义每一轮训练的总损失值
        total_loss = 0
        for data in train_set:
            img, label = data
            # 移动到设备中, 一定要这样子！
            img = img.to(device)
            label = label.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 输出结果
            y_hat = net(img)
            # 计算损失, 请注意变量名不要跟函数的名字相同
            ls = loss(y_hat, label)
            # 反向传播
            ls.backward()
            optimizer.step()
            # 计算总体损失
            total_loss = total_loss + ls.item()
        avg_loss = total_loss / len(train_set)
        # 每一轮epoch都需要进行一次测试
        test_accuracy = evaluate_accuracy_gpu(net, test_set, device)
        print(f'Epoch {i + 1}, Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.2f}')

# 0.1的学习率也能让网络拥有90%的准确率
lr, num_of_epoch, batch_size = 0.1, 10, 64
# 调用train函数，并且让模型在GPU上训练
train(net, train_set, test_set, num_of_epoch, lr, device = 'cuda' if torch.cuda.is_available() else 'cpu')

