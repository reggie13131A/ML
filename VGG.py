import torch
from torch import nn
import torchvision
# from alexNet import train, Accumulator, evaluate_accuracy_gpu

# 定义一个VGG块的函数，VGG块可以进行重复调用堆叠，确实已经实现了自定义多少个VGG块，自定义一个VGG块有多少个卷积层，自定义一个卷积层有多宽
# 参数需要：多少个卷积层，输入通道数，输出通道数
def vgg_block(num_of_conv, in_channels, out_channels):
    # 遍历num of conv次，每次都增加一个卷积层，并且增加一个激活函数
    layers = []
    for i in range(num_of_conv):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        # 添加激活函数
        layers.append(nn.ReLU())
        # in_channels只是一个中间参数，用来传递到下一层，所以每次都要更新in_channels的值
        in_channels = out_channels
    # 一个VGG块添加一个池化层
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 1024))
# 构造VGG，根据上述的conv_arch来进行构造
def vgg(conv_arch):
    in_channels = 1
    conv_blocks = []
    # 遍历上述architecture，每次都增加一个VGG块,
    for (num_of_conv, out_channels) in conv_arch:
        # 每次增加一个VGG块
        conv_blocks.append(vgg_block(num_of_conv, in_channels, out_channels))
        # 这个是为了下一层的输入与本层的输出对齐
        in_channels = out_channels

    return nn.Sequential(
        *conv_blocks,
        nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)

# 观察形状
x= torch.randn(size=(1, 1, 224, 224))
for block in net:
    x = block(x)
    print(block.__class__.__name__, 'output shape:\t', x.shape)


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize((0.5,), (0.5,))
])

# 使用torch下的dataloader模块来加载数据，并且让batch_size是64
batch_size = 64
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


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __getitem__(self, idx):
        return self.data[idx]

def train(net, train_set, test_set, epoch, lr, device):
    # 初始化参数，Xavier初始化方法是比较适配于线性层以及卷积层的
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

lr, num_of_epoch, batch_size = 0.05, 10, 64
train(net, train_set, test_set, num_of_epoch, lr, device = 'cuda' if torch.cuda.is_available() else 'cpu')