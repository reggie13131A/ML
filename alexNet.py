import torch
from torch import nn
import torchvision

# 使用sequential定义Alexnet, sequential是torch下的一个容器类，在里面的layer可以被叠罗汉一样叠起来
# 并且自动执行
AlexNet = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096),
    nn.ReLU(),
    # 在这里跟了一个dropout层，防止过拟合
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 10),
    # 不要这个softmax回归，加上了他会导致准确率只有0.1
    # nn.Softmax(dim=1)
)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    # 可以将这行代码揭开注释，数据被压缩在了1与-1之间，看看是什么效果
    # torchvision.transforms.Normalize((0.5,), (0.5,))
])

batch_size = 64
train_set = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST(root='./mnlist', train=True, transform=transform), batch_size=batch_size,
    shuffle=True)
test_set = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST(root='./mnlist', train=False, transform=transform), batch_size=batch_size,
    shuffle=True)


# 自己写一个训练函数
def train(net, train_set, test_set, epoch, lr, device):
    def init_weight(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weight)
    print('training on', device)
    # 将模型移动到device中去
    net.to(device)
    # 定义优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
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
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.2f}')


# 自己写一个accuracy函数,
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


lr, num_of_epoch, batch_size = 0.0005, 10, 64
train(AlexNet, train_set, test_set, num_of_epoch, lr, device='cuda' if torch.cuda.is_available() else 'cpu')