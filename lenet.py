import torch
import torchvision
from torch import nn

class Reshape(nn.Module):
    def forward(self, x):
        # 这里又是一个转维度的操作, batch_size, color_channel, height, weight
        return x.view(-1,1,28,28)

LeNet = nn.Sequential(
    Reshape(),
    # 输入1通道，输出6通道
    nn.Conv2d(1,6,kernel_size=5,padding=2),
    # 激活函数
    nn.ReLU(),
    nn.AvgPool2d(2, stride=2),
    nn.Conv2d(6,16,kernel_size=5),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2,stride=2),
    # 最后图片变成了5*5的形式，展开以后就是16*5*5
    nn.Flatten(),
    nn.Linear(16*5*5,120),
    nn.ReLU(),
    nn.Linear(120,84),
    nn.ReLU(),
    nn.Linear(84,10)
)

X = torch.randn(size=(1,1,28,28), dtype=torch.float)
for lay in LeNet:
    X = lay(X)
    # 你其实可以看到，他最后的几个线性层没有说一下子讲的特别低，降低的很缓和的
    print(lay.__class__.__name__,'output shape: \t',X.shape)

def evaluate_accuracy(data_iter, net, device):
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

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

batch_size = 64
train_set = torch.utils.data.DataLoader(torchvision.datasets.FashionMNIST(root='./mnlist', train=True, transform=transform), batch_size=batch_size, shuffle=True)
test_set = torch.utils.data.DataLoader(torchvision.datasets.FashionMNIST(root='./mnlist', train=False, transform=transform), batch_size=batch_size, shuffle=True)

# 打印一下，方便看看模型长啥样
for X, y in train_set:
    print(X.shape, X.min(), X.max(), y.shape, y.min(), y.max())
    break

def train(net, train_iter, test_iter, num_of_epoch, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            # xavier_uniform初始化，是一种均匀分布初始化而避免产生梯度爆炸的方法，m是你的模型，m.weight是你模型的参数
            nn.init.xavier_uniform_(m.weight)
    # net,是你模型的net，net.apply，是将某个函数递归调用到模型的所有子模块去；
    # 这里的net.apply，就是为所有的参数调用一次初始化函数
    net.apply(init_weights)
    print('training on', device)
    # 将模型挪到device的内存中去；这里的device就是你的GPU
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 多酚类问题
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_of_epoch):
        net.train()
        total_loss = 0.0
        for X, y in train_iter:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            total_loss += l.item()
        avg_loss = total_loss / len(train_iter)
        test_accuracy = evaluate_accuracy(test_iter, net, device)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.2f}')

# 这里要将学习率调小，因为太大了会导致梯度消失
lr, num_of_epoch, batch_size = 0.001, 10, 64
train(LeNet, train_set, test_set, num_of_epoch, lr, device = 'cuda' if torch.cuda.is_available() else 'cpu')