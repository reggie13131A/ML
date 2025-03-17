import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim  # 导入优化器模块

# 下载mnlist, 并封装成为函数, 返回经过dataloader处理的批次数据
def download_mnlist(batch_size):
    mnlist_train = torchvision.datasets.FashionMNIST(root='./mnlist', download=True, train=True, transform=transforms.ToTensor())
    mnlist_test = torchvision.datasets.FashionMNIST(root='./mnlist', download=True, train=False, transform=transforms.ToTensor())

    # 添加两个data_loader，来加载训练集与测试集
    train_set = torch.utils.data.DataLoader(dataset=mnlist_train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_set = torch.utils.data.DataLoader(dataset=mnlist_test, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_set, test_set

train_set, test_set = download_mnlist(256)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_input = 28 * 28
num_output = 10
num_hiddenUnit = 256
# 初始化参数
first_layer_parameters = nn.Parameter(torch.randn(num_input, num_hiddenUnit, requires_grad=True))
bias1 = nn.Parameter(torch.randn(num_hiddenUnit, requires_grad=True))
second_layer_parameters = nn.Parameter(torch.randn(num_hiddenUnit, num_output, requires_grad=True))
bias2 = nn.Parameter(torch.randn(num_output,requires_grad=True))

params = [first_layer_parameters, bias1, second_layer_parameters, bias2]

# 定义一个relu函数
def relu(x):
    a = torch.zeros_like(x)
    return torch.max(x, a)

# 多层感知机的详细实现,@符号是矩阵乘法，x是一个批量的图片，图片已经使用张量来进行表示，x的维度是[batch_size, 1, 28, 28]
# 跟前面softmax的网络不一样;
def net(x):
    # 第0层，类似于flatten
    layer_zero = x.reshape((-1, num_input))
    # 第1层，线性层，将784变成256
    layer_one = layer_zero @ first_layer_parameters + bias1
    # 第二层，非线性激活层
    h = relu(layer_one)
    layer_three = h @ second_layer_parameters + bias2
    return layer_three

# 多层感知机的简单实现
simple_net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_input, num_hiddenUnit),
                nn.ReLU(),
                nn.Linear(num_hiddenUnit, num_output)
            ).to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(params, lr=0.001)

# 训练函数
def train(model, train_set, epochs=20):
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (X, y) in enumerate(train_set):
            optimizer.zero_grad()
            if model == net:
                output = net(X)
            loss = loss_function(output, y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            running_loss += loss.item()
            if batch_idx % 100 == 99:  # 每100批次打印一次损失
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0
    print("Training finished!")

# 测试函数
def test(model, test_set):
    correct = 0
    total = 0
    with torch.no_grad():  # 不计算梯度
        for X, y in test_set:
            if model == net:
                output = net(X)
            else:
                output = model(X)
            _, predicted = torch.max(output.data, 1)  # 取最大值作为预测
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')

# 运行训练和测试
train(net, train_set)  # 使用手动实现的net训练
test(net, test_set)    # 使用手动实现的net测试