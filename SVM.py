import torch
import torchvision
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

batch_size = 1
train_set = torch.utils.data.DataLoader(torchvision.datasets.FashionMNIST(root='./mnlist', train=True, transform=transform), batch_size=batch_size, shuffle=True)
test_set = torch.utils.data.DataLoader(torchvision.datasets.FashionMNIST(root='./mnlist', train=False, transform=transform), batch_size=batch_size, shuffle=True)

# 提取训练集和测试集数据
def extract_data(dataset):
    X, y = [], []
    for images, labels in dataset:
        X.append(images.view(images.size(0), -1).numpy())  # 展平图像
        y.append(labels.numpy())
    return np.concatenate(X), np.concatenate(y)

X_train, y_train = extract_data(train_set)
X_test, y_test = extract_data(test_set)

# 应为(60000, 28*28)
print("训练集形状:", X_train.shape)
# 应为(10000, 28*28)
print("测试集形状:", X_test.shape)

# 创建 SVM 模型
svm_model = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("测试集准确率:", accuracy)