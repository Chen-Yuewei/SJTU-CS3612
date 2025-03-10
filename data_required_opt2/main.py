import os
import numpy as np
from dataset import get_data, normalize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt




# 加载FashionMNIST数据集并进行预处理
def preprocess(X_train, X_test, Y_train, Y_test, batchSize):
    x = torch.from_numpy(X_train)
    y = torch.from_numpy(Y_train)
    y = torch.squeeze(y, dim=-1)
    trainSet = TensorDataset(x, y)

    a = torch.from_numpy(X_test)
    b = torch.from_numpy(Y_test)
    b = torch.squeeze(b, dim=-1)
    testSet = TensorDataset(a, b)

    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # trainSet = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    # testSet = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batchSize, shuffle=True)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=batchSize, shuffle=False)

    return trainLoader, testLoader

# 构建LeNet模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d((2, 2), stride=2)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d((2, 2), stride=2)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()

        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)       
        a = x.view(x.size(0), -1)       # a用作后续可视化
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        b = x.view(x.size(0), -1)       # b用作后续可视化
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)

        return x, a, b

# 定义损失函数和优化器，并进行训练
def train(trainLoader, drawTrainingLoss=False, drawTrainingAccuracy=False, testSameTime=False):
    """
    :param trainLoader: 预处理好的训练数据集
    :param drawTrainingLoss: 是否绘制随训练进程变化的training loss，默认为False（不绘制）
    :param drawTrainingAccuracy: 是否绘制随训练进程变化的training accuracy，默认为False（不绘制）
    :param testSameTime: 是否随训练进程同步进行测试，并绘制testing loss & testing accuracy，默认为False（不绘制）
    :return: 返回训练好的网络
    """

    if drawTrainingLoss:
        lossList = []
    if drawTrainingAccuracy:
        accuracyList = []
    if testSameTime:
        testLossList = []
        testAccuracyList = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()
    epochs = 50
    for epoch in range(epochs):
        running_loss = 0.0
        if drawTrainingAccuracy:
            correct = 0.0            
        for i, data in enumerate(trainLoader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs, _, _ = net(inputs)
            # print(outputs.shape)
            # print(labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if drawTrainingAccuracy:
                tmp = torch.argmax(outputs, dim=1)
                flag = tmp.eq(labels)
                correct += torch.mean(flag.float())

            if i % 32 == 31:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 32))
                if drawTrainingLoss:
                    lossList.append( running_loss / 32)                
                if drawTrainingAccuracy:
                    accu = correct / 32
                    print('testing accuracy: ', accu)       # 当前epoch正确率
                    accuracyList.append(100 * accu)
                    correct = 0
                running_loss = 0.0

        if testSameTime:
            testAc, testLs = test(testLoader)
            testAccuracyList.append(testAc)
            testLossList.append(testLs)
    print('Finished Training')

    # drawing pics
    if drawTrainingLoss:
        plt.figure()
        plt.plot([i+1 for i in range(epochs)], lossList)
        plt.xlabel('training epoch')
        plt.ylabel('training loss')
    if drawTrainingAccuracy:
        plt.figure()
        plt.plot([i+1 for i in range(epochs)], accuracyList)
        plt.xlabel('training epoch')
        plt.ylabel('training accuracy (%)')
    if testSameTime:
        plt.figure()
        plt.plot([i+1 for i in range(epochs)], testLossList)
        plt.xlabel('training epoch')
        plt.ylabel('testing loss')
        plt.figure()
        plt.plot([i+1 for i in range(epochs)], testAccuracyList)
        plt.xlabel('training epoch')
        plt.ylabel('testing accuracy (%)')
    if drawTrainingAccuracy or drawTrainingLoss or testSameTime:
        plt.show()
    return net


# 测试模型的准确率
def test(testLoader):
    correct = 0
    loss = 0.0
    total = 0
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            outputs, _, _ = net(images)
            loss += torch.nn.functional.cross_entropy(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    cor = 100 * correct / total
    avgloss = loss / total
    print('Accuracy of the network on the 1000 test images: %d %%' % cor)
    print('Avg loss of the network on the 1000 test images: %f ' % avgloss)
    return cor, avgloss

# 定义PCA算法
def myPCA(X, K):
    """
    对数据进行PCA降维
    :param X: 数据矩阵，每一行代表一个样本，每一列代表一个特征
    :param K: 降维后的维度
    :return: 降维后的特征矩阵
    """
    # 去中心化
    X = X - np.mean(X, axis=0)
    # 计算协方差矩阵
    cov = np.cov(X, rowvar=False)
    # 计算特征值和特征向量
    eig_vals, eig_vecs = np.linalg.eig(cov)
    # 对特征值从大到小排序
    idx = np.argsort(-eig_vals)
    eig_vecs = eig_vecs[:, idx]
    # 取前K个特征向量作为投影矩阵
    proj = eig_vecs[:, :K]
    # 进行降维
    X_pca = np.dot(X, proj)
    return X_pca

def PCA_visualize(testLoader):
    # 选取3层输出进行可视化
    features_x = []     # 最终层
    features_a = []     # 卷积层conv2
    features_b = []     # 全连接层fc1
    labels = []
    net.eval()
    with torch.no_grad():
        for images, target in testLoader:
            x, a, b = net(images)
            features_x.append(x.numpy())
            features_a.append(a.numpy())
            features_b.append(b.numpy())
            labels.append(target.numpy())
    features_x = np.concatenate(features_x, axis=0)
    features_a = np.concatenate(features_a, axis=0)
    features_b = np.concatenate(features_b, axis=0)
    labels = np.concatenate(labels, axis=0)
    features_x_pca = myPCA(features_x, 2)
    features_a_pca = myPCA(features_a, 2)
    features_b_pca = myPCA(features_b, 2)

    plt.figure()
    plt.scatter(features_x_pca[:, 0], features_x_pca[:, 1], c=labels, s=10)
    plt.colorbar()
    plt.title('PCA - final layer')

    plt.figure()
    plt.scatter(features_a_pca[:, 0], features_a_pca[:, 1], c=labels, s=10)
    plt.colorbar()
    plt.title('PCA - conv layer 3')

    plt.figure()
    plt.scatter(features_b_pca[:, 0], features_b_pca[:, 1], c=labels, s=10)
    plt.colorbar()
    plt.title('PCA - fc layer 1')

    plt.show()

    print("visualizing...")
    return 0


def tSNE_visualize(testLoader):
    features_x = []     # 最终层
    features_a = []     # 卷积层conv2
    features_b = []     # 全连接层fc1
    labels = []
    net.eval()
    with torch.no_grad():
        for images, target in testLoader:
            x, a, b = net(images)
            # print(x.shape)
            # print(a.shape)
            # print(b.shape)
            features_x.append(x.numpy())
            features_a.append(a.numpy())
            features_b.append(b.numpy())
            labels.append(target.numpy())
    features_x = np.concatenate(features_x, axis=0)
    features_a = np.concatenate(features_a, axis=0)
    features_b = np.concatenate(features_b, axis=0)
    labels = np.concatenate(labels, axis=0)


   
    features_x_tsne = myTSNE(features_x)
    features_a_tsne = myTSNE(features_a)
    features_b_tsne = myTSNE(features_b)

    plt.figure()
    plt.scatter(features_x_tsne[:, 0], features_x_tsne[:, 1], c=labels, s=10)
    plt.colorbar()
    plt.title('t-SNE - final layer')

    plt.figure()
    plt.scatter(features_a_tsne[:, 0], features_a_tsne[:, 1], c=labels, s=10)
    plt.colorbar()
    plt.title('t-SNE - conv layer 3')

    plt.figure()
    plt.scatter(features_b_tsne[:, 0], features_b_tsne[:, 1], c=labels, s=10)
    plt.colorbar()
    plt.title('t-SNE - fc layer 1')

    plt.show()

    return 0

def myTSNE(X, dim_new=2, perp=30.0):
    """
    t-SNE是一种可视化高维数据的算法，将高维数据映射到低维空间中。
    该函数实现基于t-SNE的降维，将输入的高维数据X映射到二维平面上。
    :param X: 输入的高维数据，形状为(n, d)
    :param dim_new: 映射到的低维空间的维度，默认为2
    :param initial_dims: 高维数据的初始维度，默认为50
    :param perp: t-SNE中的困惑度参数，用于控制局部结构的数量，默认为30.0
    """
    # 初始化参数
    X = X - np.mean(X, axis=0) # 特征维度均值归一化
    X = X / np.std(X, axis=0) # 特征维度标准差归一化
    n = X.shape[0] # 样本数量
    max_iter = 1000 # 最大迭代次数
    eta = 500 # 学习率
    min_gain = 0.01 # 最小增益
    Y = np.random.randn(n, dim_new) # 初始化低维空间
    dY = np.zeros((n, dim_new))
    iY = np.zeros((n, dim_new))
    gains = np.ones((n, dim_new))

    # 计算高斯概率分布，用于t-SNE中的相似度计算
    P = x2p(X, perp, 1e-5) 
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4 # 对高斯概率分布进行缩放
    P = np.maximum(P, 1e-12) # 防止出现除以0的情况

    # 迭代
    for iter in range(max_iter):
        sum_Y = np.sum(np.square(Y), axis=1)
        num = -2.0 * np.dot(Y, Y.T)
        num = 1.0 / (1.0 + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.0
        Q = num / np.sum(num)   # 计算低维空间中的相似度分布
        PQ = P - Q              # 降维前后相似度分布之间的差异
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (dim_new, 1)).T * (Y[i, :] - Y), axis=0) # 计算梯度
        if iter < 20:
            momentum = 0.5
        else:
            momentum = 0.8
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)     # 计算每个样本在低维空间中移动的距离
        Y = Y + iY              # 更新低维空间
        Y = Y - np.mean(Y, axis=0)      # 特征维度均值归一化
        if (iter+1) % 10 == 0:
            print('Iteration {}: error is {}'.format(iter+1, np.sum(PQ * PQ)))
        if iter == 100:
            P = P / 4
    return Y

def x2p(X, perp, tol=1e-5):
    # 初始化参数
    n = X.shape[0]
    d = X.shape[1]
    sum_X = np.sum(np.square(X), axis=1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perp)
    # 开始迭代
    for i in range(n):
        if i % 500 == 0:
            print('computing p-values for point ', i, ' of ', n, '...')
        beta_min = -np.inf
        beta_max = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        H, thisP = hbeta(Di, beta[i])
        H_diff = H - logU
        tries = 0
        while np.abs(H_diff) > tol and tries < 50:
            if H_diff > 0:
                beta_min = beta[i]
                if beta_max == np.inf or beta_max == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + beta_max) / 2
            else:
                beta_max = beta[i]
                if beta_min == np.inf or beta_min == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + beta_min) / 2
            H, thisP = hbeta(Di, beta[i])
            H_diff = H - logU
            tries += 1
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
    return P

def hbeta(D, beta):
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    if sumP==0:
        P /= 1
    else:
        P = P / sumP
    return H, P

#############################################################################################################


# 自行设计的卷积神经网络

class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*3*3, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        a = x.view(x.size(0), -1)       # a用作可视化, conv layer 3
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 128*3*3)

        x = self.fc1(x)
        b = x.view(x.size(0), -1)       # b用作可视化, fc layer 1
        x = self.relu(x)
        x = self.fc2(x)
        return x, a, b



#############################################################################################################

if __name__ == '__main__':
    ######################## Get train/test dataset ########################
    # get data from set
    X_train, X_test, Y_train, Y_test = get_data('E:/03-02/Machine Learning/project/mine/data_required_opt2/dataset')
    
    # preprocess, data -> loader
    batchSize = 32
    trainLoader, testLoader = preprocess(X_train, X_test, Y_train, Y_test, batchSize)
    
    # generate
    # net = LeNet()
    net = myNet()

    # train
    net = train(trainLoader, drawTrainingAccuracy=False, drawTrainingLoss=False, testSameTime=False)

    # test & visualization
    # test(testLoader)
    # PCA_visualize(testLoader)
    # tSNE_visualize(testLoader)

    

