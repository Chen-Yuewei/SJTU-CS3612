import os
import numpy as np
from dataset import get_data,normalize
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import cv2
# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        
        # 编码器
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # self.encoder_fc1 = nn.Linear(256 * 2 * 2, latent_dim)
        # self.encoder_fc2 = nn.Linear(256 * 2 * 2, latent_dim)
        self.encoder_fc1 = nn.Linear(256, latent_dim)
        self.encoder_fc2 = nn.Linear(256, latent_dim)
       
        # 解码器
        self.decoder_fc = nn.Linear(latent_dim, 256 * 2 * 2)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        
    def encode(self, x):
        x = self.encoder_conv(x)
        # print(x.shape)      # 64 * 256 * 1 * 1
        x = x.view(x.size(0), -1)
        mu = self.encoder_fc1(x)
        logvar = self.encoder_fc2(x)
        return mu, logvar
    
    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(x.size(0), 256, 2, 2)
        x = self.decoder_conv(x)
        return x
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
# 定义训练函数
def train_vae(model, dataloader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    for y in dataloader:
        x = y[0]        # ??????????????????
        # print(x)
        # x = x.to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(x)
        recon_loss = criterion(x_recon, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(dataloader.dataset)
    return train_loss
# 定义测试函数
def test_vae(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            x_recon, mu, logvar = model(x)
            recon_loss = criterion(x_recon, x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            test_loss += loss.item()
    test_loss /= len(dataloader.dataset)
    return test_loss



if __name__ == '__main__':
    ######################## Get train dataset ########################
    X_train = get_data('dataset')
    # print(X_train.shape)
    ########################################################################
    ######################## Implement you code here #######################
    ########################################################################
    X_train = np.array(X_train)
    Xtrain = torch.from_numpy(X_train)
    dataset = TensorDataset(Xtrain)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    

    # 初始化模型
    model = VAE(latent_dim=20)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    # 训练模型
    for epoch in range(10):
        train_loss = train_vae(model, dataloader, optimizer, criterion, device)
        # test_loss = test_vae(model, dataloader, criterion, device)
        # print('Epoch {}, Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch+1, train_loss, test_loss))
    # 生成新图片
    n = 4
    a = 0.2
    z = torch.zeros([4, 20])
    z[0][0] = z[1][1] = z[2][2] = z[3][3] = a
    z[0][1] = z[1][2] = z[2][3] = z[3][4] = 1-a
    print(z)

    x_gen = model.decode(z).detach().cpu()
    # print(x_gen.shape)          # torch.Size([n, 3, 32, 32])

    x_gen = x_gen.tolist()

    plt.figure()
    for i in range(n):
        pic_gen = np.array(x_gen[i])
        # print(pic_gen.shape)        # (3, 32, 32)
        pic_gen = np.transpose(pic_gen, (1, 2, 0))
        # print(pic_gen.shape)        # (32, 32, 3)

        plt.subplot(2, 2, i+1)
        plt.imshow(pic_gen)
    plt.show()

