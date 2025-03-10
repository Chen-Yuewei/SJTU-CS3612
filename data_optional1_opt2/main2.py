import os
import numpy as np
from dataset import get_data,normalize
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

# 定义VAE的编码器和解码器
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc21 = nn.Linear(512, 64)
        self.fc22 = nn.Linear(512, 64)
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = x.view(-1, 256 * 4 * 4)
        h1 = nn.functional.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(64, 512)
        self.fc2 = nn.Linear(512, 256 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1)
    def forward(self, z):
        z = nn.functional.relu(self.fc1(z))
        z = nn.functional.relu(self.fc2(z))
        z = z.view(-1, 256, 4, 4)
        z = nn.functional.relu(self.deconv1(z))
        z = nn.functional.relu(self.deconv2(z))
        z = nn.functional.relu(self.deconv3(z))
        z = torch.sigmoid(self.deconv4(z))
        return z
# 定义VAE的损失函数
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x.view(-1, 32 * 32 * 3), x.view(-1, 32 * 32 * 3), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
# 定义训练过程
def train_vae(encoder, decoder, train_loader, optimizer, epoch):
    encoder.train()
    decoder.train()
    for batch_idx, data in enumerate(train_loader):
        # print(batch_idx)
        # print(data)
        data = Variable(data[0])
        if torch.cuda.is_available():
            data = data.cuda()
        optimizer.zero_grad()
        mu, logvar = encoder(data)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        recon_batch = decoder(z)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
# 定义随机生成图片的函数
def generate_images(decoder, n_images):
    with torch.no_grad():
        z = torch.randn(n_images, 64)
        if torch.cuda.is_available():
            z = z.cuda()
        images = decoder(z).detach().cpu()
        images = vutils.make_grid(images, padding=0, normalize=True, nrow=3)
        return images



if __name__ == '__main__':
    ######################## Get train dataset ########################
    X_train = get_data('dataset')
    ########################################################################
    ######################## Implement you code here #######################
    ########################################################################
    # 定义训练参数
    batch_size = 64
    n_epochs = 100
    
    # X_train = np.array(X_train)
    Xtrain = torch.from_numpy(X_train)
    # print(Xtrain.type())
    dataset = TensorDataset(Xtrain)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    # 初始化VAE的编码器和解码器
    encoder = Encoder()
    decoder = Decoder()
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    # 定义优化器
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0001)
    # 训练VAE
    for epoch in range(1, n_epochs + 1):
        train_vae(encoder, decoder, dataloader, optimizer, epoch)
        # 每个epoch结束后生成一些图片
        images = generate_images(decoder, n_images=6)
        vutils.save_image(images, 'generated_images/epoch_{}.png'.format(epoch))