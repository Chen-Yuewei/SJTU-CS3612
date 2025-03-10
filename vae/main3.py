import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from dataset import get_data, normalize
import torchvision.utils as vutils
from matplotlib import pyplot as plt


# 定义VAE模型
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(32 * 8 * 8, 16)  # 编码器输出的均值
        self.fc_logvar = nn.Linear(32 * 8 * 8, 16)  # 编码器输出的对数方差
        self.decoder = nn.Sequential(
            nn.Linear(16, 32 * 8 * 8),
            nn.Unflatten(1, (32, 8, 8)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

# 定义损失函数
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 定义训练函数
def train(epoch, batch_size, opt_lr):
    # 训练模型
    num_epochs = epoch
    batch_size = batch_size
    optimizer = optim.Adam(model.parameters(), lr=opt_lr)

    model.train()
    for epoch in range(num_epochs+1):
        for i in range(0, len(train_data_tensor), batch_size):
            batch = train_data_tensor[i:i + batch_size]
            recon_batch, mu, logvar = model(batch)
            loss = loss_function(recon_batch, batch, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs+1}], Loss: {loss.item():.4f}")

        # 随机生成图像
        # if epoch % 20 == 0:
        #     images = generate_images_fix(model.decoder, z)
        #     for i in range(len(images)):
        #         vutils.save_image(images[i], 'gen/epoch_{}_{}.png'.format(epoch+1, i+1))
    return model


# 定义随机生成图片的函数
def generate_images(decoder, n_images):
    with torch.no_grad():
        z = torch.randn(n_images, 16)
        if torch.cuda.is_available():
            z = z.cuda()
        images = decoder(z).detach().cpu()
        images = vutils.make_grid(images, padding=0, normalize=True, nrow=3)
        return images
def generate_images_fix(decoder, z):
    with torch.no_grad():
        if torch.cuda.is_available():
            z = z.cuda()
        images = decoder(z).detach().cpu()
        # images = vutils.make_grid(images, padding=0, normalize=True, nrow=3)
        return images

# 定义图像重构函数
def reconstuct(n=1):
    for j in range(n):
        # 随机选取一张图片
        i = int(np.random.rand()*1000)
        img = [train_data[i]]
        img = np.array(img)
        img_tensor = torch.from_numpy(img).float()
        z = model.encoder(img_tensor)
        recon, mu, logvar = model(img_tensor)
        img_new = recon.tolist()[0]

        img = img[0].transpose(1, 2, 0)
        img_new = np.array(img_new)
        img_new = img_new.transpose(1, 2, 0)

        plt.figure(figsize=(4,2))
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.title('original pic {}'.format(i))
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(img_new)
        plt.title('reconstructed')
        plt.axis('off')
        plt.savefig('recon/recon_{}.png'.format(i))
    return img_new, z


# 定义图像融合函数
def fusion_z(n=1):
    for i in range(n):
        z1 = torch.randn(1, 16)
        z2 = torch.randn(1, 16)
        imgList = []
        imgList.append(generate_images_fix(model.decoder, z1)[0])
        for a in [0.8, 0.6, 0.4, 0.2]:
            z = a * z1 + (1 - a) * z2 
            imgList.append(generate_images_fix(model.decoder, z)[0])
        imgList.append(generate_images_fix(model.decoder, z2)[0])
        images = vutils.make_grid(imgList, padding=0, normalize=True)
        vutils.save_image(images, 'fusion/{}.png'.format(i+1))

def fusion_z_4pic():
    z1 = torch.randn(1, 16)
    z2 = torch.randn(1, 16)
    z3 = torch.randn(1, 16)
    z4 = torch.randn(1, 16)
    imgList = []
    for a in [1,0.8,0.6,0.4,0.2,0]:
        for b in [1,0.8,0.6,0.4,0.2,0]:
            z = a*b*z1 + a*(1-b)*z2 + (1-a)*b*z3 + (1-a)*(1-b)*z4
            imgList.append(generate_images_fix(model.decoder, z)[0])
    images = vutils.make_grid(imgList, padding=0, normalize=True, nrow=6)
    vutils.save_image(images, 'fusion/test2.png')



if __name__ == '__main__':
    # 将numpy数组转换为张量
    train_data = get_data('dataset')
    train_data_tensor = torch.from_numpy(train_data).float()

    # 创建模型和优化器
    model = VAE()
    # z = torch.randn(6, 16)
    model = train(100, 32, 0.0008)

    # 随机重建10张图片
    # reconstuct(10)

    # 随机生成图片，进行融合
    # fusion_z_4pic()
    # fusion_z(5)






