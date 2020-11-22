import torch
from torch import nn
import torch.nn.functional as F
from gan.spectral_normalization import SpectralNorm

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        #Hint: Hint: Apply spectral normalization to convolutional layers. Input to SpectralNorm should be your conv nn module
        ####################################
        self.conv1 = nn.Conv2d(3, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bnorm2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bnorm3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bnorm4 = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bnorm2(self.conv2(x))
        x = self.bnorm3(self.conv3(x))
        x = self.bnorm4(self.conv4(x))
        x = self.conv5(x)

        return x



class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        self.noise_dim = noise_dim
        self.tconv1 = nn.ConvTranspose2d(noise_dim, 1024, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0))
        self.bnorm1 = nn.BatchNorm2d(1024)
        self.tconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bnorm2 = nn.BatchNorm2d(512)
        self.tconv3 = nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bnorm3 = nn.BatchNorm2d(256)
        self.tconv4 = nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bnorm4 = nn.BatchNorm2d(128)
        self.tconv5 = nn.ConvTranspose2d(128, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.bnorm1(self.tconv1(x))
        x = self.bnorm2(self.tconv2(x))
        x = self.bnorm3(self.tconv3(x))
        x = self.bnorm4(self.tconv4(x))
        x = self.tanh(self.tconv5(x))
        # print(x.shape)
        
        return x
    

