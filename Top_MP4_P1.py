import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import zipfile
import pickle
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from gan.train import train
from gan.models import Discriminator, Generator
from gan.losses import discriminator_loss, generator_loss
from gan.losses import ls_discriminator_loss, ls_generator_loss



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-c", "--config", required=True,
    #                     help="JSON configuration string for this operation")
    parser.add_argument("-ep", "--num_epochs", required=True, default=50,
                        help="number of epochs")
    parser.add_argument("-bs", "--batch_size", required=True, default=128,
                        help="Batch size")
    parser.add_argument("-dp", "--data_path", required=True, default='/celeba_data/',
                        help="data path. in NGC /<data mount point>")
    parser.add_argument("-rp", "--result_path", required=True, default='/results/',
                        help="result path. in NGC /results")
    parser.add_argument("-m", "--Mode", required=True, default='GAN',
                        help="Mode of operation: GAN/LSGAN")
    parser.add_argument("-ss", "--scale_size", required=True, default=64,
                        help="scale size")
    parser.add_argument("-lr", "--learning_rate", required=True, default=2e-4,
                        help="scale size")
    # Grab the Arguments
    conf_data = parser.parse_args()
    # args.config = args.config.replace("\'", "\"")
    # conf_data = json.loads(unquote(args.config))


    # ========Input parameters
    # path = os.getcwd()
    # print('The os.getcwd(): ', path)
    num_epochs = int(conf_data.num_epochs)
    batch_size = int(conf_data.batch_size)
    data_path = conf_data.data_path
    result_path = conf_data.result_path
    scale_size = conf_data.scale_size
    learning_rate = conf_data.learning_rate
    Mode = conf_data.Mode

    NOISE_DIM = 100


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'


    # Prepare datasets and data loader

    if 1:
        with zipfile.ZipFile(data_path + '/celeba_data.zip', 'r') as zip_ref:
            zip_ref.extractall('/raid/celeba')
            celeba_root = './raid/celeba'

    celeba_train = ImageFolder(root=celeba_root, transform=transforms.Compose([
      transforms.Resize(scale_size),
      transforms.ToTensor(),
    ]))

    celeba_loader_train = DataLoader(celeba_train, batch_size=batch_size, drop_last=True)

    if mode == 'GAN':
        # ================Training GAN
        D = Discriminator().to(device)
        G = Generator(noise_dim=NOISE_DIM).to(device)

        D_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate, betas = (0.5, 0.999))
        G_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas = (0.5, 0.999))

        # original gan
        train(D, G, D_optimizer, G_optimizer, discriminator_loss,
                  generator_loss, num_epochs=num_epochs, show_every=150,
                  train_loader=celeba_loader_train, device=device)

        f = open(result_path + "./result_GAN.pkl", "wb")
        pickle.dump([D, G], f)
        f.close()

    if mode == 'LSGAN':
        # ================Training LS-GAN
        D = Discriminator().to(device)
        G = Generator(noise_dim=NOISE_DIM).to(device)

        D_optimizer = torch.optim.Adam(D.parameters(), lr=num_epochs, betas = (0.5, 0.999))
        G_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas = (0.5, 0.999))

        # ls-gan
        train(D, G, D_optimizer, G_optimizer, ls_discriminator_loss,
                  ls_generator_loss, num_epochs=num_epochs, show_every=200,
                  train_loader=celeba_loader_train, device=device)

        f = open(result_path + "./result_LSGAN.pkl", "wb")
        pickle.dump([D, G], f)
        f.close()