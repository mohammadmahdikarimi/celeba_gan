import torch
from gan.spectral_normalization import SpectralNorm

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        #Hint: Hint: Apply spectral normalization to convolutional layers. Input to SpectralNorm should be your conv nn module
        ####################################
        self.conv1 = nn.Conv2d(3, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bnorm1 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bnorm1 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bnorm1 = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

        self.lrelu = nn.LeakyReLU(0.2)
        ####################################
        
        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        
        
        ##########       END      ##########
        
        return x

class Alexnet_Classifier(nn.Module):
    def __init__(self):
        #super() allows you to build classes that easily extend the functionality of previously built classes without implementing their functionality again.
        super(Alexnet_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bnorm1 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bnorm1 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bnorm1 = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

        self.lrelu = nn.LeakyReLU(0.2)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.fc1 = nn.Linear(in_features=9216, out_features=4096, bias=True)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.fc3 = nn.Linear(in_features=4096, out_features=NUM_CLASSES, bias=True)
        self.drop = nn.Dropout(p=0.5, inplace=False)
        print("Alex")


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        print("t1_Alex: ",x.size())
        x = x.view(x.size()[0], 9216)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        self.noise_dim = noise_dim
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        
        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        
        
        ##########       END      ##########
        
        return x
    

