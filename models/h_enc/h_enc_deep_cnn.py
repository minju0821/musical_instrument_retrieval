import torch
from torch import nn

class ConvNet(nn.Module):
    def __init__(self, activ, out_classes):
        super(ConvNet, self).__init__()

        if activ=='LReLU':
            relu = nn.LeakyReLU(negative_slope=0.33)
        elif activ=='PReLU':
            relu = nn.PReLU()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            relu,
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            relu
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            relu,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            relu
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            relu,
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            relu
        )

        self.pool_drop = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Dropout2d(p=0.25)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            relu,
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            relu,
            nn.AdaptiveMaxPool2d((1, 1))
        )

        self.linear1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=1024),
            relu,
            nn.Dropout(p=0.5)
        )

        # self.linear2 = nn.Sequential(
        #     nn.Linear(in_features=1024, out_features=out_classes),
        #     # relu,
        # )

        self.conv1.apply(self.init_weights)
        self.conv2.apply(self.init_weights)
        self.conv3.apply(self.init_weights)
        self.conv4.apply(self.init_weights)
        self.linear1.apply(self.init_weights)
        # self.linear2.apply(self.init_weights)

    # for all conv. & linear layers w/ zero biases
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.)
    
    def forward(self, x):
        x = x.unsqueeze(dim=1).permute(0, 1, 3, 2)
        
        x = self.pool_drop(self.conv1(x.float()))
        x = self.pool_drop(self.conv2(x))
        x = self.pool_drop(self.conv3(x))

        x = self.conv4(x)
        x = self.linear1(x)
        # x = self.linear2(x)

        return x
