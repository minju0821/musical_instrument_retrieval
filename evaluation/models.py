import torch
from torch import nn

# default model architecture for f_enc -> last linear layer output dimension is 953 (for 953 classes)
class ConvNet(nn.Module):
    def __init__(self, out_classes):
        super(ConvNet, self).__init__()

        self.conv1 = self.seq(1, 32)
        self.conv2 = self.seq(32, 64)
        self.conv3 = self.seq(64, 128)
        self.conv4 = self.seq(128, 256).append(
            nn.AdaptiveMaxPool2d((1, 1))
        )

        self.pool_drop = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Dropout2d(p=0.25)
        )

        self.linear1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=1024),
            nn.LeakyReLU(negative_slope=0.33),
            nn.Dropout(p=0.5)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=out_classes),
        )

        self.conv1.apply(self.init_weights)
        self.conv2.apply(self.init_weights)
        self.conv3.apply(self.init_weights)
        self.conv4.apply(self.init_weights)
        self.linear1.apply(self.init_weights)
        self.linear2.apply(self.init_weights)

    def seq(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(negative_slope=0.33),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(negative_slope=0.33)
        )

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
        l1_out = self.linear1(x)
        l2_out = self.linear2(l1_out)

        return l2_out, l1_out
