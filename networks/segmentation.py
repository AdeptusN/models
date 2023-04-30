"""Segmentation model based on UNet by AdeptusN"""

import torch
import torch.nn as nn
from AdeptusN.modules.conv_modules import Conv3x3, Conv5x5


class UNet(nn.Module):
    """
    UNet model for segmentation with changeable number of layers
    """
    def __init__(
            self, in_channels=3, out_channels=1, features=(64, 128, 256, 512)
    ):
        """
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels in the out mask
            features: tuple of layers activation maps numbers
        """
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)

        # Encoder
        for feature in features:
            self.downs.append(Conv5x5(
                in_channels, feature,
                batch_norm=True, dropout=False,
                activation_func=nn.LeakyReLU())
            )
            in_channels = feature

        # Decoder
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))

            self.ups.append(Conv5x5(
                feature*2, feature,
                batch_norm=True, dropout=False,
                activation_func=nn.ReLU())
            )

        self.bottleneck = Conv3x3(
            features[-1], features[-1]*2,
            batch_norm=True, dropout=False,
            activation_func=nn.ReLU()
        )
        self.classifier = nn.Sequential(
            # Conv5x5(features[0], features[0], batch_norm=True, dropout=False, activation_func=nn.ReLU()),
            nn.Conv2d(features[0], out_channels, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()  # - откомментить, если используется самописная функция активации

    def forward(self, x):
        """
        Forward propagation method of neural network.
        Args:
            x: mini-batch of data

        Returns:
            Result of network working
        """
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i + 1](concat_skip)

        out = self.classifier(x)
        out = self.sigmoid(out)

        return out
    
    @staticmethod
    def load():
        pass


if __name__ == "__main__":
    model = UNet()
