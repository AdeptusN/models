"""Autoencoder based on UNet by AdeptusN"""

import torch
import torch.nn as nn

from adeptus_modules.conv_modules import Conv3x3, Conv5x5


class UNetAutoencoder(nn.Module):
    """Autoencoder with skip connections."""

    def __init__(self, in_channels=3, out_channels=3):
        super(UNetAutoencoder, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_module1 = nn.Sequential(
            Conv5x5(in_channels=in_channels, out_channels=64, activation_func=nn.LeakyReLU()),
            Conv5x5(in_channels=64, out_channels=64, activation_func=nn.LeakyReLU())
        )

        self.conv_module2 = Conv3x3(in_channels=64, out_channels=128, batch_norm=True, activation_func=nn.LeakyReLU())
        self.conv_module3 = Conv3x3(in_channels=128, out_channels=256, batch_norm=True, activation_func=nn.LeakyReLU())
        self.conv_module4 = Conv3x3(in_channels=256, out_channels=512, batch_norm=True, activation_func=nn.LeakyReLU())
        self.conv_module5 = Conv3x3(in_channels=512, out_channels=512, batch_norm=True, activation_func=nn.LeakyReLU())

        self.bottle_neck = Conv3x3(in_channels=512, out_channels=512,
                                   batch_norm=True, activation_func=nn.ReLU())

        self.deconv_module1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv_conv_module1 = Conv3x3(in_channels=512*2, out_channels=512,
                                           batch_norm=True, activation_func=nn.ReLU())
        self.deconv_module2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv_conv_module2 = Conv3x3(in_channels=512*2, out_channels=256,
                                           batch_norm=True, activation_func=nn.ReLU())
        self.deconv_module3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv_conv_module3 = Conv3x3(in_channels=256*2, out_channels=128,
                                           batch_norm=True, activation_func=nn.ReLU())
        self.deconv_module4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv_conv_module4 = Conv3x3(in_channels=128*2, out_channels=64,
                                           batch_norm=True, activation_func=nn.ReLU())
        self.deconv_module5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv_conv_module5 = nn.Sequential(
            Conv5x5(in_channels=64*2, out_channels=32, batch_norm=True, activation_func=nn.ReLU()),
            Conv5x5(in_channels=32, out_channels=32, batch_norm=True, activation_func=nn.ReLU())
        )

        # self.final_conv = Conv3x3(in_channels=32, out_channels=out_channels,
        #                           batch_norm=True, activation_func=nn.Tanh())

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward propagation method of neural network.
        Args:
            x: mini-batch of data

        Returns:
            Data from latent representation of autoencoder.
        """
        skip_connections = []

        out = self.conv_module1(x)
        skip_connections.append(out)
        out = self.max_pool(out)

        out = self.conv_module2(out)
        skip_connections.append(out)
        out = self.max_pool(out)

        out = self.conv_module3(out)
        skip_connections.append(out)
        out = self.max_pool(out)

        out = self.conv_module4(out)
        skip_connections.append(out)
        out = self.max_pool(out)

        out = self.conv_module5(out)
        skip_connections.append(out)
        out = self.max_pool(out)

        out = self.bottle_neck(out)

        out = self.deconv_module1(out)
        out = torch.cat((out, skip_connections[4]), axis=1)
        out = self.deconv_conv_module1(out)

        out = self.deconv_module2(out)
        out = torch.cat((out, skip_connections[3]), axis=1)
        out = self.deconv_conv_module2(out)

        out = self.deconv_module3(out)
        out = torch.cat((out, skip_connections[2]), axis=1)
        out = self.deconv_conv_module3(out)

        out = self.deconv_module4(out)
        out = torch.cat((out, skip_connections[1]), axis=1)
        out = self.deconv_conv_module4(out)

        out = self.deconv_module5(out)
        out = torch.cat((out, skip_connections[0]), axis=1)
        out = self.deconv_conv_module5(out)

        out = self.final_conv(out)

        return out

    @staticmethod
    def load():
        pass
