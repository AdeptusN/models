"""Autoencoder model by AdeptusN"""

import torch
import torch.nn as nn

from adeptus_modules.conv_modules import Conv3x3, Conv5x5


class Autoencoder(nn.Module):
    """Autoencoder with 5 downscale and upscale layer and with ability to become variational.
    Input image is desired to be 256x192."""

    def __init__(self, in_channels=3, out_channels=3, latent_dim=200, variational=False):
        """

        Args:
            in_channels: number of channels of input image
            out_channels: number of channels of output image
            latent_dim: latent dimension size
        """
        super(Autoencoder, self).__init__()
        self.variational = variational

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_module1 = nn.Sequential(
            Conv5x5(in_channels=in_channels, out_channels=64, activation_func=nn.LeakyReLU(), batch_norm=True),
            Conv5x5(in_channels=64, out_channels=64, activation_func=nn.LeakyReLU(), batch_norm=True)
        )

        self.conv_module2 = Conv3x3(in_channels=64, out_channels=128, batch_norm=True, activation_func=nn.LeakyReLU())
        self.conv_module3 = Conv3x3(in_channels=128, out_channels=256, batch_norm=True, activation_func=nn.LeakyReLU())
        self.conv_module4 = Conv3x3(in_channels=256, out_channels=512, batch_norm=True, activation_func=nn.LeakyReLU())
        self.conv_module5 = Conv3x3(in_channels=512, out_channels=512, batch_norm=True, activation_func=nn.LeakyReLU())

        self.bottle_neck = Conv3x3(in_channels=512, out_channels=512,
                                   batch_norm=True, activation_func=nn.LeakyReLU())

        self.latent_dim = latent_dim
        self.latent_linear = nn.Linear(512 * 8 * 6, self.latent_dim)

        if variational:
            self.mu = nn.Linear(512 * 8 * 6, self.latent_dim)
            self.log_var = nn.Linear(512 * 8 * 6, self.latent_dim)

        self.decode_input = nn.Linear(self.latent_dim, 512 * 8 * 6)

        self.deconv_module1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv_conv_module1 = Conv3x3(in_channels=512, out_channels=512,
                                           batch_norm=True, activation_func=nn.ReLU())
        self.deconv_module2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv_conv_module2 = Conv3x3(in_channels=512, out_channels=256,
                                           batch_norm=True, activation_func=nn.ReLU())
        self.deconv_module3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv_conv_module3 = Conv3x3(in_channels=256, out_channels=128,
                                           batch_norm=True, activation_func=nn.ReLU())
        self.deconv_module4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv_conv_module4 = Conv3x3(in_channels=128, out_channels=64,
                                           batch_norm=True, activation_func=nn.ReLU())
        self.deconv_module5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv_conv_module5 = nn.Sequential(
            Conv5x5(in_channels=64, out_channels=32, batch_norm=True, activation_func=nn.ReLU()),
            Conv5x5(in_channels=32, out_channels=out_channels, batch_norm=True, activation_func=nn.ReLU())
        )

    def _encode(self, x):
        x = self.conv_module1(x)
        x = self.max_pool(x)

        x = self.conv_module2(x)
        x = self.max_pool(x)

        x = self.conv_module3(x)
        x = self.max_pool(x)

        x = self.conv_module4(x)
        x = self.max_pool(x)

        x = self.conv_module5(x)
        x = self.max_pool(x)

        x = self.bottle_neck(x)

        x = self.latent_linear(x)

        if self.variational:
            mu = self.mu(x)
            log_var = self.log_var(x)
            return mu, log_var

        return x

    def _sampler(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.distributions.Normal(0, 1).sample()
        return std * eps + mu

    def extract_features(self, x):
        assert self.variational, "This method is used only when variational=True"
        mu, log_var = self._encode(x)
        sample = self._sampler(mu, log_var)
        features = self.decode_input(sample)
        features = torch.reshape(features, (-1, 512, 8, 6))
        return features

    def _decode(self, z):
        out = self.decode_input(z)
        out = torch.reshape(out, (-1, 512, 8, 6))

        out = self.deconv_module1(out)
        out = self.deconv_conv_module1(out)

        out = self.deconv_module2(out)
        out = self.deconv_conv_module2(out)

        out = self.deconv_module3(out)
        out = self.deconv_conv_module3(out)

        out = self.deconv_module4(out)
        out = self.deconv_conv_module4(out)

        out = self.deconv_module5(out)
        out = self.deconv_conv_module5(out)

        # z = self.final_conv(z)
        return out

    def forward(self, x):
        """
        Forward propagation method of neural network.
        Args:
            x: mini-batch of data

        Returns:
            Data upscaled from latent representation
        """
        if self.variational:
            mu, log_var = self._encode(x)
            z = self._sampler(mu, log_var)
        else:
            z = self._encode(x)

        out = self._decode(z)

        return out
