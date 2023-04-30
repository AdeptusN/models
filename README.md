# models
Repository of NN models by AdeptusN

## autoencoder.py
Autoencoder with 5 downscale and upscale layers. Might be turned into variational autoencoder.

    Autoencoder(
        in_channels=3,                     #number of input channels
        out_channels=3,                    #number of output channels
        latent_dim=200,                    #latent dimension size
        variational=False                  #turn into variational
    )

## unet_autoencoder.py
Autoencoder with skip connections.

    UNetAutoencoder(
        in_channels=3,                     #number of input channels
        out_channels=3                     #number of output channels
    )

## segmentation.py
Changeable UNet model for segmentation.

    UNet(
        in_channels=3,                     #number of input channels
        out_channels=1,                    #number of output channels
        features=(64, 128, 256, 512)       #tuple of layers activation maps features number
    )