import tensorflow as tf
from layers import Residual, PixelShuffle


class Generator(tf.keras.Model):
    # Input is a 96x96 bicubic downsample of a 384x384 cropped image
    def __init__(self, cfg):
        super(Generator, self).__init__()
        num_blks = 4
        num_filters = 64

        self.block0 = tf.keras.Sequential()
            # (N, 3, 96, 96) -> (N, 64, 96, 96)
        self.block0.add(tf.keras.layers.Conv2D(num_filters, 9, padding="same", kernel_initializer=cfg.init))
        self.block0.add(tf.keras.layers.PReLU())

        # Add k resnet blocks
        for blk in range(num_blks):
            self.block0.add(Residual(cfg, num_filters))
        
        self.block0.add(tf.keras.layers.Conv2D(num_filters, 3, padding="same", kernel_initializer=cfg.init))
        self.block0.add(tf.keras.layers.BatchNormalization())

        self.block1 = tf.keras.Sequential([
            # (N, 64, 96, 96) -> (N, 256, 96, 96)
            tf.keras.layers.Conv2D(num_filters*4, 3, padding="same", kernel_initializer=cfg.init),
            # (N, 256, 96, 96) -> (N, 64, 192, 192)
            PixelShuffle(2),
            tf.keras.layers.PReLU(),
            # (N, 64, 192, 192) -> (N, 256, 192, 192)
            tf.keras.layers.Conv2D(num_filters*4, 3, padding="same", kernel_initializer=cfg.init),
            # (N, 256, 192, 192) -> (N, 64, 384, 384)
            PixelShuffle(2),
            tf.keras.layers.PReLU(),
            # (N, 64, 384, 384) -> (N, 3, 384, 384)
            tf.keras.layers.Conv2D(cfg.num_channels, 3, padding="same", kernel_initializer=cfg.init),
        ])

        self.pad = tf.keras.layers.Conv2D(num_filters, 1, padding="same")
    
    def call(self, x_in):
        x_out = self.block0(x_in)
        x_pad = self.pad(x_in)
        return self.block1(x_out + x_pad)


class Discriminator(tf.keras.Model):

    def __init__(self, cfg):
        super(Discriminator, self).__init__()
        num_filters = 64
        num_fc = 1024

        # Channels in, channels out, filter size, stride
        self.model = tf.keras.Sequential([
            # (N, 3, 384, 384) -> (N, 64, 384, 384)
            tf.keras.layers.Conv2D(num_filters, 3, padding="same", kernel_initializer=cfg.init),
            tf.keras.layers.LeakyReLU(),
            # (N, 64, 384, 384) -> (N, 64, 192, 192)
            tf.keras.layers.Conv2D(num_filters, 3, 2, padding="same", kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            # (N, 64, 192, 192) -> (N, 128, 192, 192)
            tf.keras.layers.Conv2D(num_filters*2, 3, padding="same", kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            # (N, 128, 192, 192) -> (N, 128, 96, 96)
            tf.keras.layers.Conv2D(num_filters*2, 3, 2, padding="same", kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            # (N, 128, 96, 96) -> (N, 256, 96, 96)
            tf.keras.layers.Conv2D(num_filters*4, 3, padding="same", kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            # (N, 256, 96, 96) -> (N, 256, 48, 48)
            tf.keras.layers.Conv2D(num_filters*4, 3, 2, padding="same", kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            # (N, 256, 48, 48) -> (N, 512, 48, 48)
            tf.keras.layers.Conv2D(num_filters*8, 3, padding="same", kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            # (N, 512, 48, 48) -> (N, 512, 24, 24)
            tf.keras.layers.Conv2D(num_filters*8, 3, 2, padding="same", kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            # (N, 512, 24, 24) -> (N, 512*24*24) -> (N, 294912)
            tf.keras.layers.Flatten(),
            # (N, 294912) -> (N, 1024)
            tf.keras.layers.Dense(num_fc, kernel_initializer=cfg.init),
            tf.keras.layers.LeakyReLU(),
            # (N, 1024) -> (N, 1)
            tf.keras.layers.Dense(1, kernel_initializer=cfg.init)
        ])

    def call(self, x_in):
        x_out = self.model(x_in)
        return tf.keras.activations.sigmoid(x_out)

