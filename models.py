import tensorflow as tf
from layers import Residual, PixelShuffle


class Generator(tf.keras.Model):
    # Input is a 24x24 bicubic downsample of a 96x96 cropped image
    def __init__(self, cfg):
        super(Generator, self).__init__()
        num_blks = 16
        num_filters = 64

        self.block0 = tf.keras.Sequential()
            # (N, 3, 24, 24) -> (N, 64, 24, 24)
        self.block0.add(tf.keras.layers.Conv2D(num_filters, 9, padding="same", kernel_initializer=cfg.init))
        self.block0.add(tf.keras.layers.PReLU())

        # Add k resnet blocks
        for blk in range(num_blks):
            self.block0.add(Residual(cfg, num_filters))
        
        self.block0.add(tf.keras.layers.Conv2D(num_filters, 3, padding="same", kernel_initializer=cfg.init))
        self.block0.add(tf.keras.layers.BatchNormalization())

        self.block1 = tf.keras.Sequential([
            # (N, 64, 24, 24) -> (N, 256, 24, 24)
            tf.keras.layers.Conv2D(num_filters*4, 3, padding="same", kernel_initializer=cfg.init),
            # (N, 256, 24, 24) -> (N, 64, 48, 48)
            PixelShuffle(2),
            tf.keras.layers.PReLU(),
            # (N, 64, 48, 48) -> (N, 256, 48, 48)
            tf.keras.layers.Conv2D(num_filters*4, 3, padding="same", kernel_initializer=cfg.init),
            # (N, 256, 48, 48) -> (N, 64, 96, 96)
            PixelShuffle(2),
            tf.keras.layers.PReLU(),
            # (N, 64, 96, 96) -> (N, 3, 96, 96)
            tf.keras.layers.Conv2D(cfg.num_channels, 3, padding="same", kernel_initializer=cfg.init),
        ])

        self.pad = tf.keras.layers.Conv2D(num_filters, 1, padding="same")

        # Create saver
        self.save_path = cfg.save_dir + cfg.extension + 'GEN'
        self.ckpt_prefix = self.save_path + '/ckpt'
        self.saver = tf.train.Checkpoint(block0=self.block0, block1=self.block1)
    
    def call(self, x_in):
        x_out = self.block0(x_in)
        x_pad = self.pad(x_in)
        return self.block1(x_out + x_pad)
    
    def save(self):
        self.saver.save(file_prefix=self.ckpt_prefix)
    
    def load(self):
        self.saver.restore(tf.train.latest_checkpoint(self.save_path))


class Discriminator(tf.keras.Model):
    # Input is a (N, 96, 96, 3) image, of either SR or cropped HR descent
    def __init__(self, cfg):
        super(Discriminator, self).__init__()
        num_filters = 64
        num_fc = 1024

        # Channels in, channels out, filter size, stride
        self.model = tf.keras.Sequential([
            # (N, 3, 96, 96) -> (N, 64, 96, 96)
            tf.keras.layers.Conv2D(num_filters, 3, padding="same", kernel_initializer=cfg.init),
            tf.keras.layers.LeakyReLU(),
            # (N, 64, 96, 96) -> (N, 64, 48, 48)
            tf.keras.layers.Conv2D(num_filters, 3, 2, padding="same", kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            # (N, 64, 48, 48) -> (N, 128, 48, 48)
            tf.keras.layers.Conv2D(num_filters*2, 3, padding="same", kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            # (N, 128, 48, 48) -> (N, 128, 24, 24)
            tf.keras.layers.Conv2D(num_filters*2, 3, 2, padding="same", kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            # (N, 128, 24, 24) -> (N, 256, 24, 24)
            tf.keras.layers.Conv2D(num_filters*4, 3, padding="same", kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            # (N, 256, 24, 24) -> (N, 256, 12, 12)
            tf.keras.layers.Conv2D(num_filters*4, 3, 2, padding="same", kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            # (N, 256, 12, 12) -> (N, 512, 12, 12)
            tf.keras.layers.Conv2D(num_filters*8, 3, padding="same", kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            # (N, 512, 12, 12) -> (N, 512, 6, 6)
            tf.keras.layers.Conv2D(num_filters*8, 3, 2, padding="same", kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            # (N, 512, 6, 6) -> (N, 512*6*6) -> (N, 18432)
            tf.keras.layers.Flatten(),
            # (N, 18432) -> (N, 1024)
            tf.keras.layers.Dense(num_fc, kernel_initializer=cfg.init),
            tf.keras.layers.LeakyReLU(),
            # (N, 1024) -> (N, 1)
            tf.keras.layers.Dense(1, kernel_initializer=cfg.init)
        ])
    
        # Create saver
        self.save_path = cfg.save_dir + cfg.extension + 'GEN'
        self.ckpt_prefix = self.save_path + '/ckpt'
        self.saver = tf.train.Checkpoint(model=self.model)

    def call(self, x_in):
        x_out = self.model(x_in)
        return tf.keras.activations.sigmoid(x_out)
    
    def save(self):
        self.saver.save(file_prefix=self.ckpt_prefix)
    
    def load(self):
        self.saver.restore(tf.train.latest_checkpoint(self.save_path))

