import tensorflow as tf

 
class Residual(tf.keras.layers.Layer):

    def __init__(self, cfg, filters):
        super(Residual, self).__init__()
        self.layer = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, 1, padding="same", kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters, 3, 1, padding="same", kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization()
        ])

    def call(self, x_in):
        x_out = self.layer(x_in)
        return x_out + x_in

class PixelShuffle(tf.keras.layers.Layer):

    def __init__(self, size):
        super(PixelShuffle, self).__init__()
        self.size = size
    
    def call(self, x_in):
        x_out = tf.nn.depth_to_space(x_in, self.size)
        return x_out
