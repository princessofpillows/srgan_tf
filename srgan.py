import tensorflow as tf
import numpy as np
import h5py, os, cv2
from tqdm import trange
from datetime import datetime
from pathlib import Path
from config import get_config
from preprocessing import package_data
from models import Generator, Discriminator


tf.enable_eager_execution()
cfg = get_config()

class SRGAN(object):

    def __init__(self, cfg):
        super(SRGAN, self).__init__()
        
        # Training stats
        self.global_step = tf.train.get_or_create_global_step()
        self.epoch = tf.Variable(0)
        
        # Load models
        self.generator = Generator(cfg)
        self.discriminator = Discriminator(cfg)
        self.vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=cfg.hr_resolution + (3,), pooling=None)
        self.gen_optim = tf.train.AdamOptimizer(cfg.learning_rate)
        self.disc_optim = tf.train.AdamOptimizer(cfg.learning_rate)

        self.build_writers()
        self.preprocessing()

    def preprocessing(self):
        if cfg.package_data:
            # Package data into H5 format
            package_data(cfg)
        # Load data
        cwd = os.getcwd()
        f = h5py.File(cwd + cfg.data_dir + '/data.h5', 'r')
        #lr = f['lr'][:]
        hr = f['hr'][:].astype(np.float32)
        f.close()

        s = np.arange(len(hr))
        np.random.shuffle(s)
        # shuffle randomly
        #lr = np.asarray(lr)[s]
        hr = np.asarray(hr)[s]

        # Normalize
        #hr = ((hr - hr.mean()) / hr.std()).astype(np.float32)

        self.size = len(hr)
        self.data_tr = tf.data.Dataset.from_tensor_slices((hr))

    def build_writers(self):
        if not Path(cfg.save_dir).is_dir():
            os.mkdir(cfg.save_dir)
        if cfg.extension is None:
            cfg.extension = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        self.log_path = cfg.log_dir + cfg.extension
        self.writer = tf.contrib.summary.create_file_writer(self.log_path)
        self.writer.set_as_default()

        self.save_path = cfg.save_dir + cfg.extension
        self.ckpt_prefix = self.save_path + '/ckpt'
        self.saver = tf.train.Checkpoint(generator=self.generator, gen_optim=self.gen_optim, discriminator=self.discriminator, 
                                        disc_optim=self.disc_optim, global_step=self.global_step, epoch=self.epoch)

    def logger(self, tape, mse_loss, vgg_loss, adv_loss, percept_loss, images):
        with tf.contrib.summary.record_summaries_every_n_global_steps(cfg.log_freq, self.global_step):
            # Log vars
            tf.contrib.summary.scalar('SRGAN/mse_loss', mse_loss)
            tf.contrib.summary.scalar('SRGAN/vgg_loss', vgg_loss)
            tf.contrib.summary.scalar('SRGAN/adv_loss', adv_loss)
            tf.contrib.summary.scalar('SRGAN/percept_loss', percept_loss)

            # Log weights
            slots = self.gen_optim.get_slot_names()
            for variable in tape.watched_variables():
                    tf.contrib.summary.scalar(variable.name, tf.nn.l2_loss(variable))
                    for slot in slots:
                        slotvar = self.gen_optim.get_slot(variable, slot)
                        if slotvar is not None:
                            tf.contrib.summary.scalar(variable.name + '/' + slot, tf.nn.l2_loss(slotvar))

            # Log a generated image
            tf.contrib.summary.image('SRGAN/generated', images)
    
    def log_img(self, img, name):
        if self.global_step.numpy() % (cfg.log_freq * 5) == 0:
            with tf.contrib.summary.always_record_summaries():
                img = tf.cast(img, tf.float32)
                tf.contrib.summary.image(name, img, max_images=3)

    def update(self, hr_crop, hr_ds):
        # Construct graph
        with tf.GradientTape(persistent=True) as tape:
            sr = self.generator(hr_ds)
            # mse_loss = tf.keras.losses.mean_squared_error(hr_crop, sr)

            sr_vgg_logits = self.vgg(sr)
            hr_vgg_logits = self.vgg(hr_crop)
            vgg_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(hr_vgg_logits, sr_vgg_logits))

            sr_disc_logits = self.discriminator(sr)
            hr_disc_logits = self.discriminator(hr_crop)

            # Comparing HR logits with 1's labels and SR logits with 0's labels
            adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=hr_disc_logits, labels=tf.ones_like(sr_disc_logits))
                                     + tf.nn.sigmoid_cross_entropy_with_logits(logits=sr_disc_logits, labels=tf.zeros_like(sr_disc_logits)))


            # percept_loss = tf.reduce_sum(mse_loss) + tf.reduce_sum(mse_loss) + vgg_loss + 1e-3 * adv_loss
            percept_loss = vgg_loss + 1e-3 * adv_loss
        

            self.logger(tape, mse_loss, vgg_loss, adv_loss, percept_loss, [sr[0], hr[0]])

        # Compute/apply gradients for generator with perceptual loss
        gen_grads = tape.gradient(percept_loss, self.generator.weights)
        gen_grads_and_vars = zip(gen_grads, self.generator.weights)
        self.gen_optim.apply_gradients(gen_grads_and_vars)

        # Compute/apply gradients for discriminator with adversarial loss
        disc_grads = tape.gradient(adv_loss, self.discriminator.weights)
        disc_grads_and_vars = zip(disc_grads, self.discriminator.weights)
        self.disc_optim.apply_gradients(disc_grads_and_vars)

        self.global_step.assign_add(1)
    

    def pretrain_update(self, hr_crop, hr_ds):
        # Construct graph
        with tf.GradientTape() as tape:
            sr = self.generator(hr_ds)
            mse_loss = tf.keras.losses.mean_squared_error(hr_crop, sr)

        # Compute/apply gradients for generator with MSE
        gen_grads = tape.gradient(mse_loss, self.generator.weights)
        gen_grads_and_vars = zip(gen_grads, self.generator.weights)
        self.gen_optim.apply_gradients(gen_grads_and_vars)

        self.global_step.assign_add(1)

    def train(self):
        if Path(self.save_path).is_dir():
            self.saver.restore(tf.train.latest_checkpoint(self.save_path))
        epoch = self.epoch.numpy()
        for epoch in trange(epoch, cfg.epochs):
            # Uniform shuffle
            batch = self.data_tr.shuffle(self.size).batch(cfg.batch_size)
            for hr in batch:
                # Normalize
                hr = tf.image.per_image_standardization(hr)
                # Random 384x384 crop
                hr_crop = tf.image.random_crop(hr, (cfg.batch_size,) + cfg.crop_resolution + (3,))
                # Apply gaussian blur and downsample to 96x96
                hr_crop_blur = []
                for i in range(len(hr_crop)):
                    hr_crop_blur.append(cv2.GaussianBlur(hr_crop[i].numpy(), (3, 3), 0))
                hr_ds = tf.image.resize(hr_crop_blur, cfg.lr_resolution, tf.image.ResizeMethod.BICUBIC)
                self.update(hr_crop, hr_ds)
            self.epoch.assign_add(1)
            if epoch % cfg.save_freq == 0:
                self.saver.save(file_prefix=self.ckpt_prefix)

def main():
    srgan = SRGAN(cfg)
    srgan.train()

if __name__ == '__main__':
    main()

