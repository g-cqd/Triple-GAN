import tensorflow as tf


class Triple_GAN:
    def __init__(self, epoch, dataset_name, batch_size):

        self.dataset_name = dataset_name
        self.epoch = epoch
        self.batch = batch_size

        pass

    def discriminator( self, x, y, scope='discriminator', is_training=True, reuse=False):

        pass

if __name__ == "__main__":
    Triple_GAN()
    pass