import tensorflow                           as tf
import keras                                as ks

class MonoGAN_Monitor( ks.callbacks.Callback ):
    def __init__( self, image_number = 100, dimension = 128 ):
        self.image_number = image_number
        self.dimension  = dimension

    def on_epoch_end( self, epoch, logs = None ):
        random_latent_vectors = tf.random.normal( shape = ( self.image_number, self.dimension ) )
        generated_images = self.model.generator( random_latent_vectors )
        generated_images *= 255
        generated_images.numpy()
        for i in range( self.num_img ):
            img = ks.preprocessing.image.array_to_img( generated_images[i] )
            img.save( "./gan-outputs/generated_img_{i}_{epoch}.png".format( i = i, epoch = epoch ) )
