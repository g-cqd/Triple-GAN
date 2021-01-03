import tensorflow                           as tf
import keras                                as ks

class TripleGAN( ks.Model ):
    def __init__( self, discriminator, generator, classifier, dimension ):
        super( TripleGAN, self ).__init__()
        self.discriminator = discriminator
        self.classifier = classifier
        self.generator = generator
        self.dimension = dimension

    def compile( self, dr_opt, gr_opt, cr_opt, loss_function ):
        super( TripleGAN, self ).compile()
        self.dr_opt = dr_opt
        self.gr_opt = gr_opt
        self.cr_opt = cr_opt
        self.loss_function = loss_function

    def train_step( self, real_images ):

        if isinstance( real_images, tuple ):
            real_images = real_images[0]

        # Sample random points in the latent space
        batch_size = tf.shape( real_images )[0]
        random_latent_vectors = tf.random.normal( shape = ( batch_size, self.dimension ) )

        # Decode them to fake images
        generated_images = self.generator( random_latent_vectors )

        # Combine them with real images
        combined_images = tf.concat(
            [generated_images, real_images],
            axis = 0
        )

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones( ( batch_size, 1 ) ), tf.zeros( ( batch_size, 1 ) )],
            axis = 0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform( tf.shape( labels ) )

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator( combined_images )
            dr_loss = self.loss_function( labels, predictions )
        
        grads = tape.gradient( dr_loss, self.discriminator.trainable_weights )
        self.dr_opt.apply_gradients(
            zip( grads, self.discriminator.trainable_weights )
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal( shape = ( batch_size, self.dimension ) )

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros( ( batch_size, 1 ) )

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            gr_loss = self.loss_function(misleading_labels, predictions)

        grads = tape.gradient( gr_loss, self.generator.trainable_weights )
        self.gr_opt.apply_gradients( zip( grads, self.generator.trainable_weights ) )

        return {
            "dr_loss": dr_loss,
            "gr_loss": gr_loss
        }
