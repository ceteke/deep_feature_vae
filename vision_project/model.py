import tensorflow as tf
from vision_project.vgg19 import Vgg19


class Model(object):
    def __init__(self, x, vgg_path, num_batches, learning_rate=5e-4, alpha=1, beta=0.5):
        self.x = x
        self.learning_rate = learning_rate
        self.vgg_path = vgg_path
        self.alpha = alpha
        self.beta = beta

        self.vgg = Vgg19(vgg_path)
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.is_train = tf.placeholder(tf.bool)

        decay_lr = tf.train.exponential_decay(self.learning_rate, self.global_step, num_batches, 0.5)
        self.optimizer = tf.train.AdamOptimizer(decay_lr)
        self.build_model()

    def conv_bn_layer(self, input, filters, kernel_size, stride=2, bn=True, activation=tf.nn.leaky_relu):
        output = tf.layers.conv2d(input, filters=filters, kernel_size=kernel_size, strides=stride,
                                  activation=activation)
        if bn:
            return tf.layers.batch_normalization(output, training=self.is_train)
        else:
            return output

    def sq_err(self, t1, t2):
        return tf.reduce_mean(tf.square(t1 - t2))

    def build_model(self):
        el1 = self.conv_bn_layer(self.x, 32, 4)
        el2 = self.conv_bn_layer(el1, 64, 4)
        el3 = self.conv_bn_layer(el2, 128, 4)
        el4 = self.conv_bn_layer(el3, 256, 4)

        el4 = tf.layers.Flatten()(el4)

        mu = tf.layers.dense(el4, 100)
        sigma = tf.layers.dense(el4, 100)

        epsilon = tf.random_normal((tf.shape(mu)[0], 100), mean=0.0, stddev=1.0)
        self.z = mu + epsilon * tf.exp(.5 * sigma)

        fc = tf.layers.dense(self.z, 4096)
        fc = tf.reshape(fc, [-1, 4, 4, 256])

        dl1 = tf.image.resize_nearest_neighbor(fc, size=(8,8))
        dl2 = self.conv_bn_layer(dl1, 128, 3, 1)
        dl2 = tf.image.resize_nearest_neighbor(dl2, size=(16, 16))
        dl3 = self.conv_bn_layer(dl2, 64, 3, 1)
        dl3 = tf.image.resize_nearest_neighbor(dl3, size=(32, 32))
        dl4 = self.conv_bn_layer(dl3, 32, 3, 1)
        dl4 = tf.image.resize_nearest_neighbor(dl4, size=(64, 64))
        output = self.conv_bn_layer(dl4, 3, 3, 1, False, None)

        vgg_layers = self.vgg(self.x)
        vgg_layers_hat = self.vgg(output, reuse=True)

        self.perceptual_loss = self.sq_err(vgg_layers[2], vgg_layers_hat[2])
        self.perceptual_loss += self.sq_err(vgg_layers[3], vgg_layers_hat[3])
        self.perceptual_loss += self.sq_err(vgg_layers[4], vgg_layers_hat[4])
        self.perceptual_loss *= 0.5

        self.kl_loss = tf.reduce_mean(-0.5*tf.reduce_sum(1+sigma-tf.square(mu)-tf.exp(sigma),1))
        self.loss = self.alpha*self.kl_loss + self.beta*self.perceptual_loss

        self.train_op = self.optimizer.minimize(self.loss, self.global_step)