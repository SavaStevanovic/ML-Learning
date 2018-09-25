import tensorflow as tf


class TfLinearRegresion(object):
    def __init__(self, x_dimension, learning_rate=0.01, random_seed=None):
        self.x_dimension = x_dimension
        self.learning_rate = learning_rate
        self.graph = tf.Graph()
        # model
        with self.graph.as_default():
            tf.random_seed = random_seed

            self.build()

            self.init_op = tf.global_variables_initializer()

    def build(self):

        self.X = tf.placeholder(dtype=tf.float32, shape=(
            None, self.x_dimension), name='x_input')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None), name='y_input')

        print(self.X)
        print(self.y)

        w = tf.Variable(tf.zeros(shape=(1)), name='weight')
        b = tf.Variable(tf.zeros(shape=1), name='bias')

        print(w)
        print(b)

        self.z_net = tf.squeeze(w*self.X+b, name='z_net')

        print(self.z_net)

        sqr_errors = tf.square(self.y-self.z_net, name='sqr_error')

        print(sqr_errors)

        self.mean_cost = tf.reduce_mean(sqr_errors, name='mean_cost')

        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate, name='GradientDescent')
        self.optimizer = optimizer.minimize(self.mean_cost)
