import tensorflow as tf


class NeuralNetwork(object):
    def __init__(self, class_count, feature_count,learning_rate=0.001, random_seed=None):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(random_seed)
            self.x = tf.placeholder(dtype=tf.float32, shape=(
                None, feature_count), name='x_input')

            self.y = tf.placeholder(dtype=tf.int32, shape=None, name='y_input')
            self.y_onehot = tf.one_hot(indices=self.y, depth=class_count)
            h1 = tf.layers.dense(inputs=self.x, units=50,
                                activation=tf.tanh, name='layer_1')
            h2 = tf.layers.dense(inputs=h1, units=50,
                                activation=tf.tanh, name='layer_2')
            activations = tf.layers.dense(inputs=h2, units=class_count,
                                activation=None, name='layer_3')
            self.predictions={
                'classes':tf.argmax(activations,axis=1,name='predicted_classes'),
                'probabilities':tf.nn.softmax(activations,name='softmax_tensor')
            }

            self.cost=tf.losses.softmax_cross_entropy(onehot_labels=self.y_onehot,logits=activations)

            self.optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

            self.train_op=self.optimizer.minimize(loss=self.cost)

            self.init_op = tf.global_variables_initializer()
        

