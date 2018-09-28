import tensorflow as tf
import numpy as np


def conv_layer(input_tensor, name,
               kernel_size, n_output_channels,
               padding_mode='SAME', strides=(1, 1, 1, 1)):

    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()
        n_input_channels = input_shape[-1]

        weights_shape = list(kernel_size)+[n_input_channels, n_output_channels]
        weights = tf.get_variable(name='_weights', shape=weights_shape)

        print(weights)

        biases = tf.get_variable(
            name='_biases', initializer=tf.zeros(shape=[n_output_channels]))
        print(biases)

        conv = tf.nn.conv2d(input=input_tensor, filter=weights,
                            strides=strides, padding=padding_mode)
        print(conv)

        conv = tf.nn.bias_add(value=conv, bias=biases,
                              name='net_pre-activation')
        print(conv)

        conv = tf.nn.relu(conv, name='activation')
        print(conv)

        return conv


def fc_layer(input_tensor, name, n_output_units, activation_fn=None):
    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()[1:]
        n_input_units = np.prod(input_shape)
        if len(input_shape) > 1:
            input_tensor = tf.reshape(input_tensor, shape=(-1, n_input_units))

        weights_shape = [n_input_units, n_output_units]
        weights = tf.get_variable(name='_weights', shape=weights_shape)
        print(weights)

        biases = tf.get_variable(
            name='_biases', initializer=tf.zeros(shape=[n_output_units]))
        print(biases)

        layer = tf.matmul(input_tensor, weights)
        print(layer)

        layer = tf.nn.bias_add(layer, biases, name='net_pre-activation')
        print(layer)

        if activation_fn is None:
            return layer

        layer = activation_fn(layer, name='activation')
        print(layer)

        return layer


def build_cnn(learning_rate):
    tf_x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='tf_x')
    tf_y = tf.placeholder(dtype=tf.int32, shape=[None], name='tf_y')

    tf_x_image = tf.reshape(tf_x, shape=[-1, 28, 28, 1], name='tf_x_reshaped')
    tf_y_onehot = tf.one_hot(indices=tf_y, depth=10,
                             dtype=tf.int32, name='tf_y_onehot')

    print('\nBuilding first layer:')
    h1 = conv_layer(input_tensor=tf_x_image, name='conv_1',
                    kernel_size=(5, 5), n_output_channels=32,
                    padding_mode='VALID')

    h1_pool = tf.nn.max_pool(h1, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')

    print('\nBuilding second layer:')
    h2 = conv_layer(input_tensor=h1_pool, name='conv_2',
                    kernel_size=(5, 5), n_output_channels=64,
                    padding_mode='VALID')

    h2_pool = tf.nn.max_pool(h2, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')

    print('\nBuilding third layer:')
    h3 = fc_layer(input_tensor=h2_pool, name='fc_3',
                  n_output_units=1024,
                  activation_fn=tf.nn.relu)

    keep_proba = tf.placeholder(dtype=tf.float32, name='fc_keep_prob')
    h3_drop = tf.nn.dropout(h3, keep_prob=keep_proba, name='dropout_layer')

    h4 = fc_layer(input_tensor=h3_drop, name='fc_4',
                  n_output_units=10, activation_fn=None)

    predictions = {
        'probabilities': tf.nn.softmax(h4, name='probabilities'),
        'labels': tf.cast(tf.argmax(h4, axis=1), tf.int32, name='labels')
    }

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=h4, labels=tf_y_onehot), name='cross_entropy_loss')

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = optimizer.minimize(cross_entropy_loss, name='train_op')

    correct_predictions = tf.equal(
        predictions['labels'], tf_y, name='correct_predictions')

    accuracy = tf.reduce_mean(
        tf.cast(correct_predictions, tf.float32), name='accuracy')
