import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import Data_loading as dl
import numpy as np
import matplotlib.pyplot as plt
import TensorFlow.Training as tr
import TensorFlow.TfConvolutionalNetwork as cnn
import TensorFlow.ModelIO as mio

X_data, y_data = dl.load_mnist('./TensorFlow/mnist/', kind='train')
print('Rows: %d, Columns: %d' % (X_data.shape[0], X_data.shape[1]))

X_test, y_test = dl.load_mnist('./TensorFlow/mnist/', kind='t10k')
print('Rows: %d, Columns: %d' % (X_test.shape[0], X_test.shape[1]))

X_train, y_train = X_data[:50000, :], y_data[:50000]
X_valid, y_valid = X_data[50000:, :], y_data[50000:]

print('Training:   ', X_train.shape, y_train.shape)
print('Validation: ', X_valid.shape, y_valid.shape)
print('Test Set:   ', X_test.shape, y_test.shape)

mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train_centered = (X_train-mean_vals)/std_val
X_valid_centered = (X_valid-mean_vals)/std_val
X_test_centered = (X_test-mean_vals)/std_val

random_seed = 123
learning_rate = 1e-4
graph = tf.Graph()
with graph.as_default():
    tf.set_random_seed(random_seed)
    cnn.build_cnn(learning_rate=learning_rate)

    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    tr.train(sess, training_set=(X_train_centered, y_train), validation_set=(
        X_valid_centered, y_valid), initialize=True, random_seed=123, epochs=20)
    mio.save(saver, sess, epoch=20)

graph2 = tf.Graph()
with graph2.as_default():
    tf.set_random_seed(random_seed)
    cnn.build_cnn(random_seed)
    saver = tf.train.Saver()

with tf.Session(graph=graph2) as sess:
    mio.load(saver=saver, sess=sess, epoch=20, path='./trained-models/')

    predictions= tr.predict(sess=sess,X_test=X_test_centered,return_proba=False)

    print('Test Accuracy: %.3f%%' % (100*np.sum(predictions == y_test)/len(y_test))) 