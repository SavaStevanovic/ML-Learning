import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import Data_loading as dl
import numpy as np
import TensorFlow.TfNeuralNetwork as nn
import matplotlib.pyplot as plt
import TensorFlow.Training as tr

X_train, y_train = dl.load_mnist('./TensorFlow/mnist/', kind='train')
print('Rows: %d, Columns: %d' % (X_train.shape[0], X_train.shape[1]))

X_test, y_test = dl.load_mnist('./TensorFlow/mnist/', kind='t10k')
print('Rows: %d, Columns: %d' % (X_test.shape[0], X_test.shape[1]))

mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train_centered = (X_train-mean_vals)/std_val
X_test_centered = (X_test-mean_vals)/std_val

del X_test, X_train

print(X_train_centered.shape, y_train.shape)
print(X_test_centered.shape, y_test.shape)

model = nn.NeuralNetwork(len(set(y_train)), X_train_centered.shape[1])
with tf.Session(graph=model.graph) as sess:

    training_costs = tr.train_batch_model(
        sess, model, X_train_centered, X_test_centered, y_train, y_test, num_epochs=100, batch_size=64)