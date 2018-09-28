import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import Data_loading as dl
import numpy as np
import matplotlib.pyplot as plt
import TensorFlow.Training as tr
import TensorFlow.TfLayersConvolutionalNetwork as cnn
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

epoch=20

cnn1 = cnn.ConvNN(random_seed=123, epochs=epoch)

cnn1.train(training_set=(X_train_centered, y_train),
          validation_set=(X_test_centered, y_test),
          initialize=True,
          )
cnn1.save(epoch=epoch)

del cnn1

cnn2 = cnn.ConvNN(random_seed=123)
cnn2.load(epoch=epoch, path='./tflayers-model/')

print(cnn2.predict(X_test_centered[:10, :]))

preds = cnn2.predict(X_test_centered)
print('Test Accuracy: %.2f%%' % (100 *
                                 np.sum(y_test == preds)/len(y_test)))
