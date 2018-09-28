import tensorflow as tf
import numpy as np
from Data_loading import make_random_data
import matplotlib.pyplot as plt

graph = tf.Graph()
# model
with graph.as_default():
    tf.set_random_seed(123)

    x = tf.placeholder(shape=(None), dtype=tf.float32, name='x')
    y = tf.placeholder(shape=(None), dtype=tf.float32, name='y')

    weight = tf.Variable(tf.random_normal(
        shape=(1, 1), stddev=0.25), name='weight')

    bias = tf.Variable(0., name='bias')

    y_pred = tf.add(weight*x, bias, name='y_pred')

    cost = tf.reduce_mean(tf.square(y-y_pred), name='cost')

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

    train_op = optimizer.minimize(cost, name='train_op')

# data loading

x_data, y_data = make_random_data()

plt.plot(x_data, y_data, 'o')
plt.show()

# train/test splits
X_train, y_train = x_data[:100], y_data[:100]
X_test, y_test = x_data[100:], y_data[100:]

# training
n_epochs = 500
training_costs = []

with graph.as_default():
    saver = tf.train.Saver()
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    for e in range(n_epochs+1):
        c, _ = sess.run(['cost:0', 'train_op'], feed_dict={
            'x:0': X_train,  'y:0': y_train})
        training_costs.append(c)
        if e % 50 == 0:
            print('Epoch %4d: %.4f' % (e, c))

    saver.save(sess, './trained-models/regresion_model')

plt.plot(training_costs)
plt.show()
x_arr = np.arange(-2, 4, 0.1)

graph_to_load=tf.Graph()
with tf.Session(graph=graph_to_load) as sess:
    loader=tf.train.import_meta_graph('./trained-models/regresion_model.meta')
    loader.restore(sess,'./trained-models/regresion_model')
    y_arr=sess.run('y_pred:0',feed_dict={'x:0': x_arr})

plt.figure()
plt.plot(X_train,y_train,'bo')
plt.plot(X_test,y_test,'bo',alpha=0.3)
plt.plot(x_arr,y_arr[0,:], '-r', lw=3)
plt.show()