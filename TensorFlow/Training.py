import tensorflow as tf
import Data_batching as db
import numpy as np


def train_model(sess, model, X_train, y_train, num_epochs=10):
    sess.run(model.init_op)

    training_costs = []
    for _ in range(num_epochs):
        _, cost = sess.run([model.optimizer, model.mean_cost], feed_dict={
            model.X: X_train, model.y: y_train})
        training_costs.append(cost)

    return training_costs


def train_batch_model(sess, model, X_train, X_test_centered, y_train, y_test, num_epochs=10, batch_size=64):
    sess.run(model.init_op)

    for epoch in range(num_epochs):
        training_costs = []
        batch_generator = db.create_batch_generator(
            X_train, y_train, batch_size=batch_size, shuffle=True)

        for batch_X, batch_y in batch_generator:
            feed = {model.x: batch_X, model.y: batch_y}
            _, batch_cost = sess.run(
                [model.train_op, model.cost], feed_dict=feed)
        training_costs.append(batch_cost)
        print(' -- Epoch %2d  Avg. Training Loss: %.4f' %
              (epoch+1, np.mean(training_costs)))
    feed = {model.x: X_test_centered, model.y: y_test}
    y_pred = sess.run(model.predictions['classes'], feed_dict=feed)
    print('Test Accuracy: %.2f%%' %
          (100*np.sum(y_pred == y_test)/y_test.shape[0]))


def predict_model(sess, model, X_test):
    y_pred = sess.run(model.z_net,
                      feed_dict={model.X: X_test})
    return y_pred
