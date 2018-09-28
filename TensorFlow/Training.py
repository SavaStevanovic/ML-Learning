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


def train(sess, training_set, validation_set=None, initialize=True, epochs=20, shuffle=True, dropout=0.5, random_seed=None):

    X_data = np.array(training_set[0])
    y_data = np.array(training_set[1])
    training_loss = []

    if initialize:
        sess.run(tf.global_variables_initializer())

    np.random.seed(random_seed)
    for epoch in range(1, epochs+1):
        batch_gen = db.create_batch_generator(X_data, y_data, shuffle=shuffle)

        avg_loss = 0.0
        for i, (batch_x, batch_y) in enumerate(batch_gen):
            feed = {'tf_x:0': batch_x, 'tf_y:0': batch_y,
                    'fc_keep_prob:0': dropout}

            loss, _ = sess.run(
                ['cross_entropy_loss:0', 'train_op'], feed_dict=feed)
            avg_loss += loss

        training_loss.append(avg_loss/(i+1))
        print('Epoch %02d Training Avg. Loss: %7.3f' %
              (epoch, avg_loss), end=' ')

        if validation_set is not None:
            feed = {'tf_x:0': validation_set[0], 'tf_y:0': validation_set[1],
                    'fc_keep_prob:0': 1.}
            validation_acc = sess.run('accuracy:0', feed_dict=feed)
            print(' Validation Acc: %7.3f' % validation_acc)
        else:
            print()


def predict(sess, X_test, return_proba=False):
    feed = {'tf_x:0': X_test, 'fc_keep_prob:0': 1.}
    if return_proba:
        return sess.run('probabilities:0', feed_dict=feed)
    return sess.run('labels:0', feed_dict=feed)
