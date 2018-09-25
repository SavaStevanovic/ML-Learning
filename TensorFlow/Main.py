import TensorFlow.Training as tr
from TensorFlow.TfLinearRegresion import TfLinearRegresion
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

X_train = np.arange(10).reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1,
                    2.0, 5.0, 6.3,
                    6.6, 7.4, 8.0,
                    9.0])

model = TfLinearRegresion(X_train.shape[1])
with tf.Session(graph=model.graph) as sess:
    training_costs = tr.train_model(sess, model, X_train, y_train)

    plt.plot(range(1,len(training_costs)+1),training_costs)
    plt.tight_layout()
    plt.xlabel('Epoch') 
    plt.ylabel('Training Cost') 
    plt.show() 

    plt.scatter(X_train,y_train,marker='s',s=50,label='TrainingData')
    plt.plot(range(X_train.shape[0]),tr.predict_model(sess, model,X_train),color='gray',marker=0,markersize=6,linewidth=3,label='Model')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend
    plt.tight_layout()
    plt.show()