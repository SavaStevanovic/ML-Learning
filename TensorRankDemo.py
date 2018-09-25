import tensorflow as tf
import numpy as np

g = tf.Graph()

# define the computation graph
with g.as_default():
     # define tensors t1, t2, t3
     t1 = tf.constant(np.pi)
     t2 = tf.constant([1, 2, 3, 4])
     t3 = tf.constant([[1, 2], [3, 4]])

     # get their ranks
     r1 = tf.rank(t1)
     r2 = tf.rank(t2)
     r3 = tf.rank(t3)

     # get their shapes
     s1 = t1.get_shape()
     s2 = t2.get_shape()
     s3 = t3.get_shape()
     print('Shapes:', s1, s2, s3)

with tf.Session(graph=g) as sess:
     print('Ranks:',
           r1.eval(),
           r2.eval(),
           r3.eval())