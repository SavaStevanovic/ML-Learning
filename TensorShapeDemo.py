
import tensorflow as tf 
import numpy as np 

g = tf.Graph() 
with g.as_default(): 
    arr = np.array([[1., 2., 3., 3.5], 
                    [4., 5., 6., 6.5], 
                    [7., 8., 9., 9.5]]) 
    T1 = tf.constant(arr, name='T1') 
    print(T1) 
    s = T1.get_shape() 
    print('Shape of T1 is', s) 
    T2 = tf.Variable(tf.random_normal( 
        shape=s)) 
    print(T2) 
    T3 = tf.Variable(tf.random_normal( 
        shape=(s.as_list()[0],))) 
    print(T3) 


with g.as_default(): 
    T4 = tf.reshape(T1, shape=[1, 1, -1], 
                    name='T4') 
    print(T4) 
    T5 = tf.reshape(T1, shape=[1, 3, -1], 
                    name='T5') 
    print(T5) 


with tf.Session(graph = g) as sess: 
    print(sess.run(T4)) 
    print() 
    print(sess.run(T5)) 


with g.as_default(): 
    T6 = tf.transpose(T5, perm=[2, 1, 0], 
                    name='T6')

    print(T6) 
    T7 = tf.transpose(T5, perm=[0, 2, 1], 
                    name='T7') 
    print(T7)


with g.as_default(): 
    t5_splt = tf.split(T5, 
                    num_or_size_splits=2, 
                    axis=2, name='T8') 
    print(t5_splt) 


g = tf.Graph() 
with g.as_default(): 
    t1 = tf.ones(shape=(5, 1), 
                dtype=tf.float32, name='t1') 
    t2 = tf.zeros(shape=(5, 1), 
                dtype=tf.float32, name='t2') 
    print(t1) 
    print(t2) 
with g.as_default(): 
    t3 = tf.concat([t1, t2], axis=0, name='t3') 
    print(t3) 
    t4 = tf.concat([t1, t2], axis=1, name='t4') 
    print(t4) 


with tf.Session(graph=g) as sess: 
    print(t3.eval()) 
    print() 
    print(t4.eval())