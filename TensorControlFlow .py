import tensorflow as tf


graph = tf.Graph()
with graph.as_default():
    tf_x = tf.placeholder(shape=None, dtype=tf.float32, name='tf_x')
    tf_y = tf.placeholder(shape=None, dtype=tf.float32, name='tf_y')

    # if
    r = tf.cond(tf.less(tf_x, tf_y),
                lambda: tf.add(tf_x, tf_y, name='result_add'),
                lambda: tf.subtract(tf_x, tf_y, name='result_sub'))

    # case
    def f1(): return tf.constant(-1)

    def f0(): return tf.constant(0)

    def f2(): return tf.constant(1)

    c = tf.case([(tf.less(tf_x, tf_y), f1),
                 (tf.equal(tf_x, tf_y), f0)], default=f2)

    #while
    i=tf.constant(0)
    f_cond=lambda i:tf.less(i,100)
    f_add=lambda i: tf.add(i,1)
    w=tf.while_loop(cond=f_cond,body=f_add,loop_vars=[i])


with tf.Session(graph=graph) as sess:
    print('Object:', r)
    x, y = 1., 2.
    print('x < y: %s -> Result:' % (x < y),
          r.eval(feed_dict={'tf_x:0': x, 'tf_y:0': y}))

    x, y = 2., 1.
    print('x < y: %s -> Result:' % (x < y),
          r.eval(feed_dict={'tf_x:0': x, 'tf_y:0': y}))

    x, y = 2., 1.
    print('%.2f compared to %.2f: -> Result:' % (x, y),
          c.eval(feed_dict={'tf_x:0': x, 'tf_y:0': y}))

    x, y = 1., 1.
    print('%.2f compared to %.2f: -> Result:' % (x, y),
          c.eval(feed_dict={'tf_x:0': x, 'tf_y:0': y}))

    x, y = 2., 3.
    print('%.2f compared to %.2f: -> Result:' % (x, y),
          c.eval(feed_dict={'tf_x:0': x, 'tf_y:0': y}))
    
    print(w.eval())
