import os
import struct
import numpy as np


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:

        magic, n = struct.unpack('>II',  lbpath.read(8))

        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:

        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))

        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

        images = ((images / 255.) - .5) * 2
    return images, labels


def make_random_data(): 
    x = np.random.uniform(low=-2, high=4, size=200) 
    y = [] 
    for t in x: 
        r = np.random.normal(loc=0.0, 
                            scale=(0.5 + t*t/3), 
                            size=None) 
        y.append(r) 
    return  x, 1.726*x -0.84 + np.array(y) 