import os
import sys
import datetime
import numpy
import theano
import theano.tensor as T
import lasagne
import pickle as cPickle
from reader import Reader
from conv_network import CNN


def test_cnn(n_kerns=(16, 16, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8), batch_size=32):
    rng = numpy.random.RandomState(234555)

    rd = Reader('~/data/kaggle/nerve/test', batch_size=batch_size, multiply_image=False, train_size=1.0)
    x = T.tensor4('x', dtype=theano.config.floatX)   # the data is presented as rasterized images

    image_shape = rd.get_img_shape()
    cnn = CNN(rng, x, n_kerns, image_shape)
    if os.path.isfile('model.bin'):
        print 'using erlier model'
        f = open('model.bin', 'rb')
        cnn.__setstate__(cPickle.load(f))
        f.close()

    test_prediction = lasagne.layers.get_output(cnn.dense_layer, deterministic=True)
    test_model = theano.function([x], test_prediction)

    n_batches = len(rd.images)/batch_size
    for i in xrange(n_batches):
        test_images, dummy_masks = rd.get_train_images()
        predicted_masks = test_model(test_images)

if __name__ == '__main__':
    test_cnn()