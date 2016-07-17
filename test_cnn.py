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

from EllipseToMaskConverter import EllipseToMaskConverter
from Mask2RunLengthConverter import Mask2RunLengthConverter

def test_cnn(n_kerns=(16, 16, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8), batch_size=32):
    rng = numpy.random.RandomState(234555)

    rd = Reader('../data/test', batch_size=batch_size, multiply_image=False, train_size=1.0)
    x = T.tensor4('x', dtype=theano.config.floatX)   # the data is presented as rasterized images

    image_shape = rd.get_img_shape()
    cnn = CNN(rng, x, n_kerns, image_shape)
    if os.path.isfile('model.bin'):
        print 'using earlier model'
        f = open('model.bin', 'rb')
        cnn.__setstate__(cPickle.load(f))
        f.close()

    test_prediction = lasagne.layers.get_output(cnn.dense_layer, deterministic=True)
    test_model = theano.function([x], test_prediction)

    predicted_values = []
    n_batches = len(rd.images)/batch_size
    for i in xrange(n_batches):
        test_images, dummy_masks = rd.get_train_images()
        predicted_values.extend(test_model(test_images))

    denormalized_predicted_values = denormalize_ellipses(predicted_values, image_shape)

    denormalized_predicted_ellipses = convert_to_ellipses(denormalized_predicted_values)

    ellipse2mask_converter = EllipseToMaskConverter()
    mask2runlength_converter = Mask2RunLengthConverter()
    predicted_runlength = []
    for ellipse in denormalized_predicted_ellipses:
        mask = ellipse2mask_converter.convertFromRotatedRectangle(ellipse, image_shape[0], image_shape[1])
        runlength = mask2runlength_converter.convert(mask)
        predicted_runlength.append(runlength)

    with open('result.csv', 'w') as file:
        file.write('img,pixel\n')
        i = 1
        for i in range(len(predicted_runlength)):
            runlength = predicted_runlength[i]
            line = map(str, runlength)
            line = ' '.join(line)
            file.write(str(i) + ',' + line + '\n')

    print('kunec')


def denormalize_ellipses(predicted_ellipses, image_shape):
    denormalized_predicted_masks = []
    for i in range(len(predicted_ellipses)):
        ellipse = predicted_ellipses[i]
        ellipse[ellipse > 1.0] = 1.0
        for j in range(len(ellipse) - 1):
            ellipse[j] = int(ellipse[j] * image_shape[j % 2] + 0.5)
        ellipse[4] *= 360
        denormalized_predicted_masks.append(ellipse)
    return denormalized_predicted_masks


def convert_to_ellipses(predicted_values):
    predicted_ellipses = []
    for value in predicted_values:
        if value[0:len(value)-1].any(0):
            # jak beda dobre maski to usunac podmiane na 1 i odkomentowac continue
            tmp = value[0:len(value)-1]
            tmp[tmp <= 0] = 1
            # continue
        ellipse = ((int(value[0]), int(value[1])), (int(value[2]), int(value[3])), value[4])
        predicted_ellipses.append(ellipse)
    return predicted_ellipses


if __name__ == '__main__':
    test_cnn()