import numpy as np
import os
import glob
import cv2
import theano
from sklearn.cross_validation import train_test_split
from Mask2EllipseConverter import Mask2EllipseConverter
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

IMAGE_MULTIPLIER = 8
IMAGE_SHAPE = (420.0, 580.0)
IMAGE_SHAPE = (IMAGE_SHAPE[0]/2.0, IMAGE_SHAPE[1]/2.0)

class Reader(object):
    def __init__(self, img_path, batch_size, train_size=0.8):
        '''
        :param img_path: path to train or test images
        :param batch_size: number of images in minibatch (must be multiplication of 64 )
        '''
        assert batch_size % IMAGE_MULTIPLIER ** 2 == 0, ('minibatch must be multiplication of %d' % IMAGE_MULTIPLIER ** 2)
        self.minibatch = batch_size
        self.mask2ellipse = Mask2EllipseConverter()
        self.img_path = os.path.expanduser(img_path)
        self.images = glob.glob(self.img_path+'/*[0-9].tif')
        self.train_images, test_valid_images, \
                   dummy1, dummy2 = train_test_split(self.images,
                                                     np.ones(len(self.images)),
                                                     train_size=train_size,
                                                     random_state=1234)
        self.test_images, self.valid_images, \
                  dummy1, dummy2 = train_test_split(test_valid_images,
                                                    np.ones(len(test_valid_images)),
                                                    train_size=0.5, random_state=1234)

    def get_train_images(self, img_iterator):
        return self._get_images(self.train_images, img_iterator)

    def get_test_images(self, img_iterator):
        return self._get_images(self.test_images, img_iterator)

    def get_valid_images(self, img_iterator):
        return self._get_images(self.valid_images, img_iterator)

    def get_length_of_training_data(self):
        return len(self.train_images)

    def get_length_of_testing_data(self):
        return len(self.test_images)

    def get_length_of_valid_data(self):
        return len(self.valid_images)

    def get_number_of_all_images(self):
        return len(self.images)*IMAGE_MULTIPLIER**2

    def get_number_of_training_images(self):
        return len(self.train_images)*IMAGE_MULTIPLIER**2

    def get_img_shape(self):
        return IMAGE_SHAPE[0] - IMAGE_MULTIPLIER, IMAGE_SHAPE[1] - IMAGE_MULTIPLIER

    def _get_images(self, image_type, img_iterator):
        images = image_type
        crop_img_shape = (np.asarray(IMAGE_SHAPE)-IMAGE_MULTIPLIER)
        multiplied_img = np.zeros(np.append(self.minibatch, crop_img_shape), dtype=theano.config.floatX)
        number_of_images_to_read = self.minibatch/IMAGE_MULTIPLIER**2
        multiplied_mask = np.copy(multiplied_img)
        transformed_multiplied_mask = np.zeros((self.minibatch, 5), dtype=theano.config.floatX)
        img_num = 0
        for i in xrange(number_of_images_to_read):
            img_iterator += i
            img = cv2.imread(images[img_iterator], 0)
            mask = cv2.imread(images[img_iterator].replace('.', '_mask.'), 0)
            img = cv2.resize(img, None, fx=0.5, fy=0.5)
            mask = cv2.resize(mask, None, fx=0.5, fy=0.5)

            for y_axis in xrange(IMAGE_MULTIPLIER):
                for x_axis in xrange(IMAGE_MULTIPLIER):
                    multiplied_img[img_num] = img[y_axis:y_axis + crop_img_shape[0],
                                              x_axis:x_axis+crop_img_shape[1]] / 255.0
                    multiplied_mask[img_num] = mask[y_axis:y_axis + crop_img_shape[0],
                                               x_axis:x_axis+crop_img_shape[1]]
                    temp_transformed_multiplied_mask = \
                        self.mask2ellipse.convert(mask[y_axis:y_axis + crop_img_shape[0],
                                                  x_axis:x_axis+crop_img_shape[1]])[2]
                    temp_transformed_multiplied_mask = self._convert_tuples_to_list(temp_transformed_multiplied_mask)

                    transformed_multiplied_mask[img_num] = temp_transformed_multiplied_mask / \
                        (IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[0], IMAGE_SHAPE[1], 360.0)
                    img_num += 1
        multiplied_img = np.asarray(np.expand_dims(multiplied_img, 1), dtype=theano.config.floatX)
        transformed_multiplied_mask = np.asarray(transformed_multiplied_mask, dtype=theano.config.floatX)
        return multiplied_img, transformed_multiplied_mask, img_iterator

    @staticmethod
    def _convert_tuples_to_list(tuples):
        result_list = []
        for element in tuples:
            if type(element) == float:
                element = [element]
            result_list += element
        return np.asarray(result_list, dtype=theano.config.floatX)


if __name__ == '__main__':
    rd = Reader('~/data/kaggle/nerve/train', 128)
    train = rd.get_train_images(0)
    test = rd.get_test_images(0)
