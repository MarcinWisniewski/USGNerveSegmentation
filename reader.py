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

IMAGE_CROP = 8
IMAGE_MULTIPLIER = IMAGE_CROP**2
IMAGE_SHAPE = (420.0/2.0, 580.0/2.0)


class Reader(object):
    def __init__(self, img_path, batch_size, multiply_image=True, train_size=0.8):
        '''
        :param img_path: path to train or test images
        :param batch_size: number of images in minibatch (must be multiplication of 64 )
        '''
        self.multiply_image = multiply_image
        self.minibatch = batch_size
        self.mask2ellipse = Mask2EllipseConverter()
        self.img_path = os.path.expanduser(img_path)
        self.images = glob.glob(self.img_path+'/*[0-9].tif')
        #self.images = glob.glob(self.img_path+'/10_100.tif')

        if train_size < 1.0:
            self.train_images, test_valid_images, \
                       dummy1, dummy2 = train_test_split(self.images,
                                                         np.ones(len(self.images)),
                                                         train_size=train_size,
                                                         random_state=1234)
            self.test_images, self.valid_images, \
                      dummy1, dummy2 = train_test_split(test_valid_images,
                                                        np.ones(len(test_valid_images)),
                                                        train_size=0.5, random_state=1234)
        else:
            self.train_images = self.images
            self.test_images, self.valid_images = None, None

        self.train_images_buffer = None
        self.test_images_buffer = None
        self.valid_images_buffer = None
        self.train_images_index = 0
        self.test_images_index = 0
        self.valid_images_index = 0

    def get_train_images(self):
        if self.train_images_index >= len(self.train_images):
            self.train_images_index = 0
        if self.train_images_buffer is None:
            self.train_images_buffer = self._fill_image_buffers(self.train_images, self.train_images_index)
            self.train_images_index += 1

        while len(self.train_images_buffer[0]) < self.minibatch:
            images_to_add = self._fill_image_buffers(self.train_images, self.train_images_index)
            self.train_images_buffer[0] = np.append(self.train_images_buffer[0], images_to_add[0], axis=0)
            self.train_images_buffer[1] = np.append(self.train_images_buffer[1], images_to_add[1], axis=0)
            self.train_images_index += 1

        minibatch_images = (self.train_images_buffer[0][:self.minibatch],
                            self.train_images_buffer[1][:self.minibatch])
        self.train_images_buffer[0] = np.delete(self.train_images_buffer[0], range(self.minibatch), axis=0)
        self.train_images_buffer[1] = np.delete(self.train_images_buffer[1], range(self.minibatch), axis=0)
        return minibatch_images

    def get_test_images(self):
        if self.test_images_index >= len(self.test_images):
            self.test_images_index = 0
        if self.test_images_buffer is None:
            self.test_images_buffer = self._fill_image_buffers(self.test_images, self.test_images_index)
            self.test_images_index += 1

        while len(self.test_images_buffer[0]) < self.minibatch:
            images_to_add = self._fill_image_buffers(self.test_images, self.test_images_index)
            self.test_images_buffer[0] = np.append(self.test_images_buffer[0], images_to_add[0], axis=0)
            self.test_images_buffer[1] = np.append(self.test_images_buffer[1], images_to_add[1], axis=0)
            self.test_images_index += 1

        minibatch_images = (self.test_images_buffer[0][:self.minibatch],
                            self.test_images_buffer[1][:self.minibatch])
        self.test_images_buffer[0] = np.delete(self.test_images_buffer[0], range(self.minibatch), axis=0)
        self.test_images_buffer[1] = np.delete(self.test_images_buffer[1], range(self.minibatch), axis=0)
        return minibatch_images

    def get_valid_images(self):
        if self.valid_images_index >= len(self.valid_images):
            self.valid_images_index = 0
        if self.valid_images_buffer is None:
            self.valid_images_buffer = self._fill_image_buffers(self.valid_images, self.valid_images_index)
            self.valid_images_index += 1

        while len(self.valid_images_buffer[0]) < self.minibatch:
            images_to_add = self._fill_image_buffers(self.valid_images, self.valid_images_index)
            self.valid_images_buffer[0] = np.append(self.valid_images_buffer[0], images_to_add[0], axis=0)
            self.valid_images_buffer[1] = np.append(self.valid_images_buffer[1], images_to_add[1], axis=0)
            self.valid_images_index += 1

        minibatch_images = (self.valid_images_buffer[0][:self.minibatch],
                            self.valid_images_buffer[1][:self.minibatch])
        self.valid_images_buffer[0] = np.delete(self.valid_images_buffer[0], range(self.minibatch), axis=0)
        self.valid_images_buffer[1] = np.delete(self.valid_images_buffer[1], range(self.minibatch), axis=0)
        return minibatch_images

    def get_length_of_training_data(self):
        return len(self.train_images)*IMAGE_MULTIPLIER//self.minibatch - 1

    def get_length_of_testing_data(self):
        return len(self.test_images)*IMAGE_MULTIPLIER//self.minibatch - 1

    def get_length_of_valid_data(self):
        return len(self.valid_images)*IMAGE_MULTIPLIER//self.minibatch - 1

    def get_number_of_all_images(self):
        return len(self.images)*IMAGE_MULTIPLIER

    def get_number_of_training_images(self):
        return len(self.train_images)*IMAGE_MULTIPLIER

    def get_img_shape(self):
        return IMAGE_SHAPE[0] - IMAGE_CROP, IMAGE_SHAPE[1] - IMAGE_CROP

    def _fill_image_buffers(self, image_type, img_iterator):
        images = image_type
        img = cv2.imread(images[img_iterator], 0)
        if os.path.isfile(images[img_iterator].replace('.', '_mask.')):
            mask = cv2.imread(images[img_iterator].replace('.', '_mask.'), 0)
        else:
            mask = np.zeros(img.shape, dtype='uint8')

        img, mask = self._resize_img_and_mask(img, mask)
        if self.multiply_image:
            multiplied_img, multiplied_mask = self._multiply_images(img, mask)
        else:
            multiplied_img, multiplied_mask = img[: -IMAGE_CROP, : -IMAGE_CROP]/255.0, mask[: -IMAGE_CROP, : -IMAGE_CROP]
            multiplied_mask = np.expand_dims(multiplied_mask, axis=0)
            multiplied_img = np.asarray(np.expand_dims(multiplied_img, axis=0), dtype=theano.config.floatX)

        transformed_multiplied_mask = self._convert_mask_to_ellipse_params(multiplied_mask)
        multiplied_img = np.asarray(np.expand_dims(multiplied_img, 1), dtype=theano.config.floatX)
        transformed_multiplied_mask = np.asarray(transformed_multiplied_mask, dtype=theano.config.floatX)
        return [multiplied_img, transformed_multiplied_mask]

    def _multiply_images(self, img, mask):
        crop_img_shape = np.asarray(self.get_img_shape())
        multiplied_img = np.zeros(np.append(IMAGE_MULTIPLIER, crop_img_shape), dtype=theano.config.floatX)
        multiplied_mask = np.copy(multiplied_img)
        multiplied_mask = np.asarray(multiplied_mask, dtype='uint8')
        img_num = 0
        for y_axis in xrange(IMAGE_CROP):
            for x_axis in xrange(IMAGE_CROP):
                multiplied_img[img_num] = img[y_axis:y_axis + crop_img_shape[0],
                                              x_axis:x_axis + crop_img_shape[1]] / 255.0
                multiplied_mask[img_num] = mask[y_axis:y_axis + crop_img_shape[0],
                                                x_axis:x_axis + crop_img_shape[1]]
                img_num += 1

        return multiplied_img, multiplied_mask

    def _convert_mask_to_ellipse_params(self, multiplied_masks):
        transformed_multiplied_mask = np.zeros((len(multiplied_masks), 2), dtype=theano.config.floatX)
        for image_num in xrange(len(multiplied_masks)):
            temp_transformed_multiplied_mask = \
                    self.mask2ellipse.convert(multiplied_masks[image_num])

            transformed_multiplied_mask[image_num] = temp_transformed_multiplied_mask[:2] / \
                                                     ((IMAGE_SHAPE[0]-IMAGE_CROP),
                                                      (IMAGE_SHAPE[1]-IMAGE_CROP))
                                                     #IMAGE_SHAPE[0]-IMAGE_CROP,
                                                     #IMAGE_SHAPE[1]-IMAGE_CROP,
                                                     #360.0)
        return transformed_multiplied_mask

    @staticmethod
    def _resize_img_and_mask(img, mask):
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        mask = cv2.resize(mask, None, fx=0.5, fy=0.5)
        return img, mask


if __name__ == '__main__':
    rd = Reader('~/data/kaggle/nerve/train', 32, multiply_image=True)
    iters = rd.get_length_of_training_data()
    for i in xrange(iters):
        train = rd.get_train_images()
        print i

    iters = rd.get_length_of_testing_data()
    for i in xrange(iters):
        test = rd.get_test_images()
        print i

    print iters

    iters = rd.get_length_of_valid_data()
    for i in xrange(iters):
        valid = rd.get_valid_images()
        print i
    print iters