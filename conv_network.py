__author__ = 'Marcin'
import lasagne


class CNN(object):
    def __init__(self, rng, input, n_kerns, img_shape):

        self.layer0_input = input

        self.cnn_layer = lasagne.layers.InputLayer(shape=(None, 1, int(img_shape[0]), int(img_shape[1])),
                                                   input_var=self.layer0_input)

        self.cnn_layer = lasagne.layers.Conv2DLayer(incoming=self.cnn_layer, num_filters=n_kerns[0],
                                                    filter_size=(5, 5), stride=(4, 4), pad='same',
                                                    W=lasagne.init.GlorotNormal(),
                                                    nonlinearity=lasagne.nonlinearities.rectify)

        self.cnn_layer = lasagne.layers.Conv2DLayer(incoming=self.cnn_layer, num_filters=n_kerns[1],
                                                    filter_size=(3, 3), pad='same',
                                                    W=lasagne.init.GlorotNormal(),
                                                    nonlinearity=lasagne.nonlinearities.rectify)
        self.cnn_layer = lasagne.layers.MaxPool2DLayer(self.cnn_layer, pool_size=(2, 2))

        self.cnn_layer = lasagne.layers.Conv2DLayer(incoming=self.cnn_layer, num_filters=n_kerns[2],
                                                    filter_size=(3, 3), pad='same',
                                                    W=lasagne.init.GlorotNormal(),
                                                    nonlinearity=lasagne.nonlinearities.rectify)
        self.cnn_layer = lasagne.layers.MaxPool2DLayer(self.cnn_layer, pool_size=(2, 2))

        self.cnn_layer = lasagne.layers.Conv2DLayer(incoming=self.cnn_layer, num_filters=n_kerns[3],
                                                    filter_size=(3, 3), pad='same',
                                                    W=lasagne.init.GlorotNormal(),
                                                    nonlinearity=lasagne.nonlinearities.rectify)

        self.dense_layer = lasagne.layers.DenseLayer(lasagne.layers.dropout(self.cnn_layer, p=.5),
                                                     W=lasagne.init.GlorotNormal(),
                                                     num_units=16, nonlinearity=lasagne.nonlinearities.rectify)
        self.dense_layer = lasagne.layers.DenseLayer(lasagne.layers.dropout(self.dense_layer, p=.5),
                                                     W=lasagne.init.GlorotNormal(),
                                                     num_units=16, nonlinearity=lasagne.nonlinearities.rectify)

        self.dense_layer = lasagne.layers.DenseLayer(self.dense_layer, num_units=5,
                                                     W=lasagne.init.GlorotNormal(),
                                                     nonlinearity=lasagne.nonlinearities.identity)

        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):

    def __getstate__(self):
        return lasagne.layers.get_all_param_values(self.dense_layer)

    def __setstate__(self, weights):
        lasagne.layers.set_all_param_values(self.dense_layer, weights)

    def get_number_of_parameters(self):
        return lasagne.layers.count_params(self.dense_layer)