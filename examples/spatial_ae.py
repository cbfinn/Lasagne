#!/usr/bin/env python

"""
Usage example employing Lasagne for pose detection using our dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import sys
import os
import time
#import caffe

import numpy as np
import pickle
import theano
import theano.tensor as T

import lasagne
import lasagne.layers.dnn

from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import SoftmaxDNNLayer as SoftmaxLayer
from lasagne.utils import floatX

# Image conversions
from skimage import color
import scipy.misc


# ################## Prepare the dataset ##################
def load_dataset():
    def load_images(filename, mean_img = None):
        data = np.float32(np.load(filename))

        if mean_img is None:
            mean_img = data.mean(axis=0)

        data_submean = data-mean_img # Subtract mean

        tgt_imgs = np.float32(np.zeros((data.shape[0],3600)))
        for i in range(data.shape[0]):
          img = color.rgb2gray(scipy.misc.imresize(np.transpose(data_submean[i], (2,1,0)), 0.25))
          tgt_imgs[i,:] = np.reshape(img.T,(3600,))

        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        #return data / np.float32(256), tgt_imgs / np.float32(256)
        return data, tgt_imgs / np.float32(256), mean_img
        #return data / np.float32(256)

    def load_labels(filename):
        data = np.float32(np.load(filename))
        # The labels are vectors of floats now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train, y_train, mean_img = load_images('/home/cfinn/data/train_legopush2.npy')
    #y_train = load_labels('/home/cfinn/data/labels_test.npy')
    X_val, y_val, _ = load_images('/home/cfinn/data/val_legopush2.npy', mean_img)
    #y_val = load_labels('/home/cfinn/data/labels_test_v.npy')

    # We reserve the last 100 training examples for validation.
    #X_train, X_val = X_train[:-100], X_train[-100:]
    #y_train, y_val = y_train[:-100], y_train[-100:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val #, y_train, X_val, y_val #, X_test, y_test


# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

def build_cnn(input_var=None, get_fp=False):
    # As a third model, we'll create a CNN of three convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    net = {}
    net['input'] = lasagne.layers.InputLayer(shape=(None, 3, 240, 240),
                                             input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    net['conv1'] =ConvLayer(net['input'],
            num_filters=64,filter_size=7,
            nonlinearity=lasagne.nonlinearities.rectify,
            stride=2,flip_filters=False,
            W=lasagne.init.Normal())
    params = pickle.load(open('/home/cfinn/code/Lasagne/Recipes/modelzoo/blvc_googlenet.pkl'))['param values']
    net['conv1'].W.set_value(params[0])
    net['conv1'].b.set_value(params[1])

    net['conv2'] = ConvLayer(net['conv1'],
            num_filters=32,filter_size=5,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(std=0.001))
    net['conv3'] = ConvLayer(net['conv2'],
            num_filters=32,filter_size=5,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(std=0.001))  # This controls the initial softmax distr

    net['reshape1'] = ReshapeLayer(net['conv3'],shape=(-1,1,109,109))
    net['sfx'] = SoftmaxLayer(net['reshape1'],algo='accurate')
    net['reshape2'] = ReshapeLayer(net['sfx'], shape=(-1,109*109))

    net['fp'] = DenseLayer(net['reshape2'],num_units=2,
            W=lasagne.init.Expectation(width=109,height=109),
            b=lasagne.init.Constant(val=0.0),nonlinearity=lasagne.nonlinearities.linear)
    if get_fp:
        return net['fp']
    net['reshape3'] = ReshapeLayer(net['fp'],shape=(-1,32*2))
    net['recon'] = DenseLayer(net['reshape3'],num_units=3600,nonlinearity=lasagne.nonlinearities.linear,
            W=lasagne.init.Normal(),b=lasagne.init.Constant(val=0.0))

    for key in net['fp'].params.keys():
        net['fp'].params[key].remove('trainable')
        if 'regularizable' in net['fp'].params[key]:
            net['fp'].params[key].remove('regularizable')
    for key in net['conv1'].params.keys():
        net['conv1'].params[key].remove('trainable')
        if 'regularizable' in net['conv1'].params[key]:
            net['conv1'].params[key].remove('regularizable')

    #lrs = {conv1: 0.001, conv2: 0.001, conv3: 0.001, fc_final: 0.005}
    #lrs = {net['conv1/7x7_s2']: 0.001, net['conv2']: 0.001, net['conv3']: 0.001,
    #       net['recon']: 0.005, net['fc_images']: 0.0}
    #lrs = {net['conv1/7x7_s2']: 0.0, net['conv2']: 0.0, net['conv3']: 0.0,
    #       net['fc1_smaller']: 0.0, net['fc_images']: 0.0}

    return net['recon']


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def test_network(get_fp=False):
    X_train, y_train, X_val, y_val = load_dataset()
    input_var = T.tensor4('inputs')
    target_var = T.fmatrix('targets')

    print("Building model and compiling functions...")
    network = build_cnn(input_var)

    # And load them again later on like this:
    with np.load('trained_spatial_model.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    reconstruction = np.array(lasagne.layers.get_output(network, floatX(X_val)).eval())

    if not get_fp:
        return reconstruction, y_val

    # Retrieve feature points
    network = build_cnn(input_var, get_fp=True)
    with np.load('trained_spatial_model.npz') as f:
        # Exclude last to params this time (reconstruction weights and biases)
        param_values = [f['arr_%d' % i] for i in range(len(f.files)-2)]
    lasagne.layers.set_all_param_values(network, param_values)
    fp = np.array(lasagne.layers.get_output(network, floatX(X_val)).eval())

    return reconstruction, y_val, fp, X_val

# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(num_epochs=78):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val = load_dataset()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.fmatrix('targets')

    train_batch_size = 25
    val_batch_size = 25

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_cnn(input_var)

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    #lr = theano.shared(0.001,dtype=theano.shared.floatX)

    #updates = lasagne.updates.momentum(
    #        loss, params, learning_rate=0.001, momentum=0.9)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.sum()/2.0/train_batch_size

    lr = theano.shared(np.array(0.0001, dtype=theano.config.floatX))
    lr_decay = np.array(0.5, dtype=theano.config.floatX)
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(
            loss, params, learning_rate=lr)
    #updates = lasagne.updates.momentum(
    #        loss, params, learning_rate=0.001, momentum=0.9)
    #updates = theano.compat.OrderedDict()
    #for layer, learning_rate in lrs.items():
    #    updates.update(lasagne.updates.momentum(loss, layer.get_params(), learning_rate,
    #    momentum = 0.9))

    test_prediction = lasagne.layers.get_output(network)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    #test_loss = test_loss.sum()/2.0

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], test_loss)

    # Finally, launch the training loop.
    print("Starting training...")
    for epoch in range(num_epochs):
        if epoch == 5:
            lr.set_value(lr.get_value()*lr_decay)
            print('dropping learning rate')

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, train_batch_size, shuffle=False):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            if epoch == 0 and train_batches == 0:
                print("train error: "+str(train_err))
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, val_batch_size, shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            val_err += err.sum()/2.0/val_batch_size
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("(Ran {} iterations)".format(train_batches*(epoch+1)))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

    # After training, we compute and print the test error:
    #test_err = 0
    #test_batches = 0
    #for batch in iterate_minibatches(X_test, y_test, 25, shuffle=False):
    #    inputs, targets = batch
    #    err = val_fn(inputs, targets)
    #    test_err += err
    #    test_batches += 1
    #print("Final results:")
    #print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    #import ipdb; ipdb.set_trace()

    # Optionally, you could now dump the network weights to a file like this:
    np.savez('trained_spatial_model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['num_epochs'] = int(sys.argv[1])
        #if len(sys.argv) > 2:
        #    kwargs['model'] = sys.argv[1]
        main(**kwargs)
