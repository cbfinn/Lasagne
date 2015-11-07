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

import numpy as np
import pickle
import theano
import theano.tensor as T

import lasagne
import lasagne.layers.dnn


# ################## Prepare the dataset ##################
def load_dataset():
    def load_images(filename):
        data = np.load(filename)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        #data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_labels(filename):
        data = np.float32(np.load(filename))
        # The labels are vectors of floats now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_images('/home/cfinn/data/img_data_test.npy')
    y_train = load_labels('/home/cfinn/data/labels_test.npy')
    X_val = load_images('/home/cfinn/data/img_data_test_v.npy')
    y_val = load_labels('/home/cfinn/data/labels_test_v.npy')

    # We reserve the last 100 training examples for validation.
    #X_train, X_val = X_train[:-100], X_train[-100:]
    #y_train, y_val = y_train[:-100], y_train[-100:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val #, X_test, y_test


# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of three convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 3, 240, 240),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    conv1 = lasagne.layers.dnn.Conv2DDNNLayer(
            network, num_filters=64, filter_size=(7, 7),
            nonlinearity=lasagne.nonlinearities.rectify,
            stride=2, W=lasagne.init.Normal())

    # Set conv1 params
    params = pickle.load(open('/home/cfinn/code/Lasagne/Recipes/modelzoo/blvc_googlenet.pkl'))['param values']
    lasagne.layers.set_all_param_values(conv1, params[0:2])

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    conv2 = lasagne.layers.dnn.Conv2DDNNLayer(
            conv1, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal())

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    conv3 = lasagne.layers.dnn.Conv2DDNNLayer(
            conv2, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal())
    network = lasagne.layers.ReshapeLayer(
            conv3, shape=(-1,1,109,109))  # put channels into num

    softmax = lasagne.layers.dnn.SoftmaxDNNLayer(network,
            algo='accurate')

    network = lasagne.layers.ReshapeLayer(
            softmax, shape=(-1,109*109))  # collapse image.

    # A fully-connected layer (special weights here, 0 learning rate)
    exp_fc = lasagne.layers.DenseLayer(
            network,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Expectation(width=109,height=109),
            b=lasagne.init.Constant(val=0.0))
    for key in exp_fc.params.keys():
        exp_fc.params[key].remove('trainable')
        if 'regularizable' in exp_fc.params[key]:
            exp_fc.params[key].remove('regularizable')

    network = lasagne.layers.ReshapeLayer(
            exp_fc, shape=(-1,32*2))

    # And, finally, the 9-unit output layer:
    fc_final = lasagne.layers.DenseLayer(
            network, num_units=9,
            W=lasagne.init.Normal())
    lrs = {conv1: 0.001, conv2: 0.001, conv3: 0.001, fc_final: 0.005}

    return fc_final, lrs


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


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(num_epochs=100):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val = load_dataset()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.fmatrix('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network,lrs = build_cnn(input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    #loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()

    #loss += 0.005*lasagne.regularization.regularize_network_params(network,
    #        lasagne.regularization.l2)

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    #lr = theano.shared(0.001,dtype=theano.shared.floatX)
    #lr = theano.shared(np.array(0.005, dtype=theano.config.floatX))
    #lr_decay = np.array(0.1, dtype=theano.config.floatX)

    updates = theano.compat.OrderedDict()
    for layer, learning_rate in lrs.items():
        updates.update(lasagne.updates.momentum(loss, layer.get_params(), learning_rate,
        momentum = 0.9))

    #params = lasagne.layers.get_all_params(network, trainable=True)
    #updates = lasagne.updates.momentum(
    #        loss, params, learning_rate=lr, momentum=0.9)
    #updates = lasagne.updates.adam(
    #        loss, params, learning_rate=lr)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    #test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
    test_loss = lasagne.objectives.squared_error(test_prediction,
                                                 target_var)
    #test_loss = test_loss.mean()
    #test_dist = lasagne.objectives.euc_distance(test_prediction,target_var)
    # As a bonus, also create an expression for the classification accuracy:
    #test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
    #                  dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates) #, profile=True)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], test_loss)#, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        #if epoch == 80:
        #    lr.set_value(lr.get_value()*lr_decay)
        #    print('dropping learning rate')

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 25, shuffle=True):
            #print("batch " + str(i))
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            if epoch == 0 and train_batches == 0:
                print("train error: "+str(train_err))
            #import ipdb; ipdb.set_trace()
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_dist = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 25, shuffle=False):
            inputs, targets = batch
            #import ipdb; ipdb.set_trace()
            err = val_fn(inputs, targets)

            point_dim = 3 # We're in 3D
            num = err.shape[0]
            num_points = err.shape[1] / point_dim
            total_dist = 0.0
            for i in range(num):
                for n in range(num_points):
                    offset = n*point_dim
                    my_dist = 0
                    for k in range(point_dim):
                        my_dist += err[i,offset+k]
                    total_dist += np.sqrt(my_dist)
            euc_dist = total_dist / (num*num_points)

            val_err += err.mean()
            val_dist += euc_dist
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation distance:\t\t{:.6f}".format(val_dist / val_batches))

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
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)
