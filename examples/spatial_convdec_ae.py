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

        tgt_imgs = np.float32(np.zeros((data.shape[0],3600)))
        for i in range(data.shape[0]):
          # NOTE: rgb2gray converts to an image in the range 0-1, so no need to scale later.
          img = color.rgb2gray(scipy.misc.imresize(np.transpose(data[i], (2,1,0)), 0.25))
          tgt_imgs[i,:] = np.reshape(img.T,(3600,))

        # For subtracting the mean of the training images
        if mean_img is None:
            mean_img = tgt_imgs.mean(axis=0)
        tgt_imgs -= mean_img

        # NOTE: we leave data unscaled because of the conv1 weights.
        return data, tgt_imgs, mean_img

    # We can now download and read the training and test set images and labels.
    X_train, y_train, mean_img = load_images('/home/cfinn/data/train_legopush2.npy')
    X_val, y_val, _ = load_images('/home/cfinn/data/val_legopush2.npy', mean_img)

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

def build_cnn(input_var=None, get_fp=False, sfx_temp=1.0, n_fp=16, with_var=False):
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
            W=lasagne.init.Normal(), pad=3)
    params = pickle.load(open('/home/cfinn/code/Lasagne/Recipes/modelzoo/blvc_googlenet.pkl'))['param values']
    net['conv1'].W.set_value(params[0])
    net['conv1'].b.set_value(params[1])

    net['conv2'] = ConvLayer(net['conv1'],
            num_filters=32,filter_size=5,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(std=0.001), pad='same')
    net['conv3'] = ConvLayer(net['conv2'],
            num_filters=n_fp,filter_size=5,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(std=0.001), pad='same')  # This controls the initial softmax distr

    net['reshape1'] = ReshapeLayer(net['conv3'],shape=(-1,1,120,120))
    net['sfx'] = SoftmaxLayer(net['reshape1'],algo='accurate', temp=sfx_temp)
    net['reshape2'] = ReshapeLayer(net['sfx'], shape=(-1,120*120))

    net['fp'] = DenseLayer(net['reshape2'],num_units=2,
            W=lasagne.init.Expectation(width=120,height=120),
            b=lasagne.init.Constant(val=0.0),nonlinearity=lasagne.nonlinearities.linear)
    #net['reshape3'] = ReshapeLayer(net['fp'],shape=(-1,n_fp*2))

    #if with_var:
        #net['fp.^2'] = lasagne.layers.ElemwiseMergeLayer([net['reshape3'],net['reshape3']],
                #merge_function=T.mul)
        #net['fp^2'] = lasagne.layers.DenseLayer(net['reshape2'],num_units=2,
                #W=lasagne.init.Expectation(width=109,height=109,option='x^2y^2'),
                #b=lasagne.init.Constant(val=0.0),nonlinearity=lasagne.nonlinearities.linear)
        #net['reshape4'] = lasagne.layers.ReshapeLayer(net['fp^2'],shape=(-1,z_dim))
        #net['fp_var'] = lasagne.layers.ElemwiseSumLayer([net['fp.^2'],net['reshape4']], coeffs=[-1,1])
        ##net['encoding'] = lasagne.layers.ConcatLayer([net['fp'], net['fp_var']], axis=1)
    #else:
    #net['encoding'] = net['reshape3']

    #if input_var == None:
    #    x_imgs = lasagne.layers.InputLayer(shape=(None, n_fp, 60, 60),
    #                                      input_var=None)
    #    y_imgs = lasagne.layers.InputLayer(shape=(None, n_fp, 60, 60),
    #                                      input_var=None)
    #else:
    #    xy_img = lasagne.init.Expectation(option='xy',width=60,height=60).sample(shape=(60*60,2))
    #    batch_size = 25
    #    x_img = np.tile(xy_img[:,0].reshape([60,60]), (batch_size, n_fp, 1, 1))
    #    y_img = np.tile(xy_img[:,1].reshape([60,60]), (batch_size, n_fp, 1, 1))
    #    x_imgs = lasagne.layers.InputLayer(shape=(None, n_fp, 60, 60),
    #                                      input_var=x_img)
    #    y_imgs = lasagne.layers.InputLayer(shape=(None, n_fp, 60, 60),
    #                                      input_var=y_img)
    rank = 1

    x_fp = lasagne.layers.SliceLayer(net['fp'], indices=slice(0,1), axis=1) # batch_size * n_fp x 1
    y_fp = lasagne.layers.SliceLayer(net['fp'], indices=slice(1,2), axis=1) # batch_size * n_fp x 1
    w_init = lasagne.init.Constant(1.0/np.sqrt(rank))
    if rank > 1:
        x_fp = lasagne.layers.ReshapeLayer(x_fp, shape=(-1, n_fp)) # batch_size x n_fp
        y_fp = lasagne.layers.ReshapeLayer(y_fp, shape=(-1, n_fp)) # batch_size x n_fp
        w1x = theano.shared(floatX(w_init((n_fp, rank))))
        w2x = theano.shared(floatX(w_init((rank, 3600))))
        w1y = theano.shared(floatX(w_init((n_fp, rank))))
        w2y = theano.shared(floatX(w_init((rank, 3600))))
        weights_x = T.dot(w1x,w2x); weights_y = T.dot(w1y,w2y);
    else:
        weights_x = theano.shared(floatX(w_init((1,3600))))
        weights_y = theano.shared(floatX(w_init((1,3600))))
    x_fp_imgs = lasagne.layers.DenseLayer(x_fp, num_units=60*60, # batch_size*n_fp x 3600
                                          W=weights_x,
                                          b=lasagne.init.Constant(0.0),
                                          nonlinearity=None)
    y_fp_imgs = lasagne.layers.DenseLayer(y_fp, num_units=60*60, # batch_size*n_fp x 3600
                                          W=weights_y,
                                          b=lasagne.init.Constant(0.0),
                                          nonlinearity=None)
    x_fp_imgsr = ReshapeLayer(x_fp_imgs,shape=(-1,n_fp,60,60))
    y_fp_imgsr = ReshapeLayer(y_fp_imgs,shape=(-1,n_fp,60,60))

    #deconv0x = lasagne.layers.ElemwiseSumLayer([x_imgs,x_fp_imgsr], coeffs=[-1,1])
    #deconv0y = lasagne.layers.ElemwiseSumLayer([y_imgs,y_fp_imgsr], coeffs=[-1,1])
    all_maps = lasagne.layers.ConcatLayer([x_fp_imgsr,y_fp_imgsr], axis=1)
    deconv1 = ConvLayer(all_maps, num_filters=32,filter_size=1,
            nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.Normal(std=0.001))
    deconv2 = ConvLayer(deconv1, num_filters=32,filter_size=5,
            nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.Normal(std=0.001), pad='same')
    deconv3 = ConvLayer(deconv2, num_filters=32,filter_size=5,
            nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.Normal(std=0.001), pad='same')
    net['recon'] = ConvLayer(deconv3, num_filters=1,filter_size=5,
            nonlinearity=None, W=lasagne.init.Normal(), pad='same')
    net['recon'] = lasagne.layers.FlattenLayer(net['recon'])

    #net['recon'] = DenseLayer(net['encoding'],num_units=3600,nonlinearity=lasagne.nonlinearities.linear,
    #        W=lasagne.init.Normal(),b=lasagne.init.Constant(val=0.0))

    for key in net['fp'].params.keys():
        net['fp'].params[key].remove('trainable')
        if 'regularizable' in net['fp'].params[key]:
            net['fp'].params[key].remove('regularizable')
    for key in net['conv1'].params.keys():
        net['conv1'].params[key].remove('trainable')
        if 'regularizable' in net['conv1'].params[key]:
            net['conv1'].params[key].remove('regularizable')
    # Commenting this out makes training better??
    #for key in y_fp_imgs.params.keys():
    #    y_fp_imgs.params[key].remove('trainable')
    #    if 'regularizable' in y_fp_imgs.params[key]:
    #        y_fp_imgs.params[key].remove('regularizable')
    #for key in x_fp_imgs.params.keys():
    #    x_fp_imgs.params[key].remove('trainable')
    #    if 'regularizable' in x_fp_imgs.params[key]:
    #        x_fp_imgs.params[key].remove('regularizable')

    #lrs = {conv1: 0.001, conv2: 0.001, conv3: 0.001, fc_final: 0.005}
    #lrs = {net['conv1/7x7_s2']: 0.001, net['conv2']: 0.001, net['conv3']: 0.001,
    #       net['recon']: 0.005, net['fc_images']: 0.0}
    #lrs = {net['conv1/7x7_s2']: 0.0, net['conv2']: 0.0, net['conv3']: 0.0,
    #       net['fc1_smaller']: 0.0, net['fc_images']: 0.0}
    if get_fp:
        return net['recon'], net['fp']
    else:
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

def test_network(weights_file='trained_spatial_model.npz', get_fp=False, get_sharp_fp=False):
    print("Loading data...")
    X_train, y_train, X_val, y_val = load_dataset()
    input_var = T.tensor4('inputs')
    target_var = T.fmatrix('targets')

    print("Building model and compiling functions...")
    network, network_fp = build_cnn(input_var, get_fp=True)

    with np.load(weights_file) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    reconstruction = np.array(lasagne.layers.get_output(network, floatX(X_val)).eval())

    if not get_fp:
        return reconstruction, y_val

    # Retrieve feature points
    # TODO - this might not be necessary
    with np.load(weights_file) as f:
        # Exclude last to params this time (reconstruction weights and biases)
        param_values = [f['arr_%d' % i] for i in range(len(f.files)-2)]
    lasagne.layers.set_all_param_values(network_fp, param_values)
    fp = np.array(lasagne.layers.get_output(network_fp, floatX(X_val)).eval())

    if not get_sharp_fp:
        return reconstruction, y_val, fp, X_val

    _, network_fp_sharp = build_cnn(input_var, get_fp=True, sfx_temp=0.001)
    with np.load(weights_file) as f:
        # Exclude last to params this time (reconstruction weights and biases)
        param_values = [f['arr_%d' % i] for i in range(len(f.files)-2)]
    lasagne.layers.set_all_param_values(network_fp_sharp, param_values)
    fp_sharp = np.array(lasagne.layers.get_output(network_fp_sharp, floatX(X_val)).eval())

    return reconstruction, y_val, fp, X_val, fp_sharp

# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(num_epochs=78, output_dir='spatial_ae/', weights_file='', resume=False):
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

    if resume:
        # Find weights file and init network
        import glob
        weights_file = max(glob.iglob(output_dir + 'trained_*.npz'), key=os.path.getctime)
        # Need to get string file path to weights file
        f = np.load(output_dir+'learning_curve.npz')
        train_curve = f['train_loss'].tolist()
        val_curve = f['val_loss'].tolist()
    else:
        train_curve = []
        val_curve = []

    if weights_file:
        with np.load(weights_file) as f:
            param_values =  [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)


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
        #if epoch == 5:  # Used for 32 fp
            #lr.set_value(lr.get_value()*lr_decay)
            #print('dropping learning rate')

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
        train_curve.append(train_err/train_batches)
        val_curve.append(val_err/val_batches)
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("(Ran {} iterations)".format(train_batches*(epoch+1)))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot learning curves
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot(train_curve, color='seagreen')
    plt.hold(True)
    plt.plot(val_curve,color='dodgerblue')
    plt.legend(['training loss', 'validation loss'])
    plt.xlabel('epochs',fontsize=14); plt.ylabel('reconstruction error',fontsize=14)
    plt.savefig(output_dir+'learning_curve')
    plt.hold(False)
    # Save learning curve data
    np.savez(output_dir+'learning_curve.npz', train_loss=train_curve,
             val_loss=val_curve)

    # Optionally, you could now dump the network weights to a file like this:
    filename = 'trained_spatialae_' + str(val_err) + '.npz'
    np.savez(output_dir+filename, *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    # Arguments - integers.
    import argparse
    parser = argparse.ArgumentParser(description='Command line options')
    parser.add_argument('--num_epochs', type=int, dest='num_epochs')
    parser.add_argument('--dir', dest='output_dir') # Note, this must include the trailing slash
    parser.add_argument('--resume', action='store_true') # False by default
    args = parser.parse_args(sys.argv[1:])
    main(**{k:v for (k,v) in vars(args).items() if v is not None})
