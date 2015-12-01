import sys
import os
import numpy as np
import pickle
import theano
import theano.tensor as T
import lasagne as nn
from lasagne.layers.dnn import Conv2DDNNLayer, SoftmaxDNNLayer
import time
from PIL import Image
from scipy.stats import norm
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.utils import floatX

from skimage import color
import scipy.misc

import h5py

# ############################################################################
# Implementation of variational autoencoder (AEVB) algorithm as in:
# [1] arXiv:1312.6114 [stat.ML] (Diederik P Kingma, Max Welling 2013)

# ################## Download and prepare the MNIST dataset ##################
# For the linked MNIST data, the autoencoder learns well only in binary mode.
# This is most likely due to the distribution of the values. Most pixels are
# either very close to 0, or very close to 1.
#
# Running this code with default settings should produce a manifold similar
# to the example in this directory. An animation of the manifold's evolution
# can be found here: https://youtu.be/pgmnCU_DxzM

def load_dataset():
    def load_h5_images(filename, mean_img = None):
        f = h5py.File(filename, 'r')
        rgb0 = np.array(f['rgb_frames-1'],dtype=np.float32)
        rgb1 = np.array(f['rgb_frames+0'],dtype=np.float32)
        rgb2 = np.array(f['rgb_frames+1'],dtype=np.float32)

        # Only use rgb2
        tgt_imgs = np.float32(np.zeros((rgb2.shape[0],3600)))
        for i in range(rgb2.shape[0]):
          # NOTE: rgb2gray converts to an image in the range 0-1, so no need to scale later.
          img = color.rgb2gray(scipy.misc.imresize(np.transpose(rgb2[i], (2,1,0)), 0.25))
          tgt_imgs[i,:] = np.reshape(img.T,(3600,))

        # For subtracting the mean of the training images
        if mean_img is None:
            mean_img = tgt_imgs.mean(axis=0)
        tgt_imgs -= mean_img
        # NOTE: we leave data unscaled because of the conv1 weights.
        return rgb0,rgb1,rgb2, tgt_imgs, mean_img

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
        #return data / np.float32(256)

    # We can now download and read the training and test set images and labels.
    #X_train, y_train, mean_img = load_images('/home/cfinn/data/train_legopush2.npy')
    #X_val, y_val, _ = load_images('/home/cfinn/data/val_legopush2.npy', mean_img)
    rgb0_train, rgb1_train, rgb2_train, y_train, mean_img = load_h5_images('/media/drive2tb/fpcontroldata/train_ricebowl_09-08.h5')
    rgb0_val, rgb1_val, rgb2_val, y_val, _ = load_h5_images('/media/drive2tb/fpcontroldata/val_ricebowl_09-08.h5', mean_img)

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return rgb0_train, rgb1_train, rgb2_train, y_train, rgb0_val, rgb1_val, rgb2_val, y_val #, y_train, X_val, y_val #, X_test, y_test

# ############################# Output images ################################
# image processing using PIL

def get_image_array(X, index, shp=(28,28), channels=1):
    ret = ((X[index]+0.5) * 255.).reshape(channels,shp[0],shp[1]) \
            .transpose(2,1,0).clip(0,255).astype(np.uint8)
    if channels == 1:
        ret = ret.reshape(shp[1], shp[0])
    return ret

def get_image_pair(X, Xpr, channels=1, idx=-1):
    mode = 'RGB' if channels == 3 else 'L'
    shp=X[0][0].shape
    i = np.random.randint(X.shape[0]) if idx == -1 else idx
    orig = Image.fromarray(get_image_array(X, i, shp, channels), mode=mode)
    ret = Image.new(mode, (orig.size[0], orig.size[1]*2))
    ret.paste(orig, (0,0))
    new = Image.fromarray(get_image_array(Xpr, i, shp, channels), mode=mode)
    ret.paste(new, (0, orig.size[1]))
    return ret

# ############################# Batch iterator ###############################

def iterate_minibatches(in0, in1, in2, targets, batchsize, shuffle=False):
    assert len(in0) == len(targets)
    if shuffle:
        indices = np.arange(len(targets))
        np.random.shuffle(indices)
    for start_idx in range(0, len(targets) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield np.concatenate((in0[excerpt],in1[excerpt],in2[excerpt]), axis=0), targets[excerpt]


# ##################### Custom layer for middle of VCAE ######################
# This layer takes the mu and sigma (both DenseLayers) and combines them with
# a random vector epsilon to sample values for a multivariate Gaussian

class GaussianSampleLayer(nn.layers.MergeLayer):
    def __init__(self, mu, logsigma, rng=None, **kwargs):
        self.rng = rng if rng else RandomStreams(nn.random.get_rng().randint(1,2147462579))
        super(GaussianSampleLayer, self).__init__([mu, logsigma], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        mu, logsigma = inputs
        shape=(self.input_shapes[0][0] or inputs[0].shape[0],
                self.input_shapes[0][1] or inputs[0].shape[1])
        if deterministic:
            return mu
        return mu + T.exp(logsigma) * self.rng.normal(shape)

class GaussianSigSampleLayer(nn.layers.MergeLayer):
    def __init__(self, mu, sigma, rng=None, **kwargs):
        self.rng = rng if rng else RandomStreams(nn.random.get_rng().randint(1,2147462579))
        super(GaussianSigSampleLayer, self).__init__([mu, sigma], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        mu, sigma = inputs
        shape=(self.input_shapes[0][0] or inputs[0].shape[0],
                self.input_shapes[0][1] or inputs[0].shape[1])
        if deterministic:
            return mu
        return mu + sigma * self.rng.normal(shape)


# ############################## Build Model #################################
# encoder has 1 hidden layer, where we get mu and sigma for Z given an inp X
# continuous decoder has 1 hidden layer, where we get mu and sigma for X given code Z
# once we have (mu, sigma) for Z, we sample L times
# Then L separate outputs are constructed and the final layer averages them

def build_vae(input_var,batch_size=25, L=2, targetvar_shape=3600, channels=1, z_dim=32):
    n_fp = z_dim/2

    net = {}
    net['input'] = nn.layers.InputLayer(shape=(None, 3, 240, 240),
                                             input_var=input_var)

    # Encoder producing mu and sigma (not log sigma)
    net['conv1'] = Conv2DDNNLayer(net['input'],
            num_filters=64,filter_size=7,
            nonlinearity=nn.nonlinearities.rectify,
            stride=2,flip_filters=False,
            W=nn.init.Normal())

    params = pickle.load(open('/home/cfinn/code/Lasagne/Recipes/modelzoo/blvc_googlenet.pkl'))['param values']
    net['conv1'].W.set_value(params[0])
    net['conv1'].b.set_value(params[1])

    net['conv2'] = Conv2DDNNLayer(net['conv1'],
            num_filters=32,filter_size=5,
            nonlinearity=nn.nonlinearities.rectify,
            W=nn.init.Normal(std=0.001))
    net['conv3'] = Conv2DDNNLayer(net['conv2'],
            num_filters=n_fp,filter_size=5,
            nonlinearity=nn.nonlinearities.rectify,
            W=nn.init.Normal(std=0.001))  # This controls the initial softmax distr
    #conv3_2 = nn.

    net['reshape1'] = nn.layers.ReshapeLayer(net['conv3'],shape=(-1,1,109,109))
    net['sfx'] = SoftmaxDNNLayer(net['reshape1'],algo='accurate', temp=1.0)
    net['reshape2'] = nn.layers.ReshapeLayer(net['sfx'], shape=(-1,109*109))

    # Get mean
    net['fp'] = nn.layers.DenseLayer(net['reshape2'],num_units=2,
            W=nn.init.Expectation(width=109,height=109,option='xy'),
            b=nn.init.Constant(val=0.0),nonlinearity=nn.nonlinearities.linear)
    net['reshape3'] = nn.layers.ReshapeLayer(net['fp'],shape=(-1,z_dim))
    fp0 = nn.layers.SliceLayer(net['reshape3'], indices=slice(0,batch_size), axis=0)
    fp1 = nn.layers.SliceLayer(net['reshape3'], indices=slice(batch_size,2*batch_size), axis=0)
    fp_prior = nn.layers.ElemwiseSumLayer([fp0,fp1], coeffs=[1,2])
    l_enc_mu = nn.layers.SliceLayer(net['reshape3'], indices=slice(2*batch_size,3*batch_size), axis=0)

    # Construct variance
    net['fp.^2'] = nn.layers.ElemwiseMergeLayer([l_enc_mu,l_enc_mu],
            merge_function=T.mul)
    sfx2 = nn.layers.SliceLayer(net['reshape2'], indices=slice(2*batch_size*n_fp,None),axis=0)
    net['fp^2'] = nn.layers.DenseLayer(sfx2,num_units=2,
            W=nn.init.Expectation(width=109,height=109,option='x^2y^2'),
            b=nn.init.Constant(val=0.0),nonlinearity=nn.nonlinearities.linear)
    net['reshape4'] = nn.layers.ReshapeLayer(net['fp^2'],shape=(-1,z_dim))
    net['fp_var'] = nn.layers.ElemwiseSumLayer([net['fp.^2'],net['reshape4']], coeffs=[-1,1])
    # Clip variance to be at least 0.000001
    relu_shift = 0.000001
    l_enc_sigma = nn.layers.NonlinearityLayer(net['fp_var'],
        nonlinearity = lambda a: T.nnet.relu(a-relu_shift)+relu_shift)

    for key in net['fp^2'].params.keys():
        net['fp^2'].params[key].remove('trainable')
        if 'regularizable' in net['fp^2'].params[key]:
            net['fp^2'].params[key].remove('regularizable')
    for key in net['fp'].params.keys():
        net['fp'].params[key].remove('trainable')
        if 'regularizable' in net['fp'].params[key]:
            net['fp'].params[key].remove('regularizable')
    for key in net['conv1'].params.keys():
        net['conv1'].params[key].remove('trainable')
        if 'regularizable' in net['conv1'].params[key]:
            net['conv1'].params[key].remove('regularizable')

    l_dec_mu_list = []
    l_dec_logsigma_list = []
    l_output_list = []
    # tie the weights of all L versions so they are the "same" layer
    W_dec_hid = None
    b_dec_hid = None
    W_dec_mu = None
    b_dec_mu = None
    W_dec_ls = None
    b_dec_ls = None
    # Take L samples
    for i in xrange(L):
        l_Z = GaussianSigSampleLayer(l_enc_mu, net['fp_var'], name='Z')
        #l_dec_hid = nn.layers.DenseLayer(l_Z, num_units=n_hid,
                #nonlinearity = T.nnet.softplus,
                #W=nn.init.Normal() if W_dec_hid is None else W_dec_hid,
                #b=nn.init.Constant(0.) if b_dec_hid is None else b_dec_hid,
                #name='dec_hid')
        l_dec_mu = nn.layers.DenseLayer(l_Z, num_units=targetvar_shape,
                nonlinearity = None,
                W = nn.init.Normal() if W_dec_mu is None else W_dec_mu,
                b = nn.init.Constant(0) if b_dec_mu is None else b_dec_mu,
                name = 'dec_mu')
        # relu_shift is for numerical stability - if training data has any
        # dimensions where stdev=0, allowing logsigma to approach -inf
        # will cause the loss function to become NAN. So we set the limit
        # stdev >= exp(-1 * relu_shift)
        relu_shift = 10
        l_dec_logsigma_coef = nn.layers.DenseLayer(l_Z, num_units=1, #targetvar_shape,
                W = nn.init.Normal() if W_dec_ls is None else W_dec_ls,
                b = nn.init.Constant(0) if b_dec_ls is None else b_dec_ls,
                nonlinearity = lambda a: T.nnet.relu(a+relu_shift)-relu_shift,
                name='dec_logsigma')

        # Constant variance for the entire image
        l_dec_logsigma = nn.layers.DenseLayer(l_dec_logsigma_coef, num_units=targetvar_shape,
                W = nn.init.Constant(1), b = nn.init.Constant(0), nonlinearity=None, name='dec_output_ls')
        for key in l_dec_logsigma.params.keys():
            l_dec_logsigma.params[key].remove('trainable')
            if 'regularizable' in l_dec_logsigma.params[key]:
                l_dec_logsigma.params[key].remove('regularizable')

        l_output = GaussianSampleLayer(l_dec_mu, l_dec_logsigma,
                name='dec_output')
        l_dec_mu_list.append(l_dec_mu)
        l_dec_logsigma_list.append(l_dec_logsigma)
        l_output_list.append(l_output)
        if W_dec_mu is None:
            #W_dec_hid = l_dec_hid.W
            #b_dec_hid = l_dec_hid.b
            W_dec_mu = l_dec_mu.W
            b_dec_mu = l_dec_mu.b
            W_dec_ls = l_dec_logsigma_coef.W
            b_dec_ls = l_dec_logsigma_coef.b
    # Why is this here? (elementwise mean over all samples...)
    l_output = nn.layers.ElemwiseSumLayer(l_output_list, coeffs=1./L, name='output')
    return l_enc_mu, l_enc_sigma, l_dec_mu_list, l_dec_logsigma_list, l_output_list, l_output, fp_prior

# ############################## Main program ################################

def log_likelihood(tgt, mu, ls):
    return T.sum(-(np.float32(0.5 * np.log(2 * np.pi)) + ls)
            - 0.5 * T.sqr(tgt - mu) / T.exp(2 * ls))

# If resume is true, it wlil look in output_dir for the learning curve data and weights.
def main(L=2, z_dim=32, num_epochs=300, output_dir='vae/', weights_file='', resume=False):
    print("Loading data...")
    rgb0_train,rgb1_train, rgb2_train, y_train, rgb0_val,rgb1_val,rgb2_val, y_val = load_dataset()
    #X_train, X_val = load_dataset()
    #width, height = X_train.shape[2], X_train.shape[3]
    width = 60
    height = 60
    input_var = T.tensor4('inputs')
    target_var = T.fmatrix('targets')

    # Create VAE model
    print("Building model and compiling functions...")
    print("L = {}, z_dim = {}".format(L, z_dim))
    #x_dim = width * height
    l_z_mu, l_z_s, l_x_mu_list, l_x_ls_list, l_x_list, l_x, l_z_prior = \
           build_vae(input_var, L=L, z_dim=z_dim)

    if resume:
        # Find weights file and init network
        import glob
        weights_file = max(glob.iglob(output_dir + 'params_*.npz'), key=os.path.getctime)
        # Need to get string file path to weights file
        f = np.load(output_dir+'learning_curve.npz')
        train_curve = f['train_loss'].tolist()
        val_curve = f['val_loss'].tolist()
        val_l2_curve = f['val_l2_loss'].tolist()
    else:
        train_curve = []
        val_curve = []
        val_l2_curve = []

    if weights_file:
        with np.load(weights_file) as f:
            param_values =  [f['arr_%d' % i] for i in range(len(f.files))]
        nn.layers.set_all_param_values(l_x, param_values)

    def build_loss(deterministic):
        layer_outputs = nn.layers.get_output([l_z_mu, l_z_s] + l_x_mu_list + l_x_ls_list
                + l_x_list + [l_x] + [l_z_prior], deterministic=deterministic)
        z_mu =  layer_outputs[0]
        z_s =  layer_outputs[1]
        x_mu =  layer_outputs[2:2+L]
        x_ls =  layer_outputs[2+L:2+2*L]
        x_list =  layer_outputs[2+2*L:2+3*L]
        x = layer_outputs[2+3*L]
        z_prior = layer_outputs[3+3*L:]

        # Loss expression has two parts as specified in [1]
        # kl_div = KL divergence between p_theta(z) and p(z|x)
        # - divergence between prior distr and approx posterior of z given x
        # - or how likely we are to see this z when accounting for Gaussian prior
        # logpxz = log p(x|z)
        # - log-likelihood of x given z
        # - in continuous case, is log-likelihood of seeing the target x under the
        #   Gaussian distribution parameterized by dec_mu, sigma = exp(dec_logsigma)

        prior_var = 0.1 # Variance of the prior

        # Standard prior with variance prior_var
        #kl_div = 0.5 * T.sum(1 + T.log(z_s/prior_var) - T.sqr(z_mu)/prior_var - z_s/prior_var)
        # Prior with no mean, just variance (not sure if this makes sense)
        #kl_div = 0.5 * T.sum(1 + T.log(z_s/prior_var) - z_s/prior_var)
        # Uniform prior
        #kl_div = 0.5*T.sum(1+T.log(2*np.pi*z_s))
        # Constant velocity prior
        kl_div = 0.5 * T.sum(1 + T.log(z_s/prior_var) - T.sqr(z_mu-z_prior)/prior_var - z_s/prior_var)

        # Use downsampled target_var instead of input_var
        logpxz = sum(log_likelihood(target_var, mu, ls)
            for mu, ls in zip(x_mu, x_ls))/L
        #logpxz = sum(log_likelihood(input_var.flatten(2), mu, ls)
        #    for mu, ls in zip(x_mu, x_ls))/L
        prediction = x_mu[0] if deterministic else T.sum(x_mu, axis=0)/L
        loss = -1 * (logpxz + kl_div)
        return loss, prediction

    # If there are dropout layers etc these functions return masked or non-masked expressions
    # depending on if they will be used for training or validation/test err calcs
    loss, _ = build_loss(deterministic=False)
    test_loss, test_prediction = build_loss(deterministic=True)

    # ADAM updates
    params = nn.layers.get_all_params(l_x, trainable=True)
    updates = nn.updates.adam(loss, params, learning_rate=1e-4)
    # Use target_var in loss, not just input_var
    #train_fn = theano.function([input_var], loss, updates=updates)
    #val_fn = theano.function([input_var], test_loss)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_prediction])

    print("Starting training...")
    batch_size = 25

    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(rgb0_train,rgb1_train,rgb2_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            this_err = train_fn(inputs, targets)
            #print this_err
            train_err += this_err
            train_batches += 1
        val_err = 0
        val_batches = 0
        l2_err = 0
        for batch in iterate_minibatches(rgb0_val, rgb1_val, rgb2_val, y_val, batch_size, shuffle=False):
            inputs, targets = batch
            [err, pred] = val_fn(inputs, targets)
            l2loss = ((pred-targets)**2).sum()/2.0/batch_size
            l2_err += l2loss
            val_err += err
            val_batches += 1
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        train_curve.append(train_err/train_batches)
        val_curve.append(val_err/val_batches)
        val_l2_curve.append(l2_err/val_batches)
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation l2 loss:\t\t{:.6f}".format(l2_err / val_batches))

    #test_err = 0
    #test_batches = 0
    #for batch in iterate_minibatches(X_test, batch_size, shuffle=False):
    #    err = val_fn(batch)
    #    test_err += err
    #    test_batches += 1
    #test_err /= test_batches
    #print("Final results:")
    #print("  test loss:\t\t\t{:.6f}".format(test_err))
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
    plt.xlabel('epochs',fontsize=14); plt.ylabel('-log likelihood',fontsize=14)
    plt.savefig(output_dir+'learning_curve')
    plt.hold(False)
    plt.plot(val_l2_curve, color='mediumaquamarine')
    plt.xlabel('epochs',fontsize=14); plt.ylabel('val l2 reconstruction error',fontsize=14)
    plt.savefig(output_dir+'val_l2_curve')
    # Save learning curve data
    np.savez(output_dir+'learning_curve.npz', train_loss=train_curve,
             val_loss=val_curve, val_l2_loss=val_l2_curve)

    # save some example pictures so we can see what it's done
    #example_batch_size = 20
    #X_comp0 = rgb0_val[:example_batch_size]
    #X_comp1 = rgb1_val[:example_batch_size]
    #X_comp2 = rgb2_val[:example_batch_size]
    #y_comp = y_val[:example_batch_size]
    #y_comp = np.reshape(y_comp, (example_batch_size, 1, width, height))
    #pred_fn = theano.function([input_var], test_prediction)
    #X_pred = pred_fn(X_comp).reshape(-1, 1, width, height)
    #for i in range(20):
        #get_image_pair(y_comp, X_pred, idx=i, channels=1).save(output_dir+'output_{}.jpg'.format(i))

    # save the parameters so they can be loaded for next time
    print("Saving")
    fn = 'params_{:.6f}'.format(val_err/val_batches)
    np.savez(output_dir+fn + '.npz', *nn.layers.get_all_param_values(l_x))

    # sample from latent space if it's 2d
    if z_dim == 2:
        # functions for generating images given a code (used for visualization)
        # for an given code z, we deterministically take x_mu as the generated data
        # (no Gaussian noise is used to either encode or decode).
        z_var = T.vector()
        generated_x = nn.layers.get_output(l_x_mu_list[0], {l_z_mu:z_var},
                deterministic=True)
        gen_fn = theano.function([z_var], generated_x)
        im = Image.new('L', (width*19,height*19))
        for (x,y),val in np.ndenumerate(np.zeros((19,19))):
            z = np.asarray([norm.ppf(0.05*(x+1)), norm.ppf(0.05*(y+1))],
                    dtype=theano.config.floatX)
            x_gen = gen_fn(z).reshape(-1, 1, width, height)
            im.paste(Image.fromarray(get_image_array(x_gen,0)), (x*width,y*height))
            im.save('gen.jpg')

def test_network(L=4, z_dim=32,output_dir='vae/', get_fp=False):
    print("Loading data...")
    rgb0_train,rgb1_train, rgb2_train, y_train, rgb0_val,rgb1_val,rgb2_val, y_val = load_dataset()

    import glob
    weights_file = max(glob.iglob(output_dir + 'params_*.npz'), key=os.path.getctime)

    input_var = T.tensor4('inputs')
    target_var = T.fmatrix('targets')

    # Create VAE model
    print("Building model and compiling functions...")
    print("L = {}, z_dim = {}".format(L, z_dim))
    #x_dim = width * height
    l_z_mu, l_z_s, l_x_mu_list, l_x_ls_list, l_x_list, l_x, l_z_prior = \
           build_vae(input_var, L=L, z_dim=z_dim)

    with np.load(weights_file) as f:
        param_values =  [f['arr_%d' % i] for i in range(len(f.files))]
    nn.layers.set_all_param_values(l_x, param_values)

    # Get entire validation set
    X_vals = []
    y_vals = []
    reconstructions = []
    fp = []
    for batch in iterate_minibatches(rgb0_val,rgb1_val,rgb2_val, y_val, 25, shuffle=False):
        X_val, y_val = batch
        X_vals.append(X_val)
        y_vals.append(y_val)
        reconstructions.append(np.array(nn.layers.get_output(l_x, floatX(X_val), deterministic=True).eval()))
        fp.append(np.array(nn.layers.get_output(l_z_mu, floatX(X_val)).eval()))

    if not get_fp:
        return reconstructions, y_vals

    return reconstructions, y_vals, fp, rgb2_val


if __name__ == '__main__':
    # Arguments - integers.
    import argparse
    parser = argparse.ArgumentParser(description='Command line options')
    parser.add_argument('--num_epochs', type=int, dest='num_epochs')
    parser.add_argument('--L', type=int, dest='L')
    parser.add_argument('--z_dim', type=int, dest='z_dim')
    parser.add_argument('--dir', dest='output_dir') # Note, this must include the trailing slash
    parser.add_argument('--resume', action='store_true') # False by default
    args = parser.parse_args(sys.argv[1:])
    main(**{k:v for (k,v) in vars(args).items() if v is not None})
