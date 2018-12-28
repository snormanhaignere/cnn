import numpy as np
import scipy.io as sio
import h5py
import tensorflow as tf
import pickle
import os
from matplotlib import pyplot as plt
import importlib
import inspect
from copy import deepcopy


def weights_tnorm(shape, sig=0.1, seed=0):
    W = tf.Variable(tf.truncated_normal(
        shape, stddev=sig, mean=sig, seed=seed))
    return W

def weights_norm(shape, sig=0.1, seed=0):
    W = tf.Variable(tf.random_normal(shape, stddev=sig, mean=0, seed=seed))
    return W

def poisson(prediction, response):
    return tf.reduce_mean(tf.reduce_sum(prediction - response * tf.log(prediction + 1e-5), 1), name='poisson')


def conv1d(k, Y):
    # convolution function
    # k: batch x time x feature
    # W: time x feature x output channel
    # returns: batch x time x output channel
    return tf.nn.conv1d(k, Y, stride=1, padding='SAME')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def act(name):
    if name == 'relu':
        fn = tf.nn.relu
    elif name == 'identity':
        fn = tf.identity
    else:
        raise NameError('No matching activation')
    return fn


def seed_to_randint(seed):
    np.random.seed(seed)
    return np.random.randint(1e9)

# def param_bases(global_params, basis_params, neuron_params, n_bases, n_neurons, strf_rank, neuron_rank):

# def kern2D_wrapper(type, P):

#     if type='param':
#         W_list = []
#         for i in range(P['n_feats']):
#             X = param_kern(P['global_params'], P['basis_params'], P['neuron_params'], P['n_bases'], P['n_neurons'], P['strf_rank'], P['neuron_rank'])
#             dims = np.int32(X.shape)
#             W_list.append(tf.reshape(X, [dims[0], dims[1], 1, dims[2]]))
#         W = tf.concat(W_list, 3)

#     elif type='nonparam':
#         W = kern2D(P['n_t'], P['n_freq'], P['n_feats'], P['n_kern'], P['sig'], rank=P['rank'], seed=P['seed']):
#     else:
#         raise NameError('Conditional fell through')

#     return W


def kern2D(n_t, n_freq, n_feats, n_kern, sig, rank=None, seed=0, distr='tnorm'):

    if distr == 'tnorm':
        fn = weights_tnorm
    elif distr == 'norm':
        fn = weights_norm
    else:
        raise NameError('No matching distribution')

    print('started creating feats')
    print(rank)
    if rank is None:
        W = fn([n_t, n_freq, n_feats, n_kern], sig=sig, seed=seed_to_randint(seed))
    else:
        # import pdb; pdb.set_trace()
        W_list = []
        for i in range(n_kern):
            for k in range(n_feats):
                x = tf.matmul(fn([n_t, rank], sig=sig, seed=seed_to_randint(seed)+k*2+i*2*n_feats), fn([rank, n_freq], sig=sig, seed=seed_to_randint(seed)+1+k*2+i*2*n_feats))
                W_list.append(tf.reshape(x, [n_t, n_freq, 1]))
        print(len(W_list))
        print(tf.concat(W_list, 2).shape)
        W = tf.reshape(tf.concat(W_list, 2), [n_t, n_freq, n_feats, n_kern])
        print(W.shape)

    print('finished creating feats')
    return W

class Net:

    def __init__(self, data_dims, n_freqs, n_feats, sr_Hz, layers, loss_type='squared_error',
                 weight_scale=0.1, seed=0, log_dir=None, log_id=None, optimizer='Adam'):

        # dimensionality of feature sand data
        self.n_stim = data_dims[0]
        self.n_tps_input = data_dims[1]
        self.n_resp = data_dims[2]
        self.n_freqs = n_freqs
        self.n_feats = n_feats

        # place holders for input features and data
        self.F = tf.placeholder(
            'float32', shape=[None, self.n_tps_input, self.n_freqs, self.n_feats])
        self.D = tf.placeholder(
            'float32', shape=[None, self.n_tps_input, self.n_resp])

        # other parameters
        self.weight_scale = weight_scale
        self.seed = seed

        # layer parameters
        self.layers = layers
        self.n_layers = len(layers)
        for i in range(self.n_layers):
            if self.layers[i]['time_win_sec']==0:
                self.layers[i]['time_win_smp'] = 1
            else:
                self.layers[i]['time_win_smp'] = np.int32(
                    np.round(sr_Hz * self.layers[i]['time_win_sec']))
            if not('sign' in self.layers[i]):
                self.layers[i]['sign'] = 'none'

        assert(self.layers[self.n_layers - 1]['n_kern'] == self.n_resp)

        # other misc parameters
        self.optimizer = optimizer
        self.loss_type = loss_type
        self.train_loss = []
        self.val_loss = []
        self.test_loss = []
        self.iteration = []
        self.learning_rate = tf.placeholder('float32')
        self.best_loss = 1e100
        self.best_loss_index = 0

        # directory and file info
        if log_dir is None:
            self.log_dir = '/Users/svnh2/Desktop/projects/scratch'
        else:
            self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if log_id is None:
            self.log_id = 'seed' + str(seed)
        else:
            self.log_id = log_id

    def initialize(self):

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)

    def build(self, initialize=True):

        self.W = []
        self.b = []
        self.L = []
        for i in range(self.n_layers):

            if self.layers[i]['sign'] == 'pos':
                fn = tf.abs
            if self.layers[i]['sign'] == 'neg':
                fn = lambda x: -tf.abs(x)
            elif self.layers[i]['sign'] == 'none':
                fn = tf.identity

            if i == 0:
                X = self.F
                n_input_feats = self.n_feats

            else:
                X = self.layers[i - 1]['Y']
                n_input_feats = np.int32(X.shape[3])

            if self.layers[i]['type'] == 'conv2d':

                # pad input to ensure causality
                pad_size = np.int32(
                    np.floor(self.layers[i]['time_win_smp'] / 2))
                X_pad = tf.pad(X, [[0, 0], [pad_size, 0], [0, 0], [0, 0]])

                self.layers[i]['W'] = fn(kern2D(self.layers[i]['time_win_smp'], self.layers[i]['freq_win_smp'], n_input_feats, self.layers[i]['n_kern'],
                                                self.weight_scale, seed=seed_to_randint(self.seed)+i, rank=self.layers[i]['rank'], distr='norm'))
                self.layers[i]['b'] = tf.abs(kern2D(1, 1, 1, self.layers[i]['n_kern'],
                                                    self.weight_scale, seed=seed_to_randint(self.seed)+i+self.n_layers, distr='norm'))
                self.layers[i]['Y'] = act(self.layers[i]['act'])(conv2d(X_pad, self.layers[i]['W']) + self.layers[i]['b'])

            elif self.layers[i]['type'] == 'conv1d':

                # pad input to ensure causality
                pad_size = np.int32(
                    np.floor(self.layers[i]['time_win_smp'] / 2))
                X_pad = tf.pad(X, [[0, 0], [pad_size, 0], [0, 0], [0, 0]])

                # useful properties
                n_tps_curr_layer = np.int32(X_pad.shape[1])
                n_freqs_last_layer = np.int32(X_pad.shape[2])

                # weights
                self.layers[i]['W'] = fn(kern2D(self.layers[i]['time_win_smp'], n_freqs_last_layer, n_input_feats, self.layers[i]['n_kern'],
                                                self.weight_scale, seed=seed_to_randint(self.seed)+i, rank=self.layers[i]['rank'], distr='norm'))
                self.layers[i]['b'] = tf.abs(kern2D(1, 1, 1, self.layers[i]['n_kern'],
                                                    self.weight_scale, seed=seed_to_randint(self.seed)+i+self.n_layers, distr='norm'))

                # unwrap frequency and feature dimension
                # -> stim x time x (freq/feature)
                # -> time x (freq/feature) x kernel
                print(X_pad.shape)
                print(self.layers[i]['W'].shape)
                X_pad_unwrap = tf.reshape(X_pad, [-1, n_tps_curr_layer, n_freqs_last_layer*n_input_feats])
                W_unwrap = tf.reshape(self.layers[i]['W'], [self.layers[i]['time_win_smp'], 
                                                            n_freqs_last_layer*n_input_feats, 
                                                            self.layers[i]['n_kern']])
                b_unwrap = tf.reshape(self.layers[i]['b'], [1, 1, self.layers[i]['n_kern']])
                print(X_pad_unwrap.shape)
                print(W_unwrap.shape)

                # 1d convolution with unwrapped
                Y_unwrap = act(self.layers[i]['act'])(conv1d(X_pad_unwrap, W_unwrap) + b_unwrap)
                
                # re-wrap
                # -> stim x time x freq x output kernels
                self.layers[i]['Y'] = tf.reshape(Y_unwrap, [-1, n_tps_curr_layer, 1, self.layers[i]['n_kern']])

            else:
                raise NameError('No matching layer type')

        # remove padding-induced extensions
        self.Y = self.layers[self.n_layers - 1]['Y'][:, 0:self.n_tps_input, :, :]

        # remove frequency dimension
        print(self.Y.shape)
        assert(self.Y.shape[2]==1)
        assert(self.Y.shape[3]==self.n_resp)
        self.Y = tf.reshape(self.Y, [-1, self.n_tps_input, self.n_resp])

        # loss
        if self.loss_type == 'squared_error':
            self.error = tf.reduce_mean(tf.square(self.D - self.Y))
        elif self.loss_type == 'poisson':
            self.error = poisson(self.D, self.Y)
        else:
            raise NameError('Loss must be squared_error or poisson')

        self.penalty = tf.zeros((1,))

        self.loss = self.error + self.penalty

        # gradient optimizer
        if self.optimizer == 'Adam':
            self.train_step = tf.train.AdamOptimizer(
                self.learning_rate).minimize(self.loss)
        elif self.optimizer == 'GradientDescent':
            self.train_step = tf.train.GradientDescentOptimizer(
                self.learning_rate).minimize(self.loss)
        elif self.optimizer == 'RMSProp':
            self.train_step = tf.train.RMSPropOptimizer(
                self.learning_rate).minimize(self.loss)
        else:
            raise NameError('No matching optimizer')

        # initialize global variables
        if initialize:
            self.initialize()

    def feed_dict(self, F, D=None, inds=None, learning_rate=None):

        if inds is None:
            inds = np.arange(F.shape[0])

        d = {self.F: F[inds, :, :, :]}

        if not (D is None):
            d[self.D] = D[inds, :, :]

        if not (learning_rate is None):
            d[self.learning_rate] = learning_rate

        return d

    def save(self):
        fname = self.log_dir + '/' + self.log_id + '-model.ckpt'
        self.saver.save(self.sess, fname)

    def load(self):
        fname = self.log_dir + '/' + self.log_id + '-model.ckpt'
        self.saver.restore(self.sess, fname)

    def train(self, F, D, learning_rate=0.5, max_iter=300, eval_interval=30, batch_size=None,
              train_val_test=None, early_stopping_steps=5, print_iter=True):

        if train_val_test is None:
            train_val_test = np.zeros(D.shape[0])

        # samples used for training, validation, and testing
        train_inds = np.where(train_val_test == 0)[0]
        val_inds = np.where(train_val_test == 1)[0]
        test_inds = np.where(train_val_test == 2)[0]

        # dictionaries for validation and test
        # can initialize them now, because they are not batch dependent
        if len(val_inds) > 0:
            val_dict = self.feed_dict(F, D=D, inds=val_inds)
        if len(test_inds) > 0:
            test_dict = self.feed_dict(F, D=D, inds=test_inds)

        # by default batch size equals the size of the training data
        if batch_size is None:
            batch_size = len(train_inds)

        with self.sess.as_default():

            # indices for this batch
            batch_inds = np.arange(0, batch_size)

            # evaluate loss before any training
            if len(self.train_loss) == 0:
                train_dict = self.feed_dict(F, D=D, inds=train_inds[batch_inds],
                                            learning_rate=learning_rate)
                self.train_loss.append(self.loss.eval(feed_dict=train_dict))
                if len(val_inds) > 0:
                    self.val_loss.append(self.error.eval(feed_dict=val_dict))
                if len(test_inds) > 0:
                    self.test_loss.append(self.error.eval(feed_dict=test_dict))
                self.iteration = [0]
            
            not_improved = 0
            for i in range(max_iter):

                # update
                train_dict = self.feed_dict(F, D=D, inds=train_inds[batch_inds],
                                            learning_rate=learning_rate)
                self.train_step.run(feed_dict=train_dict)

                # evaluate loss
                if np.mod(i + 1, eval_interval) == 0:
                    self.iteration.append(
                        self.iteration[len(self.iteration) - 1] + eval_interval)
                    if print_iter:
                        print(self.iteration[len(self.iteration) - 1])
                    train_loss = self.loss.eval(feed_dict=train_dict)
                    self.train_loss.append(train_loss)
                    if len(val_inds) > 0:
                        validation_loss = self.error.eval(feed_dict=val_dict)
                        stop_loss = validation_loss
                        self.val_loss.append(validation_loss)
                    if len(test_inds) > 0:
                        self.test_loss.append(
                            self.error.eval(feed_dict=test_dict))

                    # early stopping / saving
                    if early_stopping_steps > 0:

                        # prefer to use validation loss
                        if len(val_inds) > 0:
                            stop_loss = validation_loss
                        else:
                            stop_loss = train_loss

                        # check if improved, if so save
                        if stop_loss < self.best_loss:
                            self.best_loss = stop_loss
                            self.best_loss_index = len(self.val_loss) - 1
                            self.save()
                            not_improved = 0
                        else:
                            not_improved += 1

                        # check if we should stop
                        if not_improved == early_stopping_steps:
                            not_improved = 0
                            break

                # update batch
                batch_inds = batch_inds + batch_size
                if batch_inds[-1] > len(train_inds):
                    np.random.shuffle(train_inds)
                    batch_inds = np.arange(batch_size)

            # load the best loss
            if early_stopping_steps > 0:
                self.load()

    def predict(self, F, sess=None):

        if sess is None:
            sess = self.sess

        with sess.as_default():

            return self.Y.eval(feed_dict=self.feed_dict(F))

    def layer_vals(self, sess=None):

        if sess is None:
            sess = self.sess

        with sess.as_default():

            layers = []
            for i in range(self.n_layers):
                layer = {}
                layer['W'] = self.layers[i]['W'].eval()
                if 'b' in self.layers[i]:
                    layer['b'] = self.layers[i]['b'].eval()
                layers.append(layer)

            return layers


class MultiNet(Net):

    def __init__(self, nets, log_dir=None, log_id=None):

        self.nets = nets
        self.weights = np.ones(len(nets)) / len(nets)
        self.D = nets[0].D

        # other misc parameters
        self.train_loss = []
        self.val_loss = []
        self.test_loss = []
        self.iteration = []
        self.learning_rate = tf.placeholder('float32')
        self.best_loss = 1e100
        self.best_loss_index = 0

        # directory and file info
        if log_dir is None:
            self.log_dir = '/Users/svnh2/Desktop/projects/scratch'
        else:
            self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if log_id is None:
            self.log_id = 'seed' + str(self.nets[0].seed)

    def build(self):

        for i in range(len(self.nets)):
            self.nets[i].build(initialize=False)
        #             if i == 0:
        #                 self.Y = self.nets[i].Y * self.weights[i]
        #             else:
        #                 self.Y = self.Y + self.nets[i].Y * self.weights[i]

        self.Y = self.nets[0].Y * self.weights[0] + \
            self.nets[1].Y * self.weights[1]

        # loss
        self.loss = tf.reduce_mean(tf.square(self.D - self.Y))

        # gradient optimizer
        self.train_step = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # initialize global variables
        self.initialize()

    def feed_dict(self, F, D=None, inds=None, learning_rate=None):

        assert(len(self.nets) == len(F))

        if inds is None:
            inds = np.arange(F[0].shape[0])

        d = {}
        for i in range(len(F)):
            d[self.nets[i].F] = F[i][inds, :, :]

        if not (D is None):
            for i in range(len(self.nets)):
                d[self.nets[i].D] = D[inds, :, :]

        if not (learning_rate is None):
            d[self.learning_rate] = learning_rate

        return d

    def layer_vals(self):

        nets = []
        for i in range(len(self.nets)):
            nets.append(self.nets[i].layer_vals(sess=self.sess))

        return nets
