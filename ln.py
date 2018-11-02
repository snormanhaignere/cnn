import numpy as np
import scipy.io as sio
import h5py
import tensorflow as tf
import pickle
import os
import importlib
import inspect
import sys
from copy import deepcopy
from matplotlib import pyplot as plt

# import personal code
if os.getcwd().find('/Users/svnh2') != -1:
    root_directory = '/Users/svnh2/Desktop/projects'
elif os.getcwd().find('/home/svnh') != -1:
    root_directory = '/mindhive/nklab/u/svnh'
elif os.getcwd().find('/home/svnh') != -1:
    root_directory = '/mindhive/nklab/u/svnh'
elif os.getcwd().find('/mindhive/nklab3') != -1:
    root_directory = '/mindhive/nklab/u/svnh'
else:
    raise NameError('No root directory found')
sys.path.append(root_directory) 
from general_analysis import misc
from general_analysis import plot
import cnn

def fit(
    F, D, sr, time_win_sec, log_dir=os.getcwd(), log_id='id0', O={}, 
    overwrite=False, plot_figures=True, sort_weights_PC1=False, sort_data_PC1=False):
    
    # default optional parameters
    P = {}
    P['rank'] = None
    P['weight_scale'] = 1e-6
    P['learning_rate'] = 0.001
    P['early_stopping_steps'] = 10
    P['train_val_test_frac'] = [0.6, 0.2, 0.2]
    P['eval_interval'] = 5
    P['max_iter'] = 1000
    P['act'] = 'relu'
    P['seed'] = 0
    P['stim_labels'] = None
    P['print_iter'] = True

    # replace defaults with optional parameters
    P = misc.param_handling(P, O, maxlen=100, delimiter='/', omit_key_from_idstring=['stim_labels', 'print_iter'])

    # feature dimensionality
    n_stims = F.shape[0]
    n_tps = F.shape[1]
    n_feats = F.shape[2]

    # data dimensionality
    if len(D.shape) == 2:
        D = np.reshape(D, [n_stims, n_tps, 1])
    n_resp = D.shape[2]
    data_dims = [n_stims, n_tps, n_resp]
    assert(D.shape[0]==n_stims)
    assert(D.shape[1]==n_tps)

    # create train, validation, test splits
    if P['stim_labels'] is None:
        labels = np.zeros((n_stims,1))
    else:
        labels = P['stim_labels']
    train_val_test = misc.partition_within_labels(labels, P['train_val_test_frac'], seed=P['seed'])

    # directory to save results
    idstring = 'win-' + misc.num2str(time_win_sec) + '_sr-' + misc.num2str(sr)
    if P['idstring']:
        idstring = idstring + '_' + P['idstring']
    save_directory = misc.mkdir(log_dir + '/' + idstring)

    # file with the key stats
    stats_file = save_directory + '/stats_' + log_id + '.p'

    if os.path.exists(stats_file) and not(overwrite):

        S = pickle.load(open(stats_file, "rb"))

    else:

        # create single layer CNN with above parameters
        layer = {}
        layer['type'] = 'conv'
        layer['n_kern'] = n_resp
        layer['time_win_sec'] = time_win_sec
        layer['act'] = P['act']
        layer['rank'] = P['rank']
        layers = []
        layers.append(layer)
        
        # initialize, build, and train
        tf.reset_default_graph()
        n_weights = (time_win_sec*sr*n_feats)
        net = cnn.Net(
            data_dims, n_feats, sr, deepcopy(layers), loss_type='squared_error',
            weight_scale=P['weight_scale']/n_weights, seed=P['seed'], 
            log_dir=save_directory, log_id=log_id)
        net.build()
        net.train(
            F, D, max_iter=P['max_iter'],
            eval_interval=P['eval_interval'], learning_rate=P['learning_rate'], 
            train_val_test=train_val_test, early_stopping_steps=P['early_stopping_steps'], 
            print_iter=P['print_iter'])

        S = {}
        S['train_loss'] = net.train_loss
        S['val_loss'] = net.val_loss
        S['test_loss'] = net.test_loss
        S['W'] = net.layer_vals()[0]['W']
        S['Y'] = net.predict(F)
        S['train_val_test'] = train_val_test

        pickle.dump(S, open(stats_file, "wb"))

    print('Train loss:', S['train_loss'][-1])
    print('Val loss:', S['val_loss'][-1])
    print('Test loss:', S['test_loss'][-1])

    S['test_corr'] = np.corrcoef(S['Y'][S['train_val_test']==2, :, 0].flatten(), D[S['train_val_test']==2, :, 0].flatten())[0,1]
    
    if plot_figures:

        # loss
        plt.plot(S['train_loss'])
        plt.plot(S['val_loss'])
        plt.plot(S['test_loss'])
        plt.legend(['Train', 'Val', 'Test'])
        plt.xlabel('Eval Iter')
        plt.ylabel('Loss')
        plt.savefig(save_directory + '/loss_' + log_id + '.pdf', bbox_inches='tight')
        plt.show()

        # predictions
        xi = np.where(S['train_val_test']==2)[0]
        if sort_data_PC1:
            [U,E,V] = np.linalg.svd(np.transpose(D[xi,:,0]))
            feat_weights = V[0,:] * np.sign(U[:,0].mean())
            xi = xi[np.flipud(np.argsort(feat_weights))]
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plot.imshow(D[xi,:,0])
        plt.title('Data')
        plt.ylabel('Test stims')
        plt.xlabel('Time')
        plt.subplot(1,2,2)
        plot.imshow(S['Y'][xi,:,0])
        plt.title('Prediction')
        plt.ylabel('Test stims')
        plt.xlabel('Time')
        del xi
        plt.savefig(save_directory + '/predictions_' + log_id + '.pdf', bbox_inches='tight')
        plt.show()

        # plot weights, optionally sort by first PC
        if sort_weights_PC1:
            [U,E,V] = np.linalg.svd(S['W'][:,:,0])
            feat_weights = V[0,:] * np.sign(U[:,0].mean())
            xi = np.flipud(np.argsort(feat_weights))
        else:
            xi = np.arange(0, n_feats)
        plt.figure(figsize=(5,5))
        plot.imshow(np.fliplr(np.transpose(S['W'][:,xi,0])))
        plt.title('Weights')
        plt.xlabel('Time')
        plt.ylabel('Feats')
        del xi
        plt.savefig(save_directory + '/weights_' + log_id + '.pdf', bbox_inches='tight')
        plt.show()

    return S