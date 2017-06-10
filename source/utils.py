#from __future__ import division,print_function
import math, os, json, sys, re
# import cPickle as pickle
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter, attrgetter, methodcaller
from collections import OrderedDict
import itertools
from itertools import chain
import time 

import pandas as pd
import PIL
from PIL import Image
from numpy.random import random, permutation, randn, normal, uniform, choice
from numpy import newaxis
import scipy
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread
from sklearn.metrics import confusion_matrix
import bcolz
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold

from IPython.lib.display import FileLink
from tqdm import tqdm 
import cv2
#import theano
#from theano import shared, tensor as T
#from theano.tensor.nnet import conv2d, nnet
#from theano.tensor.signal import pool

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
#from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
#from keras.utils.layer_utils import layer_from_config
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import fbeta_score
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint


#from vgg16 import *
from vgg16bn import *
np.set_printoptions(precision=4, linewidth=100)





to_bw = np.array([0.299, 0.587, 0.114])



def create_base_vgg(extend_top=False,size=(100,100)):
    model = Vgg16BN(size).model


    for i in range (15): #excluding the six 512 blocks at the top, was 15
        model.pop()

    for l in model.layers:
        l.trainable = False 
    
    
    if extend_top:
        for l in get_lrg_layers(): 
            model.add(l)
            
            
    return model


def make_staged_predictions(model,num_parts, do_augmentation=False):
    partial_preds = []
    nb_aug = 1

    vgg_common = create_base_vgg()
    vgg_common.compile(Adam(), 'categorical_crossentropy', metrics=['accuracy'])


    data = load_array("data/cache/xtest_100x100.dat")

    chunck_size = len(data)/num_parts

    print "data has shape: ", data.shape
    print "breaking data into chunks of ", chunck_size
    

    # do prediction on normal photos anyways
    for i in range(1,num_parts+2):
        print "predicting part ",i
        vgg_test_feat = vgg_common.predict(data[(i-1)*chunck_size:min(len(data),i*chunck_size)], batch_size=32, verbose=0)
        partial_preds.append(model.predict(vgg_test_feat, batch_size=32, verbose=0))
        del vgg_test_feat   

    
    normal_image_preds = np.vstack(partial_preds)
    
    
    if not do_augmentation:
        return   normal_image_preds  

    
    
    else: ## test augmentation 
        augmented_predictions = normal_image_preds 

        test_gen = image.ImageDataGenerator(  #image.ImageDataGenerator()
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        channel_shift_range=0.05,
        horizontal_flip=True,
        vertical_flip=True)





        for n in range(nb_aug):
            print "augmentation round {} of {}".format(n+1,nb_aug) 
            partial_preds = []

            for i in range(1,num_parts+2):
                print "predicting part {} of {}".format(i,num_parts)      
                
                start_index = (i-1)*chunck_size
                end_index = min(len(data),i*chunck_size)
                num_items = end_index-start_index
                
                
                test_batch = test_gen.flow(data[start_index:end_index],batch_size=16,shuffle=False)
                
                vgg_test_feat = vgg_common.predict_generator(test_batch, steps=math.ceil(num_items/16.0) ,verbose=0) 
                
#                 print ""
                print "items needed vs. items generated: ",num_items,len(vgg_test_feat) 
                
                # vgg_test_feat = vgg_test_feat[:num_items] #to make sure we only take as much as we need for this batch 
                
                partial_preds.append(model.predict(vgg_test_feat, batch_size=32, verbose=0))
                
                
            augmented_predictions += np.vstack(partial_preds)
            
        augmented_predictions /= (nb_aug+1) #including normal photo predictions
        return augmented_predictions

    
    

def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
    def mf(x):
        p2 = np.zeros_like(p)
        for i in range(17):
            p2[:, i] = (p[:, i] > x[i]).astype(np.int)
        score = fbeta_score(y, p2, beta=2, average='samples')
        return score

    x = [0.2]*17
    for i in range(17):
        best_i2 = 0
        best_score = 0
        for i2 in range(resolution):
            threshold = float(i2) / resolution
            x[i] = threshold
            score = mf(x)
            if score > best_score:
                best_i2 = threshold
                best_score = score

        x[i] = best_i2
        if verbose:
            print(i, best_i2, best_score)

    return x

def f2_score(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method will generate bogus results
    return fbeta_score(y_true, y_pred, beta=2, average='samples')

    

def stretch_n(bands, lower_percent=0, higher_percent=100):
    # out = np.zeros_like(bands)
    out = np.zeros_like(bands).astype(np.float32)

    n = bands.shape[2]
    for i in range(n):
        a = 0  # np.min(band) # original:  255
        b = 1  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t

    return out.astype(np.float32)




def gray(img):
    return np.rollaxis(img,0,3).dot(to_bw)
def to_plot(img):
    return np.rollaxis(img, 0, 3).astype(np.uint8)
def plot(img):
    plt.imshow(to_plot(img))


def floor(x):
    return int(math.floor(x))
def ceil(x):
    return int(math.ceil(x))

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


def do_clip(arr, mx):
    clipped = np.clip(arr, (1-mx)/1, mx)
    return clipped/clipped.sum(axis=1)[:, np.newaxis]


def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, batch_size=4, class_mode='categorical',
                target_size=(224,224)):
    return gen.flow_from_directory(dirname, target_size=target_size,
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)


def onehot(x):
    return to_categorical(x)


def wrap_config(layer):
    return {'class_name': layer.__class__.__name__, 'config': layer.get_config()}


def copy_layer(layer): return layer_from_config(wrap_config(layer))


def copy_layers(layers): return [copy_layer(layer) for layer in layers]


def copy_weights(from_layers, to_layers):
    for from_layer,to_layer in zip(from_layers, to_layers):
        to_layer.set_weights(from_layer.get_weights())


def copy_model(m):
    res = Sequential(copy_layers(m.layers))
    copy_weights(m.layers, res.layers)
    return res


def insert_layer(model, new_layer, index):
    res = Sequential()
    for i,layer in enumerate(model.layers):
        if i==index: res.add(new_layer)
        copied = layer_from_config(wrap_config(layer))
        res.add(copied)
        copied.set_weights(layer.get_weights())
    return res


def adjust_dropout(weights, prev_p, new_p):
    scal = (1-prev_p)/(1-new_p)
    return [o*scal for o in weights]


def get_data(path, target_size=(224,224)):
    batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)
    return np.concatenate([batches.next() for i in range(batches.nb_sample)])


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]


def mk_size(img, r2c):
    r,c,_ = img.shape
    curr_r2c = r/c
    new_r, new_c = r,c
    if r2c>curr_r2c:
        new_r = floor(c*r2c)
    else:
        new_c = floor(r/r2c)
    arr = np.zeros((new_r, new_c, 3), dtype=np.float32)
    r2=(new_r-r)//2
    c2=(new_c-c)//2
    arr[floor(r2):floor(r2)+r,floor(c2):floor(c2)+c] = img
    return arr


def mk_square(img):
    x,y,_ = img.shape
    maxs = max(img.shape[:2])
    y2=(maxs-y)//2
    x2=(maxs-x)//2
    arr = np.zeros((maxs,maxs,3), dtype=np.float32)
    arr[floor(x2):floor(x2)+x,floor(y2):floor(y2)+y] = img
    return arr


def vgg_ft(out_dim):
    vgg = Vgg16()
    vgg.ft(out_dim)
    model = vgg.model
    return model

def vgg_ft_bn(out_dim):
    vgg = Vgg16BN()
    vgg.ft(out_dim)
    model = vgg.model
    return model


def get_classes(path):
    batches = get_batches(path+'train', shuffle=False, batch_size=1)
    val_batches = get_batches(path+'valid', shuffle=False, batch_size=1)
    test_batches = get_batches(path+'test', shuffle=False, batch_size=1)
    return (val_batches.classes, batches.classes, onehot(val_batches.classes), onehot(batches.classes),
        val_batches.filenames, batches.filenames, test_batches.filenames)


def split_at(model, layer_type):
    layers = model.layers
    layer_idx = [index for index,layer in enumerate(layers)
                 if type(layer) is layer_type][-1]
    return layers[:layer_idx+1], layers[layer_idx+1:]




class MixIterator(object):
    def __init__(self, iters):
        self.iters = iters
        self.multi = type(iters) is list
        if self.multi:
            self.N = sum([it[0].N for it in self.iters])
        else:
            self.N = sum([it.N for it in self.iters])

    def reset(self):
        for it in self.iters: it.reset()

    def __iter__(self):
        return self

    def next(self, *args, **kwargs):
        if self.multi:
            nexts = [[next(it) for it in o] for o in self.iters]
            n0s = np.concatenate([n[0] for n in o])
            n1s = np.concatenate([n[1] for n in o])
            return (n0, n1)
        else:
            nexts = [next(it) for it in self.iters]
            n0 = np.concatenate([n[0] for n in nexts])
            n1 = np.concatenate([n[1] for n in nexts])
            return (n0, n1)

