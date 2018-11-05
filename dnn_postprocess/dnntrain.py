import numpy as np
import pandas as pd
import sys
import kaldi_io
import pickle
from sklearn.model_selection import train_test_split
from sklearn import mixture,preprocessing
import _pickle as cPickle
import h5py
import pdb

##-------------------------- set gpu using tf ---------------------------
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


from keras import optimizers
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, Reshape, Flatten, Merge, ELU
from keras.regularizers import l2
from keras.utils import to_categorical, multi_gpu_model
from keras.callbacks import EarlyStopping


## io
#ipath2train_ivectors = "ark:copy-vector scp:/home/jerry/research/ap17_olr/lsid/exp/mtlbnfivector/ivectors_train_2048_400/ivector.scp ark,t:-|"
#ipath2test_ivectors = "ark:copy-vector scp:/home/jerry/research/ap17_olr/lsid/exp/mtlbnfivector/ivectors_dev_1s_2048_400/ivector.scp ark,t:-|"

## lang to int (to generate labels)
lang2num_dict = {'Kazak': 0, 'Tibet': 1, 'Uyghu': 2, 'ct-cn': 3, 'id-id':4, 'ja-jp':5, 'ko-kr':6, 'ru-ru':7, 'vi-vn':8, 'zh-cn':9}

lang2spkidlen_dict = {'Kazak': 11, 'Tibet': 14, 'Uyghu': 11,'ct-cn':10, 'id-id':10, 'ja-jp':10, 'ko-kr':10, 'ru-ru':10, 'vi-vn':10, 'zh-cn':10}

def load_train_data_and_preprocess(ipath2train_ivectors):
  """
    load ivectors, generate labels, flush them, split into train and val sets
    Return: a list of train set tuple, a list of val set tuple where the first item is label(int), second is a float vector(list)

  """
  assert ipath2train_ivectors, "Empty ipath2train_ivectors!"
  train_set = [] # a list of tuples where the first item is lanid, the second item is numpy vector (ivector)
  for uttid, vec in kaldi_io.read_vec_flt_ark(ipath2train_ivectors):
    langid = uttid[:5]
    assert langid in lang2num_dict, "Invalid ivector utterance id: %s" %(uttid)
    train_set.append( (lang2num_dict[langid], vec) )

  ## Note that normalization is skipped here
  ##  as we can easily and already have normlized ivectors in kaldi

  ## split into train and validation sets (includes flushing samples and splitting)
  trainset, validset = train_test_split(train_set, test_size=0.15, stratify=list(zip(*train_set))[0] )
  print("train set size is: %d, valid set size is:%d " %(len(trainset), len(validset)))
  return trainset, validset

# deprecated. There is a bug
# # ipath2indom_testspk is a list of in domain test_spkids
# def load_train_data_and_preprocess2(ipath2train_ivectors, ipath2indom_testspk):
#   """
#     Split train and val sets by speakerID

#     load ivectors, generate labels, split into train and val sets
#     Return: a list of train set tuple, a list of val set tuple where the first item is label(int), second is a float vector(list)
#   """
#   assert ipath2train_ivectors, "Empty ipath2train_ivectors!"
#   assert ipath2indom_testspk, "Empty ipath2indom_testspk!"
#   with open(ipath2indom_testspk, 'r') as f:
#      indom_testspk_list = [ line.rstrip('\n') for line in f ]

#   assert indom_testspk_list, "Empty test spk list!"

#   trainset = [] # a list of tuples where the first item is lanid, the second item is numpy vector (ivector)
#   indom_testset = []
#   for uttid, vec in kaldi_io.read_vec_flt_ark(ipath2train_ivectors):
#     langid = uttid[:5]
#     assert langid in lang2num_dict, "Invalid ivector utterance id: %s" %(uttid)
#     spkid = uttid[:lang2spkidlen_dict[langid] ]
    
#     if spkid in indom_testspk_list:
#       indom_testset.append( (lang2num_dict[langid], vec) )
#     else:
#       trainset.append( (lang2num_dict[langid], vec) )
#   print("train set size is: %d, in domain test set size is:%d " %(len(trainset), len(indom_testset)))
#   return trainset, indom_testset
  

# deprecated. There is a bug
# def load_train_data_and_preprocess3(ipath2train_ivectors, ipath2valid_ivectors):
#   """
#     load ivectors, generate labels
#     Return: a list of train set tuple, a list of val set tuple where the first item is label(int), second is a float vector(list)
#   """
#   assert ipath2train_ivectors, "Empty ipath2train_ivectors!"
#   assert ipath2valid_ivectors, "Empty ipath2valid_ivectors!"
#   trainset = []
#   indom_testset = []
#   for uttid, vec in kaldi_io.read_vec_flt_ark(ipath2train_ivectors):
#     langid = uttid[:5]
#     assert langid in lang2num_dict, "Invalid ivector utterance id: %s" %(uttid)
#     trainset.append( (lang2num_dict[langid], vec))

#   for uttid, vec in kaldi_io.read_vec_flt_ark(ipath2valid_ivectors):
#     langid = uttid[:5]
#     assert langid in lang2num_dict, "Invalid ivector utterance id: %s" %(uttid)
#     indom_testset.append( (lang2num_dict[langid], vec))

#   print("train set size is: %d, test set size is: %d " %(len(trainset), len(indom_testset)))
#   return trainset, indom_testset


def build_dnn_classifier():
  """
    build a DNN classifier for language identification
      the input dim should be euqal to ivector dim
      the output dim should be equal to number of languages
    Return: a model
  """
  ## create model
  model = Sequential()
  model.add(Dense(512, input_dim=100, activation='relu'))
  #model.add(Dense(10, input_dim=10, activation='relu'))
  #model.add(Dense(1024, input_dim=1024, activation='relu'))
  #model.add(Dense(1024, input_dim=1024, activation='relu'))
  model.add(Dense(10, activation='softmax'))
  ## Compile model
  parallel_model = multi_gpu_model(model, gpus=2)
  optimizer = optimizers.Adam(lr=0.0006)
  parallel_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  return parallel_model


def train_dnn_classifier(model, trainset, validset, opath2model):
  """
    train model and save it to hardisk
  """

  train_label, train_feat = zip(*trainset)
  ## convert labels to categorical one-hot encoding
  train_label = to_categorical(train_label, num_classes=10)

  valid_label, valid_feat = zip(*validset)
  ## convert labels to categorical one-hot encoding
  valid_label = to_categorical(valid_label, num_classes=10)

  # convert list to numpy array
  train_feat = np.asarray(train_feat)
  valid_feat = np.asarray(valid_feat)

  callbacks = [EarlyStopping(monitor='val_loss', patience=0, verbose=0)]
  print(">> Start to train dnn...")
  model.fit(train_feat, train_label,
            validation_data=[valid_feat, valid_label],
            epochs=2, 
            batch_size=128)
            #callbacks=callbacks)

  # model.save causes error as it is multi-gpu decorated.
  true_model = model.get_layer('sequential_1')
  true_model.save(opath2model)



if __name__ == '__main__':
  # ipath2train_ivectors="ark,t:copy-vector scp:exp/mtlbnfivector/ivectors_train_2048_400/ivector.scp ark,t:-|"
  # opath2model="exp/mtlbnfivector/dnn.mdl"
  # ipath2indom_testspk = "dnn_postprocess/valid_spk"
  assert len(sys.argv) == 3, "Invalid number of input arguments"
  ipath2train_ivectors = sys.argv[1]
  opath2model = sys.argv[2]
  #ipath2indom_testspk = sys.argv[3]
  # ipath2valid_ivectors = sys.argv[3]

  trainset, validset = load_train_data_and_preprocess(ipath2train_ivectors)
  # trainset, validset = load_train_data_and_preprocess2(ipath2train_ivectors, ipath2indom_testspk)
  # trainset, validset = load_train_data_and_preprocess3(ipath2train_ivectors, ipath2valid_ivectors)
  model = build_dnn_classifier()
  train_dnn_classifier(model, trainset, validset, opath2model)

