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


from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Dense, Reshape, Flatten, Merge, ELU
from keras.regularizers import l2
from keras.utils import to_categorical, multi_gpu_model



## lang to int (to generate labels)
uttprefix2num_dict = {'Kazak': 0, 'Tibet': 1, 'Uyghu': 2, 'ct-cn': 3, 'id-id':4, 'ja-jp':5, 'ko-kr':6, 'ru-ru':7, 'vi-vn':8, 'zh-cn':9}

def load_test_data_and_preprocess(ipath2testset):
  """
    load ivectors, generate labels
    Return: a list of test set tuple (the first item is label(int), second is a float vector(list))

  """
  assert ipath2testset, "Empty ipath2testset!"
  testset = []
  for uttid, vec in kaldi_io.read_vec_flt_ark(ipath2testset):
    langid = uttid[:5]
    assert langid in uttprefix2num_dict, "Invalid ivector utterance id: %s" %(uttid)
    testset.append( (uttprefix2num_dict[langid], vec) )

  ## Note that normalization is skipped here
  ##  as we can easily and already have normlized ivectors in kaldi
  print("test set size is: %d" %len(testset) )
  return testset


def test_dnn_classifier(model, testset):
  """
    test model on testset
    Return: a 2-d numpy array. Each row vector is 10 dim, which represents the dnn output

  """
  test_label, test_feat = zip(*testset)
  ## convert labels to categorical one-hot encoding
  test_label = to_categorical(test_label, num_classes=10)

  # covnert list to numpy
  test_feat = np.asarray(test_feat)

  print(">> Start to test dnn...")
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  ## score = (loss, accuracy) and it doesnt affected by batch_size below
  score = model.evaluate(test_feat, test_label, batch_size=512)
  print(score)
  # make prediction
  predict_result = model.predict(test_feat)
  return predict_result


def save_predict_result(predict_result, opath2result):
  """
    format result and save it to opath2result
    the output should be formated as follows.
    
    Kazak Kazak_F0101033 6.049543
    Tibet Kazak_F0101033 1.100619
    Uyghu Kazak_F0101033 5.243432
    ct Kazak_F0101033 -0.9423696
    id Kazak_F0101033 -3.326587
    ja Kazak_F0101033 -0.8206892
    ko Kazak_F0101033 -3.42079
    ru Kazak_F0101033 3.852252
    vi Kazak_F0101033 -2.77112
    zh Kazak_F0101033 -0.8504514
    ...
 
    Note that the first two columns should have exactly the same format as above.
    Otherwise, it will cause logical bugs when the result is further used in other bash sciprts.

  """
  lang_list = ['Kazak', 'Tibet', 'Uyghu', 'ct', 'id', 'ja', 'ko', 'ru', 'vi', 'zh']
  lang_len = len(lang_list)
  print(">> Save score result to %s" %opath2result)
  with open(opath2result, 'w') as f:
    for utt_idx, (uttid, _) in enumerate(kaldi_io.read_vec_flt_ark(ipath2testset)):
      for lang_idx, lang in enumerate(lang_list):
        f.write("%s %s %.6f\n" %(lang_list[lang_idx], uttid, predict_result[utt_idx, lang_idx]))
  

if __name__ == '__main__':

  assert len(sys.argv) == 4, "Invalid number of input arguments"
  ipath2model = sys.argv[1]
  ipath2testset = sys.argv[2]
  # ipath2outdom_testset = sys.argv[3]
  opath2result = sys.argv[3]

  # ipath2model = 'exp/mtlbnfivector/dnn.mdl'
  # ipath2indom_testset = 'ark:copy-vector scp:/home/jerry/research/ap17_olr/lsid/exp/mtlbnfivector/ivectors_valid_1s_2048_400/ivector.scp ark,t:-|'
  # ipath2outdom_testset = 'ark:copy-vector scp:/home/jerry/research/ap17_olr/lsid/exp/mtlbnfivector/ivectors_dev_1s_2048_400/ivector.scp ark,t:-|'
  # opath2result = 'exp/mtlbnfivector/score/dev_1s/foo_dnn'

  testset = load_test_data_and_preprocess(ipath2testset)
  model = load_model(ipath2model)
  predict_result = test_dnn_classifier(model, testset)
  save_predict_result(predict_result, opath2result)

