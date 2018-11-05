# Author: Jerry Peng 2018


# In this script, we draw the statistics of utterances.
# Typically, we count the number of voiced frames for each utterance
# and draw histograms of the whole train/test set.
# Finally, we produce the percent of voiced frames over all frames.

# Note that it draws the two histograms in the same figure

# Input: vad.scp/vad.ark for all utterances
# Output: a figure and a double number

import pdb
import os
import re, sys
import matplotlib.pyplot as plt
import kaldi_io
import numpy as np


# Draw histogram for voiced/all frames
def draw_histogram(numvoicedframes, idx=0, title=""):
  assert (idx == 0 or idx == 1), "Invalid idx %d" %idx
  assert numvoicedframes, "Empty numvoicedframes"
  nframe_list = [ numvoicedframes[uttid][idx] for uttid in numvoicedframes.keys() ]
  num_bins = range(0, 1000)
  #num_bins = 100
  plt.hist(nframe_list, num_bins, facecolor='blue', alpha=0.5)
  plt.xlabel("number of frames per utterance")
  plt.ylabel("number of utterance")
  plt.title(title)
  plt.show()



# get ratio
def get_ratio(numvoicedframes):
  assert numvoicedframes, "Empty numvoicedframes"
  nvoiced_all  = 0.0
  nwhole_all = 0.0
  for uttid in numvoicedframes.keys():
    nvoiced, nwhole = numvoicedframes[uttid]
    nvoiced_all = nvoiced_all + nvoiced
    nwhole_all = nwhole_all + nwhole
  return float(nvoiced_all) / nwhole_all




def main():
  #assert len(sys.argv) == 2, "Improper number of arguments."

  #ipath2vad_scp=sys.argv[1]
  # for debugging
  #ipath2vad_scp="ark:copy-vector scp:/home/jerry/research/ap17_olr/lsid/_vad/vad_dev_1s.1.scp ark,t:-|"
 
  ipath2train_vad_scp="ark:copy-vector scp:/home/jerry/research/ap17_olr/lsid/data/train/vad.scp ark,t:- |"
  ipath2dev1s_vad_scp="ark:copy-vector scp:/home/jerry/research/ap17_olr/lsid/data/dev_1s/vad.scp ark,t:- |"
  
  # use dict rather than list for further investigation of some specific utterance
  # but there is no need to realize this so far
  train_numvoicedframes=dict()

  for uttid, vad_vec in kaldi_io.read_vec_int_ark(ipath2train_vad_scp):
    assert (uttid not in train_numvoicedframes), "Duplicated utterance %s in %s" %(uttid, ipath2train_vad_scp)
    train_numvoicedframes[uttid] = (np.sum(vad_vec), vad_vec.shape[0])

  dev1s_numvoicedframes=dict()

  for uttid, vad_vec in kaldi_io.read_vec_int_ark(ipath2dev1s_vad_scp):
    assert (uttid not in dev1s_numvoicedframes), "Duplicated utterance %s in %s" %(uttid, ipath2dev1s_vad_scp)
    dev1s_numvoicedframes[uttid] = (np.sum(vad_vec), vad_vec.shape[0])

  ## draw histogram for voiced frames for train
  #draw_histogram(train_numvoicedframes, 0, "distribution of number of voiced frames per utterance")

  ## draw histogram of voiced frames for dev_1s

  idx=0
  train_nframe_list = [ train_numvoicedframes[uttid][idx] for uttid in train_numvoicedframes.keys() ]
  train_num_bins = range(0, 1000)

  dev1s_nframe_list = [ dev1s_numvoicedframes[uttid][idx] for uttid in dev1s_numvoicedframes.keys() ]
  dev1s_num_bins = range(0, 1000)

  #num_bins = 100
  title = "distribution of number of frames per utterance"
  plt.hist(train_nframe_list, train_num_bins, facecolor='blue', alpha=0.5)
  plt.hist(dev1s_nframe_list, dev1s_num_bins, facecolor='red', alpha=0.5)
  plt.xlabel("number of frames per utterance")
  plt.ylabel("number of utterance")
  plt.title(title)
  plt.show()

  ## draw histogram for all frames
  #draw_histogram(numvoicedframes, 1, "distrbution of number of frames per utterance")

  #pdb.set_trace()
  # get voiced/whole frames ratio
  print("The nvoiced/nwhole ratio is %.4f" %get_ratio(numvoicedframes))



if __name__ == '__main__':
  main()

