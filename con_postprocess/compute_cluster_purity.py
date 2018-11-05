# Jerry Peng 2018

# In this script, it reads in two files: utt2cluster and utt2spk (contains labels)
# which  should have the same number of lines. And then it computes the purity of clusters

import pdb
import numpy as np
import kaldi_io
from sklearn import metrics

# io 
ipath2utt2cluster = "exp/mfccivector_trainvaliddev1/conivectors_dev_1s_256_100_no_cmn/utt2cluster_bytrain2_lkg_complete.txt"
ipath2utt2spk = "data/dev_1s/utt2spk"


def read_samples(ipath2utt2xxx):
  label2int = {}
  samples = {} # sample
  with open(ipath2utt2xxx, 'r') as f:
    for line in f:
      uttid, label = line.rstrip().split(' ')
      if label in label2int:
        samples[uttid] = label2int[label]
      else:
        label2int[label] = len(label2int) + 1
        samples[uttid] = label2int[label]

  return label2int, samples


def compute_purity(ipath2utt2cluster, ipath2utt2spk):
  _, clustered_samples = read_samples(ipath2utt2cluster)
  _, groundtruth_samples = read_samples(ipath2utt2spk)

  labels_pred = []
  labels_true = []
  # clustered_samples should be a subset of groundtruth_samples
  #pdb.set_trace()
  assert len(clustered_samples) <= len(groundtruth_samples), "Error: #utts in %s > #utts in %s" \
      %(ipath2utt2cluster, ipath2utt2spk)
  for uttid, label in clustered_samples.items():
    assert uttid in groundtruth_samples, "utt %s does not exist in %s" %(uttid, ipath2utt2spk)
    labels_pred.append(label)
    labels_true.append(groundtruth_samples[uttid])

  contingency_matrix = metrics.cluster.contingency_matrix(np.array(labels_true), np.array(labels_pred) )
  return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


if __name__ == '__main__':
  print("Purity: %.4f" %compute_purity(ipath2utt2cluster, ipath2utt2spk))
