# Author: Jerry Peng

# In this script, given a score file, 
# we select utterances that have being misclassfied to a wrong language
# and output a list of them


# Input: exp/conivector/score/total/foo_wccn
# Output: each row is a utterance id that has been misclassfied

import pdb
import os
import re, sys
from signal import signal, SIGPIPE, SIG_DFL
import kaldi_io
from kaldi_utt_hist import *

# pdb.set_trace()
# They are related to the input file's content and format
# map the first five characters of uttid to language
utt2lan={'ct-cn':'ct', 'id-id':'id', 'ja-jp':'ja', 'ko-kr':'ko', 'ru-ru':'ru', 'vi-vn':'vi', 'zh-cn':'zh', 'Kazak':'Kazak', 'Tibet':'Tibet', 'Uyghu':'Uyghu'}

# foo_score = "exp/conivector/score/total/foo_wccn"


assert len(sys.argv) == 3, "Improper number of arguments."
foo_score=sys.argv[1]
ipath2vad_scp=sys.argv[2]

# use for store ten lines as there are 10 languages decision for each test utterance
numlan = 10
scores = ['0'] * numlan
maxscore = float("-inf")
targetscore = 0
correct_uttids = []
wrong_uttids = []
with open(foo_score) as f:
  for idx, line in enumerate(f):
    lan_idx = idx%numlan
    lan, uttid, scores[lan_idx] = line.strip().split(' ')
    if maxscore < float(scores[lan_idx]):
      maxscore = float(scores[lan_idx])
    assert uttid[:5] in utt2lan, "unexpected uttid: %s" %uttid
    if lan == utt2lan[uttid[:5]]:
      targetscore = float(scores[lan_idx])
    if numlan-1 == idx%numlan:
      if targetscore >= maxscore:
        correct_uttids.append(uttid)
      else:
        wrong_uttids.append(uttid)
      maxscore = float("-inf")
      targetscore = 0

# pdb.set_trace()

# if 'wrong' == wrong_or_correct:
#   for uttid in wrong_uttids:
#     print(uttid)
# else:
#   for uttid in correct_uttids:
#     print(uttid)

numvoicedframes_correct = dict()
numvoicedframes_wrong = dict()

for uttid, vad_vec in kaldi_io.read_vec_int_ark(ipath2vad_scp):
  assert (uttid not in numvoicedframes_correct), "Duplicated utterance %s in %s" %(uttid, ipath2vad_scp)
  assert (uttid not in numvoicedframes_wrong), "Duplicated utterance %s in %s" %(uttid, ipath2vad_scp)
  if uttid in correct_uttids:
    numvoicedframes_correct[uttid] = (np.sum(vad_vec), vad_vec.shape[0])
  if uttid in wrong_uttids:
    numvoicedframes_wrong[uttid] = (np.sum(vad_vec), vad_vec.shape[0])

draw_histogram(numvoicedframes_correct, 0, "distribution of number of voiced frames per utterance(correct utterance)")
print("The nvoiced/nwhole ratio is %.4f" %get_ratio(numvoicedframes_correct))

draw_histogram(numvoicedframes_wrong, 0, "distribution of number of voiced frames per utterance(wrong utterance)")
print("The nvoiced/nwhole ratio is %.4f" %get_ratio(numvoicedframes_wrong))

