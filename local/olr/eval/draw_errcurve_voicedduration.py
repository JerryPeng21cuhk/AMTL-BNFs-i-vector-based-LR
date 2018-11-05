# Author: Jerry Peng

# In this script, we draw the statistics of utterances.
# Specifically, we first get the list of misclassfied utterances
# and then draw error rate curve w.r.t #voicedframes

# The result shows a consistent decrease of error rate with the increase of #voicedframes,

# Input: exp/conivector/score/total/foo_wccn  <ipath2vad-rspecifier>

import pdb
import os
import re, sys
from signal import signal, SIGPIPE, SIG_DFL
import kaldi_io
from kaldi_utt_hist import *

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


numvoicedframes_correct = dict()
numvoicedframes_wrong = dict()

for uttid, vad_vec in kaldi_io.read_vec_int_ark(ipath2vad_scp):
  assert (uttid not in numvoicedframes_correct), "Duplicated utterance %s in %s" %(uttid, ipath2vad_scp)
  assert (uttid not in numvoicedframes_wrong), "Duplicated utterance %s in %s" %(uttid, ipath2vad_scp)
  if uttid in correct_uttids:
    numvoicedframes_correct[uttid] = np.sum(vad_vec)
  if uttid in wrong_uttids:
    numvoicedframes_wrong[uttid] = np.sum(vad_vec)


nvf_wrong = numvoicedframes_wrong.values()
nvf_correct =  numvoicedframes_correct.values()
nvf_ratio = []
curve_x = range(20, 300, 2)
for x in curve_x:
  nutt_wrong = sum(i <= x for i in nvf_wrong)
  nutt_correct = sum(i <= x for i in nvf_correct)
  if (0 == nutt_wrong + nutt_correct):
    nvf_ratio.append(0)
  else:
    nvf_ratio.append( float(nutt_wrong) / (nutt_wrong + nutt_correct))

plt.plot(curve_x, nvf_ratio);
plt.xlabel("number of voiced frames")
plt.ylabel("ratio of number of wrong classified utts over all utts")
plt.show()
