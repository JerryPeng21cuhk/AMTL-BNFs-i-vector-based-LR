# jerry peng 2018

# In this script, it reads in a foo_xxxx file, and get the third column which is a column of score
# and apply softmax on the score to handle the problem of severe mismatch between Cavg and EER
# as eer do not apply softmat to normalize scores


import sys
import os
import pdb
import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


cnt = 0
score = [0] * 10
with open(sys.argv[1], 'r') as f:
  for line in f:
    score[cnt] = float(line.rstrip().split(' ')[-1])
    if cnt == 9:
      cnt = 0
      score = softmax(score)
      for i in score:
        print("%.4f" %i)
      score = [0] *10
    else:
      cnt = cnt + 1