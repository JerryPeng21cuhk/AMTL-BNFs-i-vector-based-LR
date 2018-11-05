# Author: Jerry Peng

# In this script, given a trained AMTL nnet3 dir, it reads the log files:
# compute_prob_train.*.log and compute_prob_valid.*.log.
# and draw llk_curve and accuracy_curve respectively.

# This is used to check intermediate results.
# The results shows that there is no noticeable effect when increasing the
# grl weight.


import pdb
import os
import glob
import re, sys
import matplotlib.pyplot as plt


#ipath2log="/home/jerry/research/ap17_olr/lsid/exp/nnet3/multi_bnf/log"




def get_digits(x):
    try:
        digits=re.search('\w\.(\d+?)\.log', x).group(1)
        return(int(digits))
    except:
        print('Cannot capture digits of file: %s' %x)
        print('Set digits of file:%s to 1000' %x)
        return(1000)


def get_train(ipath2log):
    flist_prob_train = glob.glob(os.path.join(ipath2log, 'compute_prob_train.*.log'))
    flist_prob_train = sorted(flist_prob_train, key=get_digits)

    output_likelihood = {}
    output_accuracy = {}

    for f in flist_prob_train:
        #pdb.set_trace()
        for line in open(f).readlines():
            match = re.search('log-likelihood.*(output[-\d]*)\' is ([-+]?\d*\.\d+|\d+) per', line)     
            if match:
                output_node = match.group(1)
                likelihood = float(match.group(2))
                try:
                    output_likelihood[output_node].append(likelihood)
                except:
                    output_likelihood[output_node] = [likelihood]

            match = re.search('accuracy.*(output[-\d]*)\' is ([-+]?\d*\.\d+|\d+) per', line)     
            if match:
                output_node = match.group(1)
                likelihood = float(match.group(2))
                try:
                    output_accuracy[output_node].append(likelihood)
                except:
                    output_accuracy[output_node] = [likelihood]

    return output_likelihood, output_accuracy


def get_valid(ipath2log):
    flist_prob_train = glob.glob(os.path.join(ipath2log, 'compute_prob_valid.*.log'))
    flist_prob_train = sorted(flist_prob_train, key=get_digits)

    output_likelihood = {}
    output_accuracy = {}

    for f in flist_prob_train:
        #pdb.set_trace()
        for line in open(f).readlines():
            match = re.search('log-likelihood.*(output[-\d]*)\' is ([-+]?\d*\.\d+|\d+) per', line)     
            if match:
                output_node = match.group(1)
                likelihood = float(match.group(2))
                try:
                    output_likelihood[output_node].append(likelihood)
                except:
                    output_likelihood[output_node] = [likelihood]

            match = re.search('accuracy.*(output[-\d]*)\' is ([-+]?\d*\.\d+|\d+) per', line)     
            if match:
                output_node = match.group(1)
                likelihood = float(match.group(2))
                try:
                    output_accuracy[output_node].append(likelihood)
                except:
                    output_accuracy[output_node] = [likelihood]

    return output_likelihood, output_accuracy
                

if __name__ == '__main__':
    logdir = sys.argv[1]
    valid_likelihood, valid_accuracy = get_valid(logdir)
    train_likelihood, train_accuracy = get_train(logdir)
    assert(len(valid_likelihood) == len(train_likelihood))
    assert(len(valid_accuracy) == len(train_accuracy))
    assert(len(valid_accuracy) == len(valid_likelihood))
    #pdb.set_trace()
    for output_node in train_likelihood.keys():
        iters=len(valid_accuracy[output_node])
        #pdb.set_trace()
        plt.figure()
        plt.subplot(211)
        plt.title(output_node)
        plt.plot(range(iters), train_likelihood[output_node])
        plt.hold(True)
        plt.plot(range(iters), valid_likelihood[output_node])
        plt.subplot(212)
        plt.plot(range(iters), train_accuracy[output_node])
        plt.hold(True)
        plt.plot(range(iters), valid_accuracy[output_node])
        plt.show()
    print('valid_accuracy', valid_accuracy)
    print('train_accuracy', train_accuracy)
