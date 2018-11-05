#!/usr/bin/env python
# Copyright 2016  Tsinghua University
#                 (Author: Yixiang Chen, Lantian Li, Dong Wang)
# Licence: Apache 2.0


import sys
from math import *
#import pdb
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# Load the result file in the following format
#           lang0     lang1     lang2     lang3     lang4     lang5     lang6     lang7     lang8     lang9   
# <utt-id>  <score0>  <score1>  <score2>  <score3>  <score4>  <score5>  <score6>  <score7>  <score8>  <score9>

# The language identity is defined as: 
langlist = ['ct-cn', 'id-id', 'ja-jp', 'ko-kr', 'ru-ru', 'vi-vn', 'zh-cn', 'Kazak', 'Tibet', 'Uyghu']

langnum = 10
dictl = {'0':1, '1':2, '2':3, '3':4, '4':5, '5':6, '6':7, '7':8, '8':9, '9':10}

# Load scoring file and label.scp.
def Loaddata(fin, langnum):

    x = []
    for i in range(langnum+1):
        x.append(0)

    if fin == '-':
	fin = sys.stdin
    else:
        fin = open(fin, 'r')
    lines = fin.readlines()
    fin.close()

    data = []
    for line in lines[1:]:
        part = line.split()
        x[0] = part[0].split('g')[1].split('_')[0]
        for i in range(langnum):
            x[i+1] = part[i + 1]
        data.append(x)
        x = []
        for i in range(langnum+1):
            x.append(0)

    datas = []
    for ll in data:
        for lb in range(langnum):
            datas.append([dictl[ll[0][0]], lb + 1, float(ll[lb + 1])])
            
    # score normalized to [0, 1] 
    for i in range(len(datas) / langnum):
        sum = 0
        for j in range(langnum):
            k = i * langnum + j
            sum += exp(datas[k][2])
        for j in range(langnum):
            k = i * langnum + j
            datas[k][2] = exp(datas[k][2]) / sum


    return datas

# Compute Cavg.
# data: matrix for result scores, assumed within [0,1].
# threshold: 0.1 default(Need adjustment for other task).
def CountConfusionMatrix(data, threshold = 0.1, lgn = 10):

    target_Cavg = [0.0] * lgn
    # target_Cavg: P_Target * P_Miss + sum(P_NonTarget*P_FA)
    ConfMatrix = []

    for language in range(lgn):
        P_FA = [0.0] * lgn
        P_Miss = 0.0
        # compute P_FA and P_Miss
        LTm = 0.0; LTs = 0.0; LNm = 0.0; LNs = [0.0] * lgn;
        for line in data:
            language_label = language + 1
            if line[0] == language_label:
                if line[1] == language_label:
                    LTm += 1
                #    if line[2] < threshold:
                #        LTs += 1
                for t in range(lgn):
		    if line[1] == t + 1:
                        if line[2] > threshold:
                            LNs[t] += 1
        LNm = LTm
	ConfMatrix.append([LNs, LNm])

    #pdb.set_trace()
    return ConfMatrix 


def ConfMatrix_Display(ConfMatrix):
    rows = []
    for row,ntotal in ConfMatrix:
	#srow = [str(item) for item in row]
	#print(', '.join(srow) + ' :%d'%ntotal)
	rows.append(row)
    # Compute confusion matrix
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    #pdb.set_trace()
    plot_confusion_matrix(np.array(rows), classes=langlist,
                          title='Confusion matrix, without normalization')
    
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(np.array(rows), classes=langlist, normalize=True,
                          title='Normalized confusion matrix')
    
    plt.show()



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
	cm = cm.astype('int')
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



if __name__ == '__main__':

    fin=sys.argv[1]
    data = Loaddata(fin, langnum)
    
    ConfMatrix = CountConfusionMatrix(data, threshold=0.2, lgn=langnum)
    ConfMatrix_Display(ConfMatrix)

