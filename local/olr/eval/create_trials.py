#import pdb
import sys
import os
import re
import signal
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

lang_dict = {'ct-cn':'ct', 'id-id':'id', 'ja-jp':'ja', 'ko-kr':'ko', 'ru-ru':'ru', 'vi-vn':'vi', 'zh-cn':'zh', 'Kazak':'Kazak', 'Tibet':'Tibet', 'Uyghu':'Uyghu'}

def match(utt, lang):
    return lang_dict[utt[0:5]] == lang

train_spk_ivector=sys.argv[1]
dev_ivector=sys.argv[2]
fo=sys.argv[3]
if fo == '-':
    fo = sys.stdout
else:
    fo = open(fo, 'w')

langs = [line.rstrip('\n').split(' ')[0] for line in open(train_spk_ivector, 'r')]

utts = [line.rstrip('\n').split(' ')[0] for line in open(dev_ivector, 'r')]
#pdb.set_trace()

for utt in utts:
    for lang in langs:
	if match(utt, lang):
	    fo.write(lang + ' ' + utt + ' ' + 'target\n')
	else:
	    fo.write(lang + ' ' + utt + ' ' + 'nontarget\n')
