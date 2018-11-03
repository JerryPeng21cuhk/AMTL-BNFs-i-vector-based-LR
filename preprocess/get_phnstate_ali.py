# Author: Jerry Peng 2018


# This sciprt reads in an audio list from stdin and generates the coresponding senone state alignment
# to stdout.
#
# To use this script, please make sure phnrec is installed in your system.
#   phnrec: https://speech.fit.vutbr.cz/software/phoneme-recognizer-based-long-temporal-context

# Each line of the input should be formatted as follows.
# <wavid> <wavpath> <frame-num>
# Here, <frame-num> means the number of frames. It can be specified by first generating mfcc for a given wav and then use kaldi: feat-to-len to compute the frame number.
# This is to generate frame labels.

# Note: Please replace some absolute paths before running this script

# Usage example:
#   1. Sequential method
#     paste -d'\t' \
#       <(cut data/train/wav.scp -f1 -d' ') \
#       <(cut data/train/wav.scp -f2- -d' ') \
#       <(cut data/train/feats.len -f2 -d' ') | \
#       python get_phnstate_ali.py > exp/ali/ali.ark
#
#   2. Parallel method (parallel should be installed)
#     paste -d'\t' \
#       <(cut data/train/wav.scp -f1 -d' ') \
#       <(cut data/train/wav.scp -f2- -d' ') \
#       <(cut data/train/feats.len -f2 -d' ') | \
#       parallel --progress --joblog exp/ali/log -j 15 -N10 -k --spreadstdin \
#         python preprocess/get_phnstate_ali.py > exp/ali/ali.ark
#    ## comments:
#    ##  1. 15 jobs, input 10 lines for each job
#    ##  2. parallel can be resumed if the input is unchanged by adding --resume and --joblog 
#             options


from __future__ import print_function
import pdb
import subprocess
import os
import sys
import pandas as pd
import numpy as np
from shlex import split
import tempfile
#import StringIO
import struct
import re
import signal
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

phnrec='/home/jerry/research/PhnRec/phnrec'
PHN_CZ_SPDAT_LCRC_N1500='/home/jerry/research/PhnRec/PHN_CZ_SPDAT_LCRC_N1500'

def bashcmd_excute(cmd, input=None, output=subprocess.PIPE, shell=False):
    if True==shell:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=input, shell=True)
    else:
        process = subprocess.Popen(split(cmd), stdout=subprocess.PIPE, stdin=input, shell=False)
    output, error = process.communicate()
    return output, error

# wavpath should be revised by add_downmix.py
# for each line in file wav.scp and feats.len
def phnpost_convert(wavid, wavpath, featlen, level='state'):
    #Create a temp file to store the output of sox
    tmpf = tempfile.NamedTemporaryFile()
    #Down-sampling
    #Check mono-channel or stereo
    try:
        if not os.path.isfile(wavpath):
            # a stereo
            bashcmd_excute("{0} sox -t wav - -r 8000 -t wav {1}".format(wavpath, tmpf.name), shell=True)
        else:
            # a path(mono channel)
            bashcmd_excute("sox -t wav {0} -r 8000 -t wav {1}".format(wavpath, tmpf.name))
        
        bin_post, error = bashcmd_excute("{0} -c {1} -i {2} -t post -o /dev/stdout".format(
                            phnrec,
                            PHN_CZ_SPDAT_LCRC_N1500,
                            tmpf.name,
                        ))
    finally:
        tmpf.close()
    size = struct.calcsize(">iihh")
    nSamples, sampPeriod, sampSize, paramKind = struct.unpack(">iihh", bin_post[:size])
    format = ">%df" %(nSamples * sampSize/4)
    temp = list(struct.unpack(format, bin_post[size:]))
    # state-level posterior
    posterior = np.array(temp).reshape((nSamples, int(sampSize/4)))
    if 'state' != level:
        posterior = posterior.T
        nstatetotal = posterior.shape[0] 
        nstateOfphn = 3 # Each phoneme has three states
        phn_posterior = []
        # Convert to phoneme-level posterior
        for i in range(0, nstatetotal, nstateOfphn):
            phn_posterior.append(sum(posterior[i:i+nstateOfphn]))

        posterior = np.array(phn_posterior).T
    ali = np.argmax(posterior, axis=1)
    # make sure length of label matches featlen
    alilen = len(ali)
    if featlen > alilen:
        ali = np.append(ali, np.repeat(ali[-1] ,featlen-alilen))
    else:
        ali = ali[:featlen]
    assert len(ali)==featlen, "Length mismatched!"
    print(wavid, end=' ')
    np.savetxt(sys.stdout, ali, fmt='%d', newline=' ')
    print(end='\n')
    #phnrec: decode phone state posterior
    
def main():
    for line in sys.stdin:
	try:
            wavid, wavpath, featlen = re.split(r'\t+', line)
	except:
	    assert False, line
        featlen = int(featlen)
        phnpost_convert(wavid, wavpath, featlen)

if __name__ == '__main__':
    main()
