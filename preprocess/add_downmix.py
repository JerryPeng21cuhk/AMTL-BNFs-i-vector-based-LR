# Author: jerry Peng 2018


# Some audio in AP17-olr are stereo. And among them, most contains silence in the right channel.
# This script reads in an kaldi scp file and assumes the second field in each line is path2fname.
# It revised the second field and output the revised content to stdout.

# To use this script, please make sure sox is installed in your system.

# Usage example: 
#   python add_downmix.py wav.scp


#import pdb
import subprocess
import os
import sys
import pandas as pd
from shlex import split

wav_scp = sys.argv[1]
wavscp = pd.read_csv(wav_scp, sep=' ', header=None)

for idx, row in wavscp.iterrows():
   wavid = row[0]
   wavpath = row[1]
   cmd = "soxi %s | grep Channels | awk '{print $3}'" %(wavpath)
   nchannels, error  = subprocess.check_output(cmd, shell=True)
   if nchannels == '2':
       downmix_str = "sox {0} -t wav - remix 2 |".format(wavpath)
       print(wavid + ' ' + downmix_str)
   else:
       print(wavid + ' ' + wavpath)

