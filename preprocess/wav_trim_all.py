# Author: Jerry PEng 2018

# In this script, we read in wav.scp and vad.scp, output a revised wav.scp
# which cut the origianl wav into a fixed duration sub-utterance
# vad.scp is used to prevent from creating non-speech sub-utterance
# sub-utterances are discarded once the ratio of #voicedframe/#frames is
# less than a specific ratio, etc, 0.2 in this script.

# for the details like trim step and trim duration, please refer to codes below.




import re
import pdb
import kaldi_io
import sys
import numpy as np

# args
dur=100 # frames; eacho sub-utt has #dur frames
# frame_shift = 0.01 # in second
# io used just for debugging
ipath2vad_ark = "ark: copy-vector scp:data/valid/vad.scp ark,t:- |"
ipath2wav_scp = "scp:data/valid/wav.scp"
opath2wav_scp = "data/valid/wav_1sall.scp"


#ipath2vad_ark = "ark: copy-vector scp:data/valid/vad.scp ark,t:- |"
#ipath2wav_scp = "scp:data/valid/wav.scp"
#opath2wav_scp = "data/valid/wav_justfortest.scp"

def get_vad_dict(ipath2vad_ark):
  vad_dict = dict()
  for uttid, vad_vec in kaldi_io.read_vec_int_ark(ipath2vad_ark):
    assert (uttid not in vad_dict), "Duplicated utterance %s in %s" %(uttid, ipath2vad_ark)
    vad_dict[uttid] = vad_vec
  return vad_dict


def read_wav_scp(ipath2wav_scp):
  fd = kaldi_io.open_or_fd(ipath2wav_scp)
  try:
    for line in fd:
      uttid, path2wav = line.rstrip().decode().split(' ', 1)
      yield uttid, path2wav
  finally:
    if fd is not ipath2wav_scp: fd.close()


def get_new_wav_scp(ipath2vad_ark, ipath2wav_scp, opath2wav_scp):
  vad_dict = get_vad_dict(ipath2vad_ark)
  fo = open(opath2wav_scp, "w")
  err_utt = 0
  for uttid, path2wav in read_wav_scp(ipath2wav_scp):
    assert (uttid in vad_dict), "No vad info for utterance: %s" %(uttid)
    vad_vec = vad_dict[uttid]
    nvoicedframes = sum(vad_vec) # need revision
    nframes = vad_vec.shape[0]
    if (nframes < dur):
      print("Warning: utterance %s is too short to be trimmed into an utterance with duration: %.2f" %(uttid, float(dur)/100))

    # start to trim from 0 second with step 0.5*dur
    flag = False # trim at least one subutt
    for subutt_idx, start in enumerate(range(0, nframes, int(dur*0.5)) ):
      end = start + dur
      if nframes < end and not flag :
        print("Warning: utterance %s failed to be trimmed" %(uttid))
        err_utt = err_utt + 1;
        break
      if 0.2 * dur < sum(vad_vec[start:end]):
        if re.search("sox", path2wav):
          opath2wav = path2wav + " sox -t wav -  -t wav - trim %.2f %.2f |" %(float(start*0.01), float(end*0.01))
        else:
          opath2wav = "sox %s -t wav - trim %.2f %.2f |" %(path2wav, float(start*0.01), float(end*0.01))
        subuttid = uttid + "-%05d" %subutt_idx
        fo.write(subuttid + ' ' +  opath2wav + '\n')
        flag = True
  print("Failed utterance: %d" %err_utt)

get_new_wav_scp(ipath2vad_ark, ipath2wav_scp, opath2wav_scp)


  
