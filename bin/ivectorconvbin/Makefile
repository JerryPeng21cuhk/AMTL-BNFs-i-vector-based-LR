
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi2.mk

#BINFILES = ivector-extractor-conv-init ivector-extractor-conv-acc-stats \
#		   ivector-extractor-conv-sum-acc ivector-extractor-conv-est \
#		   ivector-extract-conv \
#		   copy-tmat-conv 
BINFILES = ivector-extract-conv

OBJFILES =

TESTFILES =

ADDLIBS = ../ivectorconv/kaldi-ivector-conv.a ../ivector/kaldi-ivector.a ../hmm/kaldi-hmm.a ../gmm/kaldi-gmm.a \
    ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a \
    ../util/kaldi-util.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
