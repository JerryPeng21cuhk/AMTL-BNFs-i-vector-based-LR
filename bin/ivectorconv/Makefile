
all:

OPENFST_CXXFLAGS = 
OPENFST_LDLIBS = 
#EXTRA_CXXFLAGS = -I../

include ../kaldi2.mk

TESTFILES = 

OBJFILES = conv-ivector-extractor.o 

LIBNAME = kaldi-ivector-conv

ADDLIBS = ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../transform/kaldi-transform.a \
		../matrix/kaldi-matrix.a ../base/kaldi-base.a \
        ../util/kaldi-util.a ../hmm/kaldi-hmm.a

include ../makefiles/default_rules.mk
