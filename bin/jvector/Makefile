
all:

OPENFST_CXXFLAGS = 
OPENFST_LDLIBS = 
#EXTRA_CXXFLAGS = -I../

include ../kaldi2.mk

TESTFILES = 

#OBJFILES = jvector-extractor.o 
OBJFILES = agglomerative-clustering.o
#OBJFILES = jvector-extractor.o plda.o agglomerative-clustering.o
#OBJFILES = vivector-extractor-test.o
#OBJFILES = posterior-test.o


LIBNAME = kaldi-jvector
#LIBNAME = kaldi-plda-simplified
#LIBNAME = kaldi-jvector

ADDLIBS = ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../transform/kaldi-transform.a \
		../matrix/kaldi-matrix.a ../base/kaldi-base.a \
        ../util/kaldi-util.a ../hmm/kaldi-hmm.a

include ../makefiles/default_rules.mk
