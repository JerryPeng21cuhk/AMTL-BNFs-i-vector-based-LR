// jvectorbin/jvector-plda-pairwise-scoring-parallel.cc

// Copyright 2018 Jerry Peng

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "jvector/plda.h"
#include "util/kaldi-thread.h"

namespace kaldi {

// It does pariwise scoring for the $i row of $mat
class TaskPairwiseScoring {
  public:
    // ith row has #len items
    // It doesnt require mat to be square
    TaskPairwiseScoring(Plda *plda,
            PldaConfig *plda_config,
            std::vector<Vector<BaseFloat> > *jvectors,
            int32 i, 
            int32 len, 
            Matrix<double> *mat):
      plda_(plda), plda_config_(plda_config),
      jvectors_(jvectors),
      idx_(i), len_(len), mat_(mat) { } 
     
    void operator() () {
      Vector<BaseFloat> transformed_jvector1(plda_->Dim()),
                          transformed_jvector2(plda_->Dim());
      plda_->TransformIvector(*plda_config_, (*jvectors_)[idx_], 1, &transformed_jvector1);
      for (int32 j = idx_+1; j < len_; j++) {
        plda_->TransformIvector(*plda_config_, (*jvectors_)[j], 1, &transformed_jvector2);
        (*mat_)(idx_, j) = plda_->LogLikelihoodRatio_MultiVsMulti(
                          (Vector<double>)transformed_jvector1, 1,
                          (Vector<double>)transformed_jvector2, 1
                          );
      }
    }

    ~TaskPairwiseScoring() {
      // do nothing
    }

  private:
     Plda *plda_;
     PldaConfig *plda_config_;
     std::vector<Vector<BaseFloat> > *jvectors_;
     int32 idx_;
     int32 len_;
     Matrix<double> *mat_;
};

}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef std::string string;

  try {
    const char *usage = 
        "The parallel version of jvector-plda-pairwise-scoring\n"
        "This is a special design for speeding up pairwsely enumeration scoring a set of ivectors\n"
        "Computes log-likelihood ratios by pairwisely enumerating a set of ivectors using PLDA model\n"
        "Suppose there are N ivectors, it will generate N*(N-1)/2 scores.\n"
        "The output is a symmetric matrix. The i,j item is the plda score between i-th utt and j-th utt "
        "of the first column in file jvectors.scp/jvectors.ark\n"
        "Usage: jvector-plda-pairwise-scoring <plda> <test-jvector-rspecifier> "
        "<score-matrix-out> "
        "\n"
        "e.g.: jvector-plda-pairwise-scoring plda "
        "ark:exp/test/jvectors.ark score.mat\n";

    ParseOptions po(usage);

    bool binary = true;
    TaskSequencerConfig sequencer_opts;
    PldaConfig plda_config;

    po.Register("binary", &binary, "Write output in binary mode");
    sequencer_opts.Register(&po);
    plda_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    std::string plda_rxfilename = po.GetArg(1),
        test_jvector_rspecifier = po.GetArg(2),
        score_rxfilename = po.GetArg(3);

    Plda plda;
    ReadKaldiObject(plda_rxfilename, &plda);
    //int32 dim = plda.Dim(); // jvector dim


    SequentialBaseFloatVectorReader jvector_reader(test_jvector_rspecifier);

    std::vector<Vector<BaseFloat> > jvectors;

    for (; !jvector_reader.Done(); jvector_reader.Next()) {
      jvectors.resize(jvectors.size() + 1);
      jvectors.back() = jvector_reader.Value();
    }


    Matrix<double> scores(jvectors.size(), jvectors.size());

    {
      TaskSequencer<TaskPairwiseScoring> sequencer(sequencer_opts);

      // pairwise scoring
      for (int32 i = 0; i < jvectors.size(); i++) {
        sequencer.Run(new TaskPairwiseScoring(&plda,
                                      &plda_config,
                                      &jvectors,
                                      i,
                                      jvectors.size(),
                                      &scores) );
        KALDI_LOG << "Done #rows " << i;
      }
    }

    scores.AddMat(1.0, scores, kTrans); // make it symmetric

    WriteKaldiObject(scores, score_rxfilename, binary);
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }


}