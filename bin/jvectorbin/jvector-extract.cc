// jvectorbin/jvector-extract.cc

// Copyright 2015  Daniel Povey
//                 Srikanth Madikeri (Idiap Research Institute)
//           2018  Jerry Peng

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
#include "gmm/am-diag-gmm.h"
#include "jvector/jvector-extractor.h"
#include "util/kaldi-thread.h"

namespace kaldi {

// This class will be used to parallelize over multiple threads the job
// that this program does.  The work happens in the operator (), the
// output happens in the destructor.
  class JvectorExtractTask
  {
  public:
    JvectorExtractTask(const JvectorExtractor &extractor,
                        std::string utt,
                        const Matrix<BaseFloat> &feats,
                        const JvectorExtractorDecodeOptions &decode_opts,
                        const Posterior &ipost,
                        BaseFloatVectorWriter *jvector_writer,
                        PosteriorWriter *post_writer
                      ):
        extractor_(extractor), utt_(utt), feats_(feats), 
        decode_opts_(decode_opts), ipost_(ipost),
        jvector_writer_(jvector_writer), post_writer_(post_writer) { };

    void operator () () {
      KALDI_ASSERT(NULL != jvector_writer_);
      if (NULL == post_writer_)
        extractor_.DecodeJvector(feats_, ipost_, &jvector_);
      else
        extractor_.DecodeJvectorAndPosterior(feats_, ipost_, decode_opts_,
                      &jvector_, &opost_);
    };

    ~JvectorExtractTask() {
      KALDI_ASSERT(NULL != jvector_writer_);
      jvector_writer_->Write(utt_, Vector<BaseFloat>(jvector_));
      KALDI_VLOG(2) << "jVector for utterance " << utt_
                    << " is written successfully.";
      if (NULL != post_writer_)
      {
        post_writer_->Write(utt_, opost_);
        KALDI_VLOG(2) << "Posterior for utterance " << utt_
                      << " is written successfully.";
      }
    };

  private:
    const JvectorExtractor &extractor_;
    std::string utt_;
    Matrix<BaseFloat> feats_;
    JvectorExtractorDecodeOptions decode_opts_;
    Posterior ipost_;

    BaseFloatVectorWriter *jvector_writer_;
    PosteriorWriter *post_writer_;

    // No need to initialize these variables
    Posterior opost_;
    Vector<double> jvector_;
  };
}

int main(int argc, char const *argv[])
{
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {

    const char *usage = 
      "Extract jVectors (and Posteriors) for utterances, using a trained jVector extractor,\n"
      "and features and Gaussian-level posteriors\n"
      "Usage: jvector-extract [options] <model-in> <feature-rspecifier> "
      "<posteriors-rspecifier> <jvector-wspecifier> [<posteriors-wspecifier>]\n"
      "e.g.: \n"
      " fgmm-global-gselect-to-post 1.ubm '$feats' 'ark:gunzip -c gselect.1.gz|' ark:- | \\\n"
      "  jvector-extract final.ie '$feats' ark,s,cs:- ark,t:jvectors.1.ark ark:posts.1.ark\n"
      " or: fgmm-global-gselect-to-post 1.ubm '$feats' 'ark:gunzip -c gselect.1.gz|' ark:- | \\\n"
      "  jvector-extract final.ie '$feats' ark,s,cs:- ark,t:jvectors.1.ark\n";

    ParseOptions po(usage);
    JvectorExtractorDecodeOptions decode_opts;
    TaskSequencerConfig sequencer_config;
    decode_opts.Register(&po);
    sequencer_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4 && po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string jvector_extractor_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        ipost_rspecifier = po.GetArg(3),
        jvector_wspecifier = po.GetArg(4);


    JvectorExtractor extractor;
    ReadKaldiObject(jvector_extractor_rxfilename, &extractor);

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader ipost_reader(ipost_rspecifier);
    BaseFloatVectorWriter jvector_writer(jvector_wspecifier);

    TaskSequencer<JvectorExtractTask> sequencer(sequencer_config);

    // Initialize opost_writer ptr as NULL
    PosteriorWriter *opost_writer = NULL;
    if (po.NumArgs() == 5) {
      //posterior writer
      std::string opost_wspecifier = po.GetArg(5);
      opost_writer = new PosteriorWriter(opost_wspecifier);
      KALDI_ASSERT(opost_writer != NULL);
    }

    int64 tot_t = 0; // total number of frames
    int32 num_done = 0, num_err = 0;

    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      if (!ipost_reader.HasKey(key)) {
        KALDI_WARN << "No posteriors for utterance " << key;
        num_err++;
        continue;
      }
      const Matrix<BaseFloat> &mat = feature_reader.Value();
      const Posterior &ipost = ipost_reader.Value(key);

      if(ipost.size() == 0 || mat.NumRows() == 0) {
        KALDI_WARN << "No feature/posterior vectors for " << key;
        continue;
      }
      if (static_cast<int32>(ipost.size()) != mat.NumRows()) {
        KALDI_WARN << "Size mismatch between posterior " << (ipost.size())
                   << " and features " << (mat.NumRows()) << " for utterance "
                   << key;
        num_err++;
        continue;
      }
      // opost_writer should be NULL for jvector extraction only
      // otherwise, it will also write posterior.
      sequencer.Run(new JvectorExtractTask(extractor, key, mat, decode_opts,
                             ipost, &jvector_writer, opost_writer));
      
      tot_t +=  ipost.size();
      num_done++;
    }
    // Destructor of "sequencer" will wait for any remaining tasks.

    KALDI_LOG << "Done " << num_done << " files, " << num_err
              << " with errors.  Total frames " << tot_t;


    return (num_done != 0 ? 0 : 1);

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }

}