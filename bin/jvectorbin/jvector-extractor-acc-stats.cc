// jvectorbin/jvector-extractor-acc-stats.cc

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
#include "gmm/am-diag-gmm.h"
#include "jvector/jvector-extractor.h"
#include "util/kaldi-thread.h"


namespace kaldi {
  // this class is used to run the command
  //  stats.AccStatsForUtterance(extractor, mat, posterior);
  // in parallel.

  class JvectorTask {
  public:
    JvectorTask(const JvectorExtractor &extractor,
                std::string utt,
                const Matrix<BaseFloat> &features,
                const JvectorExtractorDecodeOptions &decode_opts,
                const Posterior &ipost,
                PosteriorWriter *post_writer,
                JvectorStats *stats,
                const bool IsToDecodePosterior=true): 
    extractor_(extractor), utt_(utt), features_(features),
      decode_opts_(decode_opts), ipost_(ipost), post_writer_(post_writer),
      stats_(stats) { 
        if (IsToDecodePosterior) {
          p_opost_ = new Posterior();
        } else {
          p_opost_ = NULL; // NULL for no posterior decoding
        }
      }

    void operator () () {
      stats_->AccStatsForUtterance(extractor_, features_, 
        decode_opts_, ipost_, p_opost_);
    };

    ~JvectorTask() {
      if (NULL != p_opost_) {
        post_writer_->Write(utt_, *p_opost_);
        KALDI_VLOG(2) << "Posterior for utterance " << utt_ 
                    << " is written successfully.";
        delete p_opost_;
      }
     };

  private: 
    const JvectorExtractor &extractor_;
    std::string utt_;
    Matrix<BaseFloat> features_;
    JvectorExtractorDecodeOptions decode_opts_;
    Posterior ipost_;
    Posterior *p_opost_;
    PosteriorWriter *post_writer_;
    JvectorStats *stats_; 
  };
  
}



int main(int argc, char const *argv[])
{
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {

    const char *usage =
        "Accumuate stats for jVector extractor training\n"
        "Reads in features and Gaussian-level posteriors (typically from a full GMM)\n"
        "Writes out stats and update posteriors\n"
        "Support multiple threads, but won't be able to make use of too many at a time\n"
        "(e.g. more than about 4)\n"
        "Usage: jvector-extractor-acc-stats [options] <jvector-model-in> <feature-rspecifier> "
        "<posteriors-rspecifier> [<posteriors-wspecifier>] <stats-out>\n"
        "e.g.: \n"
        " fgmm-global-gselect-to-post 1.fgmm '$feats' 'ark:gunzip -c gselect.1.gz|' ark:- | \\\n"
        " jvector-extractor-acc-stats 1.ie '$feats' ark,s,cs:- ark:2.1.post 2.1.acc\n";

    ParseOptions po(usage);
    bool binary = true;

    TaskSequencerConfig sequencer_opts;
    JvectorExtractorDecodeOptions decode_opts;
    po.Register("binary", &binary, "Write output in binary mode");
    sequencer_opts.Register(&po);
    decode_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() < 4 || po.NumArgs() > 5) {
      po.PrintUsage();
      exit(1);
    }

    bool IsToDecodePosterior = true; 


    std::string jvector_extractor_rxfilename, feature_rspecifier,
      ipost_rspecifier, opost_wspecifier, accs_wxfilename;

    if (po.NumArgs() == 5 ) {
      jvector_extractor_rxfilename = po.GetArg(1);
      feature_rspecifier = po.GetArg(2);
      ipost_rspecifier = po.GetArg(3);
      opost_wspecifier = po.GetArg(4);
      accs_wxfilename = po.GetArg(5);
    } else {
      jvector_extractor_rxfilename = po.GetArg(1);
      feature_rspecifier = po.GetArg(2);
      ipost_rspecifier = po.GetArg(3);
      opost_wspecifier = "ark:/dev/null";
      accs_wxfilename = po.GetArg(4);
      IsToDecodePosterior = false;
    }



    // Initialize these Reader objects before reading the JvectorExtractor,
    // because it uses up a lot of memory and any fork() after that will
    // be in danger of causing an allocation failure.
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    JvectorExtractor extractor;
    ReadKaldiObject(jvector_extractor_rxfilename, &extractor);

    JvectorStats stats(extractor);
    PosteriorWriter opost_writer(opost_wspecifier);

    int64 tot_t = 0;
    int32 num_done = 0, num_err = 0;

    {
      TaskSequencer<JvectorTask> sequencer(sequencer_opts);

      RandomAccessPosteriorReader posteriors_reader(ipost_rspecifier);
      for (; !feature_reader.Done(); feature_reader.Next()) {
        std::string key = feature_reader.Key();
        KALDI_LOG << "Opening feature " << key;
        if (!posteriors_reader.HasKey(key)) {
          KALDI_WARN << "No posteriors for utterance " << key;
          num_err++;
          continue;
        }
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const Posterior &posterior = posteriors_reader.Value(key);

        if (static_cast<int32>(posterior.size()) != mat.NumRows()) {
          KALDI_WARN << "Size mismatch between posterior " << (posterior.size())
                     << " and features " << (mat.NumRows()) << " for utterance "
                     << key;
          num_err++;
          continue;
        }

        sequencer.Run(new JvectorTask(extractor, key, mat, decode_opts, posterior, 
                            &opost_writer, &stats, IsToDecodePosterior)); // generate posterior

        tot_t += mat.NumRows();
        num_done++;
        KALDI_LOG << "Done feature " << key;
      }

      // destructor of "sequencer" will wait for any remaining tasks that
      // have not yet completed.
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_err
              << " with errors.  Total frames " << tot_t;

    WriteKaldiObject(stats, accs_wxfilename, binary);
    
    KALDI_LOG << "Wrote stats to " << accs_wxfilename;


    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
