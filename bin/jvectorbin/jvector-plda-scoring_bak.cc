// jvectorbin/jvector-plda-scoring.cc

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


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef std::string string;
  try {
    const char *usage =
        "Computes log-likelihood ratios for trials using PLDA model\n"
        "Note: the 'trials-file' has lines of the form\n"
        "<key1> <key2>\n"
        "and the output will have the form\n"
        "<key1> <key2> [<dot-product>]\n"
        "(if either key could not be found, the dot-product field in the output\n"
        "will be absent, and this program will print a warning)\n"
        "For training examples, the input is the jVectors averaged over speakers;\n"
        "a separate archive containing the number of utterances per speaker may be\n"
        "optionally supplied using the --num-utts option; this affects the PLDA\n"
        "scoring (if not supplied, it defaults to 1 per speaker).\n"
        "\n"
        "Usage: jvector-plda-scoring <plda> <train-jvector-rspecifier> <test-jvector-rspecifier>\n"
        " <trials-rxfilename> <scores-wxfilename>\n"
        "\n"
        "e.g.: jvector-plda-scoring --num-utts=ark:exp/train/num_utts.ark plda "
        "ark:exp/train/spk_jvectors.ark ark:exp/test/jvectors.ark trials scores\n"
        "See also: jvector-compute-dot-products, jvector-compute-plda\n";

    ParseOptions po(usage);

    std::string num_utts_rspecifier;

    PldaConfig plda_config;
    plda_config.Register(&po);
    po.Register("num-utts", &num_utts_rspecifier, "Table to read the number of "
                "utterances per speaker, e.g. ark:num_utts.ark\n");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string plda_rxfilename = po.GetArg(1),
        train_jvector_rspecifier = po.GetArg(2),
        test_jvector_rspecifier = po.GetArg(3),
        trials_rxfilename = po.GetArg(4),
        scores_wxfilename = po.GetArg(5);

    //  diagnostics:
    double tot_test_renorm_scale = 0.0, tot_train_renorm_scale = 0.0;
    int64 num_train_jvectors = 0, num_train_errs = 0, num_test_jvectors = 0;

    int64 num_trials_done = 0, num_trials_err = 0;

    Plda plda;
    ReadKaldiObject(plda_rxfilename, &plda);

    int32 dim = plda.Dim();

    SequentialBaseFloatVectorReader train_jvector_reader(train_jvector_rspecifier);
    SequentialBaseFloatVectorReader test_jvector_reader(test_jvector_rspecifier);
    RandomAccessInt32Reader num_utts_reader(num_utts_rspecifier);

    typedef unordered_map<string, Vector<BaseFloat>*, StringHasher> HashType;

    // These hashes will contain the jVectors in the PLDA subspace
    // (that makes the within-class variance unit and diagonalizes the
    // between-class covariance).  They will also possibly be length-normalized,
    // depending on the config.
    HashType train_jvectors, test_jvectors;

    KALDI_LOG << "Reading train jVectors";
    for (; !train_jvector_reader.Done(); train_jvector_reader.Next()) {
      std::string spk = train_jvector_reader.Key();
      if (train_jvectors.count(spk) != 0) {
        KALDI_ERR << "Duplicate training jVector found for speaker " << spk;
      }
      const Vector<BaseFloat> &jvector = train_jvector_reader.Value();
      int32 num_examples;
      if (!num_utts_rspecifier.empty()) {
        if (!num_utts_reader.HasKey(spk)) {
          KALDI_WARN << "Number of utterances not given for speaker " << spk;
          num_train_errs++;
          continue;
        }
        num_examples = num_utts_reader.Value(spk);
      } else {
        num_examples = 1;
      }
      Vector<BaseFloat> *transformed_jvector = new Vector<BaseFloat>(dim);

      tot_train_renorm_scale += plda.TransformIvector(plda_config, jvector,
                                                      num_examples,
                                                      transformed_jvector);
      train_jvectors[spk] = transformed_jvector;
      num_train_jvectors++;
    }
    KALDI_LOG << "Read " << num_train_jvectors << " training jVectors, "
              << "errors on " << num_train_errs;
    if (num_train_jvectors == 0)
      KALDI_ERR << "No training jVectors present.";
    KALDI_LOG << "Average renormalization scale on training jVectors was "
              << (tot_train_renorm_scale / num_train_jvectors);

    KALDI_LOG << "Reading test jVectors";
    for (; !test_jvector_reader.Done(); test_jvector_reader.Next()) {
      std::string utt = test_jvector_reader.Key();
      if (test_jvectors.count(utt) != 0) {
        KALDI_ERR << "Duplicate test jVector found for utterance " << utt;
      }
      const Vector<BaseFloat> &jvector = test_jvector_reader.Value();
      int32 num_examples = 1; // this value is always used for test (affects the
                              // length normalization in the TransformIvector
                              // function).
      Vector<BaseFloat> *transformed_jvector = new Vector<BaseFloat>(dim);

      tot_test_renorm_scale += plda.TransformIvector(plda_config, jvector,
                                                     num_examples,
                                                     transformed_jvector);
      test_jvectors[utt] = transformed_jvector;
      num_test_jvectors++;
    }
    KALDI_LOG << "Read " << num_test_jvectors << " test jVectors.";
    if (num_test_jvectors == 0)
      KALDI_ERR << "No test jVectors present.";
    KALDI_LOG << "Average renormalization scale on test jVectors was "
              << (tot_test_renorm_scale / num_test_jvectors);


    Input ki(trials_rxfilename);
    bool binary = false;
    Output ko(scores_wxfilename, binary);

    double sum = 0.0, sumsq = 0.0;
    std::string line;

    while (std::getline(ki.Stream(), line)) {
      std::vector<std::string> fields;
      SplitStringToVector(line, " \t\n\r", true, &fields);
      if (fields.size() != 2) {
        KALDI_ERR << "Bad line " << (num_trials_done + num_trials_err)
                  << "in input (expected two fields: key1 key2): " << line;
      }
      std::string key1 = fields[0], key2 = fields[1];
      if (train_jvectors.count(key1) == 0) {
        KALDI_WARN << "Key " << key1 << " not present in training jVectors.";
        num_trials_err++;
        continue;
      }
      if (test_jvectors.count(key2) == 0) {
        KALDI_WARN << "Key " << key2 << " not present in test jVectors.";
        num_trials_err++;
        continue;
      }
      const Vector<BaseFloat> *train_jvector = train_jvectors[key1],
          *test_jvector = test_jvectors[key2];

      Vector<double> train_jvector_dbl(*train_jvector),
          test_jvector_dbl(*test_jvector);

      int32 num_train_examples;
      if (!num_utts_rspecifier.empty()) {
        // we already checked that it has this key.
        num_train_examples = num_utts_reader.Value(key1);
      } else {
        num_train_examples = 1;
      }


      BaseFloat score = plda.LogLikelihoodRatio_1vsMany(train_jvector_dbl,
                                                num_train_examples,
                                                test_jvector_dbl);
      plda.LogLikelihoodRatio_MultiVsMulti(train_jvector_dbl, num_train_examples,
                                          test_jvector_dbl, 1);

      sum += score;
      sumsq += score * score;
      num_trials_done++;
      ko.Stream() << key1 << ' ' << key2 << ' ' << score << std::endl;
    }

    for (HashType::iterator iter = train_jvectors.begin();
         iter != train_jvectors.end(); ++iter)
      delete iter->second;
    for (HashType::iterator iter = test_jvectors.begin();
         iter != test_jvectors.end(); ++iter)
      delete iter->second;


    if (num_trials_done != 0) {
      BaseFloat mean = sum / num_trials_done, scatter = sumsq / num_trials_done,
          variance = scatter - mean * mean, stddev = sqrt(variance);
      KALDI_LOG << "Mean score was " << mean << ", standard deviation was "
                << stddev;
    }
    KALDI_LOG << "Processed " << num_trials_done << " trials, " << num_trials_err
              << " had errors.";
    return (num_trials_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
