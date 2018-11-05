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


namespace kaldi {
class StatsOfSpeaker {
  public:
    StatsOfSpeaker(int32 dim): spk_renorm_scale_(0.0), 
        validSumJvectorSq_(true), dim_(dim), num_jvectors_(0),
        sum_jvector_(dim), sum_jvector_sq_(dim) { }
    // transform jvector
    // accumulate num_jvector, spk_renorm_scale
    // accumulate avg_jvector, sum_jvector_sq_(if IsSoreAverage is true)
    void AddJvector(const PldaConfig &config,
               const Plda &plda,
               const VectorBase<float> &jvector,
               const bool needJvectorSq=false) {
      // no need to check jvector dim
      Vector<BaseFloat> transformed_jvector(dim_);
      spk_renorm_scale_ += plda.TransformIvector(config, jvector, 1, &transformed_jvector);
      num_jvectors_++;
      sum_jvector_.AddVec(1.0, transformed_jvector);
      
      if (true == needJvectorSq) {
        // if needJvectorSq is true, validSumJvectorSq_ must be true
        KALDI_ASSERT(true == validSumJvectorSq_);
        sum_jvector_sq_.AddVec2(1.0, transformed_jvector);
      } else {
        validSumJvectorSq_ = false;
        // Now, accumulate sum_jvector_sq is meaningless.
      }
    }

    
    void GetAvgTransformedJvector(VectorBase<double> *const avg_jvector,
                                  VectorBase<double> *const avg_jvector_sq=NULL) const {

      KALDI_ASSERT(avg_jvector != NULL);
      avg_jvector->CopyFromVec(sum_jvector_);
      avg_jvector->Scale(1.0 / num_jvectors_);
      if (avg_jvector_sq != NULL) {
        KALDI_ASSERT(true == validSumJvectorSq_);
        avg_jvector_sq->CopyFromVec(sum_jvector_sq_);
        avg_jvector_sq->Scale(1.0 / num_jvectors_);
      }
    }

    int32 Dim() const { return dim_; }
    int32 GetNumJvectors() const { return num_jvectors_; }
    // Just for diagnostics
    double spk_renorm_scale_;
  private:
    bool validSumJvectorSq_;
    int32 dim_;
    int32 num_jvectors_;
    Vector<double> sum_jvector_; // the sum of a spk's jvectors
    Vector<double> sum_jvector_sq_; // the sum of a spk's (jvectors.^2)
};// end of class StatsOfSpeaker

double GetllkRatio_ScoreAverage(const Plda &plda,
                         const StatsOfSpeaker &spk_train,
                         const StatsOfSpeaker &spk_test) {
  int32 dim = spk_train.Dim();
  KALDI_ASSERT(spk_test.Dim() == dim);
  Vector<double> transformed_train_ivector(dim), transformed_train_ivector_sq(dim);
  Vector<double> transformed_test_ivector(dim), transformed_test_ivector_sq(dim);
  spk_train.GetAvgTransformedJvector(&transformed_train_ivector, &transformed_train_ivector_sq);
  spk_test.GetAvgTransformedJvector(&transformed_test_ivector, &transformed_test_ivector_sq);
  return plda.LogLikelihoodRatio_ScoreAverage(transformed_train_ivector,
                                       transformed_train_ivector_sq,
                                       transformed_test_ivector,
                                       transformed_test_ivector_sq);
}

double GetllkRatio_IvectorAverage(const Plda &plda,
                          const StatsOfSpeaker &spk_train,
                          const StatsOfSpeaker &spk_test) {
  int32 dim = spk_train.Dim();
  KALDI_ASSERT(spk_test.Dim() == dim);
  Vector<double> transformed_train_ivector(dim), transformed_test_ivector(dim);
  spk_train.GetAvgTransformedJvector(&transformed_train_ivector, NULL);
  spk_test.GetAvgTransformedJvector(&transformed_test_ivector, NULL);
  return plda.LogLikelihoodRatio_MultiVsMulti(transformed_train_ivector, 1,
                                              transformed_test_ivector, 1);
}

double GetllkRatio_Standard(const Plda &plda,
                            const StatsOfSpeaker &spk_train,
                            const StatsOfSpeaker &spk_test) {
  int32 dim = spk_train.Dim();
  KALDI_ASSERT(spk_test.Dim() == dim);
  Vector<double> transformed_train_ivector(dim), transformed_test_ivector(dim);
  spk_train.GetAvgTransformedJvector(&transformed_train_ivector, NULL);
  spk_test.GetAvgTransformedJvector(&transformed_test_ivector, NULL);
  int32 spk_train_njvector = spk_train.GetNumJvectors(),
          spk_test_njvector = spk_test.GetNumJvectors();
  return plda.LogLikelihoodRatio_MultiVsMulti(transformed_train_ivector, spk_train_njvector,
                                              transformed_test_ivector, spk_test_njvector);
}

}// end of namespace

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
        "(if either key could not be found, the dot-product field in the output "
        "will be absent, and this program will print a warning)\n"
        "There are three modes to perform scoring: "
        "scoreaverage, ivectoraverage, standard(default)\n"
        "They are specified by --score-mode\n"
        "Usage: jvector-plda-scoring [--score-mode=standard] <plda> <train-spk2utt-rspecifier> "
        "<test-spk2utt-rspecifier> <train-jvector-rspecifier> <test-jvector-rspecifier> "
        "<trials-rxfilename> <scores-wxfilename>\n"
        "\n"
        "e.g.: jvector-plda-scoring --score-mode=standard plda "
        "ark:exp/train/jvectors.ark ark:exp/test/jvectors.ark "
        "ark:data/train/spk2utt ark:data/test/spk2utt trials scores "
        "See also: jvector-compute-dot-products, jvector-compute-plda\n";

    ParseOptions po(usage);

    //default mode
    std::string score_mode="standard";

    PldaConfig plda_config;
    plda_config.Register(&po);
    po.Register("score-mode", &score_mode, "three options: (scoreaverage "
          "ivectoraverage, standard). e.g. --score-mode=standard. ");
    po.Read(argc, argv);

    if (po.NumArgs() != 7) {
      po.PrintUsage();
      exit(1);
    }

    
    // check score_mode
      // std::string mode_options[] = {"scoreaverage",
      //                               "ivectoraverage",
      //                               "standard"};
      // int i = 0;
      // while (!mode_options[i].empty()) {
      //   if (score_mode == scoreaverage)
      //     break;
      //   i++;
      // }
      // if (mode_options[i].empty()) {
      //   //invalid score_mode
      //   po.PrintUsage();
      //   exit(1);
      // }


    //new method
    typedef double (*pScoreFunc)(const Plda&, const StatsOfSpeaker&, const StatsOfSpeaker&);
    typedef std::unordered_map<std::string, pScoreFunc > ScoreFuncMap;

    pScoreFunc pscore_func = NULL;
    const ScoreFuncMap scorefunc_map = {
        {"scoreaverage", &GetllkRatio_ScoreAverage},
        {"ivectoraverage", &GetllkRatio_IvectorAverage},
        {"standard", &GetllkRatio_Standard}
    };
    // if (scorefunc_map.count(score_mode) == 0) {
    //   po.PrintUsage();
    //   exit(1);
    // } else {
    //   pscore_func = scorefunc_map.find(score_mode);
    // }

    auto item_found = scorefunc_map.find(score_mode);
    if (item_found == scorefunc_map.end()) {
      po.PrintUsage();
      exit(1);
    }
    pscore_func = item_found->second;
    // end of argument validation

    // related to StatsOfSpeaker
    bool needJvectorSq = false;
    if("scoreaverage" == score_mode) {
      needJvectorSq = true;
    } 
    


    std::string plda_rxfilename = po.GetArg(1),
        train_spk2utt_rspecifier = po.GetArg(2),
        test_spk2utt_rspecifier = po.GetArg(3),
        train_jvector_rspecifier = po.GetArg(4),
        test_jvector_rspecifier = po.GetArg(5),
        trials_rxfilename = po.GetArg(6),
        scores_wxfilename = po.GetArg(7);

    //  diagnostics:
    double tot_test_renorm_scale = 0.0, tot_train_renorm_scale = 0.0;
    int64 num_train_jvectors = 0, num_train_utt_errs = 0, 
            num_test_jvectors = 0, num_test_utt_errs = 0;

    int64 num_trials_done = 0, num_trials_err = 0;

    Plda plda;
    ReadKaldiObject(plda_rxfilename, &plda);

    int32 dim = plda.Dim();

    SequentialTokenVectorReader train_spk2utt_reader(train_spk2utt_rspecifier);
    SequentialTokenVectorReader test_spk2utt_reader(test_spk2utt_rspecifier);
    RandomAccessBaseFloatVectorReader train_jvector_reader(train_jvector_rspecifier);
    RandomAccessBaseFloatVectorReader test_jvector_reader(test_jvector_rspecifier);
    
    typedef unordered_map<string, StatsOfSpeaker*, StringHasher> HashType;

    // These hashes will contain the stats for each speaker 
    HashType train_spk_stats, test_spk_stats;

    KALDI_LOG << "Reading train jVectors";
    for (; !train_spk2utt_reader.Done(); train_spk2utt_reader.Next()) {
      std::string spk = train_spk2utt_reader.Key();
      if (train_spk_stats.count(spk) != 0) {
        KALDI_ERR << "Duplicate stats found for speaker " << spk;
      }
      const std::vector<std::string> &uttlist = train_spk2utt_reader.Value();
      if (uttlist.empty()) {
        KALDI_ERR << "Speaker " << spk << " with no utterances.";
      }
      StatsOfSpeaker *newSpeaker = new StatsOfSpeaker(dim);
      int32 num_newspk_jvectors = 0;
      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];
        if (!train_jvector_reader.HasKey(utt)) {
          KALDI_WARN << "No jVector present in input for utterance " << utt;
          num_train_utt_errs++;
        } else {
          newSpeaker->AddJvector(plda_config, plda, 
                        train_jvector_reader.Value(utt), 
                        needJvectorSq);
          num_train_jvectors++;
          num_newspk_jvectors++;
        }
      }
      if (0 == num_newspk_jvectors) {
        KALDI_ERR << "None of utterance present in " << train_jvector_rspecifier
                   << " for speaker " << spk;
      }
      tot_train_renorm_scale += newSpeaker->spk_renorm_scale_;
      train_spk_stats[spk] = newSpeaker;
    }
    KALDI_LOG << "Read " << num_train_jvectors << "train jVectors, "
        << "with " << train_spk_stats.size() << "train speakers.";
    KALDI_LOG << "Average renormalization scale on train jVectors was "
        << (tot_train_renorm_scale / num_train_jvectors);


    KALDI_LOG << "Reading test jVectors";
    for (; !test_spk2utt_reader.Done(); test_spk2utt_reader.Next()) {
      std::string spk = test_spk2utt_reader.Key();
      if (test_spk_stats.count(spk) != 0) {
        KALDI_ERR << "Duplicate stats found for speaker " << spk;
      }
      const std::vector<std::string> &uttlist = test_spk2utt_reader.Value();
      if (uttlist.empty()) {
        KALDI_ERR << "Speaker " << spk << "with no utterances.";
      }
      StatsOfSpeaker *newSpeaker = new StatsOfSpeaker(dim);
      int32 num_newspk_jvectors = 0;
      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];
        if (!test_jvector_reader.HasKey(utt)) {
          KALDI_WARN << "No jVector present in input for utterance " << utt;
          num_test_utt_errs++;
        } else {
          newSpeaker->AddJvector(plda_config, plda,
                        test_jvector_reader.Value(utt),
                        needJvectorSq);
          num_test_jvectors++;
          num_newspk_jvectors++;
        }
      }
      if (0 == num_newspk_jvectors) {
        KALDI_ERR << "None of utterance present in " << test_jvector_rspecifier
                   << " for speaker " << spk;
      }
      tot_test_renorm_scale += newSpeaker->spk_renorm_scale_;
      test_spk_stats[spk] = newSpeaker;
    }

    KALDI_LOG << "Read "<< num_test_jvectors << "train jVectors, "
        << "with " << test_spk_stats.size() << "test speakers.";
    KALDI_LOG << "Average renormalization scale on test jVectors was "
        << (tot_test_renorm_scale / num_test_jvectors);

    //scoring
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
      if (train_spk_stats.count(key1) == 0) {
        KALDI_WARN << "Key " << key1 << " not present in training jVectors.";
        num_trials_err++;
        continue;
      }
      if (test_spk_stats.count(key2) == 0) {
        KALDI_WARN << "Key " << key2 << " not present in test jVectors.";
        num_trials_err++;
        continue;
      }
      //scoring
      BaseFloat score = (*pscore_func)(plda,
                            (*train_spk_stats[key1]),
                            (*test_spk_stats[key2]));
      sum += score;
      sumsq += score * score;
      num_trials_done++;
      ko.Stream() << key1 << ' ' << key2 << ' ' << score << std::endl;
    }

    for (HashType::iterator iter = train_spk_stats.begin();
         iter != train_spk_stats.end(); ++iter)
      delete iter->second;
    for (HashType::iterator iter = test_spk_stats.begin();
        iter != test_spk_stats.end(); ++iter)
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
