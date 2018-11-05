// jvectorbin/jvector-agglomerative-cluster.cc

// Copyright 2016-2018  David Snyder
//           2017-2018  Matthew Maciejewski
//           2018       Jerry Peng

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
#include "util/stl-utils.h"
#include "jvector/agglomerative-clustering.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
      "This is a revised version of standard kaldi gglomerative-clustering.\n"
      "Suppose many sets of utterances are given, each set is assumed to uttered by a speaker\n"
      "But it's unknown that whether any two sets are uttered by the same speaker\n"
      "This program aims to cluster these speakers into a set of clusters\n"
      "In adddition, this program can be cooperated with boostrap-utterances, jvector-plda-scoring, etc.\n"
      "  boostrap-utterances can boostrap a given utterance into many small sampled utterance.\n"
      "  After jvector-extract, there will be a set of ivectors for an original utterance.\n"
      "  jvector-plda-scoring is degisned to scoring two sets of ivectors.\n"
      "  And after that, this program can be used to cluster these utterance.\n"
      "  Also, you can input utt2subutt as utt2utt(Identity map) and do not perform boostrapping,\n"
      "  then it will be the traditional ivector agglomerative clustering\n"
      "\n"
      "Cluster speakers by similarity score, used in ivector plda unsuprevised adaptation.\n"
      "Takes a score matrice with the rows/columns corresponding to "
      "the utterances in sorted order and a utt2subutt file that contains "
      "the mapping from utterances to subutterances, and outputs a list of labels "
      "in the form <utterance> <label>.  Clustering is done using agglomerative hierarchical\n"
      "clustering with a score threshold as stop criterion.  By default, the\n"
      "program reads in similarity scores, but with --read-costs=true\n"
      "the scores are interpreted as costs (i.e. a smaller value indicates\n"
      "utterance similarity).\n"
      "Usage: agglomerative-cluster [options] <score-matrix-in> "
      "<utt2subutt-rspecifier> <labels-wspecifier>\n"
      "e.g.: \n"
      " agglomerative-cluster score.mat ark:utt2subutt \n"
      "   ark,t:labels.txt\n";

    ParseOptions po(usage);
    int32 num_clusters = 1;
    BaseFloat threshold = 0.0;
    bool read_costs = false;
    std::string linkage = "complete";

    po.Register("num-clusters", &num_clusters,
      "If supplied, clustering creates exactly this many clusters"
      " and the option --threshold is ignored. num-clusters should be larger than 1");
    po.Register("threshold", &threshold, "Merge clusters if their distance"
      " is less than this threshold.");
    po.Register("read-costs", &read_costs, "If true, the first"
      " argument is interpreted as a matrix of costs rather than a"
      " similarity matrix.");
    po.Register("linkage", &linkage, "linkage of Agglomerative Clustering"
      " can be [average|complete|single]. By default, compelete is applied.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    if (num_clusters < 1) {
      po.PrintUsage();
      exit(1);
    }


    Linkage lkg;
    if (linkage == "complete") {
      lkg = lkg_complete;
    } else if (linkage == "single") {
      lkg = lkg_single;
    } else if (linkage == "average") {
      lkg = lkg_average;
    } else {
      po.PrintUsage();
      exit(1);
    }

    std::string scores_rxfilename = po.GetArg(1),
      utt2subutt_rspecifier = po.GetArg(2),
      label_wspecifier = po.GetArg(3);

    // plda score actually is not cost, it is similarity score.
    // The higher score , the higher similarity
    Matrix<BaseFloat> costs;
    ReadKaldiObject(scores_rxfilename, &costs);
    // By default, the scores give the similarity between pairs of
    // utterances.  We need to multiply the scores by -1 to reinterpet
    // them as costs (unless --read-costs=true) as the agglomerative
    // clustering code requires.
    if (!read_costs) {
      costs.Scale(-1);
      threshold = -threshold;
    }

    SequentialTokenReader utt2subutt_reader(utt2subutt_rspecifier);
    Int32Writer label_writer(label_wspecifier);
    
    std::string utt = utt2subutt_reader.Key();
    std::vector<int32> cluster_ids;
    if (1 != num_clusters) {
      AgglomerativeCluster(costs,
        std::numeric_limits<BaseFloat>::max(), num_clusters, lkg, &cluster_ids);
    } else {
      AgglomerativeCluster(costs, threshold, 1, lkg, &cluster_ids);
    }

    for (int32 i = 0; i < cluster_ids.size(); i++, utt2subutt_reader.Next()) {
      KALDI_ASSERT(!utt2subutt_reader.Done()); // Make sure utt2subutt matches cluster_ids size 
      label_writer.Write(utt2subutt_reader.Key(), cluster_ids[i]);
    }
    return 0;

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
