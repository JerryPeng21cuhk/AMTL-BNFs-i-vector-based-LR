// jvectorbin/boostrap-utterances.cc

// Copyright 2018  Jerry Peng

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
#include "matrix/kaldi-matrix.h"
#include "feat/feature-functions.h"
#include <iomanip>


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "With 3 arguments, boostraps each utterance feature into a number of sub-utterances and output utt2subutt\n"
        "\n"
        "Usage: boostrap-utterances <utt-feature-rspecifier> <subutt-feature-wspecifier> <utt2subutt>\n"
        "e.g.: boostrap-utterances scp:data/train/feats.scp ark:- ark:utt2subutt \n";

    ParseOptions po(usage);
    int32 unit_length = 10;
    po.Register("unit_length", &unit_length, "The number of frames for each unit, 10 by default.");
    int32 num_subutt = 500;
    po.Register("num_subutt", &num_subutt, "The number of sub-utterance for each input utterance, 500 by default");
    int32 subutt_length = 10;
    po.Register("subutt_length", &subutt_length, "The number of units for each subutt, 10 by default.");
    int32 min_frames = 20;
    po.Register("min_frames", &min_frames, "Minumn number of voiced frames for each input utt, 20 by default.");
    bool force_boostrap = false;
    po.Register("force_boostrap", &force_boostrap, "Forcing boostrapping even if number of frames is less than min_frames.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    // num of args    
    std::string feat_rspecifier = po.GetArg(1),
      feat_wspecifier = po.GetArg(2),
      utt2subutt_wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    BaseFloatMatrixWriter feat_writer(feat_wspecifier);
    TokenVectorWriter utt2subutt_writer(utt2subutt_wspecifier);

    int32 num_utt_done = 0, num_utt_err = 0, num_subutt_done = 0;

    for (;!feat_reader.Done(); feat_reader.Next()) {
      std::string utt = feat_reader.Key();
      const Matrix<BaseFloat> &feat = feat_reader.Value();
      const int32 nframes = feat.NumRows();
      const int32 feat_dim = feat.NumCols();
      if (nframes == 0) {
        KALDI_WARN << "Empty feature matrix for utterance " << utt;
        num_utt_err++;
        continue;
      }
      if (nframes < min_frames) {
        KALDI_WARN << "number of voiced frame " << nframes << " is less than threshold: " << min_frames;
        if (false == force_boostrap) {
          num_utt_err++;
          continue;
        } else {
          KALDI_WARN << "Forcing to boostrap utterances";

          int32 num_units = nframes;
          std::vector<std::string> subutt_token_vec;
          for (size_t i = 0; i < num_subutt; i++) {
            //for each unit
            int32 frame_idx[subutt_length*unit_length];
            Matrix<BaseFloat> unit(subutt_length*unit_length, feat_dim);
            for (size_t j = 0; j < subutt_length*unit_length; j++) {
              frame_idx[j] = Rand() % num_units;
            }
            // now aunit (new sub-utt) is created
            unit.CopyRows(feat, frame_idx);
            std::string num_str = std::to_string(i);
            num_str.insert(num_str.begin(), 4-num_str.length(), '0');
            std::string subutt_token = utt + "-sub-" + num_str;
            subutt_token_vec.push_back(subutt_token);
            feat_writer.Write(subutt_token, unit);
            num_subutt_done++;
          }
          utt2subutt_writer.Write(utt, subutt_token_vec);
          num_utt_done++;
          continue;
        }
      }
      int32 num_units = nframes / unit_length;
      // Create sub-utts
      // for each subutt
      std::vector<std::string> subutt_token_vec;
      for (size_t i = 0; i < num_subutt; i++) {
        // for each unit
        int32 frame_idx[subutt_length*unit_length]; 
        Matrix<BaseFloat> unit(subutt_length*unit_length, feat_dim);
        for (size_t j = 0; j < subutt_length; j++) {
          int32 unit_idx = Rand() % num_units,
              src_frame_idx_s = unit_idx * unit_length,
              des_frame_idx_s = j * unit_length;
          // create frame_idx for copying
          for(size_t t = 0; t < unit_length; ++t) {
            frame_idx[des_frame_idx_s + t] = src_frame_idx_s + t;
          }
        }
        // now a unit(new sub-utt) is created
        unit.CopyRows(feat, frame_idx);
        std::string num_str = std::to_string(i);
        num_str.insert(num_str.begin(), 4-num_str.length(), '0');
        std::string subutt_token = utt + "-sub-" + num_str;
        subutt_token_vec.push_back(subutt_token);
        feat_writer.Write(subutt_token, unit);
        num_subutt_done++;
      }
      utt2subutt_writer.Write(utt, subutt_token_vec);
      num_utt_done++;
    }

    KALDI_LOG << "Done boostrapping voiced utterances; processed "
              << num_utt_done << " utterances, "
              << num_utt_err << " had errors. "
              << "Created " << num_subutt_done << " sub-utterances.";
    return (num_utt_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



