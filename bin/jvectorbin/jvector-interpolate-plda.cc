// jvectorbin/jvector-interpolate-plda.cc

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
  try {
    const char *usage =
        "Interpolate two PLDA objects: plda1 and plda2 with an interpolating scalar --smoothing\n"
        "If --smoothing=1.0 the interpolated plda will be same as plda1\n"
        "\n"
        "Usage: jvector-interpolate-plda <plda1-in> <plda2-in> <interpolated-plda-out>\n"
        "e.g.: jvector-interpolate--plda --smoothing=0.5 plda1 plda2 plda.smooth0.1\n";

    ParseOptions po(usage);

    BaseFloat smoothing = 0.5;
    bool binary = true;
    // std::string mode = "standard";
    // po.Register("mode", &mode, "Mode to interpolate two pldas, standard or heuristic");
    po.Register("smoothing", &smoothing, "Factor used in smoothing within-class "
                "covariance (add this factor times between-class covar)");
    po.Register("binary", &binary, "Write output in binary mode");

    PldaConfig plda_config;
    plda_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string plda1_rxfilename = po.GetArg(1),
        plda2_rxfilename = po.GetArg(2),
        plda_interpolated_wxfilename = po.GetArg(3);

    Plda plda1, plda2, *p_plda_interpolated;
    ReadKaldiObject(plda1_rxfilename, &plda1);
    ReadKaldiObject(plda2_rxfilename, &plda2);

    if (0.0 == smoothing) {
        WriteKaldiObject(plda2, plda_interpolated_wxfilename, binary);
    } 
    else if (1.0 == smoothing) {
          WriteKaldiObject(plda1, plda_interpolated_wxfilename, binary);
      }
    else {
        // if ("standard" == mode) {
          p_plda_interpolated = new Plda(plda1, plda2, smoothing);
        // } else {
        //   p_plda_interpolated = new Plda();
        //   p_plda_interpolated->Plda_init2(plda1, plda2, smoothing);
        // }
        WriteKaldiObject(*p_plda_interpolated, plda_interpolated_wxfilename, binary);
    }
    delete p_plda_interpolated;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
