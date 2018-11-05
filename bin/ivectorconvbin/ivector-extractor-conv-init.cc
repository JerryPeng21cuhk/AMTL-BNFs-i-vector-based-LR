// ivectorbin/ivector-extractor-init.cc

// Copyright 2015  Daniel Povey
//                 Srikanth Madikeri (Idiap Research Institute)
//           2018 JerryPeng

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
#include "gmm/full-gmm.h"
#include "ivectorconv/conv-ivector-extractor.h"
#include <thread>
#include <chrono>


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Initialize ivector-extractor\n"
        "Usage:  ivector-extractor-init [options] <fgmm-in> <ivector-extractor-out>\n"
        "e.g.:\n"
        " ivector-extractor-init 4.fgmm 0.ie\n";

    bool binary = true;
    IvectorExtractorConvOptions ivector_opts;
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    ivector_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }


    std::string fgmm_rxfilename = po.GetArg(1),
        ivector_extractor_wxfilename = po.GetArg(2);
        
    FullGmm fgmm;
    ReadKaldiObject(fgmm_rxfilename, &fgmm);

    IvectorExtractorConv extractor(ivector_opts, fgmm);

    WriteKaldiObject(extractor, ivector_extractor_wxfilename, binary);
    // std::this_thread::sleep_for(std::chrono::seconds(10));
    // KALDI_LOG << "AWAKE!";
    KALDI_LOG << "Initialized iVector extractor with iVector dimension "
              << extractor.IvectorDim() << " and wrote it to "
              << ivector_extractor_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


