// jvectorbin/jvector-extractor-copy.cc

// Copyright 2018 JeryPeng
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
#include "jvector/jvector-extractor.h"

int main(int argc, char const *argv[])
{
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage = 
      "Copy j-vector model [and possibly change format]\n"
      "Usage: jvector-extractor-copy [options] <model-in> <model-out>\n"
      "e.g.:\n"
      "jvector-extractor-copy --binary=false jvector.je jvector_txt.je\n";
  
    bool binary = true;
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string jvector_extractor_rxfilename = po.GetArg(1),
    jvector_extractor_wxfilename = po.GetArg(2);

    JvectorExtractor extractor;
    KALDI_LOG << "Read j-vector extractor from " << jvector_extractor_rxfilename;
    ReadKaldiObject(jvector_extractor_rxfilename, &extractor);

    KALDI_LOG << "Write j-vector extractor to " << jvector_extractor_wxfilename;
    WriteKaldiObject(extractor, jvector_extractor_wxfilename, binary);

    return 0;
    } catch(const std::exception &e) {
      std::cerr << e.what();
      return -1;
    }
}
