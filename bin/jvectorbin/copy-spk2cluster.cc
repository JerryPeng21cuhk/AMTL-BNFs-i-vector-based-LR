// jvectorbin/copy-spk2cluster.cc

// Copyright 2018       Jerry Peng

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

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
  	const char* usage = 
  	  "Copy a utt2cluster to output.\n"
  	  "It can be used for binary conversion.\n"
  	  "Usage: copy-spk2cluster [options] <labels-rspecifier-in> "
  	  "<labels-wspecifier-out>\n"
  	  "e.g.: \n"
  	  " copy-spk2cluster ark:utt2cluster ark,t:utt2cluster.txt \n";

  	ParseOptions po(usage);
  	po.Read(argc, argv);

  	if (po.NumArgs() != 2) {
  	  po.PrintUsage();
  	  exit(1);
  	}

  	std::string label_rspecifier = po.GetArg(1),
  	  label_wspecifier = po.GetArg(2);

  	SequentialInt32Reader label_reader(label_rspecifier);
  	Int32Writer label_writer(label_wspecifier);

  	for (; !label_reader.Done(); label_reader.Next()) {
  	  label_writer.Write(label_reader.Key(), label_reader.Value());
  	}

  	return 0;
  } catch(const std::exception &e) {
  	std::cerr << e.what();
  	return -1;
  }
}
