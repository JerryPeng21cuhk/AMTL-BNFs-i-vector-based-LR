// ivector/ivector-extractor-test.cc
// 2018 JerryPeng

// 1. Test IO of extractor
// 2. Test ComputeDerivedVars()
// 3. Test IePostes()

#include "gmm/model-test-common.h"
#include "gmm/full-gmm-normal.h"
#include "ivectorconv/conv-ivector-extractor.h"
#include "util/kaldi-io.h"

namespace kaldi {

void TestIvectorExtractorConvIO(const IvectorExtractorConv &extractor) {
    std::ostringstream ostr;
    bool binary = (Rand() % 2 == 0);
    extractor.Write(ostr, binary);
    std::istringstream istr(ostr.str());
    IvectorExtractorConv extractor2;
    extractor2.Read(istr, binary);
    std::ostringstream ostr2;
    extractor2.Write(ostr2, binary);
    KALDI_ASSERT(ostr.str() == ostr2.str());
}

void TestIvectorExtractorConvStatsIO(IvectorConvStats &stats) {
  std::ostringstream ostr;
  bool binary = (Rand() % 2 == 0);
  stats.Write(ostr, binary);
  std::istringstream istr(ostr.str());
  IvectorConvStats stats2;
  stats2.Read(istr, binary);
  std::ostringstream ostr2;
  stats2.Write(ostr2, binary);
  
  if (binary) {
    // this was failing in text mode, due to differences like
    // 8.2244e+06 vs  8.22440e+06
    KALDI_ASSERT(ostr.str() == ostr2.str());
  }
}

void TestIvectorConvExtraction(const IvectorExtractorConv &extractor,
                           const MatrixBase<BaseFloat> &feats,
                           const FullGmm &fgmm,
                           Posterior *post2) {
  // if (extractor.IvectorConvDependentWeights())
  //   return;  // Nothing to do as online iVector estimator does not work in this
  //            // case.
  int32 num_frames = feats.NumRows(),
      feat_dim = feats.NumCols(),
      num_gauss = extractor.NumGauss(),
      ivector_dim = extractor.IvectorDim();
  Posterior post(num_frames);

  double tot_log_like = 0.0;
  for (int32 t = 0; t < num_frames; t++) {
    SubVector<BaseFloat> frame(feats, t);
    Vector<BaseFloat> posterior(fgmm.NumGauss(), kUndefined);
    tot_log_like += fgmm.ComponentPosteriors(frame, &posterior);
    for (int32 i = 0; i < posterior.Dim(); i++)
      post[t].push_back(std::make_pair(i, posterior(i)));
  }
    
  // The zeroth and 1st-order stats are in "utt_stats".
  IvectorExtractorConvUtteranceStats utt_stats(num_gauss, feat_dim);
  extractor.GetStats(feats, post, &utt_stats);

  Vector<double> ivector1(ivector_dim);
  SpMatrix<double> jcovar1(ivector_dim);

  extractor.GetIvectorDistribution(utt_stats, &ivector1, &jcovar1);

  IvectorExtractorConvDecodeOptions opts;
  //Conv post2->resize(num_frames);

  float objf = extractor.DecodePosteriors(opts, ivector1, jcovar1, feats, post2);



  KALDI_LOG << "ivector1 = " << ivector1;
  std::ostringstream os_post;
  WritePosterior(os_post, false, *post2);
  KALDI_LOG << "post2 = " << os_post.str();

  // objf change vs. default iVector.  note, here I'm using objf
  // and auxf pretty much interchangeably :-(
  KALDI_LOG << "objf = " << objf/num_frames;
}


void UnittestIvectorExtractorConv() {
  FullGmm fgmm;
  FullGmm fgmm2;
  int32 dim = 5 + Rand() % 5, num_comp = 20 + Rand() % 5;
  KALDI_LOG << "Num Gauss = " << num_comp;
  unittest::InitRandFullGmm(dim, num_comp, &fgmm);
  unittest::InitRandFullGmm(dim, num_comp, &fgmm2);
  FullGmmNormal fgmm_normal(fgmm);

  IvectorExtractorConvOptions ivector_opts;
  ivector_opts.ivector_dim = dim + 5;
  // ivector_opts.use_weights = (Rand() % 2 == 0);
  KALDI_LOG << "Feature dim is " << dim
            << ", ivector dim is " << ivector_opts.ivector_dim;
  // IvectorExtractorConv extractor(ivector_opts, fgmm2);
  IvectorExtractorConv extractor(ivector_opts, fgmm);

  TestIvectorExtractorConvIO(extractor);

  //Generate random feats follows gaussian normal distribution
  // num_utts, num_feats, feat_dim
  int32 num_utts = 10 + Rand() % 5;
  std::vector<Matrix<BaseFloat> > all_feats(num_utts);
  std::vector<Posterior> all_posts(num_utts);

  for (int32 utt = 0; utt < num_utts; utt++) {
    int32 num_frames = 100 + Rand() % 200;
    if (Rand() % 2 == 0) num_frames *= 10;
    if (Rand() % 2 == 0) num_frames /= 1.0;
    Matrix<BaseFloat> feats(num_frames, dim);
    fgmm_normal.Rand(&feats);
    feats.Swap(&all_feats[utt]);
   // KALDI_LOG << "Ok.1: " << utt << '\n';
    //use fgmm to decode the posteriors
    Posterior& posts = all_posts[utt];
    posts.resize(num_frames);
    for (int32 t = 0; t < num_frames; t++) {
        SubVector<BaseFloat> frame(all_feats[utt], t);
        Vector<BaseFloat> posterior(num_comp, kUndefined);
        fgmm2.ComponentPosteriors(frame, &posterior);
        // KALDI_LOG << "Ok.2: utt: " << utt << " frame: " << t <<'\n';
        for (int32 i = 0; i < num_comp; i++) {
            all_posts[utt][t].push_back(std::make_pair(i, posterior(i)));
            // KALDI_LOG << "Ok.2: utt: " << utt << " frame: " << t << " i: " << i << '\n';
          }
    }

  }
  
  // KALDI_LOG << "Ok.2\n";

  // Iterate without updating posteriors
  int32 num_iters = 2;
  int32 num_decode_iters = 1;
  int32 num_out_iters = 5;
  IvectorExtractorConvEstimationOptions est_opts(1.0);
  IvectorExtractorConvDecodeOptions decode_opts;
  for (int32 out_iter = 0; out_iter < num_out_iters; out_iter++)
  {
    for (int32 decode_iter = 0; decode_iter < num_decode_iters; decode_iter++)
    {
      //decode posteriors
      float avg_objf = 0;
      std::vector<Posterior> all_posts_new(num_utts);
      for (int32 utt = 0; utt < num_utts; utt++) {
          Vector<double> ivector(ivector_opts.ivector_dim);
          SpMatrix<double> icovar(ivector_opts.ivector_dim);

          IvectorExtractorConvUtteranceStats utt_stats(num_comp, dim);
          extractor.GetStats(all_feats[utt], all_posts[utt], &utt_stats);
          extractor.GetIvectorDistribution(utt_stats, &ivector, &icovar);
          float objf = extractor.DecodePosteriors(decode_opts, ivector, icovar,
                  all_feats[utt], &(all_posts_new[utt]) );
          objf = objf / all_feats[utt].NumRows();
          avg_objf += objf;
          // KALDI_LOG << "utt: " << utt  << "objf_avg = " << objf;
      }
      // all_posts = all_posts_new;

      avg_objf = avg_objf / num_utts;
      KALDI_LOG << "Iter: " << out_iter << "objf_avg = " << avg_objf;
    }

    for (int32 iter = 0; iter < num_iters; iter++) {
      IvectorConvStats all_stats(extractor);
      for (int32 utt = 0; utt < num_utts; utt++) {
          Vector<double> ivector(ivector_opts.ivector_dim);
          SpMatrix<double> icovar(ivector_opts.ivector_dim);

          // IvectorExtractorConvUtteranceStats utt_stats(num_comp, dim);
          // extractor.GetStats(all_feats[utt], all_posts[utt], &utt_stats);
          // all_stats.CommitStatsForUtterance(extractor, utt_stats);
          all_stats.AccStatsForUtterance(extractor, all_feats[utt], all_posts[utt]);
      }
      all_stats.Update(est_opts, &extractor);
    }

 }
}

}
int main() {
    using namespace kaldi;
    SetVerboseLevel(5);
    UnittestIvectorExtractorConv();
    std::cout << "Test OK.\n";
    return 0;
}