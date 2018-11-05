// jvector/jvector-extractor.cc

// Revised 2018 JerryPeng

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

#include <vector>

#include "jvector/jvector-extractor.h"
#include "hmm/posterior.h"
#include "base/kaldi-math.h"

namespace kaldi {

JvectorExtractor::JvectorExtractor(
    const JvectorExtractorOptions &opts,
    const FullGmm &fgmm) {
  KALDI_ASSERT(opts.jvector_dim > 0);
  gincovars_.resize(fgmm.NumGauss());
  for (int32 i = 0; i < fgmm.NumGauss(); i++) {
    const SpMatrix<BaseFloat> &inv_var = fgmm.inv_covars()[i];
    gincovars_[i].Resize(inv_var.NumRows());
    gincovars_[i].CopyFromSp(inv_var);
  }  

  Matrix<double> gmm_means;
  fgmm.GetMeans(&gmm_means);
  gmeans_.Resize(gmm_means.NumRows(), gmm_means.NumCols());
  gmeans_.CopyFromMat(gmm_means);

  KALDI_ASSERT(!gincovars_.empty());
  int32 feature_dim = gincovars_[0].NumRows(),
      num_gauss = gincovars_.size();

  T_.resize(num_gauss);
  for (int32 i = 0; i < num_gauss; i++) {
    T_[i].Resize(feature_dim, opts.jvector_dim);
    T_[i].SetRandn();
  }
  gweights_.Resize(num_gauss);
  gweights_.CopyFromVec(fgmm.weights());
  ComputeDerivedVars();
}


// JvectorExtractor::JvectorExtractor(
//     const FullGmm &fgmm,
//     const JvectorExtractor &Jvector_extractor
//     ) {
//   //Get ivector_dim
//   int32 num_gauss = ivector_extractor.NumGauss(),
//         ivector_dim = ivector_extractor.IvectorDim();
  
// }


void JvectorExtractor::GetJvectorDistribution(
    const JvectorExtractorUtteranceStats &utt_stats,
    VectorBase<double> *mean,
    SpMatrix<double> *var) const {

    Vector<double> linear(JvectorDim());
    SpMatrix<double> quadratic(JvectorDim());
    GetJvectorDistMean(utt_stats, &linear, &quadratic);

    // mean of distribution = quadratic^{-1} * linear...
    // var is the covar of i-vector
    quadratic.Invert();
    mean->AddSpVec(1.0, quadratic, linear, 0.0);
    if (var != NULL) {
        var->CopyFromSp(quadratic);
    }
}


void JvectorExtractor::GetJvectorDistMean(
    const JvectorExtractorUtteranceStats &utt_stats,
    VectorBase<double> *linear,
    SpMatrix<double> *quadratic) const {
    Vector<double> temp(FeatDim());
    int32 I = NumGauss();

    for (int32 i = 0; i < I; i++) {
      double gamma = utt_stats.gamma(i);
      if (gamma <= 0.0)
          continue;
      Vector<double> x(utt_stats.X_centre.Row(i)); // ==(sum post*features) ///not - $\gamma(i) \m_i
      // x.AddVec(-gamma, gmeans_.Row(i));
      temp.AddSpVec(1.0, gincovars_[i], x, 0.0);
      linear->AddMatVec(1.0, T_[i], kTrans, temp, 1.0); 
    }
    SubVector<double> q_vec(quadratic->Data(), JvectorDim()*(JvectorDim()+1)/2);
    q_vec.AddMatVec(1.0, U_mat_, kTrans, Vector<double>(utt_stats.gamma), 1.0);

    // Merging GetJvectorDistPrior. 
    // TODO: Try a more efficient way
    for (int32 d = 0; d < JvectorDim(); d++)
    (*quadratic)(d, d) += 1.0;
}


void JvectorExtractor::GetStats(
    const MatrixBase<BaseFloat> &feats,
    const Posterior &post,
    JvectorExtractorUtteranceStats *stats) const {
  typedef std::vector<std::pair<int32, BaseFloat> > VecType;

  int32 num_frames = feats.NumRows(), num_gauss = NumGauss(),
      feat_dim = FeatDim();
  KALDI_ASSERT(feats.NumCols() == feat_dim);
  KALDI_ASSERT(stats->gamma.Dim() == num_gauss &&
               stats->X.NumCols() == feat_dim);
  // bool update_variance = (!stats->S.empty());
  
  for (int32 t = 0; t < num_frames; t++) {
    SubVector<BaseFloat> frame(feats, t);
    const VecType &this_post(post[t]);
    SpMatrix<double> outer_prod;
    outer_prod.Resize(feat_dim);
    outer_prod.AddVec2(1.0, frame);
    for (VecType::const_iterator iter = this_post.begin();
         iter != this_post.end(); ++iter) {
      int32 i = iter->first; // Gaussian index.
      KALDI_ASSERT(i >= 0 && i < num_gauss &&
                   "Out-of-range Gaussian (mismatched posteriors?)");
      double weight = iter->second;
      stats->gamma(i) += weight;
      stats->X.Row(i).AddVec(weight, frame);
      // Update second-order stats
      stats->S[i].AddSp(weight, outer_prod);
    }
  }
  
  stats->X_centre.CopyFromMat(stats->X);
  for (int32 i = 0; i < num_gauss; i++) {
    stats->X_centre.Row(i).AddVec(-stats->gamma(i), gmeans_.Row(i));
  }
}

//TODO: Check this code
BaseFloat JvectorExtractor::GetJvectorPosteriors(
    const JvectorExtractorDecodeOptions &opts,
    const VectorBase<double> &mean,
    const SpMatrix<double> &var,
    const MatrixBase<BaseFloat> &feats,
    Posterior *post
    ) const{
    //This is to speed up training 
    // as by far we fix the posteriors in training
    // So there is no need to really decode
    if ( NULL == post )
      return 0.0;

    // typedef std::vector<std::pair<int32, BaseFloat> > VecType;
    int32 num_frames = feats.NumRows(), num_gauss = NumGauss(),
        feat_dim = FeatDim(), jvector_dim = JvectorDim();
    KALDI_ASSERT(feats.NumCols() == feat_dim);
    post->resize(num_frames);
    Vector<double> utt_consts;
    utt_consts.Resize(num_gauss, kUndefined);
    utt_consts.CopyFromVec(gconsts_);
    // loglikes -= Mean_'*Sigma_inv *T_ * mean;
    utt_consts.AddMatVec(-1.0, V_mat_, kNoTrans, mean, 1.0);
    // loglikes -= 0.5 * tr[ (var - mean*mean') * U_mat_ ]
    SpMatrix<double> var_temp(var);
    var_temp.AddVec2(1.0, mean);
    var_temp.ScaleDiag(0.5);
    SubVector<double> var_vec(var_temp.Data(),
                            jvector_dim * (jvector_dim + 1) /2);
    utt_consts.AddMatVec(-1.0, U_mat_, kNoTrans, var_vec, 1.0);
    
    // Compute utt_linear
    Matrix<double> utt_linears(num_gauss, feat_dim);
    for (int32 i = 0; i < num_gauss; ++i) {
        Vector<double> temp_vec(gmeans_.Row(i));
        temp_vec.AddMatVec(1.0, T_[i], kNoTrans, mean, 1.0);
        (utt_linears.Row(i)).AddSpVec(1.0, gincovars_[i], temp_vec, 0.0);
    }
    
    //Compute utt_quadratic
    // utt_quadratic is gincovars_mat_ which is a var member in Class JvectorExtractor

    // Compute likelihood for each frame
    BaseFloat log_sum = 0.0;
    for (int32 t = 0; t< num_frames; ++t) {
        Vector<float> log_likes(utt_consts);
        SpMatrix<float> data_sq(feat_dim);
        data_sq.AddVec2(1.0, feats.Row(t));
        data_sq.ScaleDiag(0.5);
        SubVector<float> data_sq_vec(data_sq.Data(), feat_dim * (feat_dim + 1) /2);
        log_likes.AddMatVec(-1.0, gincovars_mat_, kNoTrans, data_sq_vec, 1.0);
        

        // // Normalization
        // log_sum += loglikes.ApplySoftMax();
        // if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum))
        //     KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)"
        log_sum += VectorToPosteriorEntry(log_likes, opts.num_gselect, opts.min_post, &((*post)[t]));    
        // std::vector<std::pair<int32, BaseFloat> > post_entry;
        // VectorToPosteriorEntry(log_likes, opts.num_gselect, opts.min_post, &post_entry);    
    }
    return log_sum / num_frames;
}

    
void JvectorExtractor::DecodeJvectorAndPosterior(
    const Matrix<BaseFloat> &feats,
    const Posterior &ipost,
    JvectorExtractorDecodeOptions &decode_opts,
    Vector<double> *jvector,
    Posterior *opost) const
  {
    KALDI_ASSERT(feats.NumRows() == ipost.size());
    KALDI_ASSERT(feats.NumCols() == FeatDim());

    JvectorExtractorUtteranceStats utt_stats(NumGauss(), FeatDim());
    //GetJvectorDistribution() do not resize jvector and jcovars;
    jvector->Resize(JvectorDim());
    SpMatrix<double> jcovars(JvectorDim());
    // opost will be resized in GetJvectorPosteriors()
    // opost->resize(feats.NumRows());


    GetStats(feats, ipost, &utt_stats);

    GetJvectorDistribution(utt_stats, jvector, &jcovars);

    GetJvectorPosteriors(decode_opts, *jvector, jcovars, feats, opost);
  }


void JvectorExtractor::DecodeJvector(
    const Matrix<BaseFloat> &feats,
    const Posterior &ipost,
    Vector<double> *jvector
    ) const
  {
    KALDI_ASSERT(feats.NumRows() == ipost.size());
    KALDI_ASSERT(feats.NumCols() == FeatDim());

    JvectorExtractorUtteranceStats utt_stats(NumGauss(), FeatDim());
    //GetJvectorDistribution() do not resize jvector;
    jvector->Resize(JvectorDim());

    GetStats(feats, ipost, &utt_stats);

    GetJvectorDistribution(utt_stats, jvector, NULL);
  }





int32 JvectorExtractor::FeatDim() const {
  KALDI_ASSERT(!T_.empty());
  return T_[0].NumRows();
}
int32 JvectorExtractor::JvectorDim() const {
  KALDI_ASSERT(!T_.empty());
  return T_[0].NumCols();
}
int32 JvectorExtractor::NumGauss() const {
  return static_cast<int32>(T_.size());
}

// The format is different from kaldi's ivector implementation. ivector_offset_
// doesn't exist and hence not written
//
void JvectorExtractor::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<JvectorExtractor>");
  WriteToken(os, binary, "<w_vec>");
  gweights_.Write(os, binary);
  WriteToken(os, binary, "<M>");  
  int32 size = T_.size();
  WriteBasicType(os, binary, size);
  for (int32 i = 0; i < size; i++)
    T_[i].Write(os, binary);
  WriteToken(os, binary, "<Means>");
  int32 nrows = gmeans_.NumRows(), ncols = gmeans_.NumCols();
  WriteBasicType(os, binary, nrows);
  WriteBasicType(os, binary, ncols);
  gmeans_.Write(os, binary);
  WriteToken(os, binary, "<SigmaInv>");  
  KALDI_ASSERT(size == static_cast<int32>(gincovars_.size()));
  for (int32 i = 0; i < size; i++)
    gincovars_[i].Write(os, binary);

  WriteToken(os, binary, "</JvectorExtractor>");
}


void JvectorExtractor::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<JvectorExtractor>");
  ExpectToken(is, binary, "<w_vec>");
  gweights_.Read(is, binary);
  ExpectToken(is, binary, "<M>");  
  int32 size;
  ReadBasicType(is, binary, &size);
  KALDI_ASSERT(size > 0);
  T_.resize(size);
  for (int32 i = 0; i < size; i++)
    T_[i].Read(is, binary);
  ExpectToken(is, binary, "<Means>");
  int32 nrows, ncols;
  ReadBasicType(is, binary, &nrows);
  ReadBasicType(is, binary, &ncols);
  gmeans_.Resize(nrows, ncols);
  gmeans_.Read(is, binary);
  ExpectToken(is, binary, "<SigmaInv>");
  gincovars_.resize(size);
  // Sigma_inv_d_.resize(size);
  for (int32 i = 0; i < size; i++) {
    gincovars_[i].Read(is, binary);
    // Sigma_inv_d_[i].Resize(gincovars_[i].NumCols());
    // Sigma_inv_d_[i].CopyDiagFromSp(gincovars_[i]);
    //T_[i].MulRowsVec(Sigma_inv_d_[i]);
  }
  ExpectToken(is, binary, "</JvectorExtractor>");
  ComputeDerivedVars();
}

void JvectorExtractor::ComputeDerivedVars() {
  KALDI_LOG << "Computing derived variables for jVector extractor";
  U_mat_.Resize(NumGauss(), JvectorDim() * (JvectorDim() + 1) / 2);
  SpMatrix<double> temp_U(JvectorDim());
  for (int32 i = 0; i < NumGauss(); i++) {
    // temp_U = M_i^T Sigma_i^{-1} M_i
    temp_U.AddMat2Sp(1.0, T_[i], kTrans, gincovars_[i], 0.0);
    SubVector<double> temp_U_vec(temp_U.Data(),
                                 JvectorDim() * (JvectorDim() + 1) / 2);
    U_mat_.Row(i).CopyFromVec(temp_U_vec);
  }

  //TODO: Check V_mat_ and gconsts_ and gincovars_mat_
  V_mat_.Resize(NumGauss(), JvectorDim());
  for (int32 i = 0; i < NumGauss(); i++) {
    SubVector<double> V_vec(V_mat_, i);
    Vector<double> temp_vec;
    temp_vec.Resize(FeatDim());
    Vector<double> mean(gmeans_.Row(i));

    //temp_vec.AddSpVec(1.0, gincovars_[i], kNoTrans, mean, 0.0);
    temp_vec.AddSpVec(1.0, gincovars_[i], mean, 0.0);
    V_vec.AddMatVec(1.0, T_[i], kTrans, temp_vec, 0.0);
  }
  
  gconsts_.Resize(NumGauss());
  gconsts_.ApplyLogAndCopy(gweights_);
  gconsts_.Add(-0.5 * FeatDim() * M_LOG_2PI );
  for (int32 i = 0; i < NumGauss(); i++) {
    BaseFloat logdet = -gincovars_[i].LogPosDefDet();
    gconsts_(i) -= 0.5 * (logdet + 
        VecSpVec( gmeans_.Row(i), gincovars_[i], gmeans_.Row(i) ) );
  }

  // Update gincovars_mat_
  gincovars_mat_.Resize(NumGauss(), FeatDim() * ( FeatDim() + 1) / 2);
  for (int32 i = 0; i < NumGauss(); i++) {
    SubVector<double> temp_Sigma_inv_vec(gincovars_[i].Data(),
                                FeatDim() * ( FeatDim() + 1) / 2 );
    gincovars_mat_.Row(i).CopyFromVec(temp_Sigma_inv_vec);
  }


  KALDI_LOG << "Done.";
}








JvectorStats::JvectorStats(const JvectorExtractor &extractor) {
  int32 S = extractor.JvectorDim(), D = extractor.FeatDim(),
      I = extractor.NumGauss();
  
  gamma_.Resize(I);

  num_jvectors_ = 0.0;
  jvector_1th_.Resize(S);
  jvector_2th_.Resize(S);

  acc_1th_centre.Resize(I, D);
  acc_2th_centre.resize(I);
  for (int32 i = 0; i< I; i++)
    acc_2th_centre[i].Resize(D);
  icovars_1th.resize(I);
  for (int32 i = 0; i< I; i++)
    icovars_1th[i].Resize(S);

  Y_.resize(I);
  for (int32 i = 0; i < I; i++)
    Y_[i].Resize(D, S);
  R_.Resize(I, S * (S + 1) / 2);


}


// Revised. Add output : opost
float JvectorStats::AccStatsForUtterance(
    const JvectorExtractor &extractor,
    const MatrixBase<BaseFloat> &feats,
    const JvectorExtractorDecodeOptions &decode_opts,
    const Posterior &ipost,
    Posterior *opost) {

  CheckDims(extractor);
  
  int32 num_gauss = extractor.NumGauss(), feat_dim = extractor.FeatDim();

  if (feat_dim != feats.NumCols()) {
    KALDI_ERR << "Feature dimension mismatch, expected " << feat_dim
              << ", got " << feats.NumCols();
  }
  KALDI_ASSERT(static_cast<int32>(ipost.size()) == feats.NumRows());

  // The zeroth and 1st-order stats are in "utt_stats".
  JvectorExtractorUtteranceStats utt_stats(num_gauss, feat_dim);

  extractor.GetStats(feats, ipost, &utt_stats);
  
  float log_avg = CommitStatsForUtterance(extractor, utt_stats, decode_opts, feats, opost);
  return log_avg;
}


void JvectorStats::Read(std::istream &is, bool binary, bool add) {
  ExpectToken(is, binary, "<JvectorStats>");
  ExpectToken(is, binary, "<gamma>");
  gamma_.Read(is, binary, add);

  ExpectToken(is, binary, "<acc_1th_centre>");
  acc_1th_centre.Read(is, binary, add);

  ExpectToken(is, binary, "<acc_2th_centre>");
  int32 size;
  ReadBasicType(is, binary, &size);
  acc_2th_centre.resize(size);
  for (int32 i = 0; i < size; i++)
    acc_2th_centre[i].Read(is, binary, add);

  ExpectToken(is, binary, "<icovars_1th>");
  ReadBasicType(is, binary, &size);
  icovars_1th.resize(size);
  for (int32 i = 0; i < size; i++)
    icovars_1th[i].Read(is, binary, add);

  ExpectToken(is, binary, "<Y>");
  ReadBasicType(is, binary, &size);
  Y_.resize(size);
  for (int32 i = 0; i < size; i++)
    Y_[i].Read(is, binary, add);
  ExpectToken(is, binary, "<R>");
  R_.Read(is, binary, add);

  try {
  ExpectToken(is, binary, "<num_ivectors>");
  double num_jvectors = 0.0;
  ReadBasicType(is, binary, &num_jvectors);
  if( false == add)
  {
    num_jvectors_ = 0.0;
  }
  num_jvectors_ += num_jvectors;

  ExpectToken(is, binary, "<ivector_1th>");
  jvector_1th_.Read(is, binary, add);

  ExpectToken(is, binary, "<ivector_2th>");
  jvector_2th_.Read(is, binary, add);
  KALDI_LOG << "Read in a new version of jvector extractor Successfully.";
  } catch(const std::exception &e) {
    KALDI_LOG << "Read in an old version of jvector extractor Successfully.";
  }

  //ExpectToken(is, binary, "<Q>");
  //Q_.Read(is, binary, add);
  // ExpectToken(is, binary, "<G>");
  // G_.Read(is, binary, add);
  // ExpectToken(is, binary, "<S>");
  // ReadBasicType(is, binary, &size);
  // S_.resize(size);
  // for (int32 i = 0; i < size; i++)
  //   S_[i].Read(is, binary, add);
  // ExpectToken(is, binary, "<NumJvectors>");
  // ReadBasicType(is, binary, &num_jvectors_, add);
  // ExpectToken(is, binary, "<JvectorSum>");
  // ivector_sum_.Read(is, binary, add);
  // ExpectToken(is, binary, "<JvectorScatter>");
  // ivector_scatter_.Read(is, binary, add);
  ExpectToken(is, binary, "</JvectorStats>");
  //TODO: Add dim check for each variable read
}


void JvectorStats::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<JvectorStats>");
  WriteToken(os, binary, "<gamma>");
  gamma_.Write(os, binary);

  WriteToken(os, binary, "<acc_1th_centre>");
  acc_1th_centre.Write(os, binary);

  WriteToken(os, binary, "<acc_2th_centre>");
  int32 size = acc_2th_centre.size();
  WriteBasicType(os, binary, size);
  for (int32 i = 0; i < size; i++)
    acc_2th_centre[i].Write(os, binary);

  WriteToken(os, binary, "<icovars_1th>");
  size = icovars_1th.size();
  WriteBasicType(os, binary, size);
  for (int32 i = 0; i < size; i++)
    icovars_1th[i].Write(os, binary);

  WriteToken(os, binary, "<Y>");
  size = Y_.size();
  WriteBasicType(os, binary, size);
  for (int32 i = 0; i < size; i++)
    Y_[i].Write(os, binary);
  WriteToken(os, binary, "<R>");
  R_.Write(os, binary);

  WriteToken(os, binary, "<num_ivectors>");
  WriteBasicType(os, binary, num_jvectors_);

  WriteToken(os, binary, "<ivector_1th>");
  jvector_1th_.Write(os, binary);

  WriteToken(os, binary, "<ivector_2th>");
  jvector_2th_.Write(os, binary);

  //WriteToken(os, binary, "<Q>");
  //Q_.Write(os, binary);
  // WriteToken(os, binary, "<G>");
  // G_.Write(os, binary);
  // WriteToken(os, binary, "<S>");
  // size = S_.size();
  // WriteBasicType(os, binary, size);
  // for (int32 i = 0; i < size; i++)
  //   S_[i].Write(os, binary);
  // WriteToken(os, binary, "<NumJvectors>");
  // WriteBasicType(os, binary, num_jvectors_);
  // WriteToken(os, binary, "<JvectorSum>");
  // ivector_sum_.Write(os, binary);
  // WriteToken(os, binary, "<JvectorScatter>");
  // ivector_scatter_.Write(os, binary);
  WriteToken(os, binary, "</JvectorStats>");
}


double JvectorStats::Update(const JvectorExtractorEstimationOptions &opts,
                               JvectorExtractor *extractor) const {
  CheckDims(*extractor);
  
  double ans = 0.0;
  ans += UpdateProjections(opts, extractor);
  // KALDI_LOG << "Overall objective-function improvement per frame was " << ans;
  UpdateGweights(extractor);
  UpdateGmeans(opts, extractor);
  UpdateGincovars(opts, extractor);
  KALDI_LOG << "Variance Diagnose before whitening: ";
  JvectorVarianceDiagnostic(extractor);
  // WhitenProjections(extractor);
  // KALDI_LOG << "Variance Diagnose after whitening: ";
  // JvectorVarianceDiagnostic(extractor);
  extractor->ComputeDerivedVars();
  return ans;
}


float JvectorStats::CommitStatsForUtterance(
    const JvectorExtractor &extractor,
    const JvectorExtractorUtteranceStats &utt_stats,
    const JvectorExtractorDecodeOptions &decode_opts,
    const MatrixBase<BaseFloat> &feats,
    Posterior *opost) {
  
  int32 jvector_dim = extractor.JvectorDim();
  Vector<double> ivec_mean(jvector_dim);
  SpMatrix<double> ivec_var(jvector_dim);

  extractor.GetJvectorDistribution(utt_stats,
                                   &ivec_mean,
                                   &ivec_var);

  // TODO:
  // This is to update posteriors which is time consumming
  // So fat we fix posteriors, so there is no need to run this step
  // Instead of simply deleting this line, a check of the input opost is better
  // If null, simply return, else performing decoding
  float log_avg = extractor.GetJvectorPosteriors(decode_opts, ivec_mean,
                            ivec_var, feats, opost);
  AccStats(extractor, utt_stats, ivec_mean, ivec_var);

  return log_avg;
}

//TODO: revise it or add more funcs for CommitStatsForUtterance
//Done!
void JvectorStats::AccStats(
    const JvectorExtractor &extractor,
    const JvectorExtractorUtteranceStats &utt_stats,
    const VectorBase<double> &ivec_mean,
    const SpMatrix<double> &ivec_var) {
  subspace_stats_lock_.lock();
  
  int num_gauss = extractor.NumGauss();
  int jvector_dim = extractor.JvectorDim();
  int feat_dim = extractor.FeatDim();


  // We do the occupation stats here also.
  // Accumulate stats for gamma_
  gamma_.AddVec(1.0, utt_stats.gamma);
  
  // Stats for the linear term in T_:
  // Accumulate stats for Y_
  for  (int32 i = 0; i < num_gauss; i++) {
    Y_[i].AddVecVec(1.0, utt_stats.X_centre.Row(i),
                    Vector<double>(ivec_mean));
  }

  // Stats for the quadratic term in T_:
  SpMatrix<double> ivec_scatter(ivec_var);
  ivec_scatter.AddVec2(1.0, ivec_mean);
  SubVector<double> ivec_scatter_vec(ivec_scatter.Data(),
                                     jvector_dim * (jvector_dim + 1) / 2);
  R_.AddVecVec(1.0, utt_stats.gamma, ivec_scatter_vec);

  // Stats for acc_1th_centre;
  for (int32 i = 0; i < num_gauss; i++) {
    acc_1th_centre.Row(i).AddVec(1.0, utt_stats.X.Row(i));
    acc_1th_centre.Row(i).AddMatVec(-utt_stats.gamma(i), extractor.T_[i], kNoTrans, ivec_mean, 1.0);
  }

  //Stats for acc_2th_centre;
  Vector<double> temp(feat_dim);
  for (int32 i = 0; i < num_gauss; i++) {
    acc_2th_centre[i].AddSp(1.0, utt_stats.S[i]);
    temp.AddMatVec(1.0, extractor.T_[i], kNoTrans, ivec_mean, 0.0);
    // acc_2th_centre[i].Add(-2.0 * VecVec(temp, utt_stats.X.Row(i)) );
    acc_2th_centre[i].AddVecVec(-1.0, temp, utt_stats.X.Row(i));
    acc_2th_centre[i].AddVec2(utt_stats.gamma(i), temp);
  }

  // Stats for icovars_1th;
  for (int32 i = 0; i < num_gauss; i++) {
    icovars_1th[i].AddSp(utt_stats.gamma(i), ivec_var);
  }


  // Stats for jvector_1th_;
  num_jvectors_ += 1.0;
  jvector_1th_.AddVec(1.0, ivec_mean);
  jvector_2th_.AddVec2(1.0, ivec_mean);

  subspace_stats_lock_.unlock();
}


double JvectorStats::UpdateProjections(
    const JvectorExtractorEstimationOptions &opts,
    JvectorExtractor *extractor) const {
  int32 I = extractor->NumGauss();
  double tot_impr = 0.0;
  for (int32 i = 0; i < I; i++)
    tot_impr += UpdateProjection(opts, i, extractor);
  double count = gamma_.Sum();
  KALDI_LOG << "Overall objective function improvement for M (mean projections) "
            << "was " << (tot_impr / count) << " per frame over "
            << count << " frames.";
  return tot_impr / count;
}


double JvectorStats::UpdateProjection(
    const JvectorExtractorEstimationOptions &opts,
    int32 i,
    JvectorExtractor *extractor) const {
  int32 I = extractor->NumGauss(), S = extractor->JvectorDim();
  KALDI_ASSERT(i >= 0 && i < I);
  /*
    For Gaussian index i, maximize the auxiliary function
       Q_i(x) = tr(M_i^T Sigma_i^{-1} Y_i)  - 0.5 tr(Sigma_i^{-1} M_i R_i M_i^T)
   */
  if (gamma_(i) < opts.gaussian_min_count) {
    KALDI_WARN << "Skipping Gaussian index " << i << " because count "
               << gamma_(i) << " is below min-count.";
    return 0.0;
  }
  SpMatrix<double> R(S, kUndefined), SigmaInv(extractor->gincovars_[i]);
  SubVector<double> R_vec(R_, i); // i'th row of R; vectorized form of SpMatrix.
  SubVector<double> R_sp(R.Data(), S * (S+1) / 2);
  R_sp.CopyFromVec(R_vec); // copy to SpMatrix's memory.

  Matrix<double> M(extractor->T_[i]);
  SolverOptions solver_opts;
  solver_opts.name = "M";
  solver_opts.diagonal_precondition = true;
  // TODO: check if inversion is sufficient?
  double impr = SolveQuadraticMatrixProblem(R, Y_[i], SigmaInv, solver_opts, &M),
      gamma = gamma_(i);
  // if (i < 4) {
  //   KALDI_VLOG(1) << "Objf impr for M for Gaussian index " << i << " is "
  //                 << (impr / gamma) << " per frame over " << gamma << " frames.";
  // }
  extractor->T_[i].CopyFromMat(M);
  return impr;
}


// Note that the whitening matrix is not unique!
// In kaldi ivector-extractor.cc, they use EigenDecomp,
// Here we try CholeskyDecomp
void JvectorStats::WhitenProjections(
    JvectorExtractor *extractor) const {

  KALDI_ASSERT(num_jvectors_ > 0.0);
  Vector<double> jvector_mean(jvector_1th_);
  jvector_mean.Scale(1.0 / num_jvectors_);
  SpMatrix<double> covar(jvector_2th_);
  covar.Scale(1.0 / num_jvectors_);
  // Get Total covar
  covar.AddVec2(-1.0, jvector_mean);
  int32 dim = covar.NumRows();
  TpMatrix<double> T(dim);
  T.Cholesky(covar);


  //Update T_ the projection matrix
  int32 I = extractor->NumGauss();
  Matrix<double> temp(extractor->FeatDim(), extractor->JvectorDim());
  for (int32 i = 0; i < I; i++) {
    temp.AddMatMat(1.0, extractor->T_[i], kNoTrans, (Matrix<double>) T, kNoTrans, 0.0);
    extractor->T_[i].CopyFromMat(temp, kNoTrans);
  }

  // diagnose the whitening effect
  Vector<double> proj_jvector_mean(dim);
  T.Invert();
  proj_jvector_mean.AddTpVec(1.0, T, kNoTrans, jvector_mean, 0.0);
  KALDI_LOG << "j-vector mean after whitening is: "
              << proj_jvector_mean.Norm(2);

}

void JvectorStats::JvectorVarianceDiagnostic(
    const JvectorExtractor *extractor
  ) const {
    SpMatrix<double> W(extractor->gincovars_[0].NumRows()),
                      B(extractor->T_[0].NumRows());
    Vector<double> w(gamma_);
    w.Scale(1.0 / gamma_.Sum());

    for (int32 i = 0; i < extractor->NumGauss(); i++) {
      SpMatrix<double> gcovars(extractor->gincovars_[i]);
      gcovars.Invert();
      // do evd for diagnostic
      Vector<double> s(extractor->FeatDim());
      gcovars.Eig(&s, NULL);
      if (0 == i) {
        KALDI_LOG << "Eigenvalues of jVector covariance range from "
                << s.Min() << " to " << s.Max();
        KALDI_LOG << "max < 1.0 is sugguested.";
      }

      W.AddSp(w(i), gcovars);
      B.AddMat2(w(i), extractor->T_[i], kNoTrans, 1.0);
    }
    double trace_W = W.Trace(),
            trace_B = B.Trace();
    KALDI_LOG << "The proportion of within-Gaussian variance explained by "
      << "the jVectors is " << trace_B / (trace_B + trace_W) << ".";

}


void JvectorStats::UpdateGweights(
    // const JvectorExtractorEstimationOptions &opts,
    JvectorExtractor *extractor
  ) const {
    Vector<double> gweights;
    extractor->gweights_.CopyFromVec(gamma_);
    double tot_sum = extractor->gweights_.Sum();
    extractor->gweights_.Scale(1.0/tot_sum);

}

//TODO:  
void JvectorStats::UpdateGmeans(
    const JvectorExtractorEstimationOptions &opts,
    JvectorExtractor *extractor
  ) const {
  int32 I = extractor->NumGauss();
  for (int32 i = 0; i < I; i++) {
    if (gamma_(i) < opts.gaussian_min_count) {
      KALDI_WARN << "Skipping Gaussian index " << i << " because count "
                 << gamma_(i) << " is below min-count.";
      continue;
    }
    double temp = 1.0 / gamma_(i);
    Vector<double> temp_vec(acc_1th_centre.Row(i));
    temp_vec.Scale(temp);
    extractor->gmeans_.Row(i).CopyFromVec(temp_vec);

  }
  // Vector<double> temp(gamma_);
  // temp.InvertElements();
  // extractor->gmeans_.CopyFromMat(acc_1th_centre, kNoTrans);
  // // the item in gmeans will be come -nan if an item of gamma_ is zero.
  // extractor->gmeans_.MulRowsVec(temp);

}

void JvectorStats::UpdateGincovars(
    const JvectorExtractorEstimationOptions &opts,
    JvectorExtractor *extractor
  ) const {

  int32 I = extractor->NumGauss();
  int32 D = extractor->FeatDim();
  // int32 S = extractor->JvectorDim();

  SpMatrix<double> temp_mat(D);
  for (int32 i = 0; i < I; i++) {

    // This avoids NaN if weight is 0
    if (gamma_(i) < opts.gaussian_min_count) {
      KALDI_WARN << "Skipping Gaussian index " << i << " because count "
                 << gamma_(i) << " is below min-count.";
      continue;
    }

    temp_mat.CopyFromSp(acc_2th_centre[i]);
    temp_mat.AddMat2Sp(1.0, extractor->T_[i], kNoTrans, icovars_1th[i], 1.0);
    temp_mat.Scale(1.0/gamma_(i));
    temp_mat.AddVec2(-1.0, extractor->gmeans_.Row(i));

    temp_mat.Invert(NULL, NULL);
    extractor->gincovars_[i].CopyFromSp(temp_mat);
  }
}



void JvectorStats::CheckDims(const JvectorExtractor &extractor) const {
  int32 S = extractor.JvectorDim(), D = extractor.FeatDim(),
      I = extractor.NumGauss();
  KALDI_ASSERT(gamma_.Dim() == I);
  KALDI_ASSERT(static_cast<int32>(Y_.size()) == I);
  for (int32 i = 0; i < I; i++)
    KALDI_ASSERT(Y_[i].NumRows() == D && Y_[i].NumCols() == S);
  KALDI_ASSERT(R_.NumRows() == I && R_.NumCols() == S*(S+1)/2);
  //KALDI_ASSERT(Q_.NumRows() == 0);
  // KALDI_ASSERT(G_.NumRows() == 0);
  // S_ may be empty or not, depending on whether update_variances == true in
  // the options.
  // if (!S_.empty()) {
  //   KALDI_ASSERT(static_cast<int32>(S_.size() == I));
  //   for (int32 i = 0; i < I; i++)
  //     KALDI_ASSERT(S_[i].NumRows() == D);
  // }
  // KALDI_ASSERT(num_jvectors_ >= 0);
  // KALDI_ASSERT(ivector_sum_.Dim() == S);
  // KALDI_ASSERT(ivector_scatter_.NumRows() == S);
}



} // namespace kaldi


