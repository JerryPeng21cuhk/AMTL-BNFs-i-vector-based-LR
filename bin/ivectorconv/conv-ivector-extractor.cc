// ivector/ivector-extractor.cc

// Copyright 2013     Daniel Povey
// Copyright 2015     Srikanth Madikeri (Idiap Research Institute)
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

#include "ivectorconv/conv-ivector-extractor.h"
#include "ivector/ivector-extractor.h"

namespace kaldi {

IvectorExtractorConv::IvectorExtractorConv(
    const IvectorExtractorConvOptions &opts,
    const FullGmm &fgmm) {
  KALDI_ASSERT(opts.ivector_dim > 0);
  Sigma_inv_.resize(fgmm.NumGauss());
  for (int32 i = 0; i < fgmm.NumGauss(); i++) {
    const SpMatrix<BaseFloat> &inv_var = fgmm.inv_covars()[i];
    Sigma_inv_[i].Resize(inv_var.NumRows());
    Sigma_inv_[i].CopyFromSp(inv_var);
  }  

  Matrix<double> gmm_means;
  fgmm.GetMeans(&gmm_means);
  Means_.Resize(gmm_means.NumRows(), gmm_means.NumCols());
  Means_.CopyFromMat(gmm_means);

  KALDI_ASSERT(!Sigma_inv_.empty());
  int32 feature_dim = Sigma_inv_[0].NumRows(),
      num_gauss = Sigma_inv_.size();

  M_.resize(num_gauss);
  for (int32 i = 0; i < num_gauss; i++) {
    M_[i].Resize(feature_dim, opts.ivector_dim);
    M_[i].SetRandn();
  }
  w_vec_.Resize(fgmm.NumGauss());
  w_vec_.CopyFromVec(fgmm.weights());
  ComputeDerivedVars();
}


void IvectorExtractorConv::GetIvectorDistribution(
    const IvectorExtractorConvUtteranceStats &utt_stats,
    VectorBase<double> *mean,
    SpMatrix<double> *var) const {

    Vector<double> linear(IvectorDim());
    SpMatrix<double> quadratic(IvectorDim());
    GetIvectorDistMean(utt_stats, &linear, &quadratic);

    // mean of distribution = quadratic^{-1} * linear...
    // mean->AddSpVec(1.0, *var, linear, 0.0);
    quadratic.Invert();
    mean->AddSpVec(1.0, quadratic, linear, 0.0);
    if (var != NULL) {
        var->CopyFromSp(quadratic);
    }
}


void IvectorExtractorConv::GetIvectorDistMean(
    const IvectorExtractorConvUtteranceStats &utt_stats,
    VectorBase<double> *linear,
    SpMatrix<double> *quadratic) const {
    Vector<double> temp(FeatDim());
    int32 I = NumGauss();

    for (int32 i = 0; i < I; i++) {
    double gamma = utt_stats.gamma(i);
    if (gamma <= 0.0)
        continue;
    Vector<double> x(utt_stats.X.Row(i)); // ==(sum post*features) - $\gamma(i) \m_i
    temp.AddSpVec(1.0, Sigma_inv_[i], x, 0.0);
    linear->AddMatVec(1.0, M_[i], kTrans, temp, 1.0); 
    }
    SubVector<double> q_vec(quadratic->Data(), IvectorDim()*(IvectorDim()+1)/2);
    q_vec.AddMatVec(1.0, U_, kTrans, Vector<double>(utt_stats.gamma), 1.0);

    // Merging GetIvectorDistPrior. 
    // TODO: Try a more efficient way
    for (int32 d = 0; d < IvectorDim(); d++)
    (*quadratic)(d, d) += 1.0;
}


void IvectorExtractorConv::GetStats(
    const MatrixBase<BaseFloat> &feats,
    const Posterior &post,
    IvectorExtractorConvUtteranceStats *stats) const {
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
    // SpMatrix<double> outer_prod;
    // if (update_variance) {
    //   outer_prod.Resize(feat_dim);
    //   outer_prod.AddVec2(1.0, frame);
    // }
    for (VecType::const_iterator iter = this_post.begin();
         iter != this_post.end(); ++iter) {
      int32 i = iter->first; // Gaussian index.
      KALDI_ASSERT(i >= 0 && i < num_gauss &&
                   "Out-of-range Gaussian (mismatched posteriors?)");
      double weight = iter->second;
      stats->gamma(i) += weight;
      stats->X.Row(i).AddVec(weight, frame);
      // if (update_variance)
      //   stats->S[i].AddSp(weight, outer_prod);
    }
  }

  for (int32 i = 0; i < num_gauss; i++) {
      stats->X.Row(i).AddVec(-stats->gamma(i), Means_.Row(i));
  }
}

//TODO: Add DecodePosteriors()
BaseFloat IvectorExtractorConv::DecodePosteriors(
    const IvectorExtractorConvDecodeOptions &opts,
    const VectorBase<double> &mean,
    const SpMatrix<double> &var,
    const MatrixBase<BaseFloat> &feats,
    Posterior *post
    ) const{

    // typedef std::vector<std::pair<int32, BaseFloat> > VecType;
    int32 num_frames = feats.NumRows(), num_gauss = NumGauss(),
        feat_dim = FeatDim(), ivector_dim = IvectorDim();
    KALDI_ASSERT(feats.NumCols() == feat_dim);
    post->resize(num_frames);
    Vector<double> utt_consts;
    utt_consts.Resize(num_gauss, kUndefined);
    utt_consts.CopyFromVec(gconsts_);
    // loglikes -= Mean_'*Sigma_inv *M_ * mean;
    utt_consts.AddMatVec(-1.0, V_, kNoTrans, mean, 1.0);
    // loglikes -= 0.5 * tr[ (var - mean*mean') * U_ ]
    SpMatrix<double> var_temp(var);
    var_temp.AddVec2(1.0, mean);
    var_temp.ScaleDiag(0.5);
    SubVector<double> var_vec(var_temp.Data(),
                            ivector_dim * (ivector_dim + 1) /2);
    utt_consts.AddMatVec(-1.0, U_, kNoTrans, var_vec, 1.0);
    
    // Compute utt_linear
    Matrix<double> utt_linears(num_gauss, feat_dim);
    for (int32 i = 0; i < num_gauss; ++i) {
        Vector<double> temp_vec(Means_.Row(i));
        temp_vec.AddMatVec(1.0, M_[i], kNoTrans, mean, 1.0);
        (utt_linears.Row(i)).AddSpVec(1.0, Sigma_inv_[i], temp_vec, 0.0);
    }
    
    //Compute utt_quadratic
    // utt_quadratic is Sigma_inv_mat_ which is a var member in Class IvectorExtractorConv

    // Compute likelihood for each frame
    BaseFloat log_sum = 0.0;
    for (int32 t = 0; t< num_frames; ++t) {
        Vector<float> log_likes(utt_consts);
        SpMatrix<float> data_sq(feat_dim);
        data_sq.AddVec2(1.0, feats.Row(t));
        data_sq.ScaleDiag(0.5);
        SubVector<float> data_sq_vec(data_sq.Data(), feat_dim * (feat_dim + 1) /2);
        log_likes.AddMatVec(1.0, Sigma_inv_mat_, kNoTrans, data_sq_vec, 1.0);
        

        // // Normalization
        // log_sum += loglikes.ApplySoftMax();
        // if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum))
        //     KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)"
        log_sum += VectorToPosteriorEntry(log_likes, opts.num_gselect, opts.min_post, &((*post)[t]));    
        // std::vector<std::pair<int32, BaseFloat> > post_entry;
        // VectorToPosteriorEntry(log_likes, opts.num_gselect, opts.min_post, &post_entry);    
    }
    return log_sum;
}

















int32 IvectorExtractorConv::FeatDim() const {
  KALDI_ASSERT(!M_.empty());
  return M_[0].NumRows();
}
int32 IvectorExtractorConv::IvectorDim() const {
  KALDI_ASSERT(!M_.empty());
  return M_[0].NumCols();
}
int32 IvectorExtractorConv::NumGauss() const {
  return static_cast<int32>(M_.size());
}

// The format is different from kaldi's ivector implementation. ivector_offset_
// doesn't exist and hence not written
//
void IvectorExtractorConv::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<IvectorExtractor>");
  WriteToken(os, binary, "<w_vec>");
  w_vec_.Write(os, binary);
  WriteToken(os, binary, "<M>");  
  int32 size = M_.size();
  WriteBasicType(os, binary, size);
  for (int32 i = 0; i < size; i++)
    M_[i].Write(os, binary);
  WriteToken(os, binary, "<Means>");
  int32 nrows = Means_.NumRows(), ncols = Means_.NumCols();
  WriteBasicType(os, binary, nrows);
  WriteBasicType(os, binary, ncols);
  Means_.Write(os, binary);
  WriteToken(os, binary, "<SigmaInv>");  
  KALDI_ASSERT(size == static_cast<int32>(Sigma_inv_.size()));
  for (int32 i = 0; i < size; i++)
    Sigma_inv_[i].Write(os, binary);
  WriteToken(os, binary, "</IvectorExtractor>");
}


void IvectorExtractorConv::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<IvectorExtractor>");
  ExpectToken(is, binary, "<w_vec>");
  w_vec_.Read(is, binary);
  ExpectToken(is, binary, "<M>");  
  int32 size;
  ReadBasicType(is, binary, &size);
  KALDI_ASSERT(size > 0);
  M_.resize(size);
  for (int32 i = 0; i < size; i++)
    M_[i].Read(is, binary);
  ExpectToken(is, binary, "<Means>");
  int32 nrows, ncols;
  ReadBasicType(is, binary, &nrows);
  ReadBasicType(is, binary, &ncols);
  Means_.Resize(nrows, ncols);
  Means_.Read(is, binary);
  ExpectToken(is, binary, "<SigmaInv>");
  Sigma_inv_.resize(size);
  // Sigma_inv_d_.resize(size);
  for (int32 i = 0; i < size; i++) {
    Sigma_inv_[i].Read(is, binary);
    // Sigma_inv_d_[i].Resize(Sigma_inv_[i].NumCols());
    // Sigma_inv_d_[i].CopyDiagFromSp(Sigma_inv_[i]);
    //M_[i].MulRowsVec(Sigma_inv_d_[i]);
  }
  ExpectToken(is, binary, "</IvectorExtractor>");
  ComputeDerivedVars();
}

void IvectorExtractorConv::ComputeDerivedVars() {
  KALDI_LOG << "Computing derived variables for iVector extractor";
  U_.Resize(NumGauss(), IvectorDim() * (IvectorDim() + 1) / 2);
  SpMatrix<double> temp_U(IvectorDim());
  KALDI_LOG << "Start to compute U_";
  for (int32 i = 0; i < NumGauss(); i++) {
    // temp_U = M_i^T Sigma_i^{-1} M_i
    temp_U.AddMat2Sp(1.0, M_[i], kTrans, Sigma_inv_[i], 0.0);
    SubVector<double> temp_U_vec(temp_U.Data(),
                                 IvectorDim() * (IvectorDim() + 1) / 2);
    U_.Row(i).CopyFromVec(temp_U_vec);
  }

  KALDI_LOG << "Start to compute V_";
  //TODO: Check V_ and gconsts_ and Sigma_inv_mat_
  V_.Resize(NumGauss(), IvectorDim());
  for (int32 i = 0; i < NumGauss(); i++) {
    SubVector<double> V_vec(V_, i);
    Vector<double> temp_vec;
    temp_vec.Resize(FeatDim());
    Vector<double> mean(Means_.Row(i));

    //temp_vec.AddSpVec(1.0, Sigma_inv_[i], kNoTrans, mean, 0.0);
    temp_vec.AddSpVec(1.0, Sigma_inv_[i], mean, 0.0);
    V_vec.AddMatVec(1.0, M_[i], kTrans, temp_vec, 0.0);
  }
  
  KALDI_LOG << "Start to compute gconsts_";
  gconsts_.Resize(NumGauss());
  gconsts_.ApplyLogAndCopy(w_vec_);
  gconsts_.Add(-0.5 * FeatDim() * M_LOG_2PI );
  for (int32 i = 0; i < NumGauss(); i++) {
    BaseFloat logdet = -Sigma_inv_[i].LogPosDefDet();
    gconsts_(i) -= 0.5 * (logdet + 
        VecSpVec( Means_.Row(i), Sigma_inv_[i], Means_.Row(i) ) );
  }

  KALDI_LOG << "Start to compute Sigma_inv_mat_";
  // Update Sigma_inv_mat_
  Sigma_inv_mat_.Resize(NumGauss(), FeatDim() * ( FeatDim() + 1) / 2);
  for (int32 i = 0; i < NumGauss(); i++) {
    SubVector<double> temp_Sigma_inv_vec(Sigma_inv_[i].Data(),
                                FeatDim() * ( FeatDim() + 1) / 2 );
    Sigma_inv_mat_.Row(i).CopyFromVec(temp_Sigma_inv_vec);
  }

  KALDI_LOG << "Done.";
}








IvectorConvStats::IvectorConvStats(const IvectorExtractorConv &extractor) {
  int32 S = extractor.IvectorDim(), D = extractor.FeatDim(),
      I = extractor.NumGauss();
  
  gamma_.Resize(I);
  Y_.resize(I);
  for (int32 i = 0; i < I; i++)
    Y_[i].Resize(D, S);
  R_.Resize(I, S * (S + 1) / 2);
  // num_ivectors_ = 0;
  // ivector_sum_.Resize(S);
  // ivector_scatter_.Resize(S);
}


void IvectorConvStats::AccStatsForUtterance(
    const IvectorExtractorConv &extractor,
    const MatrixBase<BaseFloat> &feats,
    const Posterior &post) {
  typedef std::vector<std::pair<int32, BaseFloat> > VecType;

  CheckDims(extractor);
  
  int32 num_gauss = extractor.NumGauss(), feat_dim = extractor.FeatDim();

  if (feat_dim != feats.NumCols()) {
    KALDI_ERR << "Feature dimension mismatch, expected " << feat_dim
              << ", got " << feats.NumCols();
  }
  KALDI_ASSERT(static_cast<int32>(post.size()) == feats.NumRows());

  // The zeroth and 1st-order stats are in "utt_stats".
  IvectorExtractorConvUtteranceStats utt_stats(num_gauss, feat_dim);

  extractor.GetStats(feats, post, &utt_stats);
  
  CommitStatsForUtterance(extractor, utt_stats);
}


void IvectorConvStats::Read(std::istream &is, bool binary, bool add) {
  ExpectToken(is, binary, "<IvectorStats>");
  ExpectToken(is, binary, "<gamma>");
  gamma_.Read(is, binary, add);
  ExpectToken(is, binary, "<Y>");
  int32 size;
  ReadBasicType(is, binary, &size);
  Y_.resize(size);
  for (int32 i = 0; i < size; i++)
    Y_[i].Read(is, binary, add);
  ExpectToken(is, binary, "<R>");
  R_.Read(is, binary, add);
  //ExpectToken(is, binary, "<Q>");
  //Q_.Read(is, binary, add);
  // ExpectToken(is, binary, "<G>");
  // G_.Read(is, binary, add);
  // ExpectToken(is, binary, "<S>");
  // ReadBasicType(is, binary, &size);
  // S_.resize(size);
  // for (int32 i = 0; i < size; i++)
  //   S_[i].Read(is, binary, add);
  // ExpectToken(is, binary, "<NumIvectors>");
  // ReadBasicType(is, binary, &num_ivectors_, add);
  // ExpectToken(is, binary, "<IvectorSum>");
  // ivector_sum_.Read(is, binary, add);
  // ExpectToken(is, binary, "<IvectorScatter>");
  // ivector_scatter_.Read(is, binary, add);
  ExpectToken(is, binary, "</IvectorStats>");
}


void IvectorConvStats::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<IvectorStats>");
  WriteToken(os, binary, "<gamma>");
  gamma_.Write(os, binary);
  WriteToken(os, binary, "<Y>");
  int32 size = Y_.size();
  WriteBasicType(os, binary, size);
  for (int32 i = 0; i < size; i++)
    Y_[i].Write(os, binary);
  WriteToken(os, binary, "<R>");
  R_.Write(os, binary);
  //WriteToken(os, binary, "<Q>");
  //Q_.Write(os, binary);
  // WriteToken(os, binary, "<G>");
  // G_.Write(os, binary);
  // WriteToken(os, binary, "<S>");
  // size = S_.size();
  // WriteBasicType(os, binary, size);
  // for (int32 i = 0; i < size; i++)
  //   S_[i].Write(os, binary);
  // WriteToken(os, binary, "<NumIvectors>");
  // WriteBasicType(os, binary, num_ivectors_);
  // WriteToken(os, binary, "<IvectorSum>");
  // ivector_sum_.Write(os, binary);
  // WriteToken(os, binary, "<IvectorScatter>");
  // ivector_scatter_.Write(os, binary);
  WriteToken(os, binary, "</IvectorStats>");
}


double IvectorConvStats::Update(const IvectorExtractorConvEstimationOptions &opts,
                               IvectorExtractorConv *extractor) const {
  CheckDims(*extractor);
  
  double ans = 0.0;
  ans += UpdateProjections(opts, extractor);
  KALDI_LOG << "Overall objective-function improvement per frame was " << ans;
  extractor->ComputeDerivedVars();
  return ans;
}


void IvectorConvStats::CommitStatsForUtterance(
    const IvectorExtractorConv &extractor,
    const IvectorExtractorConvUtteranceStats &utt_stats) {
  
  int32 ivector_dim = extractor.IvectorDim();
  Vector<double> ivec_mean(ivector_dim);
  SpMatrix<double> ivec_var(ivector_dim);

  extractor.GetIvectorDistribution(utt_stats,
                                   &ivec_mean,
                                   &ivec_var);

  CommitStatsForM(extractor, utt_stats, ivec_mean, ivec_var);
}


void IvectorConvStats::CommitStatsForM(
    const IvectorExtractorConv &extractor,
    const IvectorExtractorConvUtteranceStats &utt_stats,
    const VectorBase<double> &ivec_mean,
    const SpMatrix<double> &ivec_var) {
  subspace_stats_lock_.lock();

  // We do the occupation stats here also.
  gamma_.AddVec(1.0, utt_stats.gamma);
  
  // Stats for the linear term in M:
  for  (int32 i = 0; i < extractor.NumGauss(); i++) {
    Y_[i].AddVecVec(1.0, utt_stats.X.Row(i),
                    Vector<double>(ivec_mean));
  }

  int32 ivector_dim = extractor.IvectorDim();
  // Stats for the quadratic term in M:
  SpMatrix<double> ivec_scatter(ivec_var);
  ivec_scatter.AddVec2(1.0, ivec_mean);
  SubVector<double> ivec_scatter_vec(ivec_scatter.Data(),
                                     ivector_dim * (ivector_dim + 1) / 2);
  R_.AddVecVec(1.0, utt_stats.gamma, ivec_scatter_vec);

  subspace_stats_lock_.unlock();
}


double IvectorConvStats::UpdateProjections(
    const IvectorExtractorConvEstimationOptions &opts,
    IvectorExtractorConv *extractor) const {
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


double IvectorConvStats::UpdateProjection(
    const IvectorExtractorConvEstimationOptions &opts,
    int32 i,
    IvectorExtractorConv *extractor) const {
  int32 I = extractor->NumGauss(), S = extractor->IvectorDim();
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
  SpMatrix<double> R(S, kUndefined), SigmaInv(extractor->Sigma_inv_[i]);
  SubVector<double> R_vec(R_, i); // i'th row of R; vectorized form of SpMatrix.
  SubVector<double> R_sp(R.Data(), S * (S+1) / 2);
  R_sp.CopyFromVec(R_vec); // copy to SpMatrix's memory.

  Matrix<double> M(extractor->M_[i]);
  SolverOptions solver_opts;
  solver_opts.name = "M";
  solver_opts.diagonal_precondition = true;
  // TODO: check if inversion is sufficient?
  double impr = SolveQuadraticMatrixProblem(R, Y_[i], SigmaInv, solver_opts, &M),
      gamma = gamma_(i);
  if (i < 4) {
    KALDI_VLOG(1) << "Objf impr for M for Gaussian index " << i << " is "
                  << (impr / gamma) << " per frame over " << gamma << " frames.";
  }
  extractor->M_[i].CopyFromMat(M);
  return impr;
}


void IvectorConvStats::CheckDims(const IvectorExtractorConv &extractor) const {
  int32 S = extractor.IvectorDim(), D = extractor.FeatDim(),
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
  // KALDI_ASSERT(num_ivectors_ >= 0);
  // KALDI_ASSERT(ivector_sum_.Dim() == S);
  // KALDI_ASSERT(ivector_scatter_.NumRows() == S);
}



} // namespace kaldi


