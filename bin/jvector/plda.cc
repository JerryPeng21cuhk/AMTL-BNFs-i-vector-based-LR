// ivector/plda.cc

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

#include <vector>
#include "jvector/plda.h"

namespace kaldi {

template<class Real>
/// This function computes a projection matrix that when applied makes the
/// covariance unit (i.e. all 1).
static void ComputeNormalizingTransform(const SpMatrix<Real> &covar,
                                        MatrixBase<Real> *proj) {
  int32 dim = covar.NumRows();
  TpMatrix<Real> C(dim);  // Cholesky of covar, covar = C C^T
  C.Cholesky(covar);
  C.Invert();  // The matrix that makes covar unit is C^{-1}, because
               // C^{-1} covar C^{-T} = C^{-1} C C^T C^{-T} = I.
  proj->CopyFromTp(C, kNoTrans);  // set "proj" to C^{-1}.
}

// interpolate between two pldas
Plda::Plda(const Plda &plda1, const Plda &plda2, const float interpolate_scalar) {
  int32 dim = plda1.Dim();
  KALDI_ASSERT(dim > 0);
  KALDI_ASSERT(plda2.Dim() == dim);
  KALDI_ASSERT(interpolate_scalar >= 0 && interpolate_scalar <= 1);
  // Interpolate means
  mean_.Resize(dim, kSetZero);
  KALDI_WARN << "2-norm of plda1 mean is: " << plda1.mean_.Norm(2.0);
  KALDI_WARN << "2-norm of plda2 mean is: " << plda2.mean_.Norm(2.0);
  Vector<double> delta_mean(plda1.mean_);
  delta_mean.AddVec(-1.0, plda2.mean_);
  KALDI_WARN << "2-norm of ( plda1 mean - plda2 mean ) is: " << delta_mean.Norm(2.0);

  // mean_.AddVec(interpolate_scalar, plda1.mean_);
  // mean_.AddVec(1.0-interpolate_scalar, plda2.mean_);

  //for debug
  mean_.AddVec(1.0, plda2.mean_);

  // Invert transform_
  Matrix<double> transform1_inverse(plda1.transform_, kNoTrans),
                  transform2_inverse(plda2.transform_, kNoTrans);
  transform1_inverse.Invert();
  transform2_inverse.Invert();

  // Get original \phi_w and \ph_b from transform_ and psi_
  SpMatrix<double> within_var(dim, kSetZero), between_var(dim, kSetZero);
  within_var.AddMat2(interpolate_scalar, transform1_inverse, kNoTrans, 0.0);
  within_var.AddMat2(1.0-interpolate_scalar, transform2_inverse, kNoTrans, 1.0);

  between_var.AddMat2Vec(interpolate_scalar, transform1_inverse, kNoTrans, plda1.psi_, 0.0);
  between_var.AddMat2Vec(1.0-interpolate_scalar, transform2_inverse, kNoTrans, plda2.psi_, 1.0);

  // Now, we do simultaneous diagnolization towards within_var and between_var
  Matrix<double> transform1(dim, dim);
  ComputeNormalizingTransform(within_var, &transform1);
  // now transform is a matrix that if we project with it,
  // within_var becomes unit.

  // between_var_proj is between_var after projecting with transform1.
  SpMatrix<double> between_var_proj(dim);
  between_var_proj.AddMat2Sp(1.0, transform1, kNoTrans, between_var, 0.0);

  Matrix<double> U(dim, dim);
  Vector<double> s(dim);
  // Do symmetric eigenvalue decomposition between_var_proj = U diag(s) U^T,
  // where U is orthogonal.
  between_var_proj.Eig(&s, &U);

  // KALDI_WARN << s;
  int32 n;
  s.ApplyFloor(0.0, &n);
  if (n > 0) {
    KALDI_WARN << "Floored " << n << " eigenvalues of between-class "
               << "variance to zero.";
  }
  // Sort from greatest to smallest eigenvalue.
  SortSvd(&s, &U);

  // The transform U^T will make between_var_proj diagonal with value s
  // (i.e. U^T U diag(s) U U^T = diag(s)).  The final transform that
  // makes within_var_ unit and between_var_ diagonal is U^T transform1,
  // i.e. first transform1 and then U^T.

  transform_.Resize(dim, dim);
  transform_.AddMatMat(1.0, U, kTrans, transform1, kNoTrans, 0.0);
  psi_ = s;

  KALDI_LOG << "Diagonal of between-class variance in normalized space is " << s;

  ComputeDerivedVars();
}

// deprecated
// Performance is really bad.

// // interpolate between two pldas
// // simply interpolate transform and psi
// // Incorrect in mathmatics
// void Plda::Plda_init2(const Plda &plda1, const Plda &plda2, const float interpolate_scalar) {
//   int32 dim = plda1.Dim();
//   KALDI_ASSERT(dim > 0);
//   KALDI_ASSERT(plda2.Dim() == dim);
//   KALDI_ASSERT(interpolate_scalar >= 0 && interpolate_scalar <= 1);
//   // Interpolate means
//   mean_.Resize(dim, kSetZero);
//   KALDI_WARN << "2-norm of plda1 mean is: " << plda1.mean_.Norm(2.0);
//   KALDI_WARN << "2-norm of plda2 mean is: " << plda2.mean_.Norm(2.0);
//   Vector<double> delta_mean(plda1.mean_);
//   delta_mean.AddVec(-1.0, plda2.mean_);
//   KALDI_WARN << "2-norm of ( plda1 mean - plda2 mean ) is: " << delta_mean.Norm(2.0);

//   // mean_.AddVec(interpolate_scalar, plda1.mean_);
//   // mean_.AddVec(1.0-interpolate_scalar, plda2.mean_);

//   //for debug
//   mean_.AddVec(1.0, plda2.mean_);


//   transform_.Resize(dim, dim);
//   transform_.AddMat(interpolate_scalar, plda1.transform_, kNoTrans);
//   transform_.AddMat(1.0-interpolate_scalar, plda2.transform_, kNoTrans);

//   psi_.Resize(dim);
//   psi_.AddVec(interpolate_scalar, plda1.mean_);
//   psi_.AddVec(1.0-interpolate_scalar, plda2.mean_);

//   ComputeDerivedVars();
// }







void Plda::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<Plda>");
  mean_.Write(os, binary);
  transform_.Write(os, binary);
  psi_.Write(os, binary);
  WriteToken(os, binary, "</Plda>");
}

void Plda::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<Plda>");
  mean_.Read(is, binary);
  transform_.Read(is, binary);
  psi_.Read(is, binary);
  ExpectToken(is, binary, "</Plda>");
  ComputeDerivedVars();
}

void Plda::ComputeDerivedVars() {
  KALDI_ASSERT(Dim() > 0);
  offset_.Resize(Dim());
  offset_.AddMatVec(-1.0, transform_, kNoTrans, mean_, 0.0);
}


/**
   This comment explains the thinking behind the function LogLikelihoodRatio.
   The reference is "Probabilistic Linear Discriminant Analysis" by
   Sergey Ioffe, ECCV 2006.

   I'm looking at the un-numbered equation between eqs. (4) and (5),
   that says
     P(u^p | u^g_{1...n}) =  N (u^p | \frac{n \Psi}{n \Psi + I} \bar{u}^g, I + \frac{\Psi}{n\Psi + I})

   Here, the superscript ^p refers to the "probe" example (e.g. the example
   to be classified), and u^g_1 is the first "gallery" example, i.e. the first
   training example of that class.  \psi is the between-class covariance
   matrix, assumed to be diagonalized, and I can be interpreted as the within-class
   covariance matrix which we have made unit.

   We want the likelihood ratio P(u^p | u^g_{1..n}) / P(u^p), where the
   numerator is the probability of u^p given that it's in that class, and the
   denominator is the probability of u^p with no class assumption at all
   (e.g. in its own class).

   The expression above even works for n = 0 (e.g. the denominator of the likelihood
   ratio), where it gives us
     P(u^p) = N(u^p | 0, I + \Psi)
   i.e. it's distributed with zero mean and covarance (within + between).
   The likelihood ratio we want is:
      N(u^p | \frac{n \Psi}{n \Psi + I} \bar{u}^g, I + \frac{\Psi}{n \Psi + I}) /
      N(u^p | 0, I + \Psi)
   where \bar{u}^g is the mean of the "gallery examples"; and we can expand the
   log likelihood ratio as
     - 0.5 [ (u^p - m) (I + \Psi/(n \Psi + I))^{-1} (u^p - m)  +  logdet(I + \Psi/(n \Psi + I)) ]
     + 0.5 [u^p (I + \Psi) u^p  +  logdet(I + \Psi) ]
   where m = (n \Psi)/(n \Psi + I) \bar{u}^g.

 */

double Plda::GetNormalizationFactor(
    const VectorBase<double> &transformed_ivector,
    int32 num_examples) const {
  KALDI_ASSERT(num_examples > 0);
  // Work out the normalization factor.  The covariance for an average over
  // "num_examples" training iVectors equals \Psi + I/num_examples.
  Vector<double> transformed_ivector_sq(transformed_ivector);
  transformed_ivector_sq.ApplyPow(2.0);
  // inv_covar will equal 1.0 / (\Psi + I/num_examples).
  Vector<double> inv_covar(psi_);
  inv_covar.Add(1.0 / num_examples);
  inv_covar.InvertElements();
  // "transformed_ivector" should have covariance (\Psi + I/num_examples), i.e.
  // within-class/num_examples plus between-class covariance.  So
  // transformed_ivector_sq . (I/num_examples + \Psi)^{-1} should be equal to
  //  the dimension.
  double dot_prod = VecVec(inv_covar, transformed_ivector_sq);
  return sqrt(Dim() / dot_prod);
}



double Plda::TransformIvector(const PldaConfig &config,
                              const VectorBase<double> &ivector,
                              int32 num_examples,
                              VectorBase<double> *transformed_ivector) const {
  KALDI_ASSERT(ivector.Dim() == Dim() && transformed_ivector->Dim() == Dim());
  double normalization_factor;
  transformed_ivector->CopyFromVec(offset_);
  transformed_ivector->AddMatVec(1.0, transform_, kNoTrans, ivector, 1.0);
  if (config.simple_length_norm)
    normalization_factor = sqrt(transformed_ivector->Dim())
      / transformed_ivector->Norm(2.0);
  else
    normalization_factor = GetNormalizationFactor(*transformed_ivector,
                                                  num_examples);
  if (config.normalize_length)
    transformed_ivector->Scale(normalization_factor);
  return normalization_factor;
}

// "float" version of TransformIvector.
float Plda::TransformIvector(const PldaConfig &config,
                             const VectorBase<float> &ivector,
                             int32 num_examples,
                             VectorBase<float> *transformed_ivector) const {
  Vector<double> tmp(ivector), tmp_out(ivector.Dim());
  float ans = TransformIvector(config, tmp, num_examples, &tmp_out);
  transformed_ivector->CopyFromVec(tmp_out);
  return ans;
}




// There is an extended comment within this file, referencing a paper by
// Ioffe, that may clarify what this function is doing.
double Plda::LogLikelihoodRatio_1vsMany(
    const VectorBase<double> &transformed_train_ivector,
    int32 n, // number of training utterances.
    const VectorBase<double> &transformed_test_ivector) const {
  int32 dim = Dim();
  double loglike_given_class, loglike_without_class;
  { // work out loglike_given_class.
    // "mean" will be the mean of the distribution if it comes from the
    // training example.  The mean is \frac{n \Psi}{n \Psi + I} \bar{u}^g
    // "variance" will be the variance of that distribution, equal to
    // I + \frac{\Psi}{n\Psi + I}.
    Vector<double> mean(dim, kUndefined);
    Vector<double> variance(dim, kUndefined);
    for (int32 i = 0; i < dim; i++) {
      mean(i) = n * psi_(i) / (n * psi_(i) + 1.0)
        * transformed_train_ivector(i);
      variance(i) = 1.0 + psi_(i) / (n * psi_(i) + 1.0);
    }
    double logdet = variance.SumLog();
    Vector<double> sqdiff(transformed_test_ivector);
    sqdiff.AddVec(-1.0, mean);
    sqdiff.ApplyPow(2.0);
    variance.InvertElements();
    loglike_given_class = -0.5 * (logdet + M_LOG_2PI * dim +
                                  VecVec(sqdiff, variance));
  }
  { // work out loglike_without_class.  Here the mean is zero and the variance
    // is I + \Psi.
    Vector<double> sqdiff(transformed_test_ivector); // there is no offset.
    sqdiff.ApplyPow(2.0);
    Vector<double> variance(psi_);
    variance.Add(1.0); // I + \Psi.
    double logdet = variance.SumLog();
    variance.InvertElements();
    loglike_without_class = -0.5 * (logdet + M_LOG_2PI * dim +
                                    VecVec(sqdiff, variance));
  }
  double loglike_ratio = loglike_given_class - loglike_without_class;
  //KALDI_WARN << "LogLikelihoodRatio_1vsMany: " << loglike_ratio;
  return loglike_ratio;
}


// Multiple test ivectors of a speaker v.s. 
//   multiple train ivectors of a enrolled speaker
double Plda::LogLikelihoodRatio_MultiVsMulti(
    const VectorBase<double> &transformed_train_ivector,
    int32 n, // number of training utterances.
    const VectorBase<double> &transformed_test_ivector,
    int32 m ) const {
  
  // Create Km+n Km Kn \detla(m+n) \delta(m) \delta(n)
  int32 dim = Dim();
  Vector<double> k_m(dim), k_n(dim), k_mn(dim);
  double delta_m=0, delta_n=0, delta_mn=0;

  for (int32 i = 0; i < dim; i++)  {
    k_m(i) = (m * psi_(i) + 1.0);
    k_n(i) = (n * psi_(i) + 1.0);
    k_mn(i) = ((m+n) * psi_(i) + 1.0);
  }
  delta_m = k_m.SumLog();
  delta_n = k_n.SumLog();
  delta_mn = k_mn.SumLog();

  for (int32 i = 0; i < dim; i++)  {
    k_m(i) = -psi_(i) / k_m(i);
    k_n(i) = -psi_(i) / k_n(i);
    k_mn(i) = -psi_(i) / k_mn(i);
  }

  // Compute the score
  double loglike_tot = -delta_mn + delta_m + delta_n;
  Vector<double> u_p2(transformed_test_ivector);
  Vector<double> u_g2(transformed_train_ivector);
  Vector<double> u_pg(transformed_test_ivector);

  u_pg.MulElements(u_g2);
  u_p2.ApplyPow(2.0);
  u_g2.ApplyPow(2.0);


  //Compute the first item
  Vector<double> k_mn_m(k_mn);
  k_mn_m.AddVec(-1.0, k_m);
  loglike_tot -= m*m * VecVec(k_mn_m, u_p2);

  // Compute the second item
  Vector<double> k_mn_n(k_mn);
  k_mn_n.AddVec(-1.0, k_n);
  loglike_tot -= n*n * VecVec(k_mn_n, u_g2);

  //Compute the last item
  loglike_tot -= 2*m*n * VecVec(k_mn, u_pg);

  // KALDI_WARN << "LogLikelihoodRatio_MultiVsMulti: " << loglike_tot/2;
  return loglike_tot/2;
}


double Plda::LogLikelihoodRatio_ScoreAverage(
    const VectorBase<double> &transformed_train_ivector,
    const VectorBase<double> &transformed_train_ivector_sq,
    const VectorBase<double> &transformed_test_ivector,
    const VectorBase<double> &transformed_test_ivector_sq
  ) const {
  int32 dim = Dim();
  Vector<double> k_2(dim), k_1(dim);
  for (int32 i = 0; i < dim; i++)  {
    k_2(i) = -psi_(i) / (2 * psi_(i) + 1.0);
    k_1(i) = -psi_(i) / (1 * psi_(i) + 1.0);
  }
  Vector<double> k_2_1(k_2);
  k_2_1.AddVec(-1.0, k_1);

  // Compute loglikelihood
  Vector<double> tmp_vec(transformed_train_ivector);
  tmp_vec.MulElements(transformed_test_ivector);

  double loglike_tot =  - VecVec(k_2_1, transformed_test_ivector_sq);
  loglike_tot -= VecVec(k_2_1, transformed_train_ivector_sq);
  loglike_tot -= 2 * VecVec(k_2, tmp_vec);
  return loglike_tot/2;
}


double Plda::LogLikelihoodRatio_IvectorAverage(
    const VectorBase<double> &transformed_train_ivector, //average
    const VectorBase<double> &transformed_test_ivector //average
  ) const {
  return LogLikelihoodRatio_MultiVsMulti(transformed_train_ivector,
                                  1,
                                  transformed_test_ivector,
                                  1);
}

void Plda::SmoothWithinClassCovariance(double smoothing_factor) {
  KALDI_ASSERT(smoothing_factor >= 0.0 && smoothing_factor <= 1.0);
  // smoothing_factor > 1.0 is possible but wouldn't really make sense.

  KALDI_LOG << "Smoothing within-class covariance by " << smoothing_factor
            << ", Psi is initially: " << psi_;
  Vector<double> within_class_covar(Dim());
  within_class_covar.Set(1.0); // It's now the current within-class covariance
                               // (a diagonal matrix) in the space transformed
                               // by transform_.
  within_class_covar.AddVec(smoothing_factor, psi_);
  /// We now revise our estimate of the within-class covariance to this
  /// larger value.  This means that the transform has to change to as
  /// to make this new, larger covariance unit.  And our between-class
  /// covariance in this space is now less.

  psi_.DivElements(within_class_covar);
  KALDI_LOG << "New value of Psi is " << psi_;

  within_class_covar.ApplyPow(-0.5);
  transform_.MulRowsVec(within_class_covar);

  ComputeDerivedVars();
}

void Plda::ApplyTransform(const Matrix<double> &in_transform) {
  KALDI_ASSERT(in_transform.NumRows() <= Dim()
    && in_transform.NumCols() == Dim());

  // Apply in_transform to mean_.
  Vector<double> mean_new(in_transform.NumRows());
  mean_new.AddMatVec(1.0, in_transform, kNoTrans, mean_, 0.0);
  mean_.Resize(in_transform.NumRows());
  mean_.CopyFromVec(mean_new);

  SpMatrix<double> between_var(in_transform.NumCols()),
                   within_var(in_transform.NumCols()),
                   psi_mat(in_transform.NumCols()),
                   between_var_new(Dim()),
                   within_var_new(Dim());
  Matrix<double> transform_invert(transform_);

  // Next, compute the between_var and within_var that existed
  // prior to diagonalization.
  psi_mat.AddDiagVec(1.0, psi_);
  transform_invert.Invert();
  within_var.AddMat2(1.0, transform_invert, kNoTrans, 0.0);
  between_var.AddMat2Sp(1.0, transform_invert, kNoTrans, psi_mat, 0.0);

  // Next, transform the variances using the input transformation.
  between_var_new.AddMat2Sp(1.0, in_transform, kNoTrans, between_var, 0.0);
  within_var_new.AddMat2Sp(1.0, in_transform, kNoTrans, within_var, 0.0);

  // Finally, we need to recompute psi_ and transform_. The remainder of
  // the code in this function  is a lightly modified copy of
  // PldaEstimator::GetOutput().
  Matrix<double> transform1(Dim(), Dim());
  ComputeNormalizingTransform(within_var_new, &transform1);
  // Now transform is a matrix that if we project with it,
  // within_var becomes unit.
  // between_var_proj is between_var after projecting with transform1.
  SpMatrix<double> between_var_proj(Dim());
  between_var_proj.AddMat2Sp(1.0, transform1, kNoTrans, between_var_new, 0.0);

  Matrix<double> U(Dim(), Dim());
  Vector<double> s(Dim());
  // Do symmetric eigenvalue decomposition between_var_proj = U diag(s) U^T,
  // where U is orthogonal.
  between_var_proj.Eig(&s, &U);

  KALDI_ASSERT(s.Min() >= 0.0);
  int32 n;
  s.ApplyFloor(0.0, &n);
  if (n > 0) {
    KALDI_WARN << "Floored " << n << " eigenvalues of between-class "
               << "variance to zero.";
  }
  // Sort from greatest to smallest eigenvalue.
  SortSvd(&s, &U);

  // The transform U^T will make between_var_proj diagonal with value s
  // (i.e. U^T U diag(s) U U^T = diag(s)).  The final transform that
  // makes within_var unit and between_var diagonal is U^T transform1,
  // i.e. first transform1 and then U^T.
  transform_.Resize(Dim(), Dim());
  transform_.AddMatMat(1.0, U, kTrans, transform1, kNoTrans, 0.0);
  psi_.Resize(Dim());
  psi_.CopyFromVec(s);
  ComputeDerivedVars();
}






void PldaStats::AddSamples(const Matrix<double> &group) {
  if (dim_ == 0) {
    Init(group.NumCols());
  } else {
    KALDI_ASSERT(dim_ == group.NumCols());
  }
  int32 n = group.NumRows(); // number of examples for this class
  Vector<double> *sum = new Vector<double>(dim_);
  sum->AddRowSumMat(1.0, group);

  mean_.AddVec(1.0, *sum);
  scatter_.AddMat2(1.0, group, kTrans, 1.0);

  class_info_.push_back(ClassInfo(sum, n));

  num_classes_ ++;
  num_examples_ += n;

}

// do this after loading all samples
void PldaStats::ComputeStats() {
  // make sure not empty!
  KALDI_ASSERT(! class_info_.empty());
  mean_.Scale(1.0 / num_examples_);
  scatter_.AddVec2(-num_examples_, mean_);

  for (size_t i = 0; i < class_info_.size(); i++)
  {
    class_info_[i].stat_1th->AddVec(-class_info_[i].num_examples,
                                    mean_);    
  }
  // Sort clas_info by num_examples_
  Sort();
  // KALDI_WARN << "mean_ " << mean_;
  // KALDI_WARN << "scatter_ " << scatter_;
}




PldaStats::~PldaStats() {
  for (size_t i = 0; i < class_info_.size(); i++)
    delete class_info_[i].stat_1th;
}

bool PldaStats::IsSorted() const {
  for (size_t i = 0; i + 1 < class_info_.size(); i++)
    if (class_info_[i+1] < class_info_[i])
      return false;
  return true;
}

void PldaStats::Init(int32 dim) {
  KALDI_ASSERT(dim_ == 0);
  dim_ = dim;
  num_classes_ = 0;
  num_examples_ = 0;

  mean_.Resize(dim);
  scatter_.Resize(dim);
  // make sure class_info_ is empty
  KALDI_ASSERT(class_info_.empty());
}





PldaEstimator::PldaEstimator(
  const PldaEstimationConfig &config,
  const PldaStats &stats):
    stats_(stats) {
  KALDI_ASSERT(stats.IsSorted());
  InitParameters(config);
}

void PldaEstimator::InitParameters(const PldaEstimationConfig &config) {

  num_factors_ = config.num_factors;
  KALDI_ASSERT(num_factors_ > 0);
  if (num_factors_ > GetMaxWithinMatRank())
    num_factors_ = GetMaxWithinMatRank();

  V_.Resize(Dim(), num_factors_);
  Sigma_inv_.Resize(Dim());
  Y_.Resize(Dim(), num_factors_);
  R_.Resize(num_factors_);
  W_.Resize(num_factors_);


  // Initialize V_ and Sigma_inv_
  V_.SetRandn();
  Sigma_inv_.CopyFromSp(stats_.scatter_);
  Sigma_inv_.Scale(1.0 / stats_.num_examples_);
  Sigma_inv_.Invert();
}

void PldaEstimator::Estimate(const PldaEstimationConfig &config,
                             Plda *plda) {
  KALDI_ASSERT(stats_.num_examples_ > 0 && "Cannot estimate with no stats");
  for (int32 i = 0; i < config.num_em_iters; i++) {
    KALDI_LOG << "Plda estimation iteration " << i
              << " of " << config.num_em_iters;
    EstimateOneIter();
  }
  GetOutput(plda);
}

void PldaEstimator::EstimateOneIter() {
  ResetPerIterStats();
  AccumulateStats();
  EstimateFromStats();
}

void PldaEstimator::ResetPerIterStats() {
  Y_.Resize(Dim(), num_factors_);
  R_.Resize(num_factors_);
  W_.Resize(num_factors_);
}

void PldaEstimator::AccumulateStats() {
  SpMatrix<double> Vt_SigInv_V(num_factors_);
  Matrix<double> Vt_SigInv(num_factors_, Dim());
  Vt_SigInv_V.AddMat2Sp(1.0, V_, kTrans, Sigma_inv_, 0.0);
  Vt_SigInv.AddMatSp(1.0, V_, kTrans, Sigma_inv_, 0.0);


  SpMatrix<double> Li_inv(num_factors_);
  Vector<double> wi(num_factors_);
  SpMatrix<double> wi_wit(num_factors_);
  Vector<double> Vt_SigInv_fi(num_factors_);
  int32 n = -1;// the current number of examples for the class.

  //E-step
  for (size_t i = 0; i < stats_.class_info_.size(); i++) {
    const ClassInfo &info = stats_.class_info_[i];
    if (info.num_examples != n) {
      n = info.num_examples;
      Li_inv.CopyFromSp(Vt_SigInv_V);
      Li_inv.Scale(n);
      //Add Identity mat
      for (size_t j = 0; j < num_factors_; j++) {
        Li_inv(j, j) += 1.0;
      }

      Li_inv.Invert();
    }
    
    Vt_SigInv_fi.AddMatVec(1.0, Vt_SigInv, kNoTrans, *(info.stat_1th), 0.0);
    wi.AddSpVec(1.0, Li_inv, Vt_SigInv_fi, 0.0);
    wi_wit.CopyFromSp(Li_inv);
    wi_wit.AddVec2(1.0, wi);

    // accumulate stats for M-step
    Y_.AddVecVec(1.0, *(info.stat_1th), wi);
    R_.AddSp(info.num_examples, wi_wit);
    W_.AddSp(1.0, wi_wit);
  }
}

// M-step and (MD-step)
void PldaEstimator::EstimateFromStats() {
  // M-step
  // Update V_ and Sigma_inv_
  SpMatrix<double> R_inv(R_);
  R_inv.Invert();
  V_.AddMatSp(1.0, Y_, kNoTrans, R_inv, 0.0);
  Sigma_inv_.CopyFromSp(stats_.scatter_);
  // Sigma_inv_.Scale(1.0 / stats_.num_examples_);
  Sigma_inv_.AddMat2Sp(-1.0, V_, kNoTrans, R_, 1.0);
  Sigma_inv_.Scale(1.0 / stats_.num_examples_);
  Sigma_inv_.Invert();

  // no improvement at all.
  // MD-step (minimum divergence step)
  // W_.Scale(1.0 / stats_.num_classes_);
  // TpMatrix<double> W_half(num_factors_);
  // W_half.Cholesky(W_);
  // Matrix<double> V_temp(Dim(), num_factors_);
  // V_temp.AddMatTp(1.0, V_, kNoTrans, W_half, kNoTrans, 0.0);
  // V_.CopyFromMat(V_temp);

}



void PldaEstimator::GetOutput(Plda *plda) {
  plda->mean_ = stats_.mean_;
  KALDI_LOG << "Norm of mean of iVector distribution is "
            << plda->mean_.Norm(2.0);

  Matrix<double> transform1(Dim(), Dim());

  SpMatrix<double> within_var(Sigma_inv_);
  within_var.Invert();
  
  //KALDI_WARN << "within_var: " << within_var;

  ComputeNormalizingTransform(within_var, &transform1);
  // now transform is a matrix that if we project with it,
  // within_var becomes unit.

  // between_var_proj is between_var after projecting with transform1.
  SpMatrix<double> between_var_proj(Dim());
  SpMatrix<double> between_var(Dim());
  between_var.AddMat2(1.0, V_, kNoTrans, 0.0);
  between_var_proj.AddMat2Sp(1.0, transform1, kNoTrans, between_var, 0.0);

  Matrix<double> U(Dim(), Dim());
  Vector<double> s(Dim());
  // Do symmetric eigenvalue decomposition between_var_proj = U diag(s) U^T,
  // where U is orthogonal.
  between_var_proj.Eig(&s, &U);

  // KALDI_WARN << s;

  int32 n;
  s.ApplyFloor(0.0, &n);
  if (n > 0) {
    KALDI_WARN << "Floored " << n << " eigenvalues of between-class "
               << "variance to zero.";
  }
  // Sort from greatest to smallest eigenvalue.
  SortSvd(&s, &U);

  // The transform U^T will make between_var_proj diagonal with value s
  // (i.e. U^T U diag(s) U U^T = diag(s)).  The final transform that
  // makes within_var_ unit and between_var_ diagonal is U^T transform1,
  // i.e. first transform1 and then U^T.

  plda->transform_.Resize(Dim(), Dim());
  plda->transform_.AddMatMat(1.0, U, kTrans, transform1, kNoTrans, 0.0);
  plda->psi_ = s;

  KALDI_LOG << "Diagonal of between-class variance in normalized space is " << s;

  // if (GetVerboseLevel() >= 2) { // at higher verbose levels, do a self-test
  //                               // (just tests that this function does what it
  //                               // should).
  //   SpMatrix<double> tmp_within(Dim());
  //   tmp_within.AddMat2Sp(1.0, plda->transform_, kNoTrans, within_var_, 0.0);
  //   KALDI_ASSERT(tmp_within.IsUnit(0.0001));
  //   SpMatrix<double> tmp_between(Dim());
  //   tmp_between.AddMat2Sp(1.0, plda->transform_, kNoTrans, between_var_, 0.0);
  //   KALDI_ASSERT(tmp_between.IsDiagonal(0.0001));
  //   Vector<double> psi(Dim());
  //   psi.CopyDiagFromSp(tmp_between);
  //   AssertEqual(psi, plda->psi_);
  // }
  plda->ComputeDerivedVars();
}

} // namespace kaldi
