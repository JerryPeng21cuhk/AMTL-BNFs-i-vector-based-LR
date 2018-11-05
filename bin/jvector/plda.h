// ivector/plda.h

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

#ifndef KALDI_JVECTOR_PLDA_H_
#define KALDI_JVECTOR_PLDA_H_

#include <vector>
#include <algorithm>
#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "gmm/model-common.h"
#include "gmm/diag-gmm.h"
#include "gmm/full-gmm.h"
#include "itf/options-itf.h"
#include "util/common-utils.h"

namespace kaldi {

/* This code implements simplified PLDA For the details, please refer paper:
    Unifying Probabilistic Linear Discriminant Analysis Variants in Biometric
    Authentication.

   This implementation of PLDA supports estimating with a between-class
   dimension less than the feature dimension. */


struct PldaConfig {
  // This config is for the application of PLDA as a transform to iVectors,
  // prior to dot-product scoring.
  bool normalize_length;
  bool simple_length_norm;
  PldaConfig(): normalize_length(true), simple_length_norm(false) { }
  void Register(OptionsItf *opts) {
    opts->Register("normalize-length", &normalize_length,
                   "If true, do length normalization as part of PLDA (see "
                   "code for details).  This does not set the length unit; "
                   "by default it instead ensures that the inner product "
                   "with the PLDA model's inverse variance (which is a "
                   "function of how many utterances the iVector was averaged "
                   "over) has the expected value, equal to the iVector "
                   "dimension.");

    opts->Register("simple-length-normalization", &simple_length_norm,
                   "If true, replace the default length normalization by an "
                   "alternative that normalizes the length of the iVectors to "
                   "be equal to the square root of the iVector dimension.");
  }
};


class Plda {
 public:
  Plda() { }

  explicit Plda(const Plda &other):
    mean_(other.mean_),
    transform_(other.transform_),
    psi_(other.psi_),
    offset_(other.offset_) {
  };

  // Initialize a new plda by interpolating two pldas
  Plda(const Plda &plda1, const Plda &plda2, const float interpolate_scalar);

  // Deprecated 
  // The performance is really bad
  // Initialize a new plda by interpolating two pldas
  // This is heuristically inspired and is not correct in mathmatics
  // void Plda_init2(const Plda &plda1, const Plda &plda2, const float interpolate_scalar);

  /// Transforms an iVector into a space where the within-class variance
  /// is unit and between-class variance is diagonalized.  The only
  /// anticipated use of this function is to pre-transform iVectors
  /// before giving them to the function LogLikelihoodRatio (it's
  /// done this way for efficiency because a given iVector may be
  /// used multiple times in LogLikelihoodRatio and we don't want
  /// to repeat the matrix multiplication
  ///
  /// If config.normalize_length == true, it will also normalize the iVector's
  /// length by multiplying by a scalar that ensures that ivector^T inv_var
  /// ivector = dim.  In this case, "num_examples" comes into play because it
  /// affects the expected covariance matrix of the iVector.  The normalization
  /// factor is returned, even if config.normalize_length == false, in which
  /// case the normalization factor is computed but not applied.
  /// If config.simple_length_normalization == true, then an alternative
  /// normalization factor is computed that causes the iVector length
  /// to be equal to the square root of the iVector dimension.
  double TransformIvector(const PldaConfig &config,
                          const VectorBase<double> &ivector,
                          int32 num_examples,
                          VectorBase<double> *transformed_ivector) const;

  /// float version of the above (not BaseFloat because we'd be implementing it
  /// twice for the same type if BaseFloat == double).
  float TransformIvector(const PldaConfig &config,
                         const VectorBase<float> &ivector,
                         int32 num_examples,
                         VectorBase<float> *transformed_ivector) const;

  /// Returns the log-likelihood ratio
  /// log (p(test_ivector | same) / p(test_ivector | different)).
  /// transformed_train_ivector is an average over utterances for
  /// that speaker.  Both transformed_train_vector and transformed_test_ivector
  /// are assumed to have been transformed by the function TransformIvector().
  /// Note: any length normalization will have been done while computing
  /// the transformed iVectors.
  double LogLikelihoodRatio_ScoreAverage(
                const VectorBase<double> &transformed_train_ivector, // average
                const VectorBase<double> &transformed_train_ivector_sq, //average of elementwise squared
                const VectorBase<double> &transformed_test_ivector, // average
                const VectorBase<double> &transformed_test_ivector_sq //average of elementwise squared
                        ) const;

  double LogLikelihoodRatio_IvectorAverage(
                const VectorBase<double> &transformed_train_ivector, //average
                const VectorBase<double> &transformed_test_ivector //average
                        ) const;

  // Newly developed
  double LogLikelihoodRatio_MultiVsMulti(const VectorBase<double> &transformed_train_ivector,
                            int32 num_train_utts,
                            const VectorBase<double> &transformed_test_ivector,
                            int32 num_test_utts)
                            const;

  // The original kaldi scoring func, deprecated
  // It is equivalent with LogLikelihoodRatio_MultiVsMulti with num_test_utts = 1
  double LogLikelihoodRatio_1vsMany(const VectorBase<double> &transformed_train_ivector,
                            int32 num_train_utts,
                            const VectorBase<double> &transformed_test_ivector)
                            const;
  /// This function smooths the within-class covariance by adding to it,
  /// smoothing_factor (e.g. 0.1) times the between-class covariance (it's
  /// implemented by modifying transform_).  This is to compensate for
  /// situations where there were too few utterances per speaker get a good
  /// estimate of the within-class covariance, and where the leading elements of
  /// psi_ were as a result very large.
  void SmoothWithinClassCovariance(double smoothing_factor);

  /// Apply a transform to the PLDA model.  This is mostly used for
  /// projecting the parameters of the model into a lower dimensional space,
  /// i.e. in_transform.NumRows() <= in_transform.NumCols(), typically for
  /// speaker diarization with a PCA transform.
  void ApplyTransform(const Matrix<double> &in_transform);

  int32 Dim() const { return mean_.Dim(); }
  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);
 protected:
  void ComputeDerivedVars(); // computes offset_.
  friend class PldaEstimator;
  friend class PldaUnsupervisedAdaptor;

  Vector<double> mean_;  // mean of samples in original space.
  Matrix<double> transform_; // of dimension Dim() by Dim();
                             // this transform makes within-class covar unit
                             // and diagonalizes the between-class covar.
  Vector<double> psi_; // of dimension Dim().  The between-class
                       // (diagonal) covariance elements, in decreasing order.

  Vector<double> offset_;  // derived variable: -1.0 * transform_ * mean_

 private:
  Plda &operator = (const Plda &other);  // disallow assignment

  /// This returns a normalization factor, which is a quantity we
  /// must multiply "transformed_ivector" by so that it has the length
  /// that it "should" have.  We assume "transformed_ivector" is an
  /// iVector in the transformed space (i.e., mean-subtracted, and
  /// multiplied by transform_).  The covariance it "should" have
  /// in this space is \Psi + I/num_examples.
  double GetNormalizationFactor(const VectorBase<double> &transformed_ivector,
                                int32 num_examples) const;

};



class PldaStats {
 public:
  PldaStats(): dim_(0) { } /// The dimension is set up the first time you add samples.

  /// This function adds training samples corresponding to
  /// one class (e.g. a speaker).  Each row is a separate
  /// sample from this group.  The "weight" would normally
  /// be 1.0, but you can set it to other values if you want
  /// to weight your training samples.
  void AddSamples(const Matrix<double> &group);

  // Adjust stats after loading all samples
  void ComputeStats();

  int32 Dim() const { return dim_; }

  int32 GetNumClasses() const { return num_classes_; }

  void Init(int32 dim);

  void Sort() { std::sort(class_info_.begin(), class_info_.end()); }
  bool IsSorted() const;
  ~PldaStats();
 protected:

  friend class PldaEstimator;

  int32 dim_; // feat dim (dim of ivector)
  int64 num_classes_;
  int64 num_examples_; // total number of examples, summed over classes.
  // double class_weight_; // total over classes, of their weight.
  // double example_weight_; // total over classes, of weight times #examples.



  Vector<double> mean_; // mean of all examples.

  SpMatrix<double> scatter_; // Sum over all examples of
                            // (example - class-mean).

  // We have one of these objects per class.
  struct ClassInfo {
    // Vector<double> *mean; // owned here, but as a pointer so
    //                       // sort can be lightweight

    Vector<double>* stat_1th; // centered 1st-order statistics
    int32 num_examples; // the number of examples in the class

    bool operator < (const ClassInfo &other) const {
      return (num_examples < other.num_examples);
    }
    ClassInfo(Vector<double> *stat_1th, int32 num_examples):
        stat_1th(stat_1th), num_examples(num_examples) { }

  };

  std::vector<ClassInfo> class_info_;
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(PldaStats);
};



//TODO:

struct PldaEstimationConfig {
  int32 num_em_iters;
  int32 num_factors;
  PldaEstimationConfig(): num_em_iters(10), num_factors(10) { }
  void Register(OptionsItf *opts) {
    opts->Register("num-em-iters", &num_em_iters,
                   "Number of iterations of E-M used for PLDA estimation");
    opts->Register("num-factors", &num_factors,
                   "Number of factors used for estimate loading matrix");
  }
};

class PldaEstimator {
 public:
  PldaEstimator(const PldaEstimationConfig &config,
                const PldaStats &stats);

  void Estimate(const PldaEstimationConfig &config,
                Plda *output);

private:
  typedef PldaStats::ClassInfo ClassInfo;

  // /// Returns the part of the objf relating to
  // /// offsets from the class means.  (total, not normalized)
  // double ComputeObjfPart1() const;

  // /// Returns the part of the obj relating to
  // /// the class means (total_not normalized)
  // double ComputeObjfPart2() const;

  // /// Returns the objective-function per sample.
  // double ComputeObjf() const;

  int32 Dim() const { return stats_.Dim(); }

  int32 GetMaxWithinMatRank() const {
    if (stats_.GetNumClasses() < stats_.Dim())
      return stats_.GetNumClasses();
    else
      return stats_.Dim();
  };

  void InitParameters(const PldaEstimationConfig &config);

  void EstimateOneIter();

  void ResetPerIterStats();

  // E-step
  void AccumulateStats();

  // M-step
  void EstimateFromStats();

  // // MD-step
  // void WhitenProjMat();



  // Copy to output. Do diagnolization
  void GetOutput(Plda *plda);

  const PldaStats &stats_;

  // SpMatrix<double> within_var_;
  Matrix<double> V_; //
  SpMatrix<double> Sigma_inv_;

  // These stats are reset on each iteration.
  // SpMatrix<double> within_var_stats_;
  // double within_var_count_; // count corresponding to within_var_stats_
  // SpMatrix<double> between_var_stats_;
  // double between_var_count_; // count corresponding to within_var_stats_

  // See Formal.pdf for details
  // These are used to update V_ and 
  Matrix<double> Y_;
  SpMatrix<double> R_;

  // Used to to whitening V_
  SpMatrix<double> W_;

  int32 num_factors_; // size of V_ is Dim()-by-num_factors

  KALDI_DISALLOW_COPY_AND_ASSIGN(PldaEstimator);
};


}  // namespace kaldi

#endif
