// jvector/jvector-extractor.h

// Revised 2018 Jerry 

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

#ifndef KALDI_JVECTOR_JVECTOR_EXTRACTOR_H_
#define KALDI_JVECTOR_JVECTOR_EXTRACTOR_H_

#include <vector>
#include <mutex>
#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "gmm/model-common.h"
#include "gmm/diag-gmm.h"
#include "gmm/full-gmm.h"
#include "itf/options-itf.h"
#include "util/common-utils.h"
#include "hmm/posterior.h"
#include "ivector/ivector-extractor.h"

namespace kaldi {

// Note, throughout this file we use SGMM-type notation because
// that's what I'm comfortable with.
// Dimensions:
//  D is the feature dim (e.g. D = 60)
//  I is the number of Gaussians (e.g. I = 2048)
//  S is the ivector dim (e.g. S = 400)

struct JvectorExtractorUtteranceStats {
  JvectorExtractorUtteranceStats(int32 num_gauss, int32 feat_dim):
      gamma(num_gauss), X(num_gauss, feat_dim), 
      X_centre(num_gauss, feat_dim) { 
        S.resize(num_gauss);
        for (int32 i = 0; i < num_gauss; i++) {
          S[i].Resize(feat_dim);
        }
      }

  Vector<double> gamma; // zeroth-order stats (summed posteriors), dimension [I]
  Matrix<double> X; // first-order stats, dimension [I][D]
  Matrix<double> X_centre; // first-order stats(centred), dimension [I][D]
  std::vector<SpMatrix<double> > S; // second-order stats. dimension [I][D][D]
};


struct JvectorExtractorOptions {
  int jvector_dim;
  JvectorExtractorOptions(): jvector_dim(400) { }
  void Register(OptionsItf *po) {
    po->Register("jvector-dim", &jvector_dim, "the j-vector subspace dimension");
  }
};

struct JvectorExtractorDecodeOptions {
  float min_post;
  int32 num_gselect;
  JvectorExtractorDecodeOptions(): 
    min_post(0.0025), num_gselect(20) { }
  void Register(OptionsItf *po) {
    po->Register("min-post", &min_post, "threshold for posterior decoding. Under min-post will be set to 0.");
    po->Register("num-gselect", &num_gselect, "Num of maximum selected components for each frame.");
  }
};


/// Options for training the JvectorExtractor, e.g. variance flooring.
struct JvectorExtractorEstimationOptions {
  double gaussian_min_count;
  JvectorExtractorEstimationOptions(double gaussian_min_count_): gaussian_min_count(gaussian_min_count_) { }
  JvectorExtractorEstimationOptions(): gaussian_min_count(100.0) { }
  void Register(OptionsItf *po) {
    po->Register("gaussian-min-count", &gaussian_min_count,
                 "Minimum total count per Gaussian, below which we refuse to "
                 "update any associated parameters.");
  }
};


// Caution: the JvectorExtractor is not the only thing required
// to get an ivector.  We also need to get posteriors from a
// FullGmm.  Typically these will be obtained in a process that involves
// using a DiagGmm for Gaussian selection, followed by getting
// posteriors from the FullGmm.  To keep track of these, we keep
// them all in the same directory, e.g. final.{ubm,dubm,ie}

class JvectorExtractor {

 public:
  friend class JvectorStats;

  JvectorExtractor() {};

  JvectorExtractor(
      const JvectorExtractorOptions &opts,
      const FullGmm &fgmm);

  JvectorExtractor(
      const FullGmm &fgmm,
      const IvectorExtractor &ivector_extractor
      );

  /// Gets the distribution over ivectors (or at least, a Gaussian approximation
  /// to it).  The output "var" may be NULL if you don't need it.  "mean", and
  /// "var", if present, must be the correct dimension (this->JvectorDim()).
  /// If you only need a point estimate of the iVector, get the mean only.
  void GetJvectorDistribution(
      const JvectorExtractorUtteranceStats &utt_stats,
      VectorBase<double> *mean,
      SpMatrix<double> *var) const;

  
  /// Gets the linear and quadratic terms in the distribution over iVectors, but
  /// only the terms arising from the Gaussian means (i.e. not the weights
  /// or the priors).
  /// Setup is log p(x) \propto x^T linear -0.5 x^T quadratic x.
  /// This function *adds to* the output rather than setting it.
  void GetJvectorDistMean(
      const JvectorExtractorUtteranceStats &utt_stats,
      VectorBase<double> *linear,
      SpMatrix<double> *quadratic) const;

  /// Adds to "stats", which are the zeroth and 1st-order
  /// stats (they must be the correct size).
  void GetStats(const MatrixBase<BaseFloat> &feats,
                const Posterior &post,
                JvectorExtractorUtteranceStats *stats) const;

  // Added by jerryPeng: Use i-vector to decode posteriors
  // Input the features of an utterance, 
  // Get the posterior of an utterance (vector<vector<pair<int32, BaseFloat> > >)
  // Return the posterior of an utterance
  // The posterior should be stored to harddish for next iteration
  BaseFloat GetJvectorPosteriors(
      const JvectorExtractorDecodeOptions &opts,
      const VectorBase<double> &mean,
      const SpMatrix<double> &var,
      const MatrixBase<BaseFloat> &feats,
      Posterior *post
      ) const;

  void DecodeJvectorAndPosterior(
      const Matrix<BaseFloat> &feats,
      const Posterior &ipost,
      JvectorExtractorDecodeOptions &decode_opts,
      Vector<double> *jvector,
      Posterior *opost
    ) const;

  void DecodeJvector(
      const Matrix<BaseFloat> &feats,
      const Posterior &ipost,
      Vector<double> *jvector
    ) const;



  int32 FeatDim() const;
  int32 JvectorDim() const;
  int32 NumGauss() const;

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  // Note: we allow the default assignment and copy operators
  // because they do what we want.
 protected:
  void ComputeDerivedVars();

  /// If we are not using weight-projection vectors, stores the Gaussian mixture
  /// weights from the UBM.  This does not affect the iVector; it is only useful
  /// as a way of making sure the log-probs are comparable between systems with
  /// and without weight projection matrices.
  Vector<double> gweights_;
  
  /// Jvector-subspace projection matrices, dimension is [I][D][S].
  /// The I'th matrix projects from ivector-space to Gaussian mean.
  /// There is no mean offset to add-- we deal with it by having
  /// a prior with a nonzero mean.
  std::vector<Matrix<double> > T_;

  /// Means and Inverse variances of speaker-adapted model, dimension [I][D][D].
  Matrix<double> gmeans_;

  std::vector<SpMatrix<double> > gincovars_;
  
  // Below are *derived variables* that can be computed from the
  // variables above.
  
  // log(pi_i) + 0.5*log|Sigma_inv_i| - 0.5 * Means_i'*Sigma_inv_i*Means_i
  Vector<double> gconsts_;

  // Each row of Sigma_inv_mat_ is a vectorization of Sigma_inv_;
  //Matrix<double> Sigma_inv_mat_;
  Matrix<float> gincovars_mat_;

  /// U_i = M_i^T \Sigma_i^{-1} M_i is a quantity that comes up
  /// in ivector estimation.  This is conceptually a
  /// std::vector<SpMatrix<double> >, but we store the packed-data 
  /// in the rows of a matrix, which gives us an efficiency 
  /// improvement (we can use matrix-multiplies).
  Matrix<double> U_mat_;

  // V_i = Means_i' * Sigma_inv_i * M_i;
  // In V_, each row is a vector: V_i
  Matrix<double> V_mat_;
};





/// JvectorStats is a class used to update the parameters of the ivector estimator.
class JvectorStats {
 public:
  friend class JvectorExtractor;

  //Warning:  must use read func to initialize it after this.
  JvectorStats() { };
  
  JvectorStats(const JvectorExtractor &extractor);
  
  float AccStatsForUtterance(const JvectorExtractor &extractor,
                            const MatrixBase<BaseFloat> &feats,
                            const JvectorExtractorDecodeOptions &decode_opts,
                            const Posterior &ipost,
                            Posterior *opost);

  void Read(std::istream &is, bool binary, bool add = false);

  void Write(std::ostream &os, bool binary) const;

  /// Returns the objf improvement per frame.
  double Update(const JvectorExtractorEstimationOptions &opts,
                JvectorExtractor *extractor) const;

  
  // Note: we allow the default assignment and copy operators
  // because they do what we want.
 protected:

  
  // This is called by AccStatsForUtterance
  float CommitStatsForUtterance(const JvectorExtractor &extractor,
                               const JvectorExtractorUtteranceStats &utt_stats,
                               const JvectorExtractorDecodeOptions &decode_opts,
                               const MatrixBase<BaseFloat> &feats,
                               Posterior *opost);




  /// This is called by CommitStatsForUtterance.  We commit the stats
  /// used to update the T matrix.
  void AccStats(const JvectorExtractor &extractor,
                       const JvectorExtractorUtteranceStats &utt_stats,
                       const VectorBase<double> &ivec_mean,
                       const SpMatrix<double> &ivec_var);
  
  // Updates M.  Returns the objf improvement per frame.
  double UpdateProjections(const JvectorExtractorEstimationOptions &opts,
                     JvectorExtractor *extractor) const;

  // This internally called function returns the objf improvement
  // for this Gaussian index.  Updates one M.
  double UpdateProjection(const JvectorExtractorEstimationOptions &opts,
                          int32 gaussian,
                          JvectorExtractor *extractor) const;

  void UpdateGweights(JvectorExtractor *extractor) const;

  void UpdateGmeans(const JvectorExtractorEstimationOptions &opts,
                     JvectorExtractor *extractor) const;

  void UpdateGincovars(const JvectorExtractorEstimationOptions &opts,
                JvectorExtractor *extractor) const;

  void JvectorVarianceDiagnostic(const JvectorExtractor *extractor) const;

  void WhitenProjections(JvectorExtractor *extractor) const;

  void CheckDims(const JvectorExtractor &extractor) const;

  /// Total auxiliary function over the training data-- can be
  /// used to check convergence, etc.

  /// This mutex guards gamma_, Y_ and R_ (for multi-threaded
  /// update)
  std::mutex subspace_stats_lock_; 
  
  /// Total occupation count for each Gaussian index (zeroth-order stats)
  Vector<double> gamma_;

  /// num of j-vectors
  /// Make it double to avoid overflow and easy weighted.
  double num_jvectors_;

  /// sum of all j-vectors
  Vector<double> jvector_1th_;

  /// sum of all j-vectors^2
  SpMatrix<double> jvector_2th_;

  /// first order occupation (centre) count for each Gaussian index
  Matrix<double> acc_1th_centre;

  /// second order occupation (centre) count for each Gaussian index
  std::vector<SpMatrix<double> > acc_2th_centre;

  /// first order icovars for each gaussian index
  std::vector<SpMatrix<double> > icovars_1th;

  /// Stats Y_i for estimating projections M.  Dimension is [I][D][S].  The
  /// linear term in M.
  std::vector<Matrix<double> > Y_;
  
  /// R_i, quadratic term for ivector subspace (M matrix)estimation.  This is a
  /// kind of scatter of ivectors of training speakers, weighted by count for
  /// each Gaussian.  Conceptually vector<SpMatrix<double> >, but we store each
  /// SpMatrix as a row of R_.  Conceptually, the dim is [I][S][S]; the actual
  /// dim is [I][S*(S+1)/2].
  Matrix<double> R_;

};



}  // namespace kaldi


#endif

