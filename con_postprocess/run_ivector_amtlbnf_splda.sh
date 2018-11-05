#!/bin/bash
# 2018 jerrypeng

# Do post process and evaluation for raw ivector based on amtl-bnf 
# note that there raw ivectors are not either normalized, centerized or whitened explicitly

# Based on the experiment result of script run_conivector_eval_tryWCCNLDAPLDA.sh
# We use the following strategy to do postprocessing

# 1. Centering train and test set individually
# 2. Whitening train and test set from whiten matrix 
# 3. Skip LDA
# 4. Train simp-PLDA#1 on training ivectors
# 5. Unsupervised PLDA adaptation on test ivectors
# 6 .final scoring on test ivectors



set -e
set +o posix

# Number of components
cnum=2048 # num of Gaussions
civ=400 # dim of i-vector
clda=9 # dim for i-vector with LDA



#testdir=test_1s
testdir=dev_1s
devdir=dev_1s

exp=exp/bnf_ivector_no_cmn



# test_dir=$exp/ivectors_task_1_2048_400_no_cmn

train_dir=$exp/ivectors_train_2048_400_no_cmn

test_dir=$exp/ivectors_dev_1s_2048_400_no_cmn



# trials="python local/olr/eval/create_trials.py $train_dir/spk_ivector.scp $test_dir/ivector.scp -"
# trials="python local/olr/eval/create_trials.py $train_dir/lan2utt $test_dir/utt2subutt -"
# trials="python local/olr/eval/create_trials_notarget.py $train_dir/lan2utt $test_dir/ivector.scp -"
trials="python local/olr/eval/create_trials.py $train_dir/lan2utt $test_dir/ivector.scp -"
mkdir -p $exp

stage=2
. ./cmd.sh
. ./path.sh
# . ./conivectorpath.sh
. ./jvectorpath.sh


set -e
. ./utils/parse_options.sh


mkdir -p $exp/score/total


if [ $stage -le 1 ]; then

  echo ">> Centering training data"
  ivector-subtract-global-mean scp:$train_dir/ivector.scp \
     ark,scp:$train_dir/ivector_centered.ark,$train_dir/ivector_centered.scp

  echo ">> Centering test data"
  ivector-subtract-global-mean scp:$test_dir/ivector.scp \
     ark,scp:$test_dir/ivector_centered.ark,$test_dir/ivector_centered.scp


  echo ">> Computing wccn matrix from training data"
  ivector-conv-compute-wccn --total-covariance-factor=0.0 \
    "ark:ivector-normalize-length scp:$train_dir/ivector_centered.scp ark:- |" \
    ark:$train_dir/utt2lan \
    $train_dir/wccn.mat

  # # # This step actually is cheating as we uses labels
  # # # In the future, we can train unsupervised wccn by setting total-covariance-factor=1.0 which actually do not require labels
  # # # So we can give it a fake utt2spk
  # # echo ">> Computing wccn matrix from test data"
  # # ivector-conv-compute-wccn --total-covariance-factor=1.0 \
  # #   "ark:ivector-normalize-length scp:$test_dir/ivector_centered.scp ark:- |" \
  # #   ark:data/${testdir}/utt2spk \
  # #   $test_dir/wccn.mat

  echo ">> Whitening training ivector by training wccn.mat"
  ivector-transform $train_dir/wccn.mat \
    "ark:ivector-normalize-length scp:$train_dir/ivector_centered.scp ark:- |" ark:- | \
    ivector-normalize-length ark:- ark:$train_dir/wccn_ivector_by_train.ark 

  echo ">> Whitening $testdir ivector by training wccn.mat"
  ivector-transform $train_dir/wccn.mat \
    "ark:ivector-normalize-length scp:$test_dir/ivector_centered.scp ark:- |" ark:- | \
    ivector-normalize-length ark:- ark:$test_dir/wccn_ivector_by_train.ark

  # train_spk_ivector="ark:ivector-mean ark:${train_dir}/lan2utt ark:$train_dir/ivector_centered.ark ark:- | ivector-normalize-length ark:- ark:-|"
  # test_utt_ivector="ark:ivector-normalize-length ark:${test_dir}/ivector_centered.ark ark:-|"

  # cat <($trials) | awk '{print $1, $2}' | \
  #   ivector-compute-dot-products - \
  #     "$train_spk_ivector" "$test_utt_ivector" $exp/score/total/foo_wccn

  # echo "i-vector with WCCN(whitening) "
  # echo 
  # printf '% 16s' 'EER% is:'
  # eer=$(awk '{print $3}' $exp/score/total/foo_wccn | paste - <($trials) | awk '{print $1, $4}' | compute-eer - 2>/dev/null)
  # printf '% 5.2f' $eer

  # python local/olr/ivector/Compute_Cavg.py  $exp/score/total/foo_wccn data/${testdir}/utt2lan

fi

# Try my simplified PLDA trained on test set
# Try different scoring scheme
if [ $stage -le 2 ]; then

  score_mode=ivectoraverage
  echo ">> Try ivector + simplified PLDA + $score_mode"

  # PLDA trained on train set
  jvector-compute-plda --num-em-iters=10 --num-factors=9 ark:${train_dir}/lan2utt \
    ark:$train_dir/wccn_ivector_by_train.ark \
    $train_dir/simp_plda 2>$train_dir/log/simp_plda.log

  # jvector-plda-scoring --simple-length-normalization=true --score-mode=$score_mode \
  #   $train_dir/simp_plda \
  #   ark:${train_dir}/lan2utt \
  #   ark:$test_dir/utt2subutt \
  #   ark:$train_dir/ivector_centered.ark \
  #   ark:$test_dir/ivector_centered.ark \
  #   "$trials | awk '{print \$1, \$2}' |" $exp/score/total/foo_simp_plda || exit 1;

  # echo "L-vector (i-vector with simp-PLDA trained on train set with complete linkage) 2-iter"
  # echo 
  # printf '% 16s' 'EER% is:'

  # # eer=$(awk '{print $3}' $exp/score/total/foo_simp_plda | paste - <($trials) | awk '{print $1, $4}' | compute-eer - 2>/dev/null)
  # # printf '% 5.2f' $eer
  # eer=$(python local/olr/eval/lid_softmax_score.py $exp/score/total/foo_simp_plda | paste - <($trials) | awk '{print $1, $4}' | compute-eer - 2>/dev/null)
  # printf '% 5.2f' $eer


  # python local/olr/ivector/Compute_Cavg.py  $exp/score/total/foo_simp_plda data/${testdir}/subutt2spk


  jvector-plda-pairwise-scoring-parallel --num-threads=20 --simple-length-normalization=true \
    $train_dir/simp_plda \
    ark:${test_dir}/wccn_ivector_by_train.ark \
    $train_dir/score_bytrain.mat


  jvector-agglomerative-cluster \
    --num-clusters=100 --read-costs=false --linkage=complete \
    $train_dir/score_bytrain.mat \
    ark:$test_dir/utt2utt \
    ark:$test_dir/utt2cluster_bytrain_lkg_complete_100

  copy-spk2cluster ark:$test_dir/utt2cluster_bytrain_lkg_complete_100 ark,t:$test_dir/utt2cluster_bytrain_lkg_complete_100.txt

  utils/utt2spk_to_spk2utt.pl $test_dir/utt2cluster_bytrain_lkg_complete_100.txt > $test_dir/cluster2utt_bytrain_lkg_complete_100

  jvector-compute-plda --num-em-iters=10 --num-factors=10 ark:${test_dir}/cluster2utt_bytrain_lkg_complete_100 \
    ark:$test_dir/wccn_ivector_by_train.ark \
    $train_dir/simp_plda_adapt_lkg_complete 2>$test_dir/log/simp_plda_adapt_lkg_complete.log

  jvector-plda-scoring --simple-length-normalization=true --score-mode=$score_mode \
    $train_dir/simp_plda_adapt_lkg_complete \
    ark:${train_dir}/lan2utt \
    ark:$test_dir/utt2utt \
    ark:$train_dir/wccn_ivector_by_train.ark \
    ark:$test_dir/wccn_ivector_by_train.ark \
    "$trials | awk '{print \$1, \$2}' |" $exp/score/total/foo_simp_plda_train_lkg_complete || exit 1;

  echo "L-vector (i-vector with adapted-simp-PLDA with complete linkage)"
  echo 
  printf '% 16s' 'EER% is:'
  eer=$(python local/olr/eval/lid_softmax_score.py $exp/score/total/foo_simp_plda_train_lkg_complete | paste - <($trials) | awk '{print $1, $4}' | compute-eer - 2>/dev/null)
  # eer=$(awk '{print $3}' $exp/score/total/foo_simp_plda_train_lkg_complete | paste - <($trials) | awk '{print $1, $4}' | compute-eer - 2>/dev/null)
  printf '% 5.2f' $eer

  python local/olr/ivector/Compute_Cavg.py  $exp/score/total/foo_simp_plda_train_lkg_complete data/${testdir}/utt2spk

fi


## deprecated
## This is a failed idea
## The idea is that:
##    Given the clustered test i-vectors and adapted in-domain plda, instead of scoring each i-vector individually, 
##      we can simply assume each ivecotr inside a cluster comes from the same language.
##      So we use adapted plda to score each cluster and each individual ivector inside shares the same score.
##    To do this, we need to gurantee a relative high clustering purity.
##    To compute purity, please refer to con_postprocess/compute_cluster_purity.py

# if [ $stage -le 3 ]; then 

#   trials_notarget="python local/olr/eval/create_trials_notarget.py $train_dir/lan2utt $test_dir/cluster2utt_bytrain2 -"
#   jvector-plda-scoring --simple-length-normalization=true --score-mode=scoreaverage \
#     $train_dir/simp_plda_adapt2 \
#     ark:${train_dir}/lan2utt \
#     ark:$test_dir/cluster2utt_bytrain2 \
#     ark:$train_dir/wccn_ivector_by_train.ark \
#     ark:$test_dir/wccn_ivector_by_train.ark \
#     "$trials_notarget | awk '{print \$1, \$2}' |" \
#     $exp/score/total/foo_simp_plda_train3_scorebynonadatedplda || exit 1;

# script=$(cat <<'EOF'
# if (@ARGV != 3) {
#   # print join(" ", @ARGV), ".\n"
#   print "usage: script utt2cluster trials foo_lan_clusterid_score ; output foo_simp_plda_score to stdout.\n"
# }
# $utt2cluster = $ARGV[0];
# $trials = $ARGV[1];
# $foo_fake = $ARGV[2];

# sub trim($)
# {
#   my $string = shift;
#   $string =~ s/^\s+//;
#   $string =~ s/\s+$//;
#   return $string;
# }

# # create a hash utt2cluster that maps from uttid to clustid
# %utt2clust = ();
# if (open(my $fh_utt2cluster, "<", $utt2cluster)) {
#   while (my $line = <$fh_utt2cluster>) {
#     chomp $line;
#     $line = trim($line);
#     my @A = split(" ", $line);
#     @A == 2 || die "Invalid line in spk2utt file: $line";
#     ($uttid,$cluster) = @A;
#     $utt2clust{$uttid} = $cluster;
#   }
# } else {
#   die "Could not open file '$utt2cluster' $!";
# }

# # read in foo_lan_clusterid_score
# # create a hash lanclust2score that maps from (lanid, clustid) to score
# %lanclust2score = ();
# if (open(my $fh_foo_fake, "<", $foo_fake)) {
#   while (my $line = <$fh_foo_fake>) {
#     chomp $line;
#     $line = trim($line);
#     my @A = split(" ", $line);
#     @A == 3 || die "Invalid line in '$foo_fake' file: $line";
#     ($lanid, $clustid, $score) = @A;
#     $lanclust2score{$lanid}{$clustid} = $score;
#   } 
# } else {
#     die "Could not open file '$foo_fake' $!";
# }


# # read in trials which should contains at least two columns
# # the first column is lanid, the second column is uttid
# # the seperator is whitespace
# # output a valid foo_xxx to std
# # which contains: <lanid> <uttid> <score>

# # the output can be used to do evaluation like Cavg or EER
# if (open(my $fh_trials, "<", $trials)) {
#   while (my $line = <$fh_trials>) {
#     chomp $line;
#     $line = trim($line);
#     my @A = split(" ", $line);
#     @A > 1 || die "Invalid line in '$trials' file: $line";
#     ($lanid, $uttid) = @A[ 0..1 ];
#     $clustid = $utt2clust{$uttid};
#     $score = $lanclust2score{$lanid}{$clustid};
#     print "$lanid $uttid $score\n";
#   }
# } else {
#   die "Could not open file '$trials' $1";
# }
# EOF
# )
# perl -e "$script"  $test_dir/utt2cluster_bytrain2.txt <($trials) $exp/score/total/foo_simp_plda_train3_scorebynonadatedplda > $exp/score/total/foo_simp_plda_train_trash;


# echo "L-vector (i-vector with simp-PLDA trained on train set)"
# echo 
# printf '% 16s' 'EER% is:'

# eer=$(awk '{print $3}' $exp/score/total/foo_simp_plda_train_trash | paste - <($trials) | awk '{print $1, $4}' | compute-eer - 2>/dev/null)
# printf '% 5.2f' $eer

# python local/olr/ivector/Compute_Cavg.py  $exp/score/total/foo_simp_plda_train_trash data/${testdir}/utt2spk



# fi


# This is used to fix the abnormal EER score
# paste $exp/score/total/foo_simp_plda_test_softmax  <($trials) | awk '{print $1, $4}' | compute-eer - 2>/dev/null
