#!/bin/bash
# 2018 jerrypeng

# Use bnf feature to train ubm ivector extractor

set -e
set +o posix

# Number of components
cnum=256 # num of Gaussions
civ=100 # dim of i-vector
# clda=9 # dim for i-vector with LDA



#featdir=_bnf
testdir=dev_1s
#exp=exp/mtlbnfivector
#exp=exp/mfccivector
exp=exp/mfccivector_trainvaliddev1
#exp=exp/mtlbnfivector_trainvaliddev1s


#train_dir=$exp/bst_conivectors_train_${cnum}_${civ}_trim1s_10subutt_10subuttlen_no_cmn
#train_dir=$exp/kaldi_ivectors_train_${cnum}_${civ}_trim1s
#train_dir=$exp/conivectors_train_${cnum}_${civ}_trim1sall_no_cmn
train_dir=$exp/conivectors_train_${cnum}_${civ}_no_cmn_alltrim1s
#train_dir=$exp/bst_conivectors_train_${cnum}_${civ}_10subutt_10subuttlen_no_cmn
#train_dir=$exp/kaldi_ivectors_train_${cnum}_${civ}
#train_dir=$exp/bst_conivectors_train_${cnum}_${civ}_1subutt_10subuttlen_no_cmn
#train_dir=$exp/bst_conivector_train_${cnum}_${civ}_10subutt_10subuttlen_no_cmn
#train_dir=$exp/conivectors_train_${cnum}_${civ}_no_cmn
#train_dir=$exp/kaldi_ivector_train_${cnum}_${civ}_trainedbytraindevall
#outdomain_test_dir=$exp/kaldi_ivector_${testdir}_${cnum}_${civ}_trainedbytraindevall
#outdomain_test_dir=$exp/bst_conivector_${testdir}_${cnum}_${civ}_1subutt_50subuttlen
#outdomain_test_dir=$exp/conivector_${testdir}_${cnum}_${civ}
#outdomain_test_dir=$exp/kaldi_ivectors_${testdir}_${cnum}_${civ}
outdomain_test_dir=$exp/conivectors_${testdir}_${cnum}_${civ}_no_cmn

#indomain_test_dir=$exp/kaldi_ivectors_valid_${cnum}_${civ}_trim1s
#indomain_test_dir=$exp/conivectors_valid_${cnum}_${civ}_trim1sall_no_cmn
#indomain_test_dir=$exp/conivectors_valid_${cnum}_${civ}_no_cmn_trim1s
#indomain_test_dir=$exp/bst_conivectors_valid_${cnum}_${civ}_10subutt_10subuttlen_no_cmn
#indomain_test_dir=$exp/conivectors_valid_${cnum}_${civ}_no_cmn
indomain_test_dir=$exp/conivectors_valid_${cnum}_${civ}_no_cmn_alltrim1s



#trials="python local/olr/eval/create_trials.py $train_dir/spk_ivector.scp $outdomain_test_dir/ivector.scp -"
#trials="python local/olr/eval/create_trials.py $train_dir/spk2subutt $outdomain_test_dir/utt2subutt -"
outdom_trials="python local/olr/eval/create_trials.py $train_dir/spk2subutt $outdomain_test_dir/ivector.scp -"
indom_trials="python local/olr/eval/create_trials.py $train_dir/spk2subutt $indomain_test_dir/ivector.scp -"
mkdir -p $exp
stage=2

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


# Try DNN scoring 

if [ $stage -le 1 ]; then

  echo ">> Try ivector + DNN scoring"
  python3 dnn_postprocess/dnntrain.py "ark,t:copy-vector scp:$train_dir/ivector.scp ark,t:-|" $exp/dnn_lanclassify.mdl
  #python3 dnn_postprocess/dnntrain.py "ark,t:copy-vector scp:$train_dir/ivector.scp ark,t:-|" $exp/dnn_lanclassify.mdl "ark,t:copy-vector scp:$indomain_test_dir/ivector.scp ark,t:-|"

fi


if [ $stage -le 2 ]; then
  python3 dnn_postprocess/dnntest.py $exp/dnn_lanclassify.mdl "ark,t:copy-vector scp:$indomain_test_dir/ivector.scp ark,t:-|" $exp/score/total/foo_dnn_indomain
  python3 dnn_postprocess/dnntest.py $exp/dnn_lanclassify.mdl "ark,t:copy-vector scp:$outdomain_test_dir/ivector.scp ark,t:-|" $exp/score/total/foo_dnn_outdomain
  echo 'ivector with DNN scoring indomain'
  echo 
  printf '% 16s' 'EER% is:'

  eer=$(awk '{print $3}' $exp/score/total/foo_dnn_indomain | paste - <($indom_trials) | awk '{print $1, $4}' | compute-eer - 2>/dev/null)
  printf '% 5.2f' $eer
  python local/olr/ivector/Compute_Cavg.py $exp/score/total/foo_dnn_indomain $indomain_test_dir/subutt2spk



  echo 'ivector with DNN scoring outdomain'
  echo 
  printf '% 16s' 'EER% is:'

  eer=$(awk '{print $3}' $exp/score/total/foo_dnn_outdomain | paste - <($outdom_trials) | awk '{print $1, $4}' | compute-eer - 2>/dev/null)
  printf '% 5.2f' $eer
  python local/olr/ivector/Compute_Cavg.py $exp/score/total/foo_dnn_outdomain data/$testdir/utt2spk


fi


## This is to testify the performance difference between mlp classifier and plda classifier
## The results are consistent to prove the severe performance degradation from domain mismatch
## However, 
##   (1) plda gives a worse result than mlp.
##   (2) plda shows a relative smaller performance gap between in-domain and out-of-domain dev set

##  From the above observation, it guides us to use unsupervised plda adaptation to convert i-vectors## into in-domain low-dimension factor-vectors (this low-dimension can be larger than #language now as there is no limit to #cluster.) Then use mlp to classify the in-domain low-dimension factor-vecotrs.
## However, we have not do the experiment now.


#if [ $stage -le 3 ]; then
#
#  score_mode="ivectoraverage";
#
#  # echo ">> Centering training data"
#  # ivector-subtract-global-mean scp:$train_dir/ivector.scp \
#  #    ark,scp:$train_dir/ivector_centered.ark,$train_dir/ivector_centered.scp
#
#  # echo ">> Centering indomain test data"
#  # ivector-subtract-global-mean scp:$indomain_test_dir/ivector.scp \
#  #   ark,scp:$indomain_test_dir/ivector_centered.ark,$indomain_test_dir/ivector_centered.scp
#
#  # echo ">> Centering outdomain test data"
#  # ivector-subtract-global-mean scp:$outdomain_test_dir/ivector.scp \
#  #   ark,scp:$outdomain_test_dir/ivector_centered.ark,$outdomain_test_dir/ivector_centered.scp
#
#  # echo ">> Whitenning training ivector by training wccn.mat"
#  # ivector-transform $train_dir/wccn.mat \
#  #   "ark:ivector-normalize-length scp:$train_dir/ivector.scp ark:- |" ark:- | \
#  #   ivector-normalize-length ark:- ark:$train_dir/wccn_ivector_by_train.ark
#
#  # echo ">> Whitening indomain test ivector by training wccn.mat"
#  # ivector-transform $train_dir/wccn.mat \
#  #   "ark:ivector-normalize-length scp:$indomain_test_dir/ivector_centered.scp ark:- |" ark:- | \
#  #   ivector-normalize-length ark:- ark:$indomain_test_dir/wccn_ivector_by_train.ark
#
#  # echo ">> Whitening outdomain test ivector by training wccn.mat"
#  # ivector-transform $train_dir/wccn.mat \
#  #   "ark:ivector-normalize-length scp:$outdomain_test_dir/ivector_centered.scp ark:- |" ark:- | \
#  #   ivector-normalize-length ark:- ark:$outdomain_test_dir/wccn_ivector_by_train.ark
#
#  # jvector-compute-plda --num-em-iters=10 --num-factors=9 ark:${train_dir}/spk2subutt \
#  #   ark:$train_dir/wccn_ivector_by_train.ark \
#  #   $train_dir/simp_plda 2>$train_dir/log/simp_plda.log
#
#  jvector-plda-scoring --simple-length-normalization=true --score-mode=$score_mode \
#      $train_dir/simp_plda \
#      ark:${train_dir}/spk2subutt \
#      ark:$indomain_test_dir/utt2subutt \
#      ark:$train_dir/wccn_ivector_by_train.ark \
#      ark:$indomain_test_dir/wccn_ivector_by_train.ark \
#      "$indom_trials | awk '{print \$1, \$2}' |" $exp/score/total/foo_plda_indomain || exit 1;
#
#  jvector-plda-scoring --simple-length-normalization=true --score-mode=$score_mode \
#      $train_dir/simp_plda \
#      ark:${train_dir}/spk2subutt \
#      ark:$outdomain_test_dir/utt2subutt \
#      ark:$train_dir/wccn_ivector_by_train.ark \
#      ark:$outdomain_test_dir/wccn_ivector_by_train.ark \
#      "$outdom_trials | awk '{print \$1, \$2}' |" $exp/score/total/foo_plda_outdomain || exit 1;
#
#  echo 'ivector with plda scoring indomain'
#  echo 
#  printf '% 16s' 'EER% is:'
#
#  eer=$(awk '{print $3}' $exp/score/total/foo_plda_indomain | paste - <($indom_trials) | awk '{print $1, $4}' | compute-eer - 2>/dev/null)
#  printf '% 5.2f' $eer
#  python local/olr/ivector/Compute_Cavg.py $exp/score/total/foo_plda_indomain $indomain_test_dir/subutt2spk
#
#
#
#  echo 'ivector with plda scoring outdomain'
#  echo 
#  printf '% 16s' 'EER% is:'
#
#  eer=$(awk '{print $3}' $exp/score/total/foo_plda_outdomain | paste - <($outdom_trials) | awk '{print $1, $4}' | compute-eer - 2>/dev/null)
#  printf '% 5.2f' $eer
#  python local/olr/ivector/Compute_Cavg.py $exp/score/total/foo_plda_outdomain data/$testdir/utt2spk
#
#
#fi
