#!/bin/bash
# 2018 jerrypeng

# This script is the combination of what I have done in past four months
# It contains these steps:
# 1. use phrec to generate phoneme posterior
# 2. Use posterior to generate ali(labels)
# 3. Use mtl-adverserial model to train a MTL-TDNN with BN layer
# 4. Train a UBM with 2048/256 mixtures
# 5. Train an i-vector extractor or j-vector extractor on BNF
# 6. Generate i-vectors for training data, generate boostrapped 10 i-vectors for test data
# 7. Perform cos-scoring, lda, kaldi-plda, simplified-plda with three scoring methods.

set -e
set +o posix

train_stage=-10
remove_egs=false
srand=0
get_egs_stage=-10
num_jobs_initial=2
num_jobs_final=8
megs_dir=
score_more=

# Number of components
cnum=2048 # num of Gaussions
civ=400 # dim of i-vector
clda=9 # dim for i-vector with LDA

exp=exp/mtlbnfivector
alidir=$exp/ali
nnetdir=$exp/nnet3/tdnn
mkdir -p $exp $nnetdir $alidir
stage=2

bnf_dim=64

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


[ ! -f local.conf ] && echo 'the file local.conf does not exist! Read README.txt for more details.' && exit 1;
. local.conf || exit 1;

num_tasks=${#task_list[@]}
feat_suffix=_hires      # The feature suffix describing features used in
                        # multilingual training
                        # _hires -> 40dim MFCC
                        # _hires_pitch -> 40dim MFCC + pitch
                        # _hires_pitch_bnf -> 40dim MFCC +pitch + BNF
echo "$0 $@"  # Print the command line for logging
if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi


# 1. Generate mfcc feature
if [ $stage -le 1 ]; then
  mfccdir=`pwd`/_mfcc
  vaddir=`pwd`/_vad
  
  echo ">> make mfcc features ---"
  for i in data/train data/test_1s ; do
    [ -f $i/wav.scp ] && [ ! -f $i/wav_org.scp ] && mv $i/wav.scp $i/wav_org.scp
    # add_downmix.py is slow. Add conditions to avoid redundant computation
    [ -f $i/wav_org.scp ] && [ ! -f $i/wav.scp ] && python preprocess/add_downmix.py $i/wav_org.scp > $i/wav.scp
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 10 --cmd "$cpu_cmd" $i $exp/log_mfcc $mfccdir
    sid/compute_vad_decision.sh --nj 4 --cmd "$cpu_cmd" $i $exp/_log_vad $vaddir
    feat-to-len scp:$i/feats.scp ark,t:$i/feats.len || exit 1;
  done
  
  # generate ali for training data (very slow and problematic)
#  echo ">> generate language alignment for train set ---"
#  nj=10
#  log=$exp/ali/log
#  paste --delimiter='\t' \
#    <(cut data/train/wav.scp -f1 -d' ') \
#    <(cut data/train/wav.scp -f2- -d' ') \
#    <(cut data/train/feats.len -f2 -d' ') | \
#    parallel --progress --joblog $log -j 15 -N10 -k --spreadstdin \
#	python preprocess/phnrec_postconvert.py > $alidir/ali.ark
  # parallel can resume if it was stopped and the input of the completed jobs is unchanged and joblog is attached
  # by add --resume option to parallel
  # 15 jobs, 10 lines for each job

  ## copy ali for training data
  cp exp/bnfivector/ali/ali.ark $alidir/
fi


num_pdf=`awk '{for(i=2;i<NF;i++) if($i>maxval) maxval=$i;} END{ print maxval;}' $alidir/ali.ark`
num_pdf=$[$num_pdf+1]


if [ $stage -le 2 ]; then
  echo ">> creating neural net configs using the xconfig parser";
  feat_dim=`feat-to-dim --print-args=false scp:data/train/feats.scp -`
  if [ -z $bnf_dim ];then
    bnf_dim=1024
  fi
  mkdir -p $nnetdir/configs
  cat <<EOF > $nnetdir/configs/network.xconfig
  input dim=$feat_dim name=input
  
  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  # the first splicing is moved before the lda layer, so no splicing here

  relu-renorm-layer name=tdnn1 input=Append(input@-2,input@-1,input,input@1,input@2) dim=1024
  relu-renorm-layer name=tdnn2 dim=1024
  relu-renorm-layer name=tdnn3 input=Append(-1,2) dim=1024
  relu-renorm-layer name=tdnn4 input=Append(-3,3) dim=1024
  relu-renorm-layer name=tdnn5 input=Append(-3,3) dim=1024
  relu-renorm-layer name=tdnn6 input=Append(-7,2) dim=1024
  relu-renorm-layer name=tdnn_bn dim=$bnf_dim
  
  # adding the layers for diffrent language's output
EOF
  # added output layer for asr task
  echo "relu-renorm-layer name=prefinal-affine-asr input=tdnn_bn dim=1024
      output-layer name=output-0 dim=$num_pdf max-change=1.5
      gradient-reversal-layer name=grl-sid input=tdnn_bn gr-weight=1.0
      relu-renorm-layer name=prefinal-affine-lid input=grl-sid dim=1024
      output-layer name=output-1 dim=641 max-change=1.5" >> $nnetdir/configs/network.xconfig
  #output-0: phnrec-tdnn; output-1: sid
  
  steps/nnet3/xconfig_to_configs.py --xconfig-file $nnetdir/configs/network.xconfig \
    --config-dir $nnetdir/configs/ \
    --nnet-edits="rename-node old-name=output-0 new-name=output"

fi


for task_index in `seq 0 $[$num_tasks-1]`; do
  multi_data_dirs[$task_index]=data/${task_list[$task_index]}/train${feat_suffix}
  multi_egs_dirs[$task_index]=exp/${task_list[$task_index]}/nnet3/egs${feat_suffix}
  multi_ali_dirs[$task_index]=exp/${task_list[$task_index]}/ali
done


if [ $stage -le 3 ]; then
    echo "$0: Generates separate egs dir per task for multi-task training"
    . $nnetdir/configs/vars || exit 1;
    local/nnet3/prepare_multilingual_egs.sh --cmd "$decode_cmd" \
      --cmvn-opts "--norm-means=false --norm-vars=false" \
      --left-context $model_left_context --right-context $model_right_context \
      $num_tasks ${multi_data_dirs[@]} ${multi_ali_dirs[@]} ${multi_egs_dirs[@]} || exit 1;
fi


if [ -z $megs_dir ];then
  megs_dir=$exp/egs
fi


if [ $stage -le 4 ] && [ ! -z $megs_dir ]; then
  echo "$0: Generate multilingual egs dir using "
  echo "separate egs dirs for multilingual training."
  if [ ! -z "$task2weight" ]; then
      egs_opts="--lang2weight '$task2weight'"
  fi
  common_egs_dir="${multi_egs_dirs[@]} $megs_dir"
  steps/nnet3/multilingual/combine_egs.sh $egs_opts \
    --cmd "$decode_cmd" \
    --samples-per-iter 400000 \
    $num_tasks ${common_egs_dir[@]} || exit 1;
fi

if [ $stage -le 5 ]; then
  #rm $exp/egs/valid_diagnostic.scp >& /dev/null
  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs 2 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=4 \
    --trainer.optimization.initial-effective-lrate=0.0015 \
    --trainer.optimization.final-effective-lrate=0.00015 \
    --trainer.optimization.minibatch-size=256,128 \
    --trainer.samples-per-iter=400000 \
    --trainer.max-param-change=2.0 \
    --trainer.srand=$srand \
    --feat-dir ${multi_data_dirs[0]} \
    --egs.dir $megs_dir \
    --use-dense-targets false \
    --targets-scp ${multi_ali_dirs[0]} \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 50 \
    --use-gpu true \
    --dir=$nnetdir  || exit 1;
    #--feat.online-ivector-dir='' \ #${multi_ivector_dirs[0]} \
fi



if [ $stage -le 6 ]; then
  echo ">> generate bnf feature"
  for i in train test_1s; do
  #for i in valid dev_1s; do
    bnfdir=_bnf/$i
    mkdir -p $bnfdir
    steps/nnet3/make_bottleneck_features.sh --use-gpu true --nj 9 --cmd "$train_cmd" \
      tdnn_bn.renorm data/$i $bnfdir $nnetdir
  done
fi


#if [ $stage -le 7 ]; then
#  echo ">> ubm training"
#  for i in train valid dev_1s; do
#    #sid/compute_vad_decision.sh --nj 4 --cmd "$cpu_cmd" data/$i $exp/_log_vad _bnf/$i
#    cp data/$i/vad.scp _bnf/$i/
#  done
#
#  utils/subset_data_dir.sh _bnf/train 18000 _bnf/train_18k
#  utils/subset_data_dir.sh _bnf/train 36000 _bnf/train_36k
#
#  #UBM training
#  sid/train_diag_ubm.sh --nj 9 --cmd "$cpu_cmd" _bnf/train_18k ${cnum} $exp/diag_ubm_${cnum}
#  sid/train_full_ubm.sh --nj 9 --cmd "$cpu_cmd" _bnf/train_36k $exp/diag_ubm_${cnum} $exp/full_ubm_${cnum}
#
#fi
#
#if [ $stage -le 8 ]; then
#  echo ">> ivector extractor training"
#  # kaldi ivector extractor
#  sid/train_ivector_extractor.sh --nj 9 --cmd "$cpu_cmd -l mem_free=5G" \
#    --num-iters 6 --ivector_dim $civ $exp/full_ubm_${cnum}/final.ubm _bnf/train \
#    $exp/extractor_${cnum}_${civ}
#fi
#
#
#if [ $stage -le 9 ]; then
#  # Extract i-vector
#  #for i in train valid  dev_1s; do
#  for i in valid  dev_1s; do
#    sid/extract_ivectors.sh --cmd "$cpu_cmd -l mem_free=5G," --nj 9 \
#      $exp/extractor_${cnum}_${civ} _bnf/$i $exp/ivectors_${i}_${cnum}_${civ}
#  done
#fi
#
#if [ $stage -le 10 ]; then
#  # cosine-distance scoring
#  for i in dev_1s; do
#    trials="python local/olr/eval/create_trials.py $exp/ivectors_train_${cnum}_${civ}/spk_ivector.scp $exp/ivectors_${i}_${cnum}_${civ}/ivector.scp -"
#    mkdir -p $exp/score/$i
#    cat <($trials) | awk '{print $1, $2}' | \
#      ivector-compute-dot-products - \
#        scp:$exp/ivectors_train_${cnum}_${civ}/spk_ivector.scp \
#        scp:$exp/ivectors_${i}_${cnum}_${civ}/ivector.scp \
#        $exp/score/$i/foo_cosine
#    
#    echo i-vector
#    echo
#    printf '% 16s' 'EER% is:'
#    eer=$(awk '{print $3}' $exp/score/$i/foo_cosine | paste - <($trials) | awk '{print $1, $4}' | compute-eer - 2>/dev/null)
#    printf '% 5.2f' $eer
#    echo
#    
#    python local/olr/ivector/Compute_Cavg.py  $exp/score/$i/foo_cosine data/$i/utt2spk
#  
#  done 
#fi
#
#if [ $stage -le 11 ]; then
#
#  testdir=dev_1s
#
#  # Demonstrate what happens if we reduce the dimension with LDA
#  ivector-compute-lda --dim=$clda  --total-covariance-factor=0.1 \
#    "ark:ivector-normalize-length scp:${exp}/ivectors_train_${cnum}_${civ}/ivector.scp  ark:- |" \
#    ark:data/train/utt2spk \
#    $exp/ivectors_train_${cnum}_${civ}/transform_${clda}.mat
#
#  ivector-transform $exp/ivectors_train_${cnum}_${civ}/transform_${clda}.mat \
#    scp:$exp/ivectors_train_${cnum}_${civ}/ivector.scp ark:- | \
#    ivector-normalize-length ark:- ark:${exp}/ivectors_train_${cnum}_${civ}/lda_${clda}.ark
#
#  ivector-transform $exp/ivectors_train_${cnum}_${civ}/transform_${clda}.mat \
#    "ark:ivector-normalize-length scp:$exp/ivectors_${testdir}_${cnum}_${civ}/ivector.scp ark:- |" ark:- | \
#    ivector-normalize-length ark:- ark:${exp}/ivectors_${testdir}_${cnum}_${civ}/lda_${clda}.ark
#
#  dir=${exp}/ivectors_train_${cnum}_${civ}
#  ivector-mean ark:data/train/spk2utt \
#    ark:$dir/lda_${clda}.ark ark:- ark,t:$dir/num_utts.ark | \
#    ivector-normalize-length ark:- ark,scp:$dir/lda_ivector.ark,$dir/lda_ivector.scp
#
#  trials="python local/olr/eval/create_trials.py $exp/ivectors_train_${cnum}_${civ}/spk_ivector.scp $exp/ivectors_${testdir}_${cnum}_${civ}/ivector.scp -"
#
#  cat <($trials) | awk '{print $1, $2}' | \
#  ivector-compute-dot-products - \
#   scp:$exp/ivectors_train_${cnum}_${civ}/lda_ivector.scp \
#   ark:$exp/ivectors_${testdir}_${cnum}_${civ}/lda_${clda}.ark \
#   $exp/score/${testdir}/foo_lda
#
#  echo 'L-vector (i-vector with LDA)'
#  echo 
#  printf '% 16s' 'EER% is:'
#  eer=$(awk '{print $3}' $exp/score/${testdir}/foo_lda | paste - <($trials) | awk '{print $1, $4}' | compute-eer - 2>/dev/null)
#  printf '% 5.2f' $eer
#
#  python local/olr/ivector/Compute_Cavg.py  $exp/score/${testdir}/foo_lda data/${testdir}/utt2spk
#
#fi
#
#
#
#
#
