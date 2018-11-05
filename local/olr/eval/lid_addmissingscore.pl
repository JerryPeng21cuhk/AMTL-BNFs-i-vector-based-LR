#!/usr/bin/env perl
# Copyright 2018 Jerry Peng

# in this script, we try to add missing scores for a set of non-speech utterances
# this is to handle the data problem of ap18-olr/test_1s 
# actually, there are two problems in all ap1x-olr dataset
# 1. some audio have two channels, and many amoung them are non-speech audio in the left channel and
#    have speech in the right channel. This is handled by preprocess/add_downmix.py
# 2. few audio in test set do not contains any speech which later failed being extracted into i-vector
# , while this competition requires all test audio be scored. 

my $usage = "$0 is to add missing scores for a set of non-speech utterances that failed to extract
ivectors and it will output a missing trial_score to stdout.
Usage: $0 <missing_uttlist> <trial_score>
For example: $0 <(cut -d' ' -f1 exp/bnf_ivector_no_cmn/score/non_speech_uttid) > foo_simp_pld_missing";

if (@ARGV != 1) {
  print $usage;
}

$missing_uttlist = $ARGV[0];
#$trial_score = $ARGV[1];

# trim the whitespace at the begining and end of a string
sub trim($) {
  my $string = shift;
  $string =~ s/^\s+//;
  $string =~ s/\s+$//;
  return $string;
}

# an array of all lang(the order matters)
@lanid = ('Kazak', 'Tibet', 'Uyghu', 'ct', 'id', 'ja', 'ko', 'ru', 'vi', 'zh');
# create an array of comp_uttid
@missing_uttid = ();
if (open( my $fh_missing_uttid, "<", $missing_uttlist)) {
  while (my $line = <$fh_missing_uttid>) {
    chomp $line;
    $line = trim($line);
    my($uttid) = split(" ", $line);
    push(@missing_uttid, $uttid);
  }
}

# output a fixed score for each missing trial
foreach $uttid (@missing_uttid) {
  foreach $lid (@lanid) {
    print "$lid $uttid 30\n";
  }
}

