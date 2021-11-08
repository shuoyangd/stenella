import "utils/submitters.tape"
import "utils/versioners.tape"

task dummy_stage2_finetune_data
    < in=$stage2_finetune
    > out {

  ln -s $in $out
}

task dummy_stage3_finetune_data
    < in=$stage3_finetune
    > out {

  ln -s $in $out
}

task dummy_blind_test_data
    < in=$blind_test_data_word_level
    > out {
  ln -s $in $out
}

# task apply_bpe_nat : subword_nmt
task apply_bpe_nat : fastBPE
    < in=(Stage:
        2=(DataSection:
          traindev=$out@dummy_stage2_finetune_data
          test=""
          blind_test=""
        )
        3=(DataSection:
          traindev=$stage3_finetune
          test=$test_data_word_level
          blind_test=$blind_test_data_word_level
        )
      )
    > out
    :: pyenv=@
    :: model=$nat_bpe_model {

  # subword-nmt apply-bpe --codes $model --input $in --output $out
  # $fastBPE/fast applybpe $out $in $model
  cat $in | spm_encode --model $model > $out
}

task generate_official_format_word_tags
    < tgt=(Stage:
        2=$out@dummy_stage2_finetune_data[side:tgt]
        3=$out@dummy_stage3_finetune_data[side:tgt]
      )
    < pe=(Stage:
        2=$out@dummy_stage2_finetune_data[side:pe]
        3=$out@dummy_stage3_finetune_data[side:pe]
      )
    > out
    :: pyenv=@ {

    paste $tgt $pe | teralign --wmt18 | cut -f 2 > $out
}

task generate_official_format_bpe_tags
    < tgt=$out@apply_bpe_nat[DataSection:traindev,side:tgt]
    < pe=$out@apply_bpe_nat[DataSection:traindev,side:pe]
    > out
    :: pyenv=@ {

    paste $tgt $pe | teralign --wmt18 | cut -f 2 > $out
}

task generate_heuristic_bpe_tags : helper_scripts
    < word_text=(Stage:
        2=$out@dummy_stage2_finetune_data[side:tgt]
        3=$out@dummy_stage3_finetune_data[side:tgt]
      )
    < word_tags=$out@generate_official_format_word_tags
    < bpe_text=$out@apply_bpe_nat[DataSection:traindev,side:tgt]
    < bpe_tags=$out@generate_official_format_bpe_tags
    > out
    :: bpe_method=@
    :: pyenv=@ {

  python $helper_scripts/construct_bpe_tags.py --word-text $word_text --word-tags $word_tags --subword-text $bpe_text --subword-tags $bpe_tags --format $bpe_method > bpe_tags.out
  linen=`wc -l bpe_tags.out | cut -d' ' -f 1`
  for i in `seq $linen` ; do
    echo "OK" >> pad
  done
  paste -d' ' pad bpe_tags.out pad > $out
}

task binarize_stage2_finetuning_data : fairseq
    < train_src=$out@apply_bpe_nat[DataSection:traindev,TrainDevDataSection:train,side:src]
    < train_tgt=$out@apply_bpe_nat[DataSection:traindev,TrainDevDataSection:train,side:tgt]
    < train_pe=$out@apply_bpe_nat[DataSection:traindev,TrainDevDataSection:train,side:pe]
    < train_tag=(RegenerateTags:
        no=""
        yes=$out@generate_heuristic_bpe_tags[Stage:2,TrainDevDataSection:train]
      )
    < valid_src=$out@apply_bpe_nat[DataSection:traindev,TrainDevDataSection:valid,side:src]
    < valid_tgt=$out@apply_bpe_nat[DataSection:traindev,TrainDevDataSection:valid,side:tgt]
    < valid_pe=$out@apply_bpe_nat[DataSection:traindev,TrainDevDataSection:valid,side:pe]
    < valid_tag=(RegenerateTags:
        no=""
        yes=$out@generate_heuristic_bpe_tags[Stage:2,TrainDevDataSection:valid]
      )
    < dict_src=$dict_src
    < dict_tgt=$dict_tgt
    > out
    :: SRC=@
    :: TRG=@
    :: pyenv=@ {


  ln -s $train_src train.$SRC
  ln -s $train_tgt train.$TRG
  ln -s $train_pe train.pe
  if [ -f $train_tag ] ; then
    ln -s $train_tag train.tag
  fi

  ln -s $valid_src valid.$SRC
  ln -s $valid_tgt valid.$TRG
  ln -s $valid_pe valid.pe
  if [ -f $valid_tag ] ; then
    ln -s $valid_tag valid.tag
  fi

  PYTHONPATH=$fairseq python $fairseq/fairseq_cli/preprocess.py --source-lang $SRC --target-lang $TRG --workers 50 \
    --trainpref train --validpref valid --srcdict $dict_src --tgtdict $dict_tgt --destdir $out --pe --pe-tags
}

task pack_stage2_finetune_data_and_model
    < data=$out@binarize_stage2_finetuning_data
    < model=$checkpoint
    > out {

  mkdir $out
  ln -s $data $out/data
  ln -s $model $out/model.pt
}

task stage2_finetune : fairseq
    < data_dir=$out@pack_stage2_finetune_data_and_model
    < code_dir="/home/v-shuding/nat/fairseq/"
    > out
    :: arch=@
    :: early_exit=@
    :: stage2_update_word_ins=@
    :: stage2_interp_masking=@
    :: stage2_mask_ins_lb_factor=@
    :: stage2_word_del_lb_factor=@
    :: pyenv=@
    :: .submitter="philly"
    :: .sku_flag="G4"
    :: .low_disk_footprint="" {

  # fairseq
  mkdir _code
  mv code.tar.gz _code
  working_dir=$PWD
  cd _code
  tar -zxvf code.tar.gz
  export CUDA_HOME="/usr/local/cuda"
  rm -r build 2> /dev/null  # if not cleaned, sometimes it will create clash
  sudo -E pip install --editable .
  cd $working_dir

  mkdir _data
  cp -r $PT_DATA_DIR/data.tar.gz _data
  working_dir=$PWD
  cd _data
  tar -zxvf data.tar.gz
  cd $working_dir

  cmd="python _code/fairseq_cli/train.py _data/data --save-dir $PT_OUTPUT_DIR --restore-file _data/model.pt --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer --ddp-backend=no_c10d --task translation_lev --criterion nat_loss --arch $arch --pe --noise random_mask --share-all-embeddings --optimizer adam --adam-betas '(0.9,0.98)' --lr 2e-5 --lr-scheduler inverse_sqrt --min-lr '1e-09' --warmup-updates 10000 --warmup-init-lr '1e-07' --label-smoothing 0.1 --dropout 0.1 --weight-decay 0.01 --apply-bert-init --log-format 'simple' --log-interval 100 --fixed-validation-seed 7 --max-tokens 4000 --validate-interval-updates 2500 --save-interval-updates 2500 --max-update 50000 --keep-best-checkpoints 1 --no-last-checkpoints --no-epoch-checkpoints --m2m-init --early-exit $early_exit --update-freq 2"

  if [ "$stage2_update_word_ins" == "no" ] ; then
    cmd="$cmd --dont-update-word-ins-head"
  fi

  if [ "$stage2_interp_masking" == "yes" ] ; then
    cmd="$cmd --interp-masking"
  fi

  if [ ! -z "$stage2_mask_ins_lb_factor" ] ; then
    cmd="$cmd --mask-ins-label-balance-factor $stage2_mask_ins_lb_factor"
  fi

  if [ ! -z "$stage2_word_del_lb_factor" ] ; then
    cmd="$cmd --word-del-label-balance-factor $stage2_word_del_lb_factor"
  fi

  echo $cmd
  PYTHONPATH=_code eval $cmd
}

task binarize_stage3_finetuning_data : fairseq
    < train_src=$out@apply_bpe_nat[Stage:3,DataSection:traindev,TrainDevDataSection:train,side:src]
    < train_tgt=$out@apply_bpe_nat[Stage:3,DataSection:traindev,TrainDevDataSection:train,side:tgt]
    < train_pe=$out@apply_bpe_nat[Stage:3,DataSection:traindev,TrainDevDataSection:train,side:pe]
    < train_tag=(RegenerateTags:
        no=""
        yes=$out@generate_heuristic_bpe_tags[Stage:3,TrainDevDataSection:train]
      )
    < valid_src=$out@apply_bpe_nat[Stage:3,DataSection:traindev,TrainDevDataSection:valid,side:src]
    < valid_tgt=$out@apply_bpe_nat[Stage:3,DataSection:traindev,TrainDevDataSection:valid,side:tgt]
    < valid_pe=$out@apply_bpe_nat[Stage:3,DataSection:traindev,TrainDevDataSection:valid,side:pe]
    < valid_tag=(RegenerateTags:
        no=""
        yes=$out@generate_heuristic_bpe_tags[Stage:3,TrainDevDataSection:valid]
      )
    < dict_src=$dict_src
    < dict_tgt=$dict_tgt
    > out
    :: SRC=@
    :: TRG=@
    :: pyenv=@ {

  ln -s $train_src train.$SRC
  ln -s $train_tgt train.$TRG
  ln -s $train_pe train.pe
  if [ -f $train_tag ] ; then
    ln -s $train_tag train.tag
  fi

  ln -s $valid_src valid.$SRC
  ln -s $valid_tgt valid.$TRG
  ln -s $valid_pe valid.pe
  if [ -f $valid_tag ] ; then
    ln -s $valid_tag valid.tag
  fi

  dict_dir=`dirname $dict_tgt`
  if [ ! -f "$dict_dir/dict.pe.txt" ] ; then
    ln -s $dict_tgt $dict_dir/dict.pe.txt
  fi

  PYTHONPATH=$fairseq python $fairseq/fairseq_cli/preprocess.py --source-lang $SRC --target-lang $TRG \
    --trainpref train --validpref valid --srcdict $dict_src --tgtdict $dict_tgt --destdir $out --pe --pe-tags
}

task pack_stage3_finetune_data_and_model
    < data=$out@binarize_stage3_finetuning_data
    < model=(Stage3Only: no=$out@stage2_finetune yes=$checkpoint)
    > out {

  mkdir $out
  ln -s $data $out/data
  if [ -f $model ] ; then
    ln -s $model $out/model.pt
  else
    ln -s $model/checkpoint_best.pt $out/model.pt
  fi
}

# task stage3_finetune : fairseq
#     < data_dir=$out@pack_stage3_finetune_data_and_model
#     < code_dir="/home/v-shuding/nat/fairseq/"
#     > out
#     :: pyenv=@
#     :: arch=@
#     :: early_exit=@
#     :: stage3_interp_masking=@
#     :: stage3_mask_ins_lb_factor=@
#     :: stage3_word_del_lb_factor=@
#     :: .submitter="philly"
#     :: .sku_flag="G8"
#     :: .low_disk_footprint="" {
#
#   # fairseq
#   mkdir _code
#   mv code.tar.gz _code
#   working_dir=$PWD
#   cd _code
#   tar -zxvf code.tar.gz
#   export CUDA_HOME="/usr/local/cuda"
#   rm -r build 2> /dev/null  # if not cleaned, sometimes it will create clash
#   sudo -E pip install --editable .
#   cd $working_dir
# 
#   mkdir _data
#   cp -r $PT_DATA_DIR/data.tar.gz _data
#   working_dir=$PWD
#   cd _data
#   tar -zxvf data.tar.gz
#   cd $working_dir
# 
#   # single gpu
#   cmd="python _code/fairseq_cli/train.py _data/data --save-dir $PT_OUTPUT_DIR --restore-file _data/model.pt --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer --ddp-backend=no_c10d --task translation_lev --criterion nat_loss --arch $arch --pe --noise random_mask --share-all-embeddings --optimizer adam --adam-betas '(0.9,0.98)' --lr 2e-5 --lr-scheduler inverse_sqrt --min-lr '1e-09' --warmup-updates 2000 --warmup-init-lr '1e-07' --label-smoothing 0.1 --dropout 0.1 --weight-decay 0.01 --apply-bert-init --log-format 'simple' --log-interval 100 --fixed-validation-seed 7 --validate-interval 1 --max-tokens 1000 --max-update 4000 --max-epoch 25 --keep-best-checkpoints 1 --no-last-checkpoints --no-epoch-checkpoints --dont-update-word-ins-head --m2m-init --early-exit $early_exit --update-freq 4"
# 
#   if [ "$stage3_interp_masking" == "yes" ] ; then
#     cmd=$cmd" --interp-masking"
#   fi
# 
#   if [ ! -z "$stage3_mask_ins_lb_factor" ] ; then
#     cmd="$cmd --mask-ins-label-balance-factor $stage3_mask_ins_lb_factor"
#   fi
# 
#   if [ ! -z "$stage3_word_del_lb_factor" ] ; then
#     cmd="$cmd --word-del-label-balance-factor $stage3_word_del_lb_factor"
#   fi
# 
#   echo $cmd
#   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=_code eval $cmd
# }

task stage3_finetune : fairseq
    < data=$out@pack_stage3_finetune_data_and_model
    > out
    :: pyenv=@
    :: arch=@
    :: early_exit=@
    :: stage3_interp_masking=@
    :: stage3_mask_ins_lb_factor=@
    :: stage3_word_del_lb_factor=@
    :: stage3_mask_ins_factor=@ {

  # single gpu
  # old param
  cmd="python $fairseq/fairseq_cli/train.py $data/data --save-dir $out --restore-file $data/model.pt --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer --ddp-backend=no_c10d --task translation_lev --criterion nat_loss --arch $arch --pe --noise random_mask --share-all-embeddings --optimizer adam --adam-betas '(0.9,0.98)' --lr 2e-5 --lr-scheduler inverse_sqrt --min-lr '1e-09' --warmup-updates 2000 --warmup-init-lr '1e-07' --label-smoothing 0.1 --dropout 0.1 --weight-decay 0.01 --apply-bert-init --log-format 'simple' --log-interval 100 --fixed-validation-seed 7 --validate-interval 1 --max-tokens 1000 --max-update 4000 --max-epoch 25 --keep-best-checkpoints 4 --no-last-checkpoints --no-epoch-checkpoints --dont-update-word-ins-head --m2m-init --early-exit $early_exit --update-freq 4"

  if [ "$stage3_interp_masking" == "yes" ] ; then
    cmd=$cmd" --interp-masking"
  fi

  if [ ! -z "$stage3_mask_ins_lb_factor" ] ; then
    cmd="$cmd --mask-ins-label-balance-factor $stage3_mask_ins_lb_factor"
  fi

  if [ ! -z "$stage3_word_del_lb_factor" ] ; then
    cmd="$cmd --word-del-label-balance-factor $stage3_word_del_lb_factor"
  fi

  if [ ! -z "$stage3_mask_ins_factor" ] ; then
    cmd="$cmd --mask-ins-factor $stage3_mask_ins_factor"
  fi

  echo $cmd
  source get-gpu 1
  PYTHONPATH=$fairseq eval $cmd
}

task binarize_test_data : fairseq
    # < test_src=$out@apply_sentencepiece[side:src]
    # < test_tgt=$out@apply_sentencepiece[side:tgt]
    < test_src=(Blind:
        no=$out@apply_bpe_nat[Stage:3,side:src,DataSection:test]
        yes=$out@apply_bpe_nat[Stage:3,side:src,DataSection:blind_test]
      )
    < test_tgt=(Blind:
        no=$out@apply_bpe_nat[Stage:3,side:tgt,DataSection:test]
        yes=$out@apply_bpe_nat[Stage:3,side:tgt,DataSection:blind_test]
      )
    # < test_tgt=$official_mt_decode_output
    < dict_src=$dict_src
    < dict_tgt=$dict_tgt
    > out
    :: SRC=@
    :: TRG=@
    :: pyenv=@ {

  ln -s $test_src test.$SRC
  ln -s $test_tgt test.$TRG

  PYTHONPATH=$fairseq python $fairseq/fairseq_cli/preprocess.py --source-lang $SRC --target-lang $TRG \
    --testpref test --srcdict $dict_src --tgtdict $dict_tgt --destdir $out
}

task score : fairseq
    < in=$out@binarize_test_data
    < checkpoint=$out@stage3_finetune
    > out
    :: SRC=@
    :: TRG=@
    :: mcd_rate=@
    :: mcd_samples=@
    :: pyenv=@ {

  source get-gpu 1
  PYTHONPATH=$fairseq fairseq-generate $in \
    --score-reference \
    --gen-subset test \
    --task translation_lev \
    --path $checkpoint/checkpoint_best.pt \
    --mcd-rate $mcd_rate \
    --mcd-samples $mcd_samples \
    --iter-decode-max-iter 9 \
    --iter-decode-eos-penalty 0 \
    --m2m-init \
    --beam 1 \
    --batch-size 100 > decode.log
  grep ^P decode.log | cut -d'-' -f2- | sort -n | \
    cut -f 2 | cut -d' ' -f2- | rev | cut -d' ' -f2- | rev > $out
}

task binarize_dev_data : fairseq
    # < test_src=$out@apply_sentencepiece[side:src]
    # < test_tgt=$out@apply_sentencepiece[side:tgt]
    < test_src=$out@apply_bpe_nat[Stage:3,side:src,DataSection:traindev,TrainDevDataSection:valid]
    < test_tgt=$out@apply_bpe_nat[Stage:3,side:tgt,DataSection:traindev,TrainDevDataSection:valid]
    # < test_tgt=$official_mt_decode_output
    < dict_src=$dict_src
    < dict_tgt=$dict_tgt
    > out
    :: SRC=@
    :: TRG=@
    :: pyenv=@ {

  ln -s $test_src test.$SRC
  ln -s $test_tgt test.$TRG

  PYTHONPATH=$fairseq python $fairseq/fairseq_cli/preprocess.py --source-lang $SRC --target-lang $TRG \
    --testpref test --srcdict $dict_src --tgtdict $dict_tgt --destdir $out
}

task list_ensemble_checkpoints
    < in=$out@stage3_finetune
    > out {

  ls -p $in/checkpoint.best* | tr '\n' ' ' > $out
}

task score_ensemble : fairseq
    < in=(EnsembleStage:
        tune=$out@binarize_dev_data
        test=$out@binarize_test_data
      )
    < checkpoint_list=$out@list_ensemble_checkpoints
    > out
    :: SRC=@
    :: TRG=@
    :: mcd_rate=@
    :: mcd_samples=@
    :: pyenv=@ {


  list=`cat $checkpoint_list`
  mkdir $out
  idx=0
  for checkpoint in $list ; do
    source get-gpu 1
    PYTHONPATH=$fairseq fairseq-generate $in \
      --score-reference \
      --gen-subset test \
      --task translation_lev \
      --path $checkpoint \
      --mcd-rate $mcd_rate \
      --mcd-samples $mcd_samples \
      --iter-decode-max-iter 9 \
      --iter-decode-eos-penalty 0 \
      --m2m-init \
      --beam 1 \
      --batch-size 50 | \
      grep ^P | cut -d'-' -f2- | sort -n | \
      cut -f 2 | cut -d' ' -f2- | rev | \
      cut -d' ' -f2- | rev \
      > $out/$idx.score
    idx=$((idx+1))
  done
}

task tune : fairseq
    < tag_in=$official_dev_tags
    < text_in=$out@apply_bpe_nat[Stage:3,side:tgt,DataSection:traindev,TrainDevDataSection:valid]
    < scores_in=$out@score_ensemble[EnsembleStage:tune]
    > out
    :: fine_grained=(FineGrainedTuning: no yes)
    :: pyenv=@ {

  scores=`ls -p $scores_in/*.score | tr '\n' ' '`
  if [ $fine_grained == "no" ] ; then
    echo "python $fairseq/scripts/powell_ensemble.py --scores $scores --text $text_in --ref $tag_in"
    python $fairseq/scripts/powell_ensemble.py --scores $scores --text $text_in --ref $tag_in > $out
  else
    echo "python $fairseq/scripts/powell_breakdown_ensemble.py --scores $scores --text $text_in --ref $tag_in"
    python $fairseq/scripts/powell_breakdown_ensemble.py --scores $scores --text $text_in --ref $tag_in > $out
  fi
}

task ensemble : fairseq
    < scores_in=$out@score_ensemble[EnsembleStage:test]
    < weight_in=$out@tune
    > out
    :: fine_grained=(FineGrainedTuning: no yes)
    :: pyenv=@ {

  scores=`ls -p $scores_in/*.score | tr '\n' ' '`
  weights=`tail -1 $weight_in`
  if [ $fine_grained == "no" ] ; then
    echo "python $fairseq/scripts/powell_ensemble.py --scores $scores --weight $weights"
    python $fairseq/scripts/powell_ensemble.py --scores $scores --weight $weights > $out
  else
    echo "python $fairseq/scripts/powell_breakdown_ensemble.py --scores $scores --weight $weights"
    python $fairseq/scripts/powell_breakdown_ensemble.py --scores $scores --weight $weights > $out
  fi
}

task postprocess_score : helper_scripts
    < text_in=(Blind:
        no=$out@apply_bpe_nat[Stage:3,side:tgt,DataSection:test]
        yes=$out@apply_bpe_nat[Stage:3,side:tgt,DataSection:blind_test]
      )
    < score_in=(Ensemble:
        no=$out@score
        yes=$out@ensemble
      )
    > out
    :: pyenv=@
    :: bpe_method=@ {

  # python $helper_scripts/debpe_lprobs.py --text $text_in --lprob $score_in --format bpe > scores_sents
  # python $helper_scripts/debpe_lprobs.py --text $text_in --lprob $score_in --format bpe | rev | cut -d' ' -f2- | rev > scores_sents

  cat $score_in | python $helper_scripts/fakescore2tags.py > bped_scores_sents
  python $helper_scripts/debpe_tags_with_deletion.py --text $text_in --tags bped_scores_sents --format $bpe_method > $out
}

task macrof1 : helper_scripts
    # < tag_in=$out@generate_tag
    < tag_in=$official_tags
    < score_in=$out@postprocess_score[Blind:no]
    > out
    :: pyenv=@ {

  mkdir ref
  mkdir sub
  cp $tag_in ref/goldlabels_mt.tags
  cp $score_in sub/predictions_mt.txt
  python /home/v-shuding/nat/scripts/word_evaluate.py ref sub > $out
}

task format_submission_tags
    < text_file=$out@dummy_blind_test_data[side:tgt]
    < label_file=$out@postprocess_score[Blind:yes]
    > out
    :: disk_footprint=@
    :: num_params=@
    :: method_name=@
    :: SRC=@
    :: TRG=@
    :: pyenv=@ {

  mkdir $out
  python /home/v-shuding/nat/scripts/format_submission_tags.py --disk-footprint $disk_footprint \
      --num-params $num_params \
      --lang-pair $SRC-$TRG \
      --method-name $method_name \
      --text-file $text_file \
      --label-file $label_file \
      --out-dir $out
}


plan run {
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: *) * (Arch: big)
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: *) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: big) * (DiffKD: *)  # big
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: yes) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (DiffKD: yes)  # m2m
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: no) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (DiffKD: no) * (DoKD: no)  # m2m+nokd+nodiffkd
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: no) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (size: 40M) * (Stage2LbFactor: lee)  # m2m ter massaged
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: no) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (size: 10M) * (Stage2LbFactor: lee)  # m2m ter massaged
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: no) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (Detokenize: no) * (DoKD: yes) * (size: 10m)  # rt 10m
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: no) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (Detokenize: no) * (DoKD: yes)  # rt
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: *) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (DiffKD: yes) * (Arch: small) * (Detokenize: *) * (DoPretrain: no)  # no pre-train
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: no) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (DiffKD: yes) * (Stage2LbFactor: *)  # m2m ru-en
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: yes) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (DiffKD: yes) * (Stage2LbFactor: def)  # m2m ro-en
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: no) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (DiffKD: simupe) * (simu_hyperparam: *) * (Stage2LbFactor: def) * (Stage3LbFactor: def)  # m2m ro-en simupe
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: yes) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (DiffKD: yes) * (Stage3LbFactor: *) * (Setup: emnlp21_best arch_big_best) * (Depre: no)  # m2m en-de
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: *) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (DiffKD: simupe simupe2) * (Stage3LbFactor: def)  # m2m en-zh
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: yes) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (Stage3LbFactor: def) * (Ensemble: yes) * (FineGrainedTuning: yes) # m2m en-zh
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: yes) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (DiffKD: yes) * (Stage3LbFactor: *)  # m2m ne-en & si-en
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: yes) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (DiffKD: yes) * (Stage3LbFactor: lee) * (Simupe: *) * (DoPretrain: aug)  # m2m et-en
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: yes) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: mid) * (DiffKD: yes) * (Stage2LbFactor: def) * (langpair: *) * (Stage3LbFactor: lee) # m2m multisrc
  # reach list_ensemble_checkpoints via (Stage2UpdateWordIns: no) * (Stage3Only: yes) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (DiffKD: yes) * (Stage2LbFactor: lee) * (langpair: sien) * (Stage3LbFactor: lee) * (Ensemble: *) # m2m multisrc ensemble list only
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: yes) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (DiffKD: yes) * (Stage2LbFactor: lee) * (langpair: eten) * (Stage3LbFactor: lee) * (Ensemble: yes) * (FineGrainedTuning: yes) # m2m multisrc ensemble run
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: yes) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (DiffKD: yes) * (Stage2LbFactor: def) * (langpair: eten) * (Stage3LbFactor: def) * (Ensemble: yes) * (FineGrainedTuning: yes) # m2m multisrc ensemble run
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: yes) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (DiffKD: yes) * (Stage2LbFactor: def) * (langpair: neen sien) * (Stage3LbFactor: lee) # m2m multisrc
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: yes) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (DiffKD: yes) * (Stage2LbFactor: def) * (langpair: ruen eten) * (Stage3LbFactor: lee) * (Stage3GapFactor: *) # m2m multisrc
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: no) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (Stage2LbFactor: lee) * (langpair: eten) * (Stage3LbFactor: lee) # m2m multisrc mvape
  # reach macrof1 via (Stage3Only: *) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (KD: *)  # KD analysis
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: no) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (DiffKD: simupe2) * (Stage3LbFactor: def)  # m2m en-zh
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: yes) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (DiffKD: yes) * (Stage2LbFactor: def) * (langpair: roen) * (Stage3LbFactor: roen)  # m2m multisrc ro-en factor exp
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: no) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (DiffKD: diffpe) * (Stage2LbFactor: def) * (langpair: roen) * (Stage3LbFactor: def)  # m2m multisrc ro-en msprod aug
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: *) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (DiffKD: official) * (Stage3LbFactor: lee) * (DoPretrain: aug_msprod)  # m2m et-en msprod aug
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: no) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (DiffKD: diffpe) * (Stage3LbFactor: def lee) * (DoPretrain: msprod)  # m2m ro-en msprod
  reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: no) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (Synthesis: *) * (Stage3LbFactor: def)  # m2m en-de abl
  # reach macrof1 via (Stage2UpdateWordIns: no) * (Stage3Only: no) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (DiffKD: yes) * (Stage3LbFactor: def)  * (DoPretrain: at) * (Depre: *)  # m2m en-de

  ### FINAL WMT21 TAG GENERATION ###
  # reach format_submission_tags via (Stage2UpdateWordIns: no) * (Stage3Only: yes) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (DiffKD: yes) * (Stage2LbFactor: lee) * (langpair: eten) * (Stage3LbFactor: lee) * (Ensemble: *) * (Blind: yes) * (FineGrainedTuning: yes)  # m2m multisrc ensemble run
  # reach format_submission_tags via (Stage2UpdateWordIns: no) * (Stage3Only: no) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (DiffKD: yes) * (Stage2LbFactor: def) * (langpair: neen) * (Stage3LbFactor: lee) * (Ensemble: no) * (Blind: yes) # m2m multisrc ensemble run
  # reach format_submission_tags via (Stage2UpdateWordIns: no) * (Stage3Only: yes) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (DiffKD: yes) * (Stage2LbFactor: def) * (langpair: roen) * (Stage3LbFactor: def) * (Ensemble: yes) * (FineGrainedTuning: yes) * (Blind: yes) # m2m multisrc ensemble run
  # reach format_submission_tags via (Stage2UpdateWordIns: no) * (Stage3Only: yes) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (Stage3LbFactor: def) * (Ensemble: yes) * (Blind: yes) * (FineGrainedTuning: yes) # m2m en-zh
  # reach format_submission_tags via (Stage2UpdateWordIns: no) * (Stage3Only: no) * (Stage3InterpMasking: no) * (RegenerateTags: yes) * (Arch: small) * (DiffKD: simupe) * (Stage3LbFactor: def) * (Ensemble: no) * (Blind: yes) # m2m en-zh
}

# Nuts and bolts:
global {
  ducttape_experimental_packages=true
  ducttape_experimental_submitters=true
  ducttape_experimental_imports=true
  ducttape_experimental_multiproc=true
}
