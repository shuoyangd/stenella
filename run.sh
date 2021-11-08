#! /bin/sh
#
# run_ende.sh
# Copyright (C) 2021 Shuoyang Ding <shuoyangd@gmail.com>
#
# Distributed under terms of the MIT license.
#

# EDIT THIS BEFORE YOU RUN IT
# BASE=/path/to/this/repo
BASE=/export/c11/shuoyangd/projects/stenella
helper_scripts=$BASE/scripts
src=en
tgt=de

# this can either be a levT model or a levT model after synthetic finetuning
# synthetic finetuning is done the same way as human translation triplet finetuning, only the data is different
checkpoint=$BASE/model/emnlp21-$src-$tgt-best.pt

mkdir -p $src-$tgt-run
cd $src-$tgt-run

# bpe data
mkdir -p processed-data
for section in train dev ; do
  cat $BASE/data/data/post-editing/$section/$src-$tgt-$section/$section.src | spm_encode --model $BASE/models/spm.128k.model > processed-data/$section.$src
  cat $BASE/data/data/post-editing/$section/$src-$tgt-$section/$section.mt | spm_encode --model $BASE/models/spm.128k.model > processed-data/$section.$tgt
  cat $BASE/data/data/post-editing/$section/$src-$tgt-$section/$section.pe | spm_encode --model $BASE/models/spm.128k.model > processed-data/$section.pe
done
cat $BASE/data/data/post-editing/test/$src-$tgt-test/test20.src | spm_encode --model $BASE/models/spm.128k.model > processed-data/test.$src
cat $BASE/data/data/post-editing/test/$src-$tgt-test/test20.mt | spm_encode --model $BASE/models/spm.128k.model > processed-data/test.$tgt
cat $BASE/data/data/post-editing/test/$src-$tgt-test/test20.pe | spm_encode --model $BASE/models/spm.128k.model > processed-data/test.pe

# generate tags
for section in train dev ; do
  paste $BASE/data/data/post-editing/$section/$src-$tgt-$section/$section.mt $BASE/data/data/post-editing/$section/$src-$tgt-$section/$section.pe | \
      teralign --wmt18 | cut -f 2 > processed-data/$section.word.tag  # word-level
  paste processed-data/$section.$tgt processed-data/$section.pe | \
      teralign --wmt18 | cut -f 2 > processed-data/$section.bpe.tag  # bpe-level
  python $helper_scripts/construct_bpe_tags.py \
      --word-text $BASE/data/data/post-editing/$section/$src-$tgt-$section/$section.mt \
      --word-tags processed-data/$section.word.tag \
      --subword-text processed-data/$section.$tgt \
      --subword-tags processed-data/$section.bpe.tag \
      --format sentencepiece > processed-data/$section.tag  # heuristic
  sed -i "s/$/ OK/" processed-data/$section.tag
  sed -i "s/^/OK /" processed-data/$section.tag
done

# binarize
mkdir -p bin
mkdir -p test-bin
PYTHONPATH=$BASE/fairseq python $BASE/fairseq/fairseq_cli/preprocess.py --source-lang $src --target-lang $tgt --workers 50 \
      --trainpref processed-data/train --validpref processed-data/dev --srcdict $BASE/models/model_dict.128k.txt --tgtdict $BASE/models/model_dict.128k.txt --destdir bin/bin --pe --pe-tags
PYTHONPATH=$BASE/fairseq python $BASE/fairseq/fairseq_cli/preprocess.py --source-lang $src --target-lang $tgt --workers 50 \
      --testpref processed-data/test --srcdict $BASE/models/model_dict.128k.txt --tgtdict $BASE/models/model_dict.128k.txt --destdir test-bin/bin

# finetune
mkdir -p output
python $BASE/fairseq/fairseq_cli/train.py bin/bin --save-dir output --restore-file $checkpoint \
    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer --ddp-backend=no_c10d --task translation_lev \
    --criterion nat_loss --arch levenshtein_transformer_m2m_small --pe --noise random_mask --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' --lr 2e-5 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
    --warmup-updates 2000 --warmup-init-lr '1e-07' --label-smoothing 0.1 --dropout 0.1 --weight-decay 0.01 \
    --apply-bert-init --log-format 'simple' --log-interval 100 --fixed-validation-seed 7 --validate-interval 1 \
    --max-tokens 1000 --max-update 4000 --max-epoch 25 --keep-best-checkpoints 4 --no-last-checkpoints \
    --no-epoch-checkpoints --dont-update-word-ins-head --m2m-init --early-exit 12,12,12 --update-freq 4

# score
PYTHONPATH=$BASE/fairseq fairseq-generate test-bin/bin \
    --score-reference \
    --gen-subset test \
    --task translation_lev \
    --path output/checkpoint_best.pt \
    --mcd-rate 0.0 \
    --mcd-samples 1 \
    --iter-decode-max-iter 9 \
    --iter-decode-eos-penalty 0 \
    --m2m-init \
    --beam 1 \
    --batch-size 100 > output/decode.log
grep ^P output/decode.log | cut -d'-' -f2- | sort -n | \
    cut -f 2 | cut -d' ' -f2- | rev | cut -d' ' -f2- | rev > output/raw_bpe_score

# postprocess score
cat output/raw_bpe_score | python $helper_scripts/fakescore2tags.py > output/raw_bpe_tags
python $helper_scripts/debpe_tags_with_deletion.py --text processed-data/test.$tgt --tags output/raw_bpe_tags --format sentencepiece > output/postprocessed_tags

# MCC
mkdir -p eval
mkdir -p eval/ref
mkdir -p eval/sub
cp $BASE/data/data/post-editing/test/$src-$tgt-test/test20.tags eval/ref/goldlabels_mt.tags
cp output/postprocessed_tags eval/sub/predictions_mt.txt
python $helper_scripts/word_evaluate.py eval/ref eval/sub > eval/mcc
cat eval/mcc
