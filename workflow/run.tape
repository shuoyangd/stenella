import "utils/submitters.tape"
import "utils/versioners.tape"

global {
  data_dir_src=(Section:
    dev=(LangPair:
      ende="/home/shuoyangd/projects/thesis-chap5-m2m/data/dev/en-de-dev/dev.src"
      enzh="/home/shuoyangd/projects/thesis-chap5-m2m/data/dev/en-zh-dev/dev.src"
      eten="/home/shuoyangd/projects/thesis-chap5-m2m/data/dev/et-en-dev/dev.src"
      roen="/home/shuoyangd/projects/thesis-chap5-m2m/data/dev/ro-en-dev/dev.src"
      neen="/home/shuoyangd/projects/thesis-chap5-m2m/data/dev/ne-en-dev/dev.src"
      sien="/home/shuoyangd/projects/thesis-chap5-m2m/data/dev/si-en-dev/dev.src"
    )
    test=(LangPair:
      ende="/home/shuoyangd/projects/thesis-chap5-m2m/data/test/en-de-test20/test20.src"
      enzh="/home/shuoyangd/projects/thesis-chap5-m2m/data/test/en-zh-test20/test20.src"
      eten="/home/shuoyangd/projects/thesis-chap5-m2m/data/test/et-en-test20/test20.src"
      roen="/home/shuoyangd/projects/thesis-chap5-m2m/data/test/ro-en-test20/test20.src"
      neen="/home/shuoyangd/projects/thesis-chap5-m2m/data/test/ne-en-test20/test20.src"
      sien="/home/shuoyangd/projects/thesis-chap5-m2m/data/test/si-en-test20/test20.src"
    )
  )

  data_dir_tgt=(Section:
    dev=(LangPair:
      ende="/home/shuoyangd/projects/thesis-chap5-m2m/data/dev/en-de-dev/dev.mt"
      enzh="/home/shuoyangd/projects/thesis-chap5-m2m/data/dev/en-zh-dev/dev.mt"
      eten="/home/shuoyangd/projects/thesis-chap5-m2m/data/dev/et-en-dev/dev.mt"
      roen="/home/shuoyangd/projects/thesis-chap5-m2m/data/dev/ro-en-dev/dev.mt"
      neen="/home/shuoyangd/projects/thesis-chap5-m2m/data/dev/ne-en-dev/dev.mt"
      sien="/home/shuoyangd/projects/thesis-chap5-m2m/data/dev/si-en-dev/dev.mt"
    )
    test=(LangPair:
      ende="/home/shuoyangd/projects/thesis-chap5-m2m/data/test/en-de-test20/test20.mt"
      enzh="/home/shuoyangd/projects/thesis-chap5-m2m/data/test/en-zh-test20/test20.mt"
      eten="/home/shuoyangd/projects/thesis-chap5-m2m/data/test/et-en-test20/test20.mt"
      roen="/home/shuoyangd/projects/thesis-chap5-m2m/data/test/ro-en-test20/test20.mt"
      neen="/home/shuoyangd/projects/thesis-chap5-m2m/data/test/ne-en-test20/test20.mt"
      sien="/home/shuoyangd/projects/thesis-chap5-m2m/data/test/si-en-test20/test20.mt"
    )
  )

  data_dir_tag=(Section:
   dev=(LangPair:
      ende="/home/shuoyangd/projects/thesis-chap5-m2m/data/dev/en-de-dev/dev.tags"
      enzh="/home/shuoyangd/projects/thesis-chap5-m2m/data/dev/en-zh-dev/dev.tags"
      eten="/home/shuoyangd/projects/thesis-chap5-m2m/data/dev/et-en-dev/dev.tags"
      roen="/home/shuoyangd/projects/thesis-chap5-m2m/data/dev/ro-en-dev/dev.tags"
      neen="/home/shuoyangd/projects/thesis-chap5-m2m/data/dev/ne-en-dev/dev.tags"
      sien="/home/shuoyangd/projects/thesis-chap5-m2m/data/dev/si-en-dev/dev.tags"
    )
    test=(LangPair:
      ende="/home/shuoyangd/projects/thesis-chap5-m2m/data/test/en-de-test20/test20.tags"
      enzh="/home/shuoyangd/projects/thesis-chap5-m2m/data/test/en-zh-test20/test20.tags"
      eten="/home/shuoyangd/projects/thesis-chap5-m2m/data/test/et-en-test20/test20.tags"
      roen="/home/shuoyangd/projects/thesis-chap5-m2m/data/test/ro-en-test20/test20.tags"
      neen="/home/shuoyangd/projects/thesis-chap5-m2m/data/test/ne-en-test20/test20.tags"
      sien="/home/shuoyangd/projects/thesis-chap5-m2m/data/test/si-en-test20/test20.tags"
    )
  )

  src_lang=(LangPair:
    ende="en"
    enzh="en"
    eten="et"
    roen="ro"
    neen="ne"
    sien="si"
  )

  tgt_lang=(LangPair:
    ende="de"
    enzh="zh"
    eten="en"
    roen="en"
    neen="en"
    sien="en"
  )

  bpe_model="/home/shuoyangd/projects/thesis-chap5-m2m/m2m/spm.128k.model"
  m2m_model="/home/shuoyangd/projects/thesis-chap5-m2m/m2m/418M_last_checkpoint.pt"
  model_dict="/home/shuoyangd/projects/thesis-chap5-m2m/m2m/model_dict.128k.txt"
  lang_pairs="/home/shuoyangd/projects/thesis-chap5-m2m/m2m/language_pairs_small_models.txt"
  mcd_rate=(MCDRate: 0.0 0.3)
  mcd_samples=(MCDSamples: 5)
  # bin_thres=(BinThres: 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1)
  ducttape_output="out"
  pyenv="conda:torch"
}

package fairseq
    :: .versioner=disk .path="/home/shuoyangd/projects/thesis-chap5-m2m/fairseq" {

  python setup.py build develop
}

package helper_scripts
    :: .versioner=disk .path="/home/shuoyangd/projects/thesis-chap5-m2m/stenella/scripts" {

}

task apply_bpe : fairseq
    > bped_src
    > bped_mt
    :: data_dir_src=@
    :: data_dir_tgt=@
    :: bpe_model=@
    :: pyenv=@ {

  python $fairseq/scripts/spm_encode.py --model $bpe_model --output_format=piece --inputs $data_dir_src --outputs $bped_src
  python $fairseq/scripts/spm_encode.py --model $bpe_model --output_format=piece --inputs $data_dir_tgt --outputs $bped_mt
}

task binarize_data : fairseq
    < bped_src=$bped_src@apply_bpe
    < bped_mt=$bped_mt@apply_bpe
    > out
    :: pyenv=@
    :: src_lang=@
    :: tgt_lang=@
    :: model_dict=@ {

  ln -s $bped_src in.$src_lang
  ln -s $bped_mt in.$tgt_lang
  python $fairseq/fairseq_cli/preprocess.py --source-lang $src_lang --target-lang $tgt_lang --thresholdsrc 0 --thresholdtgt 0 --destdir $out --srcdict $model_dict --tgtdict $model_dict --testpref in
}

task score : fairseq
    < in=$out@binarize_data
    > out
    :: m2m_model=@
    :: model_dict=@
    :: lang_pairs=@
    :: src_lang=@
    :: tgt_lang=@
    :: mcd_rate=@
    :: mcd_samples=@
    :: pyenv=@
    :: .submitter="sge"
    :: .action_flags="-m e -M dings@jhu.edu -q g.q"
    :: .resource_flags="-l 'hostname=c*,gpu=1'" {

  if [ $mcd_rate == "0.0" ] ; then
    CUDA_VISIBLE_DEVICES=`free-gpu` python $fairseq/fairseq_cli/generate.py $in --batch-size 32 --path $m2m_model --fixed-dictionary $model_dict -s $src_lang -t $tgt_lang --score-reference --task translation_multi_simple_epoch --decoder-langtok --encoder-langtok src --gen-subset test --lang-pairs $lang_pairs > $out
  else
    CUDA_VISIBLE_DEVICES=`free-gpu` python $fairseq/fairseq_cli/generate.py $in --batch-size 32 --path $m2m_model --fixed-dictionary $model_dict -s $src_lang -t $tgt_lang --score-reference --task translation_multi_simple_epoch --decoder-langtok --encoder-langtok src --gen-subset test --lang-pairs $lang_pairs --mcd-rate $mcd_rate --mcd-samples $mcd_samples > $out
  fi
}

task extract_score
    < in=$out@score
    > out {

  grep ^P $in | cut -d'-' -f2- | sort -n | cut -f2- | cut -d' ' -f2- | awk 'NF{NF-=1};1' > $out  # first prob is language tag, last is eos
}

task prep_tag
    < in=$data_dir_tag
    > out {

  awk '{for(i=2;i<=NF;i=i+2){printf "%s ", $i}{printf "%s", RS}}' $in > $out  # remove all the gap tags
}

task tune_threshold : helper_scripts
    < text_in=$bped_mt@apply_bpe[Section:dev]
    < ref_in=$out@prep_tag[Section:dev]
    < score_in=$out@extract_score[Section:dev]
    > out
    :: pyenv=@ {

  python $helper_scripts/debpe_lprobs_no_eos.py --text $text_in --lprob $score_in --format sentencepiece > scores_sents
  cat scores_sents | python $helper_scripts/powell_find_threshold.py --score scores_sents --ref $ref_in > $out
}

task postprocess_score : helper_scripts
    < text_in=$bped_mt@apply_bpe[Section:test]
    < score_in=$out@extract_score[Section:test]
    < tune_log=$out@tune_threshold
    > out
    :: pyenv=@ {

  bin_thres=`tail -1 $tune_log`
  python $helper_scripts/debpe_lprobs_no_eos.py --text $text_in --lprob $score_in --format sentencepiece > scores_sents
  cat scores_sents | python $helper_scripts/fakescore2tags_lp.py $bin_thres > $out
}

task macrof1 : helper_scripts
    < tag_in=$out@prep_tag[Section:test]
    < score_in=$out@postprocess_score
    > out
    :: pyenv=@ {

  mkdir ref
  mkdir sub
  cp $tag_in ref/goldlabels_mt.tags
  cp $score_in sub/predictions_mt.txt
  python $helper_scripts/word_evaluate.py ref sub > $out
}

plan run {
  reach macrof1 via (LangPair: eten)
}

global {
  ducttape_experimental_packages=true
  ducttape_experimental_submitters=true
  ducttape_experimental_imports=true
  ducttape_experimental_multiproc=true
}
