# -*- coding: utf-8 -*-
#
# Copyright © 2020 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2021-04-22
#
# Distributed under terms of the MIT license.

import argparse
import logging
import math
import sys

logging.basicConfig(
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

opt_parser = argparse.ArgumentParser(description="")
opt_parser.add_argument("--word-text", metavar="PATH", help="")
opt_parser.add_argument("--subword-text", metavar="PATH", help="")
opt_parser.add_argument("--word-tags", metavar="PATH", help="")
opt_parser.add_argument("--subword-tags", metavar="PATH", help="")
opt_parser.add_argument("--format", choices=['bpe', 'sentencepiece'], default='sentencepiece', help="")
opt_parser.add_argument("--ignore-initial-end-padding", default=False, action='store_true', help="")


def debpe_sentencepiece(toks):
  idx_map = {}
  new_idx = -1
  new_toks = []
  for old_idx, tok in enumerate(toks):
    if old_idx == 0 or tok.startswith("▁"):
        new_idx += 1
        idx_map[old_idx] = new_idx
        new_toks.append(tok[1:] if tok.startswith("▁") else tok)
    else:
        idx_map[old_idx] = new_idx
        new_toks[new_idx] += tok
  return idx_map, new_toks

def debpe(toks):
  idx_map = {}
  new_idx = 0
  new_toks = [""]
  for old_idx, tok in enumerate(toks):
    if tok.endswith("@@"):
      idx_map[old_idx] = new_idx
      new_toks[new_idx] += tok[:-2]
    else:
      idx_map[old_idx] = new_idx
      new_toks[new_idx] += tok
      new_toks.append("")
      new_idx += 1
  new_toks = new_toks[:-1]
  return idx_map, new_toks

def before_gap_tag_idx(idx):
  return idx * 2

def token_tag_idx(idx):
  return idx * 2 + 1

def main(options):
  word_text_file = open(options.word_text)
  subword_text_file = open(options.subword_text)
  word_tag_file = open(options.word_tags)
  subword_tag_file = open(options.subword_tags)

  linen = 0
  for word_text_line, subword_text_line, word_tag_line, subword_tag_line in \
      zip(word_text_file, subword_text_file, word_tag_file, subword_tag_file):

    word_text_line = word_text_line.strip()
    subword_text_line = subword_text_line.strip()
    word_tag_line = word_tag_line.strip()
    subword_tag_line = subword_tag_line.strip()

    if options.format == "bpe":
        idx_map, debped_subword_text_toks = debpe(subword_text_line.split())
        assert word_text_line.strip() == " ".join(debped_subword_text_toks)
    elif options.format == "sentencepiece":
        idx_map, debped_subword_text_toks = debpe_sentencepiece(subword_text_line.split())
        # sentencepiece may cause minor changes to the text so we'll be more tolerant
        word_text_line_no_repeated_space = " ".join(word_text_line.split())
        if word_text_line_no_repeated_space != " ".join(debped_subword_text_toks):
          logging.info(
            "Line {0}: the original word-level sentence \"{1}\" and debped sentence \"{2}\" is different".format(\
              linen,
              word_text_line_no_repeated_space,
              " ".join(debped_subword_text_toks),
            )
          )
        # assert word_text_line.strip() == " ".join(debped_subword_text_toks)
    else:
        raise NotImplementedError

    inv_idx_map = {}
    for k, v in idx_map.items():
      inv_idx_map[v] = inv_idx_map.get(v, []) + [k]

    word_toks = word_text_line.split()
    word_tags = word_tag_line.split()
    subword_tags = subword_tag_line.split()
    if options.ignore_initial_end_padding:
      word_tags = word_tags[1:-1]
      subwrd_tags = subword_tags[1:-1]
    new_subword_tags = []

    # if debped output and word-level text are different (most likely due to sentencepiece bug),
    # attempt to adjust the word-level indexes in the inv_idx_map
    # right now, we only process one case:
    # word sequence has broken code point and omitted by sentencepiece
    if len(debped_subword_text_toks) != len(word_text_line.split()):
      word_idx = 0
      debpe_idx = 0
      new_inv_idx_map = {}
      abort = False
      while word_idx < len(word_toks) and \
          debpe_idx < len(debped_subword_text_toks):
        if debped_subword_text_toks[debpe_idx] == word_toks[word_idx]:
          pass
        # reaching the end, don't need to do anything
        elif word_idx+1 >= len(word_toks):
          word_idx += 1
          debpe_idx += 1
          continue  # don't overwrite since the last word is corrupt
        elif word_toks[word_idx+1] == debped_subword_text_toks[debpe_idx]:
          word_idx += 1
        # nothing we can do
        else:
          logging.info("Line {0}: debpe length mismatch with word text forced me to abort".format(linen))
          sys.stdout.write(subword_tag_line + '\n')
          linen += 1
          abort = True
          break
        new_inv_idx_map[word_idx] = inv_idx_map[debpe_idx]
        word_idx += 1
        debpe_idx += 1

      if abort:
        continue
      inv_idx_map = new_inv_idx_map

    # for each word, we'll first copy the gap tag in front of it
    # and then, construct the new subword tags
    for word_idx, word_tok in enumerate(word_toks):
      # copy the gap tag
      if word_idx in inv_idx_map:
        new_subword_tags.append(word_tags[before_gap_tag_idx(word_idx)])
      else:
        continue

      # then, let's construct the new subword tag
      # we need the span first
      subword_idxes = inv_idx_map[word_idx]
      subword_word_span_bgn = min(subword_idxes)
      subword_word_span_end = max(subword_idxes)
      subword_tag_span_bgn = token_tag_idx(subword_word_span_bgn)
      subword_tag_span_end = token_tag_idx(subword_word_span_end)

      # always first copy over word tag, and then append pad
      word_tag = word_tags[token_tag_idx(word_idx)]
      new_subword_tags.extend([word_tag])
      new_subword_tags.extend(["<pad>"] * (subword_tag_span_end - subword_tag_span_bgn))

    # copy over the final, trailing gap tag
    new_subword_tags.append(word_tags[-1])
    assert len(new_subword_tags) == len(subword_tags)

    new_subword_tag_line = " ".join(new_subword_tags)
    sys.stdout.write(new_subword_tag_line + '\n')
    linen += 1


if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
      opt_parser.parse_known_args()[1]))

  main(options)
