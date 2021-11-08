# -*- coding: utf-8 -*-
#
# Copyright © 2021 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2021-07-25
#
# Distributed under terms of the MIT license.

import argparse
import logging
import math
import sys
from functools import reduce

logging.basicConfig(
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

opt_parser = argparse.ArgumentParser(description="")
opt_parser.add_argument("--text", metavar="PATH", help="")
opt_parser.add_argument("--scores", metavar="PATH", help="")
opt_parser.add_argument("--format", choices=['bpe', 'sentencepiece'], default='sentencepiece', help="")


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

def main(options):
  text_file = open(options.text)
  scores_file = open(options.scores)
  for text_line, scores_line in zip(text_file, scores_file):
    scores = [ float(score) for score in scores_line.strip().split() ]
    if options.format == "bpe":
      toks = text_line.strip().split()
      assert len(scores) == (2 * len(toks)) + 1
      idx_map, new_toks = debpe(toks)
    elif options.format == "sentencepiece":
      toks = text_line.strip().split()
      assert len(scores) == (2 * len(toks)) + 1
      idx_map, new_toks = debpe_sentencepiece(toks)
    else:
      raise NotImplementedError

    inv_idx_map = {}
    for k, v in idx_map.items():
      inv_idx_map[v] = inv_idx_map.get(v, []) + [k]

    new_scores = [ None ] * (2 * len(new_toks) + 1)
    for k, v in inv_idx_map.items():
      v = sorted(inv_idx_map[k])
      old_scores = []
      for i in v:
        old_scores.append(scores[2*i])
        old_scores.append(scores[2*i+1])
      new_ins_score = old_scores[0]
      new_del_score = reduce((lambda x, y: x * y), old_scores[1:])
      new_scores[2*k] = new_ins_score
      new_scores[2*k+1] = new_del_score
    new_scores[-1] = scores[-1]

    # sys.stdout.write(" ".join(new_toks) + "\n")
    new_scores_str = [ "{:.4f}".format(new_score) for new_score in new_scores ]
    sys.stdout.write(" ".join(new_scores_str) + "\n")


if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
      opt_parser.parse_known_args()[1]))

  main(options)
