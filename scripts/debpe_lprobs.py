# -*- coding: utf-8 -*-
#
# Copyright © 2020 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2020-10-12
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
opt_parser.add_argument("--text", metavar="PATH", help="")
opt_parser.add_argument("--lprob", metavar="PATH", help="")
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
  lprob_file = open(options.lprob)
  for text_line, lprob_line in zip(text_file, lprob_file):
    lprobs = [ float(lprob) for lprob in lprob_line.strip().split() ]
    if options.format == "bpe":
      toks = text_line.strip().split() + ["<eos>"]
      assert len(toks) == len(lprobs)
      idx_map, new_toks = debpe(toks)
    elif options.format == "sentencepiece":
      toks = text_line.strip().split() + ["▁<eos>"]
      assert len(toks) == len(lprobs)
      idx_map, new_toks = debpe_sentencepiece(toks)
    else:
      raise NotImplementedError

    inv_idx_map = {}
    for k, v in idx_map.items():
      inv_idx_map[v] = inv_idx_map.get(v, []) + [lprobs[k]]

    lprobs = [ 0.0 ] * len(new_toks)
    for k, v in inv_idx_map.items():
        lprobs[k] = sum(v)
    # sys.stdout.write(" ".join(new_toks) + "\n")
    lprobs_str = [ format(lprob, ".3f") for lprob in lprobs ]
    sys.stdout.write(" ".join(lprobs_str) + "\n")


if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
      opt_parser.parse_known_args()[1]))

  main(options)
