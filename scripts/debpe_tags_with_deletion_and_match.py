# -*- coding: utf-8 -*-
#
# Copyright © 2020 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2021-03-25
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
opt_parser.add_argument("--text", required=True, metavar="PATH", help="")
opt_parser.add_argument("--tags", required=True, metavar="PATH", help="")
opt_parser.add_argument("--format", choices=['bpe', 'sentencepiece'], default='sentencepiece', help="")
opt_parser.add_argument("--tok-text", metavar="PATH", default="", help="A tokenized version of the text, so we can match the format")


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

def debpe_sentencepiece_and_match(toks, matching_toks):
  idx_map = {}
  new_idx = -1
  new_toks = []
  for old_idx, tok in enumerate(toks):
    if old_idx == 0 or tok.startswith("▁"):
      # if this is not the first token, checked whether the previous token
      # is the same as the matching token
      if new_idx != -1:
        assert new_toks[new_idx] == matching_toks[new_idx], \
            "sentences diverge at {0}, {1}, bped sentence: {2}".format(
                " ".join(new_toks[:new_idx+1]),
                " ".join(matching_toks[:new_idx+1]),
                " ".join(toks[:old_idx])
            )
      # pre-sentencepiece text already has a space, respect it
      new_idx += 1
      idx_map[old_idx] = new_idx
      new_toks.append(tok[1:] if tok.startswith("▁") else tok)
    else:
      # pre-sentencepiece text don't have a space,
      # but if tokenized data has, match it
      if new_idx+1 < len(matching_toks) and \
            tok == matching_toks[new_idx+1]:
        new_idx += 1
        idx_map[old_idx] = new_idx
        new_toks.append(tok)
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
  tags_file = open(options.tags)
  if options.tok_text != "":
      tok_file = open(options.tok_text)
  else:
      tok_file = None

  for text_line, tags_line in zip(text_file, tags_file):
    if tok_file is not None:
        tok_line = tok_file.readline()
        matching_toks = tok_line.strip().split()
    tags = tags_line.strip().split()
    if options.format == "bpe":
      toks = text_line.strip().split()
      assert len(tags) == (2 * len(toks)) + 1
      idx_map, new_toks = debpe(toks)
    elif options.format == "sentencepiece":
      toks = text_line.strip().split()
      assert len(tags) == (2 * len(toks)) + 1
      if tok_file is not None:
        idx_map, new_toks = debpe_sentencepiece_and_match(toks, matching_toks)
      else:
        idx_map, new_toks = debpe_sentencepiece(toks)
    else:
      raise NotImplementedError

    inv_idx_map = {}
    for k, v in idx_map.items():
      inv_idx_map[v] = inv_idx_map.get(v, []) + [k]

    new_tags = [ None ] * (2 * len(new_toks) + 1)
    for k, v in inv_idx_map.items():
      v = sorted(inv_idx_map[k])
      old_tags = []
      for i in v:
        old_tags.append(tags[2*i])
        old_tags.append(tags[2*i+1])
      new_ins_tag = old_tags[0]
      new_del_tag = "BAD" if "BAD" in old_tags[1:] else "OK"
      new_tags[2*k] = new_ins_tag
      new_tags[2*k+1] = new_del_tag
    new_tags[-1] = tags[-1]

    # sys.stdout.write(" ".join(new_toks) + "\n")
    sys.stdout.write(" ".join(new_tags) + "\n")


if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
      opt_parser.parse_known_args()[1]))

  main(options)
