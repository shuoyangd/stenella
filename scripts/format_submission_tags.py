# -*- coding: utf-8 -*-
#
# Copyright © 2021 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2021-07-22
#
# Distributed under terms of the MIT license.

import argparse
import logging
import os
import pdb

logging.basicConfig(
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

opt_parser = argparse.ArgumentParser(description="Format WMT20-style Submission Tags for WMT21")
opt_parser.add_argument("--disk-footprint", required=True, type=str, help="desk top footprint, in bytes, w/o compression")
opt_parser.add_argument("--num-params", required=True, type=str, help="number of the model parameters")
opt_parser.add_argument("--lang-pair", required=True, type=str, help="language pair of the submission, e.g. en-de")
opt_parser.add_argument("--method-name", required=True, type=str, help="name of the method, e.g. LevT")
opt_parser.add_argument("--text-file", required=True, metavar="PATH", type=str, help="dir to the MT text")
opt_parser.add_argument("--label-file", required=True, metavar="PATH", type=str, help="dir to the label file")
opt_parser.add_argument("--out-dir", required=True, metavar="PATH", type=str, help="dir to the translated output (will output three files to that dir)")


def process_line(label_line, text_line):
  """
  This will process each line and give you the last three fields corresponding to this line.
  However, you still need to prepend the other fields (which requires some global info).
  """

  labels = label_line.split(' ')
  tokens = text_line.split(' ')
  tokens = filter(lambda tok: tok != '', tokens)

  word_ret = []
  gap_ret = [ (0, "gap", labels[0]) ]
  for idx, token in enumerate(tokens):
    word_ret.append((idx, token, labels[idx * 2 + 1]))
    gap_ret.append((idx, "gap", labels[idx * 2 + 2]))
  return word_ret, gap_ret


def main(options):
  word_label_file = open(os.path.join(options.out_dir, "predictions_mt.txt"), 'w')
  gap_label_file = open(os.path.join(options.out_dir, "predictions_gaps.txt"), 'w')
  word_label_file.write(str(options.disk_footprint) + '\n')
  word_label_file.write(str(options.num_params) + '\n')
  gap_label_file.write(str(options.disk_footprint) + '\n')
  gap_label_file.write(str(options.num_params) + '\n')

  wmt20_label_file = open(options.label_file)
  text_file = open(options.text_file)
  for idx, (label_line, text_line) in enumerate(zip(wmt20_label_file, text_file)):
    word_ret, gap_ret = process_line(label_line.strip(), text_line.strip())
    for item in word_ret:
      word_label_file.write("\t".join([
          options.lang_pair,
          options.method_name,
          "mt",
          str(idx),
          str(item[0]),
          item[1],
          item[2],
        ]) + '\n'
      )
    for item in gap_ret:
      gap_label_file.write("\t".join([
          options.lang_pair,
          options.method_name,
          "gap",
          str(idx),
          str(item[0]),
          item[1],
          item[2],
        ]) + '\n'
      )

  word_label_file.close()
  gap_label_file.close()


if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
      opt_parser.parse_known_args()[1]))

  main(options)
