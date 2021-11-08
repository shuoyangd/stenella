# -*- coding: utf-8 -*-
#
# Copyright © 2021 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2021-02-11
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
opt_parser.add_argument("--tag-file", "-tf", required=True, help="file that holds system predictions, one label per line")
opt_parser.add_argument("--ref-file", "-rf", required=True, help="file that holds reference ok/bad labels, one label per line")


def f1(prec, recl):
  return 2 * (prec * recl) / (prec + recl)


def main(options):
  tf = open(options.tag_file, 'r')
  rf = open(options.ref_file, 'r')

  ok_correct = 0
  ok_label_total = 0
  ok_pred_total = 0

  bad_correct = 0
  bad_label_total = 0
  bad_pred_total = 0

  for idx, (tl, rl) in enumerate(zip(tf, rf)):
    tag = tl.strip()
    rl = rl.strip()

    if tag == "OK":
      ok_pred_total += 1
      if rl == "OK":
        ok_correct += 1
        ok_label_total += 1
      else:
        bad_label_total += 1
    elif tag == "BAD":
      bad_pred_total += 1
      if rl == "BAD":
        bad_correct += 1
        bad_label_total += 1
      else:
        ok_label_total += 1
    else:
      logging.error("line {0}: tag should either have value OK/BAD, but has value {1}".format(idx, tag))

  if not (tf.read() == rf.read() == ''):
    logging.error("Your tag and reference file are of different length. You should fix that first.")

  ok_prec = ok_correct / ok_pred_total
  ok_recl = ok_correct / ok_label_total
  ok_f1 = f1(ok_prec, ok_recl)
  sys.stdout.write("p/r/f for ok label: {0:.4f}/{1:.4f}/{2:.4f}\n".format(ok_prec, ok_recl, ok_f1))

  bad_prec = bad_correct / bad_pred_total
  bad_recl = bad_correct / bad_label_total
  bad_f1 = f1(bad_prec, bad_recl)
  sys.stdout.write("p/r/f for bad label: {0:.4f}/{1:.4f}/{2:.4f}\n".format(bad_prec, bad_recl, bad_f1))

  sys.stdout.write("macro-f1: {0:.4f}\n".format(ok_f1 * bad_f1))


if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
      opt_parser.parse_known_args()[1]))

  main(options)
