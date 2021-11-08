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
opt_parser.add_argument("--score-file", "-sf", required=True, help="file that holds confidence scores, one score per line")
opt_parser.add_argument("--label-file", "-lf", required=True, help="file that holds ok/bad labels, one label per line")
opt_parser.add_argument("--log", action='store_true', default=False, help="scores are in log scale (default=False)")
opt_parser.add_argument("--log2", action='store_true', default=False, help="scores are in log scale (default=False)")
opt_parser.add_argument("--threshold", type=float, default=0.5, help="threshold for OK/BAD labels (default=0.5)")


def f1(prec, recl):
  return 2 * (prec * recl) / (prec + recl)


def main(options):
  sf = open(options.score_file, 'r')
  lf = open(options.label_file, 'r')

  ok_correct = 0
  ok_label_total = 0
  ok_pred_total = 0

  bad_correct = 0
  bad_label_total = 0
  bad_pred_total = 0

  for idx, (sl, ll) in enumerate(zip(sf, lf)):
    sl = sl.strip()
    ll = ll.strip()

    score = float(sl)
    if options.log:
      score = math.exp(score)
    elif options.log2:
      score = pow(2, score)

    if options.threshold < score <= 1:
      ok_pred_total += 1
      if ll == "OK":
        ok_correct += 1
        ok_label_total += 1
      else:
        bad_label_total += 1
    elif 0.0 < score <= options.threshold:
      bad_pred_total += 1
      if ll == "BAD":
        bad_correct += 1
        bad_label_total += 1
      else:
        ok_label_total += 1
    else:
      logging.error("line {0}: score should be in the region of 0-1, but has value {1}".format(idx, score))

  if not (sf.read() == lf.read() == ''):
    logging.error("Your score and tag file are of different length. You should fix that first.")

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
