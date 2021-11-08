# -*- coding: utf-8 -*-
#
# Copyright © 2020 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2021-02-01
#
# Distributed under terms of the MIT license.

import argparse
import logging
import math
import pdb

from collections import Counter

logging.basicConfig(
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

opt_parser = argparse.ArgumentParser(description="")
opt_parser.add_argument("--score-file", "-sf", required=True, help="file that holds confidence scores, one score per line")
opt_parser.add_argument("--label-file", "-lf", required=True, help="file that holds ok/bad labels, one label per line")
opt_parser.add_argument("--intervals", "-i", type=float, default=10, help="number of intervals for calibration (default=10)")
opt_parser.add_argument("--log", action='store_true', default=False, help="scores are in log scale (default=False)")
opt_parser.add_argument("--log2", action='store_true', default=False, help="scores are in log scale (default=False)")

def main(options):
  ok_bin = []
  bad_bin = []
  for score_line, label_line in zip(open(options.score_file), open(options.label_file)):
    if label_line.strip() == "OK":
      ok_bin.append(float(score_line.strip()))
    if label_line.strip() == "BAD":
      bad_bin.append(float(score_line.strip()))
  assert not (options.log and options.log2)

  ok_bin = sorted(ok_bin)
  bad_bin = sorted(bad_bin)

  conf_cnt_ok = Counter()
  cur_conf_lb = 0.0
  for elem in ok_bin:
    if options.log:
      elem = math.exp(elem)
    elif options.log2:
      elem = math.pow(2, elem)
    if cur_conf_lb < elem < cur_conf_lb + 1 / options.intervals:
      conf_cnt_ok[cur_conf_lb] += 1
    else:
      cur_conf_lb = math.floor(elem * options.intervals) / options.intervals
      conf_cnt_ok[cur_conf_lb] += 1

  conf_cnt_bad = Counter()
  cur_conf_lb = 0.0
  for elem in bad_bin:
    if options.log:
      elem = math.exp(elem)
    elif options.log2:
      elem = math.pow(2, elem)
    if cur_conf_lb < elem < cur_conf_lb + 1 / options.intervals:
      conf_cnt_bad[cur_conf_lb] += 1
    else:
      cur_conf_lb = math.floor(elem * options.intervals) / options.intervals
      conf_cnt_bad[cur_conf_lb] += 1

  print("OK counts")
  print(conf_cnt_ok)
  print("BAD counts")
  print(conf_cnt_bad)
  fracs = []
  ece = 0.0
  total_n = sum(conf_cnt_ok.values()) + sum(conf_cnt_bad.values())
  for key in [ idx/options.intervals for idx in range(options.intervals)]:
    if conf_cnt_ok[key] + conf_cnt_bad[key] == 0:
      print("bin range {0} - {1} doesn't have an element".format(key, key + 1 / options.intervals))
    else:
      frac = conf_cnt_ok[key] / (conf_cnt_ok[key] + conf_cnt_bad[key])
      fracs.append(frac)
      err = abs(frac - (key + 1.0/(2 * options.intervals)))
      ece += (conf_cnt_ok[key] + conf_cnt_bad[key]) / total_n * err * 100
  for frac in fracs:
      print(format(key, '.1f'), format(frac, '.3f'))
  print("ECE: {0}".format(ece))

if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
      opt_parser.parse_known_args()[1]))

  main(options)
