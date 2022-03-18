# -*- coding: utf-8 -*-
#
# Copyright © 2021 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2021-07-12
#
# Distributed under terms of the MIT license.

import argparse
import logging
import numpy as np
import scipy as sp
import os
import sys
from math import log2
from word_evaluate import evaluate

logging.basicConfig(
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

opt_parser = argparse.ArgumentParser(description="powell ensemble of binary classifiers")
opt_parser.add_argument("--score", metavar='files', type=str, required=True,
                        help="score file from the model")
opt_parser.add_argument("--ref", metavar='file', type=str,
                        help="reference tags to compute mcc")
opt_parser.add_argument("--weights", metavar='lambdas', type=float, nargs='+', default=[],
                        help="if this is provided, will output ensembled labels; otherwise, will use powell search to find weights")


def read_score_file(file_name):
  scores = []
  with open(file_name) as score_file:
    for line in score_file:
      line = line.strip()
      scores.append(np.array([ float(score) for score in line.split(' ') ]))
  return scores


def score2label(scores, stream, threshold):
  log_threshold = log2(threshold)
  for line_scores in scores:
    line = ""
    for score in line_scores:
      if score - log_threshold < 0:
        line += "BAD "
      else:
        line += "OK "
    line = line.strip()
    stream.write(line + "\n")


def mcc_wrapper(threshold, scores, ref_dir, tmp_dir="/tmp"):
  if not 0.0 < threshold < 1.0:
    return 999 # really large number, so it'll move away
  print(threshold)
  tmp_label_dir = os.path.join(tmp_dir, "tmp_qe_labels")
  tmp_label_file = open(os.path.join(tmp_dir, "tmp_qe_labels"), 'w')
  score2label(scores, tmp_label_file, threshold)
  tmp_label_file.close()
  _, _, mcc_tg = evaluate(ref_dir, ref_dir, tmp_label_dir)
  return -mcc_tg  # scipy needs to minimize, hence the negative sign


def main(options):
  scores = read_score_file(options.score)
  res = sp.optimize.minimize(mcc_wrapper,
                       np.array([ 0.5 ]),
                       (scores, options.ref),
                       method="Powell",
                       bounds=sp.optimize.Bounds(np.array([0.]), np.array([1.])),
                       options={"maxiter": 1000})

#   initial_simplex = np.vstack([np.identity(1), np.zeros(1)])
#   res = sp.optimize.minimize(mcc_wrapper,
#                        np.array([ 1. ]),
#                        (scores_ensemble, options.ref),
#                        method="Nelder-Mead",
#                        # bounds=sp.optimize.Bounds(np.array([0.]), np.array([1.])),
#                        options={"maxiter": 1000, "initial_simplex": initial_simplex})
  print(res)
  print(" ".join(["{:.4f}".format(weight) for weight in res['x']]))


if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
      opt_parser.parse_known_args()[1]))

  main(options)
