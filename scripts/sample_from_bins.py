# -*- coding: utf-8 -*-
#
# Copyright © 2021 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2021-05-04
#
# Distributed under terms of the MIT license.

import argparse
import logging
import pdb
import random
import sys

logging.basicConfig(
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

opt_parser = argparse.ArgumentParser(description="Generate sampled lines from each bin")
opt_parser.add_argument("--bin-file", required=True, type=str, metavar="PATH",
                        help="ranges of the bins, in the format like \"0.0 0.05\", one bin each line")
opt_parser.add_argument("--frac-file", required=True, type=str, metavar="PATH",
                        help="target fraction of lines in this bin, one bin each line")
opt_parser.add_argument("--metric-file", required=True, type=str, metavar="PATH",
                        help="original line number with metric for binning, sorted low to high by the metric -- each line will look like \"0.314 1984\"")
opt_parser.add_argument("--total", required=True, type=int, metavar="N",
                        help="target number of lines in the sampled file")


def read_bin_file(path):
  bins = []
  with open(path) as f:
    for line in f:
      bin_ = [ float(n) for n in line.strip().split() ]
      bins.append(tuple(bin_))
  return bins


def read_frac_file(path):
  fracs = []
  with open(path) as f:
    for line in f:
      frac = float(line.strip())
      fracs.append(frac)
  return fracs


def read_metric_file(path):
  metrics = []
  with open(path) as f:
    for line in f:
      fields = line.strip().split()
      metric = (float(fields[0]), int(fields[1]))
      metrics.append(metric)
  return metrics


def get_sample(bins, fracs, metrics, total):
  cur_bin_idx = 0
  bin_boundaries = []  # upper limit index of the bin (not include)
  for idx, metric in enumerate(metrics):
    bin_ = bins[cur_bin_idx]
    if bin_[0] <= metric[0] < bin_[1]:
      continue
    else:
      while cur_bin_idx < len(bins) - 1 and \
          not bin_[0] <= metric[0] < bin_[1]:
        bin_boundaries.append(idx)
        cur_bin_idx += 1
        bin_ = bins[cur_bin_idx]
  bin_boundaries.append(len(metrics))

  bin_cardinalities = [bin_boundaries[0]]
  for idx in range(1, len(bin_boundaries)):
    bin_cardinalities.append(bin_boundaries[idx] - bin_boundaries[idx-1])

  sample_sizes = [ int(total * frac) for frac in fracs ]
  escape = total - sum(sample_sizes)
  sample_sizes[-1] += escape

  assert len(bin_cardinalities) == len(fracs)
  repeats = [ ss // bc for ss, bc in zip(sample_sizes, bin_cardinalities) ]
  # amounts that I actually need to sample, instead of repeating
  effective_sample_sizes = [ ss % bc for ss, bc in zip(sample_sizes, bin_cardinalities) ]

  samples = []
  bin_start = 0
  for bin_idx, bin_end in enumerate(bin_boundaries):
    logging.info("processing bin {0}".format(bin_idx))
    space = metrics[bin_start: bin_end]
    bin_samples = [ sample[1] for sample in random.sample(space, effective_sample_sizes[bin_idx])]
    repeat = repeats[bin_idx]
    bin_samples.extend([ sample[1] for sample in space * repeat ])
    samples.extend(bin_samples)
    bin_start = bin_end

  return sorted(samples)


def main(options):
  bins = read_bin_file(options.bin_file)
  fracs = read_frac_file(options.frac_file)
  metrics = read_metric_file(options.metric_file)
  samples = get_sample(bins, fracs, metrics, options.total)

  for sample in samples:
    sys.stdout.write("{0}\n".format(sample))


if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
      opt_parser.parse_known_args()[1]))

  main(options)
