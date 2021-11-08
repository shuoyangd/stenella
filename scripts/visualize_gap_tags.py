#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Shuoyang Ding <shuoyangd@gmail.com>
#
# Distributed under terms of the MIT license.


import argparse
import logging
import sys

logging.basicConfig(
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

opt_parser = argparse.ArgumentParser(description="")
opt_parser.add_argument("--text", metavar="PATH", help="")
opt_parser.add_argument("--tags", metavar="PATH", help="")

def main(options):
  text_file = open(options.text)
  tags_file = open(options.tags)
  for text_line, tags_line in zip(text_file, tags_file):
    toks = text_line.strip().split(' ')
    tags = tags_line.strip().split(' ')
    new_toks = [''] * (2 * len(toks) + 1)
    new_toks[1::2] = toks
    assert len(new_toks) == len(tags)
    sys.stdout.write(", ".join(new_toks) + '\n')
    sys.stdout.write(", ".join(tags) + '\n')


if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
      opt_parser.parse_known_args()[1]))

  main(options)

