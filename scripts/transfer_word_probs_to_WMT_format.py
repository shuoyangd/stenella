# -*- coding: utf-8 -*-
#
# Copyright © 2021 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2021-07-25
#
# Distributed under terms of the MIT license.

import math
import pdb
import sys

for line in sys.stdin:
  toks = line.strip().split()
  probs = [ math.exp(float(tok)) for tok in toks ]
  wmt_scores = [0.0] * (2 * len(probs) - 1)  # this prob has eos
  wmt_scores[1::2] = probs[:-1]
  wmt_scores[-1] = probs[-1]
  wmt_scores = [ '{:.4f}'.format(score) for score in wmt_scores ]
  print(' '.join(wmt_scores))
