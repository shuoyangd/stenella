#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Shuoyang Ding <shuoyangd@gmail.com>
#
# Distributed under terms of the MIT license.

import sys

if len(sys.argv) > 1:
    thres = float(sys.argv[1])
else:
    thres = 0.5

for line in sys.stdin:
  scores = [ pow(2, float(tok)) for tok in line.strip().split() ]
  tags = [ "OK" if score > thres else "BAD" for score in scores ]
  sys.stdout.write(" ".join(tags) + "\n")
