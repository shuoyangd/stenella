#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Shuoyang Ding <shuoyangd@gmail.com>
#
# Distributed under terms of the MIT license.

import sys

for line in sys.stdin:
  scores = [ float(tok) for tok in line.strip().split() ]
  tags = [ "OK" if score < 0.5 else "BAD" for score in scores ]
  sys.stdout.write(" ".join(tags) + "\n")
