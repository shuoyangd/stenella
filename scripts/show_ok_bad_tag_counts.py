#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Shuoyang Ding <shuoyangd@gmail.com>
#
# Distributed under terms of the MIT license.

from math import log
import sys

f = open(sys.argv[1])
if len(sys.argv) >= 3:
  thresh = float(sys.argv[2])
else:
  thresh = 0.5

scores = f.readlines()
scores = [ float(l.strip()) for l in scores ]
ok_count = sum([ 1 if score > log(thresh) else 0 for score in scores ])
bad_count = len(scores) - ok_count
sys.stdout.write("#(ok) = {0}, #(bad) = {1}\n".format(ok_count, bad_count))
f.close()
