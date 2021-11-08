# -*- coding: utf-8 -*-
#
# Copyright © 2021 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2021-05-04
#
# Distributed under terms of the MIT license.

import sys

linen_file = sys.argv[1]
lines = [ int(line.strip()) for line in open(linen_file) ]

cur_out_idx = 0
for idx, line in enumerate(sys.stdin):
  idx = idx + 1
  if idx == lines[cur_out_idx]:
    sys.stdout.write(line)
    cur_out_idx += 1
    if cur_out_idx >= len(lines):
      break
