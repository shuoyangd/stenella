# -*- coding: utf-8 -*-
#
# Copyright © 2021 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2021-05-05
#
# Distributed under terms of the MIT license.

import sys

for line in sys.stdin:
  tokens = line.strip().split()
  sys.stdout.write(" ".join(tokens[0::2]) + "\n")
