# -*- coding: utf-8 -*-
#
# Copyright © 2020 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2020-06-23
#
# Distributed under terms of the MIT license.

import sys

for line in sys.stdin:
  tokens = line.strip().split()
  # tokens += [ "OK" ]
  for token in tokens:
    sys.stdout.write(token + '\n')
