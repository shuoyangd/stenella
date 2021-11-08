#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Shuoyang Ding <shuoyangd@gmail.com>
#
# Distributed under terms of the MIT license.

import copy
import sys

fwd = sys.argv[1]
bwd = sys.argv[2]

for fl, bl in zip(open(fwd), open(bwd)):
    fps = [ float(num) for num in fl.split(' ') ]
    bps = [ float(num) for num in bl.split(' ') ]
    assert len(fps) == len(bps), "{0}, {1}".format(len(fps), len(bps))
    bps_rev = list(reversed(bps))
    aps = copy.copy(fps)
    for idx in range(len(fps) - 1):
      aps[idx] = (bps_rev[idx + 1] + fps[idx]) / 2
    ap_strs = [ "{:.4f}".format(num) for num in aps ]
    sys.stdout.write( " ".join(ap_strs) + '\n' )
