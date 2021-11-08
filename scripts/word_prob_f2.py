#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Shuoyang Ding <shuoyangd@gmail.com>
#
# Distributed under terms of the MIT license.

import sys

fwd1 = sys.argv[1]
fwd2 = sys.argv[2]

for f1l, f2l in zip(open(fwd1), open(fwd2)):
    f1ps = [ float(num) for num in f1l.split(' ') ]
    f2ps = [ float(num) for num in f2l.split(' ') ]
    assert len(f1ps) == len(f2ps), "{0}, {1}".format(len(f1ps), len(f2ps))
    aps = [ p1 + p2 for p1, p2 in zip(f1ps, f2ps) ]
    ap_strs = [ "{:.4f}".format(num) for num in aps ]
    sys.stdout.write( " ".join(ap_strs) + '\n' )
