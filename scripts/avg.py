# -*- coding: utf-8 -*-
#
# Copyright © 2021 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2021-01-18
#
# Distributed under terms of the MIT license.

import sys

files = [ open(filename) for filename in sys.argv[1:] ]
for numbers in zip(*files):
    numbers = [ float(number) for number in numbers ]
    avg = sum(numbers) / len(numbers)
    print(avg)

