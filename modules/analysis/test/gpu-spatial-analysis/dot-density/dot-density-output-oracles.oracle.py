# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/dot-density/dot-density-output-oracles.oracle.py
# Independent integer-arithmetic Philox4x32-10 (Random123 algorithm), checked
# against its published all-zero known-answer vector. No production code imported.
import math
def philox(counter, key):
    mask = (1 << 32) - 1
    for _ in range(10):
        first = counter[0] * 0xD2511F53
        second = counter[2] * 0xCD9E8D57
        counter = [(second >> 32) ^ counter[1] ^ key[0], second & mask,
                   (first >> 32) ^ counter[3] ^ key[1], first & mask]
        key = [(key[0] + 0x9E3779B9) & mask, (key[1] + 0xBB67AE85) & mask]
    return counter
assert philox([0]*4, [0,0]) == [0x6627e8d5,0xe169c58d,0xbc57ac4c,0x9b00dbd8]
for rank in range(3):
    words = philox([0,rank,0,0], [123,7])
    root = math.sqrt((words[1] >> 8) / 2**24)
    fraction = (words[2] >> 8) / 2**24
    print('polygon ordered', rank, words, [1-root,root*(1-fraction),root*fraction])
print('line:', [(philox([0,rank,0,0],[7,3])[0] >> 8)/2**24 for rank in range(3)])
