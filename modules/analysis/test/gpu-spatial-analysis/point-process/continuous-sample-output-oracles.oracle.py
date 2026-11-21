# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/point-process/continuous-sample-output-oracles.oracle.py
# Independent integer Random123 Philox4x32-10 definition, with arbitrary precision products.
# The line [0,2] -> [2,2] has length 2, spacing 1/2: x = phase + ordinal/2.
import json
import struct

mask = (1 << 32) - 1
counter = [0, 0, 0, 0]
key = [1, 0x43535350]
for iteration in range(10):
    first = 0xD2511F53 * counter[0]
    second = 0xCD9E8D57 * counter[2]
    counter = [(second >> 32) ^ counter[1] ^ key[0], second & mask,
               (first >> 32) ^ counter[3] ^ key[1], first & mask]
    key = [(key[0] + 0x9E3779B9) & mask, (key[1] + 0xBB67AE85) & mask]
phase = (counter[0] >> 8) / (1 << 25)
float32 = lambda value: struct.unpack('f', struct.pack('f', value))[0]
print(json.dumps({'words': counter, 'phase': phase,
                  'positions': [coordinate for ordinal in range(4)
                                for coordinate in [float32(phase + ordinal / 2), 2]] + [0, 0]}))
