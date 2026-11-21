# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/point-process/balanced-sample-output-oracles.oracle.py
# Random123 integer definition; identical stratum-3 locations keep source-row tie order [0,2].
import json
counter = [3, 0x80000000, 0, 0]
key = [79, 0x53425350]
mask = (1 << 32) - 1
for iteration in range(10):
    first, second = 0xD2511F53 * counter[0], 0xCD9E8D57 * counter[2]
    counter = [(second >> 32) ^ counter[1] ^ key[0], second & mask,
               (first >> 32) ^ counter[3] ^ key[1], first & mask]
    key = [(key[0] + 0x9E3779B9) & mask, (key[1] + 0xBB67AE85) & mask]
shift = counter[0] % 2  # Rejection threshold for a two-member stratum is zero.
print(json.dumps({'word': counter[0], 'shift': shift, 'selected_ids': [29, [71, 83][shift]],
                  'probabilities': [1, 0.5], 'weights': [1, 2]}))
