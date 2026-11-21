// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {expectRecoveredPairLabels} from './partition-recovery-assertions';

it('rejects stale missing or out-of-range labels before canonicalizing recovered pairs', () => {
  for (const labels of [
    [0xffffffff, 0xffffffff, 0, 0],
    [0, 0, 0xffffffff, 0xffffffff],
    [0, 0, 2, 2]
  ])
    expect(() => expectRecoveredPairLabels(labels)).toThrow();
  expectRecoveredPairLabels([0, 0, 1, 1]);
  expectRecoveredPairLabels([1, 1, 0, 0]);
});
