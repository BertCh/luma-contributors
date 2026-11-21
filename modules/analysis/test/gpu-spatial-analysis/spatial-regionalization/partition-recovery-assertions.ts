// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {canonicalizePartition} from './spatial-regionalization-oracle';

/** Check both recovered components, allowing only a permutation of the two dense label IDs. */
export function expectRecoveredPairLabels(labels: readonly number[]): void {
  for (const label of labels) {
    expect(label).toBeGreaterThanOrEqual(0);
    expect(label).toBeLessThan(2);
  }
  expect(canonicalizePartition(labels)).toEqual([0, 0, 1, 1]);
}
