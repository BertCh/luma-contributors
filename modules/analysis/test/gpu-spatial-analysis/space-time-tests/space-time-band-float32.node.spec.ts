// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getSpaceTimeBandWGSL,
  validateSpaceTimeBandEdges
} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-bands';

it('rejects finite JavaScript band edges that overflow WGSL float32 literals', () => {
  for (const edge of [1e40, Number.MAX_VALUE, Infinity, NaN]) {
    expect(() => validateSpaceTimeBandEdges('bands', 'timeBands', [0, edge])).toThrow(
      'bands timeBands edges must be finite and non-negative'
    );
  }
  const edges = [0, 1, 3.4e38];
  expect(() => validateSpaceTimeBandEdges('bands', 'timeBands', edges)).not.toThrow();
  expect(getSpaceTimeBandWGSL('Time', edges)).not.toMatch(/Infinity|NaN/);
});
