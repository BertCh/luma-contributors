// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getSpatialErrorLikelihoodAllocationLengths} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-likelihood-allocations';

it('SSR-126 uses the shared default and preserves chunk-local information-probe tiles', () => {
  const base = {
    rowCount: 8192,
    edgeCount: 8192,
    predictorCount: 1,
    maximumPower: 3,
    traceMode: 'stochastic' as const,
    traceProbeCount: 3
  };
  const defaults = getSpatialErrorLikelihoodAllocationLengths(base);
  // ceil(8192/64)=128 whole-row tiles, three information probes.
  expect(defaults['likelihood norm partials']).toBe(128);
  expect(defaults['information probe partials']).toBe(384);
  const chunked = getSpatialErrorLikelihoodAllocationLengths({
    ...base,
    tileRowCount: 2,
    tileCount: 4097
  });
  // Chunks of 1 and 8191 rows require 1+ceil(8191/2)=4097 tiles, two merge groups.
  expect(chunked['likelihood norm partials']).toBe(4097);
  expect(chunked['information probe partials']).toBe(12291);
  expect(chunked['information probe merge groups']).toBe(6);
});
