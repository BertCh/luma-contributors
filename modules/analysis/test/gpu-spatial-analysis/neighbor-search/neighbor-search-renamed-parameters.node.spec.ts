// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {packGPUSpatialParameterValues} from '../../../src/analysis-core/contracts/parameter-schema';
import {GPU_NEIGHBOR_SEARCH_PARAMETER_SCHEMA} from '../../../src/gpu-spatial-analysis/neighbor-search/neighbor-search-parameters';

it('NAM-13 the neighbour-search schema packer names the replacement for retired keys', () => {
  expect(() =>
    packGPUSpatialParameterValues(GPU_NEIGHBOR_SEARCH_PARAMETER_SCHEMA, {radius: 5})
  ).toThrow('radius was renamed; use maximumDistance');
  expect(() =>
    packGPUSpatialParameterValues(GPU_NEIGHBOR_SEARCH_PARAMETER_SCHEMA, {weightKind: 1})
  ).toThrow('weightKind was renamed; use weightType');
  const packed = packGPUSpatialParameterValues(GPU_NEIGHBOR_SEARCH_PARAMETER_SCHEMA, {
    maximumDistance: 5
  });
  expect(packed[4]).toBe(5);
});
