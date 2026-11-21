// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getRegressionTileLayout,
  getRowTileMergeNodes,
  REGRESSION_TILE_MAXIMUM_TILE_COUNT,
  REGRESSION_TILE_MINIMUM_DEFAULT_ROW_COUNT
} from '../../../src/gpu-spatial-analysis/spatial-regression/regression-row-tiles';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('resolves the shared default and preserves explicit regression widths', () => {
  expect(REGRESSION_TILE_MINIMUM_DEFAULT_ROW_COUNT).toBe(64);
  expect(getRegressionTileLayout(4128)).toEqual({tileRowCount: 64, tileCount: 65});
  expect(getRegressionTileLayout(4128, 32)).toEqual({tileRowCount: 32, tileCount: 129});
  expect(getRegressionTileLayout(2 ** 24)).toEqual({tileRowCount: 4096, tileCount: 4096});
});

it('plans the actual hierarchical tile ceiling without allocating maximum partials', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const merge = {
      id: 'hierarchical-bound',
      operation: 'MergeBoundary',
      invocationCount: 1,
      bindings: [],
      body: ''
    };
    // Empty channels exercise the real count guard without a maximum-sized source or scratch.
    expect(REGRESSION_TILE_MAXIMUM_TILE_COUNT).toBe(16777216); // 4096 * 4096.
    expect(getRowTileMergeNodes(graph, merge, {tileCount: 16777216, partials: []})).toHaveLength(1);
    expect(() => getRowTileMergeNodes(graph, merge, {tileCount: 16777217, partials: []})).toThrow(
      /hierarchical-bound.*tileCount.*16777216/
    );
  } finally {
    device.destroy();
  }
});
