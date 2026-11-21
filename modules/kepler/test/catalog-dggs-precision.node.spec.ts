// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUPointToCell} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {createNullWebGPUDevice} from '../../analysis/test/utils/gpu-contributor-test-utils';
import {PLATFORM4_DGGS_BINDINGS} from '../src/analysis/catalog-platform4-dggs';
import {CatalogGpuSession} from '../src/analysis/catalog-gpu-session';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';

const points: CatalogFeatureCollection = {
  type: 'FeatureCollection',
  features: [{geometry: {type: 'Point', coordinates: [0, 0]}, properties: {v: 1}}]
};

test('NAM-06 catalog indexing, aggregation, rollup and both comparison inputs select supported precision', async () => {
  const device = createNullWebGPUDevice();
  const stop = new Error('declaration complete; no GPU submission');
  const run = vi.spyOn(CatalogGpuSession.prototype, 'run').mockImplementation(() => {
    throw stop;
  });
  const add = vi.spyOn(GPUCommandGraph.prototype, 'add');
  try {
    for (const id of [
      'dggs-point-index',
      'dggs-cell-aggregation',
      'dggs-cell-rollup',
      'dggs-cell-compare'
    ]) {
      const binding = PLATFORM4_DGGS_BINDINGS.find(candidate => candidate.id === id)!;
      const families =
        id === 'dggs-point-index'
          ? ['h3', 'quadbin', 'quadkey', 'geohash', 's2']
          : ['h3', 'quadbin'];
      for (const family of families) {
        add.mockClear();
        await expect(
          binding.runGpu!(
            {device},
            {points, before: points, after: points},
            {
              family,
              resolution: 3,
              valueProperty: 'v',
              levels: 2,
              levelStep: 1,
              measure: 'count'
            }
          )
        ).rejects.toBe(stop);
        const indexers = add.mock.calls
          .map(call => call[0])
          .filter(contributor => contributor instanceof GPUPointToCell);
        expect(indexers).toHaveLength(id === 'dggs-cell-compare' ? 2 : 1);
        for (const indexer of indexers)
          expect(indexer.props.precision).toBe(family === 'h3' ? 'double-single' : 'float32');
      }
    }
  } finally {
    add.mockRestore();
    run.mockRestore();
    device.destroy();
  }
});
