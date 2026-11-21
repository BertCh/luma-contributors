// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../analysis/src/testing/gpu-contributor-test-utils';
import {CatalogGpuSession} from '../src/analysis/catalog-gpu-session';
import {PLATFORM5_LISA_BINDINGS} from '../src/analysis/catalog-platform5-lisa';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';

test.each([
  {
    probability: 0.05,
    significant: [1, 0, 0, 1],
    probabilities: [0.05, 0.9, 0.9, 0.05],
    expected: [true, true, false, false]
  },
  {
    probability: 0.04,
    significant: [0, 1, 1, 0],
    probabilities: [0.04, 0.01, 0.01, 0.04],
    expected: [false, false, true, true]
  }
])('catalog preserves producer significance at p=$probability', async ({
  probability,
  significant,
  probabilities,
  expected
}) => {
  const device = createNullWebGPUDevice();
  const features: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [1, 2, 3, 4].map((value, index) => ({
      // Canonical x order is source rows [1,3,2,0]; publication must restore [0,1,2,3].
      geometry: {type: 'Point', coordinates: [[3, 0, 2, 1][index], 0]},
      properties: {value}
    }))
  };
  const add = vi.spyOn(GPUCommandGraph.prototype, 'add');
  // Only submission/readback are substituted. The contributor and graph wiring are real.
  const run = vi.spyOn(CatalogGpuSession.prototype, 'run').mockImplementation(() => {});
  const scalar = vi.spyOn(CatalogGpuSession.prototype, 'readScalar').mockResolvedValue(0);
  const floats = vi
    .spyOn(CatalogGpuSession.prototype, 'readFloat32')
    .mockImplementation(async output =>
      Float32Array.from(
        output.buffer.id.includes('pseudo-p-values') ? probabilities : [42, 43, 44, 45]
      )
    );
  const integers = vi
    .spyOn(CatalogGpuSession.prototype, 'readUint32')
    .mockImplementation(async output => {
      expect(output.buffer.id).toBe('catalog-local-lisa-significant');
      return Uint32Array.from(significant);
    });
  try {
    const binding = PLATFORM5_LISA_BINDINGS.find(entry => entry.id === 'local-lisa-variants')!;
    const result = await binding.runGpu!(
      {device},
      {features},
      {
        variant: 'local-getis-ord-g',
        valueProperty: 'value',
        neighborhood: 'knn',
        neighborCount: 1,
        coordinateSystem: 'planar',
        permutations: probability === 0.05 ? 39 : 49,
        seed: 4,
        significance: probability
      }
    );
    // Exact rational expectations: 1/20 < float32(.05)=.05000000074505806,
    // while 1/25 > float32(.04)=.03999999910593033. Readback rounding cannot decide these.
    expect(result.tables[0].columns['p_value']).toEqual(
      [probabilities[3], probabilities[0], probabilities[2], probabilities[1]].map(Math.fround)
    );
    expect(result.tables[0].columns['significant']).toEqual(expected);
    expect(result.tables[0].columns['source_index']).toEqual([0, 1, 2, 3]);
    expect(result.tables[0].columns['statistic']).toEqual([45, 42, 44, 43]);
    expect(integers).toHaveBeenCalledOnce();
    expect(add).toHaveBeenCalledWith(
      expect.objectContaining({
        props: expect.objectContaining({
          output: expect.objectContaining({significant: expect.anything()})
        })
      })
    );
  } finally {
    for (const spy of [integers, floats, scalar, run, add]) spy.mockRestore();
    device.destroy();
  }
});
