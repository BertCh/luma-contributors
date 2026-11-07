// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {getCatalogBinding, runCatalogOperationAsync} from '../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';

// The CPU reference is the oracle; the GPU engine must run the GPUChunked* classes and give the same
// edges at every chunking, including chunk sizes that do not divide the row count.

async function getDevice(): Promise<Device> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return device;
}

function makePoints(count: number): CatalogFeatureCollection {
  let state = 17;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  return {
    type: 'FeatureCollection',
    features: Array.from({length: count}, () => ({
      geometry: {type: 'Point', coordinates: [next() * 1000, next() * 1000]},
      properties: {}
    }))
  };
}

const CASES: {id: string; chunked: string; parameters: Record<string, unknown>}[] = [
  ...(['union', 'intersection', 'difference', 'symmetricDifference'] as const).map(operation => ({
    id: 'weights-algebra',
    chunked: 'GPUChunkedSpatialWeightsAlgebra',
    parameters: {operation, weightRule: 'sum', neighborCount: 4, distanceBand: 150}
  })),
  {
    id: 'weights-transpose',
    chunked: 'GPUChunkedSpatialWeightsTranspose',
    parameters: {neighborCount: 4, decayDistance: 200}
  },
  ...(['row', 'binary', 'double', 'variance'] as const).map(operation => ({
    id: 'weights-transform',
    chunked: 'GPUChunkedSpatialWeightsTransform',
    parameters: {operation, neighborCount: 4, decayDistance: 200}
  }))
];

for (const chunkRows of [0, 37, 1, 150]) {
  test(`chunked weights bindings match the CPU oracle (chunkRows ${chunkRows})`, async () => {
    const device = await getDevice();
    const points = makePoints(150);
    for (const {id, chunked, parameters} of CASES) {
      const binding = getCatalogBinding(id)!;
      const merged = {coordinateSystem: 'planar', chunkRows, ...parameters};
      const gpu = await runCatalogOperationAsync(binding, merged, {points}, {device});
      const cpu = await runCatalogOperationAsync(
        binding,
        merged,
        {points},
        {
          device,
          engine: 'cpu'
        }
      );
      const label = `${id} ${JSON.stringify(parameters)}`;
      expect(gpu.engine, `${label}: ${gpu.fallbackReason ?? ''}`).toBe('gpu');
      expect(gpu.dispatchedContributors).toEqual([chunked]);
      const expectedChunks = chunkRows === 0 ? 1 : Math.ceil(150 / Math.min(chunkRows, 150));
      expect(gpu.summary['chunkCount'], label).toBe(expectedChunks);
      expect(gpu.tables[0].rowCount, label).toBe(cpu.tables[0].rowCount);
      for (const column of ['source_index', 'target_index'])
        expect(gpu.tables[0].columns[column], `${label} ${column}`).toEqual(
          cpu.tables[0].columns[column]
        );
      const gpuWeights = gpu.tables[0].columns['weight'] as number[];
      (cpu.tables[0].columns['weight'] as number[]).forEach((weight, index) =>
        expect(Math.abs(gpuWeights[index] - weight), `${label} weight ${index}`).toBeLessThan(
          Math.max(1e-5, Math.abs(weight) * 1e-4)
        )
      );
    }
  }, 120000);
}
