// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPU_GEOGRAPHICAL_DETECTOR_LAYOUT} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {afterEach, expect, test, vi} from 'vitest';
import {createNullWebGPUDevice} from '../../analysis/test/utils/gpu-contributor-test-utils';
import * as gpuSessions from '../src/analysis/catalog-gpu-session';
import {CATEGORY_BINDINGS} from '../src/analysis/catalog-network-categorical-categories';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';

afterEach(() => vi.restoreAllMocks());

test('detector adapter preserves q and F when its derived p-value exceeds the supported range', async () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const layout = GPU_GEOGRAPHICAL_DETECTOR_LAYOUT;
    // [1000] and [1001,1002,1003]: SST=5, SSW=2, q=3/5, F=2*q/(1-q)=3.
    const nonCentrality = (1000 ** 2 + 1002 ** 2 - (1000 + Math.sqrt(3) * 1002) ** 2 / 4) / 1.25;
    const summary = new Float32Array(layout.length);
    summary[layout.q] = 0.6;
    summary[layout.fStatistic] = 3;
    summary[layout.nonCentrality] = nonCentrality;
    summary[layout.strataCount] = 2;
    summary[layout.observationCount] = 4;
    summary[layout.numeratorDegreesOfFreedom] = 1;
    summary[layout.denominatorDegreesOfFreedom] = 2;
    summary[layout.totalSumOfSquares] = 5;
    const uploads = new Map<string, Float32Array | Uint32Array | Int32Array>();
    const session = {
      graph: {add: vi.fn()},
      input: (
        name: string,
        values: Float32Array | Uint32Array | Int32Array,
        format: 'float32' | 'uint32'
      ) => {
        uploads.set(name, values);
        return createTransientView(graph, name, format, values.length);
      },
      output: (name: string, format: 'float32' | 'uint32', length: number) => ({
        view: createTransientView(graph, name, format, length),
        length
      }),
      run: vi.fn(),
      readFloat32: async () => summary
    } as unknown as gpuSessions.CatalogGpuSession;
    vi.spyOn(gpuSessions, 'withCatalogGpuSession').mockImplementation(
      async (_device, _id, action) => action(session)
    );
    const points: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: [1000, 1001, 1002, 1003].map((value, row) => ({
        geometry: {type: 'Point', coordinates: [row, 0]},
        properties: {category: row === 0 ? 'A' : 'B', value}
      }))
    };
    const binding = CATEGORY_BINDINGS.find(candidate => candidate.id === 'geographical-detector')!;
    const result = await binding.runGpu!(
      {device: device as Device},
      {points},
      {
        categoryProperty: 'category',
        valueProperty: 'value'
      }
    );
    expect(uploads.get('values')).toEqual(Float32Array.of(1000, 1001, 1002, 1003));
    expect(uploads.get('categories')).toEqual(Uint32Array.of(0, 1, 1, 1));
    const table = result.tables[0];
    const valueFor = (statistic: string) =>
      table.columns['value'][table.columns['statistic'].indexOf(statistic)];
    expect(valueFor('q')).toBe(Math.fround(0.6));
    expect(valueFor('fStatistic')).toBe(3);
    expect(valueFor('nonCentrality')).toBe(Math.fround(nonCentrality));
    expect(valueFor('pValue')).toBeNull();
    expect(result.summary['pValueStatus']).toBe('unavailable');
    expect(result.diagnostics).toContainEqual({
      code: 'geographical-detector-p-value-unavailable',
      severity: 'warning',
      count: 1,
      message: expect.stringMatching(/nonCentrality.*10000/)
    });
  } finally {
    device.destroy();
  }
});
