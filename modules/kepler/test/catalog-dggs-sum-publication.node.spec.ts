// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, test, vi} from 'vitest';
import {createNullWebGPUDevice} from '../../analysis/test/utils/gpu-contributor-test-utils';
import * as sessions from '../src/analysis/catalog-gpu-session';
import {PLATFORM4_DGGS_BINDINGS} from '../src/analysis/catalog-platform4-dggs';
import {getDggsCpuCell} from '../src/analysis/catalog-platform4-dggs-cells';
import {normalizeCatalogParameters} from '../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';

test('DGGS-102: production publishers suppress saturated sums with spare capacity and recover', async () => {
  const device = createNullWebGPUDevice();
  let saturated = false;
  let unequal = false;
  const inputs: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [2, 4].map(value => ({
      geometry: {type: 'Point', coordinates: [10, 10]},
      properties: {v: value}
    }))
  };
  const key = getDggsCpuCell('quadbin', 10, 10, 6);
  const allocated: string[] = [];
  const spy = vi
    .spyOn(sessions, 'withCatalogGpuSession')
    .mockImplementation(async (_device, _id, run) => {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const session = {
        graph,
        input: (_name: string, values: Float32Array, format: 'float32' | 'float32x2') =>
          createTransientView(
            graph,
            `input-${serial++}`,
            format,
            values.length / (format === 'float32x2' ? 2 : 1)
          ),
        output: (
          name: string,
          format: 'uint32' | 'uint32x2' | 'float32' | 'float32x2',
          length: number
        ) => {
          allocated.push(name);
          return {name, view: createTransientView(graph, `output-${serial++}`, format, length)};
        },
        run: () => {},
        readScalar: async (output: {name: string}) =>
          output.name.endsWith('-sum-overflow')
            ? Number(saturated)
            : output.name.endsWith('-count')
              ? 1
              : 0,
        readUint32: async (output: {name: string}) => {
          if (output.name.endsWith('-cells'))
            return Uint32Array.of(Number(key & 0xffffffffn), Number(key >> 32n));
          if (output.name.endsWith('-sums')) return Uint32Array.of(6 * 65536, 0);
          if (output.name.endsWith('-counts')) return Uint32Array.of(2);
          if (output.name === 'compare-presence') return Uint32Array.of(3);
          return Uint32Array.of(0);
        },
        readFloat32: async (output: {name: string}) =>
          Float32Array.of(
            saturated && output.name.startsWith('compare-')
              ? Number.NaN
              : unequal && (output.name === 'compare-before' || output.name === 'compare-after')
                ? 16777216
                : output.name === 'compare-delta'
                  ? unequal
                    ? 1
                    : 0
                  : output.name.endsWith('-minimums')
                    ? 2
                    : output.name.endsWith('-maximums')
                      ? 4
                      : 6
          )
      };
      return run(session as unknown as sessions.CatalogGpuSession);
    });
  try {
    for (const id of ['dggs-cell-aggregation', 'dggs-cell-rollup']) {
      const binding = PLATFORM4_DGGS_BINDINGS.find(candidate => candidate.id === id)!;
      for (const loss of [false, true, false]) {
        saturated = loss;
        const parameters = normalizeCatalogParameters(binding, {
          family: 'quadbin',
          resolution: 6,
          valueProperty: 'v',
          ...(id === 'dggs-cell-rollup' ? {levels: 1, levelStep: 1} : {})
        });
        const result = await binding.runGpu!({device}, {points: inputs}, parameters);
        const expectedRows = id === 'dggs-cell-rollup' ? 2 : 1;
        expect(result.tables[0].rowCount).toBe(expectedRows);
        expect(result.tables[0].columns['count']).toEqual(Array(expectedRows).fill(2));
        expect(result.tables[0].columns['sum']).toEqual(Array(expectedRows).fill(loss ? null : 6));
        expect(result.tables[0].columns['mean']).toEqual(Array(expectedRows).fill(loss ? null : 3));
        expect(result.tables[0].columns['sum_overflow']).toEqual(Array(expectedRows).fill(loss));
        if (id === 'dggs-cell-rollup') expect(result.tables[0].columns['level']).toEqual([0, 1]);
      }
    }
    const compare = PLATFORM4_DGGS_BINDINGS.find(
      candidate => candidate.id === 'dggs-cell-compare'
    )!;
    for (const loss of [false, true, false]) {
      saturated = loss;
      const result = compare.runGpu!(
        {device},
        {before: inputs, after: inputs},
        {
          family: 'quadbin',
          resolution: 6,
          valueProperty: 'v',
          measure: 'sum'
        }
      );
      const comparison = await result;
      expect(comparison.tables[0].rowCount).toBe(1);
      expect(comparison.tables[0].columns['before']).toEqual([loss ? null : 6]);
      expect(comparison.tables[0].columns['sum_overflow']).toEqual([loss]);
      if (loss)
        for (const name of ['after', 'delta', 'ratio', 'percent_change', 'z_score'])
          expect(comparison.tables[0].columns[name]).toEqual([null]);
      expect(comparison.tables[0].columns['status']).toEqual(['present']);
    }
    unequal = true;
    const largeInput: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: [{geometry: {type: 'Point', coordinates: [10, 10]}, properties: {v: 16777216}}]
    };
    const unequalAfter: CatalogFeatureCollection = {
      ...largeInput,
      features: [
        ...largeInput.features,
        {geometry: {type: 'Point', coordinates: [10, 10]}, properties: {v: 1}}
      ]
    };
    const unequalPeriods = await compare.runGpu!(
      {device},
      {before: largeInput, after: unequalAfter},
      {
        family: 'quadbin',
        resolution: 6,
        valueProperty: 'v',
        measure: 'sum'
      }
    );
    // Integer sums 2^24 and 2^24+1 round to the same f32 endpoint, but their delta is 1.
    expect(unequalPeriods.tables[0].columns['delta']).toEqual([1]);
    expect(unequalPeriods.tables[0].columns['percent_change']).toEqual([0.0000059604644775390625]);
    expect(allocated).toContain('compare-delta');
    expect(allocated).toContain('aggregate-sum-overflow');
    expect(allocated).toContain('level-1-sum-overflow');
    expect(allocated).toContain('compare-sum-overflow');
  } finally {
    spy.mockRestore();
  }
});
