// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import {
  getGPUCellKeyFreeBits,
  getGPUCellPeriodKeyPeriodBits,
  getGPUValueCubeMaximumBinCount,
  GPUCellPeriodKeys,
  GPUValueCube,
  packGPUCellPeriodKey,
  unpackGPUCellPeriodKeyChecked
} from '../../../src/dataframe-analysis/value-cube/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getExactFloat32Sum, computeValueCubeOracle} from './value-cube-oracle';

let serial = 0;
const view = <Format extends 'uint32' | 'float32' | 'uint32x2'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) => createTransientView(graph, `v-${serial++}`, format, length);

function withGraph<Result>(action: (graph: GPUCommandGraph) => Result): Result {
  const device = createNullWebGPUDevice();
  try {
    return action(new GPUCommandGraph(device));
  } finally {
    device.destroy();
  }
}

describe('GPUValueCube validation', () => {
  const create = (graph: GPUCommandGraph, overrides: Record<string, unknown> = {}) =>
    new GPUValueCube({
      cellCount: 4,
      periodCount: 3,
      cells: view(graph, 'uint32', 10),
      periods: view(graph, 'uint32', 10),
      values: view(graph, 'float32', 10),
      output: {means: view(graph, 'float32', 12)},
      ...overrides
    } as never);

  it('schedules clears, a scale pass, accumulation, decode and means', () => {
    withGraph(graph => {
      const cube = create(graph, {
        mask: view(graph, 'uint32', 10),
        output: {
          counts: view(graph, 'uint32', 12),
          sums: view(graph, 'float32', 12),
          means: view(graph, 'float32', 12),
          status: {invalidCount: view(graph, 'uint32', 1)}
        }
      });
      expect(cube.binCount).toBe(12);
      const ids = cube.getCommandNodes(graph).map(node => node.id);
      expect(ids).toEqual([
        'value-cube-clear-scale',
        'value-cube-clear-words',
        'value-cube-clear-counts',
        'value-cube-clear-invalid-count',
        'value-cube-scale-pass',
        'value-cube-accumulate',
        'value-cube-decode',
        'value-cube-means'
      ]);
    });
  });

  it('refuses shapes, formats and outputs it cannot honour', () => {
    const throws = (
      overrides: (graph: GPUCommandGraph) => Record<string, unknown>,
      message: RegExp
    ) => withGraph(graph => expect(() => create(graph, overrides(graph))).toThrow(message));
    throws(() => ({cellCount: 0}), /value-cube cellCount must be an integer in \[1, 4294967294\]/);
    throws(() => ({periodCount: 1.5}), /periodCount must be an integer/);
    throws(
      () => ({cellCount: 2 ** 20, periodCount: 2 ** 20}),
      /cellCount \* periodCount must be at most 4294967294/
    );
    throws(graph => ({periods: view(graph, 'uint32', 9)}), /periods must have 10 rows/);
    throws(graph => ({values: view(graph, 'uint32', 10)}), /values/);
    throws(graph => ({mask: view(graph, 'uint32', 9)}), /mask must have 10 rows/);
    throws(() => ({output: {}}), /output needs at least one of counts, sums and means/);
    throws(
      graph => ({output: {means: view(graph, 'float32', 11)}}),
      /output.means must have 12 rows/
    );
    throws(graph => ({output: {counts: view(graph, 'float32', 12)}}), /output.counts/);
    withGraph(graph => {
      const values = view(graph, 'float32', 10);
      expect(() =>
        create(graph, {cellCount: 10, periodCount: 1, values, output: {means: values}})
      ).toThrow(/outputs must not share buffers with inputs/);
    });
  });

  it('states the largest cube of a default device and refuses one bin more at declaration', () => {
    withGraph(graph => {
      const maximum = getGPUValueCubeMaximumBinCount(graph.device);
      expect(maximum).toBe(11_184_810);
      const add = (bins: number) =>
        graph.add(
          new GPUValueCube({
            id: `cube-${bins}`,
            cellCount: bins,
            periodCount: 1,
            cells: view(graph, 'uint32', 4),
            periods: view(graph, 'uint32', 4),
            values: view(graph, 'float32', 4),
            output: {counts: view(graph, 'uint32', bins)}
          })
        );
      expect(() => add(maximum)).not.toThrow();
      expect(() => add(maximum + 1)).toThrow(
        /cube-11184811 fixed-point sums needs 134217732 bytes in one binding/
      );
    });
  });
});

describe('GPUCellPeriodKeys validation and packing helpers', () => {
  const layout = 'legacy-one-bit';

  it('validates its views and period axis', () => {
    withGraph(graph => {
      const create = (overrides: Record<string, unknown> = {}) =>
        new GPUCellPeriodKeys({
          cells: view(graph, 'uint32x2', 5),
          periods: view(graph, 'uint32', 5),
          periodCount: 365,
          fill: 'ones',
          layout,
          output: {keys: view(graph, 'uint32x2', 5)},
          ...overrides
        } as never);
      expect(create().periodBits).toBe(9);
      expect(create({periodCount: 0xfffffffd}).periodBits).toBe(32);
      expect(() => create({periodCount: 0xfffffffe})).toThrow(/periodCount must be an integer in/);
      expect(() => create({fill: 'twos'})).toThrow(/fill must be 'ones' or 'zeros'/);
      expect(() => create({layout: undefined})).toThrow(/layout.*legacy-one-bit.*identity/);
      expect(() => create({cells: view(graph, 'uint32', 5)})).toThrow(/cells/);
      expect(() => create({periods: view(graph, 'uint32', 4)})).toThrow(/periods must have 5 rows/);
      expect(() => create({output: {keys: view(graph, 'uint32x2', 4)}})).toThrow(
        /output.keys must have 5 rows/
      );
    });
  });

  it('packs and unpacks exactly, and refuses cells with other bits where the period goes', () => {
    // 365 periods need 9 bits. H3-style ones fill: low 9 bits all ones.
    const cell = 0x8928308280fffffn; // an H3 resolution 9 index (ends in 18 one bits)
    const key = packGPUCellPeriodKey(cell, 300, 365, 'ones', layout);
    expect(key).toBe((cell & ~0x1ffn) | 300n);
    expect(unpackGPUCellPeriodKeyChecked(key!, 365, 'ones', layout)).toEqual({
      cellKey: cell,
      period: 300
    });
    // A cell whose low bits are not the fill is not packed and not truncated.
    expect(packGPUCellPeriodKey(cell - 1n, 3, 365, 'ones', layout)).toBeUndefined();
    expect(packGPUCellPeriodKey(cell, 365, 365, 'ones', layout)).toBeUndefined();
    // Different (cell, period) pairs get different keys, and unpacking inverts packing.
    const seen = new Map<bigint, string>();
    for (const base of [cell, cell - (1n << 9n), cell - (5n << 9n)]) {
      for (let period = 0; period < 365; period++) {
        const packed = packGPUCellPeriodKey(base, period, 365, 'ones', layout)!;
        expect(seen.has(packed)).toBe(false);
        seen.set(packed, `${base}:${period}`);
        expect(unpackGPUCellPeriodKeyChecked(packed, 365, 'ones', layout)).toEqual({
          cellKey: base,
          period
        });
      }
    }
    // S2-style zeros fill.
    expect(packGPUCellPeriodKey(0x3000000000000000n, 5, 8, 'zeros', layout)).toBe(
      0x3000000000000005n
    );
    expect(packGPUCellPeriodKey(0x3000000000000001n, 5, 8, 'zeros', layout)).toBeUndefined();
    // DPC-101: rows the kernel skips are not packed. The all-ones "no cell" key (whose low bits
    // match the 'ones' fill) would otherwise become 0xfffffffffffffffc, a groupable key; the
    // 0xffffffff "excluded upstream" period and a negative or fractional period are not periods.
    expect(packGPUCellPeriodKey(0xffffffffffffffffn, 0, 3, 'ones', layout)).toBeUndefined();
    expect(packGPUCellPeriodKey(0xffffffffffffffffn, 5, 8, 'zeros', layout)).toBeUndefined();
    expect(packGPUCellPeriodKey(0x123fn, 0xffffffff, 3, 'ones', layout)).toBeUndefined();
    expect(packGPUCellPeriodKey(0x123fn, -1, 3, 'ones', layout)).toBeUndefined();
    expect(packGPUCellPeriodKey(0x123fn, 0.5, 3, 'ones', layout)).toBeUndefined();
    expect(packGPUCellPeriodKey(0x123fn, 1, 3, 'ones', layout)).toBe(0x123dn);
    expect(getGPUCellPeriodKeyPeriodBits(2 ** 32, layout)).toBe(32);
    expect(() => getGPUCellPeriodKeyPeriodBits(0, layout)).toThrow(
      /periodCount must be an integer/
    );
    expect(getGPUCellKeyFreeBits('h3', 15)).toBe(0);
    expect(getGPUCellKeyFreeBits('quadbin', 26)).toBe(0);
    expect(getGPUCellKeyFreeBits('s2', 30)).toBe(0);
  });

  it('exact big-integer oracle is exact where binary64 and float32 sums are not', () => {
    const values = [1e8, 1, -1e8, 0.5, 1e8, -1e8];
    expect(getExactFloat32Sum(values)).toBe(1.5);
    expect(values.reduce((sum, value) => Math.fround(sum + value), 0)).not.toBe(1.5);
  });
});

it('DPC-124 cube oracle keeps a finite populated-bin mean and an empty-bin NaN', () => {
  const result = computeValueCubeOracle(
    [
      {cell: 0, period: 0, value: Math.fround(3e38), selected: true},
      {cell: 0, period: 0, value: Math.fround(3e38), selected: true}
    ],
    1,
    2
  );
  // mean([x,x]) = x. IEEE binary32(3e38) = 3.0000000054977558e38;
  // binary32(2*x) exceeds the largest finite value and rounds to Infinity.
  expect(result.counts).toEqual([2, 0]);
  expect(result.sums).toEqual([Infinity, 0]);
  expect(Number.isFinite(result.means[0])).toBe(true);
  expect(result.means).toEqual([3.0000000054977558e38, NaN]);
  expect(Number.isNaN(result.means[1])).toBe(true);
});

it('S4 cube oracle rounds the exact rational mean directly to binary32', () => {
  // Sum = 2 + 2^-23 + 2^-52. Dividing by 4 lies just ABOVE the midpoint
  // 0.5 + 2^-25; binary64 rounding of that midpoint loses the positive 2^-54 remainder.
  const result = computeValueCubeOracle(
    [1, 1 + 2 ** -23, 2 ** -52, 0].map(value => ({cell: 0, period: 0, value, selected: true})),
    1,
    2
  );
  expect(result.means).toEqual([0.5000000596046448, NaN]);
});

it('B5 composition bin means use the wide total before narrowing', async () => {
  const {getValueCubeMeanValues} = await import('./value-cube-mean-reference');
  expect(Array.from(getValueCubeMeanValues([[Math.fround(3e38), Math.fround(3e38)], []]))).toEqual([
    3.0000000054977558e38,
    NaN
  ]);
});

it('DPC-125 permits separately published slices but rejects logical and co-bound overlaps', () =>
  withGraph(graph => {
    const storage = view(graph, 'float32', 128);
    const slice = <Format extends 'float32' | 'uint32'>(format: Format, byteOffset: number) =>
      graph.createDataView(storage.buffer, {format, length: 1, byteOffset});
    const props = {
      cellCount: 1,
      periodCount: 1,
      cells: view(graph, 'uint32', 1),
      periods: view(graph, 'uint32', 1),
      values: view(graph, 'float32', 1)
    };
    expect(
      () =>
        new GPUValueCube({
          ...props,
          output: {sums: slice('float32', 0), means: slice('float32', 4)}
        })
    ).not.toThrow();
    expect(
      () =>
        new GPUValueCube({
          ...props,
          output: {sums: slice('float32', 0), means: slice('float32', 0)}
        })
    ).toThrow(/overlap/);
    expect(
      () =>
        new GPUValueCube({
          ...props,
          output: {counts: slice('uint32', 0), means: slice('float32', 4)}
        })
    ).toThrow(/overlap/);
  }));
