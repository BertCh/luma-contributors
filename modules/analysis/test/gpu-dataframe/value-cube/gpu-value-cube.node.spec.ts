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
  unpackGPUCellPeriodKey
} from '../../../src/dataframe-analysis/value-cube/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getExactFloat32Sum} from './value-cube-oracle';

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
  it('validates its views and period axis', () => {
    withGraph(graph => {
      const create = (overrides: Record<string, unknown> = {}) =>
        new GPUCellPeriodKeys({
          cells: view(graph, 'uint32x2', 5),
          periods: view(graph, 'uint32', 5),
          periodCount: 365,
          fill: 'ones',
          output: {keys: view(graph, 'uint32x2', 5)},
          ...overrides
        } as never);
      expect(create().periodBits).toBe(9);
      expect(create({periodCount: 0xfffffffd}).periodBits).toBe(32);
      expect(() => create({periodCount: 0xfffffffe})).toThrow(/periodCount must be an integer in/);
      expect(() => create({fill: 'twos'})).toThrow(/fill must be 'ones' or 'zeros'/);
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
    const key = packGPUCellPeriodKey(cell, 300, 365, 'ones');
    expect(key).toBe((cell & ~0x1ffn) | 300n);
    expect(unpackGPUCellPeriodKey(key!, 365, 'ones')).toEqual({cellKey: cell, period: 300});
    // A cell whose low bits are not the fill is not packed and not truncated.
    expect(packGPUCellPeriodKey(cell - 1n, 3, 365, 'ones')).toBeUndefined();
    expect(packGPUCellPeriodKey(cell, 365, 365, 'ones')).toBeUndefined();
    // Different (cell, period) pairs get different keys, and unpacking inverts packing.
    const seen = new Map<bigint, string>();
    for (const base of [cell, cell - (1n << 9n), cell - (5n << 9n)]) {
      for (let period = 0; period < 365; period++) {
        const packed = packGPUCellPeriodKey(base, period, 365, 'ones')!;
        expect(seen.has(packed)).toBe(false);
        seen.set(packed, `${base}:${period}`);
        expect(unpackGPUCellPeriodKey(packed, 365, 'ones')).toEqual({cellKey: base, period});
      }
    }
    // S2-style zeros fill.
    expect(packGPUCellPeriodKey(0x3000000000000000n, 5, 8, 'zeros')).toBe(0x3000000000000005n);
    expect(packGPUCellPeriodKey(0x3000000000000001n, 5, 8, 'zeros')).toBeUndefined();
    expect(getGPUCellPeriodKeyPeriodBits(2 ** 32)).toBe(32);
    expect(() => getGPUCellPeriodKeyPeriodBits(0)).toThrow(/periodCount must be an integer/);
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
