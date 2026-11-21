// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUTrajectoryRestriction} from '../../../src/gpu-spatial-analysis/trajectory-restriction/gpu-trajectory-restriction';
import {createRestrictionCompactionNodes} from '../../../src/gpu-spatial-analysis/trajectory-restriction/trajectory-restriction-compaction';
import {createOptimizationHarness} from '../../gpu-network/network-optimization/optimization-test-harness';
import {requireTestDevice} from '../../../src/testing/require-test-device';

for (const mergeIntervals of [false, true]) {
  for (const capacity of [2, 140]) {
    it(`parallel restriction compaction preserves prefix and source loss: merge=${mergeIntervals}, capacity=${capacity}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = createOptimizationHarness(device);
      const intervals = new Float32Array(528).fill(NaN);
      const positions = new Float32Array(1056).fill(NaN);
      const status = new Uint32Array(264);
      for (let row = 0; row < 132; row++) {
        const finish = row === 131 ? row : row + 1;
        intervals.set([row, finish], 4 * row);
        positions.set([row, 0, finish, 0], 8 * row);
        status[2 * row] = row === 131 ? 2 : 1;
      }
      status[254] |= 128;
      status[256] |= 64;
      status[0] |= 256;
      status[1] = 1;
      intervals[1] = 0.25;
      intervals.set([0.5, 1], 2);
      positions[2] = 0.25;
      positions.set([0.5, 7, 1, 8], 4);
      const output = {
        intervals: harness.input('float32x2', intervals),
        positions: harness.input('float32x4', positions),
        status: harness.input('uint32', status),
        relationships: harness.output('uint32x4', 2),
        compactIntervals: harness.output('float32x2', capacity),
        compactPositions: harness.output('float32x4', capacity),
        compactSources: harness.output('uint32x2', capacity),
        compactStatus: harness.output('uint32', capacity),
        compactSummary: harness.output('uint32', 3)
      };
      const contributor = new GPUTrajectoryRestriction({
        positions: harness.input('float32x2', new Float32Array(264)),
        times: harness.input('float32', new Float32Array(132)),
        offsets: harness.input('uint32', Uint32Array.of(0, 130, 132)),
        parameters: harness.input('float32', Float32Array.of(0, 132, -1, -1, 1, 1, 0)),
        intervalsPerSegment: 2,
        mergeIntervals,
        output
      });
      // Isolate the compaction passes from geometry generation, including a dropped source slot.
      const result = await harness.run({
        getCommandNodes: graph =>
          createRestrictionCompactionNodes(graph, 'compaction-fixture', contributor.props)
      });
      const required = mergeIntervals ? 4 : 133;
      const stored = Math.min(required, capacity);
      expect(result(output.compactSummary)).toEqual([required, stored, required - stored + 1]);
      // Independent hand union, split by two open endpoints and then by a track boundary.
      const merged = [
        [0, 0.25, 0, 0, 257, 0, 0],
        [0.5, 128, 0, 127, 129, 7, 0],
        [128, 130, 128, 129, 65, 0, 0],
        [130, 131, 130, 131, 1, 0, 0]
      ];
      for (let row = 0; row < capacity; row++) {
        if (row >= stored) {
          expect(result(output.compactIntervals)[2 * row]).toBeNaN();
          expect(result(output.compactIntervals)[2 * row + 1]).toBeNaN();
          expect(result(output.compactPositions).slice(4 * row, 4 * row + 4)).toEqual([
            NaN,
            NaN,
            NaN,
            NaN
          ]);
          expect(result(output.compactSources).slice(2 * row, 2 * row + 2)).toEqual([
            0xffffffff, 0xffffffff
          ]);
          expect(result(output.compactStatus)[row]).toBe(0);
          continue;
        }
        const sourceRow = row - 1;
        const [begin, finish, first, last, flags, beginY, endY] = mergeIntervals
          ? merged[row]
          : row === 0
            ? [0, 0.25, 0, 0, 257, 0, 0]
            : row === 1
              ? [0.5, 1, 0, 0, 1, 7, 8]
              : [
                  sourceRow,
                  sourceRow === 131 ? sourceRow : sourceRow + 1,
                  sourceRow,
                  sourceRow,
                  status[2 * sourceRow],
                  0,
                  0
                ];
        expect(result(output.compactIntervals).slice(2 * row, 2 * row + 2)).toEqual([
          begin,
          finish
        ]);
        expect(result(output.compactPositions).slice(4 * row, 4 * row + 4)).toEqual([
          begin,
          beginY,
          finish,
          endY
        ]);
        expect(result(output.compactSources).slice(2 * row, 2 * row + 2)).toEqual([first, last]);
        expect(result(output.compactStatus)[row]).toBe(flags);
      }
    });
  }
}

it('B3 contributor compaction clears unowned slots after track ranges shrink on resubmission', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const offsets = harness.input('uint32', Uint32Array.of(0, 3));
  const output = {
    intervals: harness.output('float32x2', 6),
    positions: harness.output('float32x4', 6),
    status: harness.output('uint32', 6),
    relationships: harness.output('uint32x4', 1),
    compactIntervals: harness.output('float32x2', 4),
    compactPositions: harness.output('float32x4', 4),
    compactSources: harness.output('uint32x2', 4),
    compactStatus: harness.output('uint32', 4),
    compactSummary: harness.output('uint32', 3)
  };
  const contributor = new GPUTrajectoryRestriction({
    positions: harness.input('float32x2', Float32Array.of(0, 0, 0, 0, 0, 0)),
    times: harness.input('float32', Float32Array.of(0, 1, 2)),
    offsets,
    parameters: harness.input('float32', Float32Array.of(0, 3, -1, -1, 1, 1, 0)),
    intervalsPerSegment: 2,
    output
  });
  const result = await harness.run(
    contributor,
    (read, frame) => {
      const count = frame === 0 ? 3 : 1;
      expect(read(output.compactSummary)).toEqual([count, count, 0]);
      harness.rewrite(offsets, Uint32Array.of(0, frame === 0 ? 1 : 0));
    },
    3
  );
  expect(result(output.compactSummary)).toEqual([0, 0, 0]);
  expect(result(output.compactIntervals)).toEqual(new Array(8).fill(NaN));
  expect(result(output.compactPositions)).toEqual(new Array(16).fill(NaN));
  expect(result(output.compactStatus)).toEqual([0, 0, 0, 0]);
});
