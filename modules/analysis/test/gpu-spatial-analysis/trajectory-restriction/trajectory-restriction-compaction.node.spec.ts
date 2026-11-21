// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUTrajectoryRestriction} from '../../../src/gpu-spatial-analysis/trajectory-restriction/gpu-trajectory-restriction';
import {createRestrictionCompactionNodes} from '../../../src/gpu-spatial-analysis/trajectory-restriction/trajectory-restriction-compaction';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

useCPUExecutorTestHooks();

function createFixture(mergeIntervals: boolean, capacity: number) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <
    Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32' | 'uint32x2' | 'uint32x4'
  >(
    format: Format,
    length: number
  ) => createTransientView(graph, `compact-${serial++}`, format, length);
  const output = {
    intervals: view('float32x2', 264),
    positions: view('float32x4', 264),
    status: view('uint32', 264),
    relationships: view('uint32x4', 2),
    compactIntervals: view('float32x2', capacity),
    compactPositions: view('float32x4', capacity),
    compactSources: view('uint32x2', capacity),
    compactStatus: view('uint32', capacity),
    compactSummary: view('uint32', 3)
  };
  const contributor = new GPUTrajectoryRestriction({
    positions: view('float32x2', 132),
    times: view('float32', 132),
    offsets: view('uint32', 3),
    parameters: view('float32', 7),
    intervalsPerSegment: 2,
    mergeIntervals,
    output
  });
  return {graph, output, contributor};
}

it('restriction compaction retires the single invocation slot-and-track walk', () => {
  const {graph, contributor} = createFixture(true, 6);
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.some(node => node.id === `${contributor.id}-compact`)).toBe(false);
  expect(nodes.some(node => node.id.includes('-compact-slot-scan'))).toBe(true);
});

for (const merge of [false, true]) {
  for (const capacity of [0, 2, 140]) {
    it(`stable compaction across a workgroup: merge=${merge}, capacity=${capacity}`, () => {
      const {graph, output, contributor} = createFixture(merge, capacity);
      const memory = createCPUGraphMemory();
      memory.write(contributor.props.offsets, Uint32Array.of(0, 130, 132));
      const intervals = new Float32Array(528).fill(NaN);
      const positions = new Float32Array(1056).fill(NaN);
      const status = new Uint32Array(264);
      for (let row = 0; row < 132; row++) {
        const finish = row === 131 ? row : row + 1;
        intervals.set([row, finish], 4 * row);
        positions.set([row, 0, finish, 0], 8 * row);
        status[2 * row] = row === 131 ? 2 : 1;
      }
      status[254] |= 128; // open end at 128, and open start of the next interval: do not merge.
      status[256] |= 64;
      // A real producer marks the first slot when the segment exhausts its two slots.
      // Segment 0 has two separated present intervals and a dropped third interval.
      status[0] |= 256;
      status[1] = 1;
      intervals[1] = 0.25;
      intervals.set([0.5, 1], 2);
      positions[2] = 0.25;
      positions.set([0.5, 7, 1, 8], 4);
      memory.write(output.intervals, intervals);
      memory.write(output.positions, positions);
      memory.write(output.status, status);
      executeCommandNodesOnCPU(
        createRestrictionCompactionNodes(graph, contributor.id, contributor.props),
        memory
      );
      const required = merge ? 4 : 133;
      const stored = Math.min(capacity, required);
      expect(Array.from(memory.readUint32(output.compactSummary))).toEqual([
        required,
        stored,
        required - stored + 1
      ]);
      const result = memory.readFloat32(output.compactIntervals);
      const resultPositions = memory.readFloat32(output.compactPositions);
      const sources = memory.readUint32(output.compactSources);
      const resultStatus = memory.readUint32(output.compactStatus);
      // Hand union: [0,.25], [.5,128), (128,130], then another track's [130,131].
      const merged = [
        [0, 0.25, 0, 0, 257, 0, 0],
        [0.5, 128, 0, 127, 129, 7, 0],
        [128, 130, 128, 129, 65, 0, 0],
        [130, 131, 130, 131, 1, 0, 0]
      ];
      for (let row = 0; row < capacity; row++) {
        if (row >= stored) {
          expect(result[2 * row]).toBeNaN();
          expect(result[2 * row + 1]).toBeNaN();
          expect(Array.from(sources.slice(2 * row, 2 * row + 2))).toEqual([0xffffffff, 0xffffffff]);
          expect(resultStatus[row]).toBe(0);
          expect(Array.from(resultPositions.slice(4 * row, 4 * row + 4))).toEqual([
            NaN,
            NaN,
            NaN,
            NaN
          ]);
          continue;
        }
        const sourceRow = row - 1;
        const [begin, finish, first, last, flags, beginY, endY] = merge
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
        expect(Number.isFinite(result[2 * row])).toBe(true);
        expect(Number.isFinite(result[2 * row + 1])).toBe(true);
        expect(Array.from(result.slice(2 * row, 2 * row + 2))).toEqual([begin, finish]);
        expect(Array.from(resultPositions.slice(4 * row, 4 * row + 4))).toEqual([
          begin,
          beginY,
          finish,
          endY
        ]);
        expect(Array.from(sources.slice(2 * row, 2 * row + 2))).toEqual([first, last]);
        expect(resultStatus[row]).toBe(flags);
      }
    });
  }
}

it('preserves the legacy cursor and latest end source for reversed ranges, then clears an empty resubmission', () => {
  const {graph, output, contributor} = createFixture(true, 6);
  const memory = createCPUGraphMemory();
  memory.write(contributor.props.offsets, Uint32Array.of(1, 0, 132));
  const intervals = new Float32Array(528).fill(NaN);
  const positions = new Float32Array(1056).fill(NaN);
  const status = new Uint32Array(264);
  for (const [row, begin, finish, flags] of [
    [0, 5, 10, 65],
    [1, 2, 3, 1],
    [2, 9, 12, 129],
    [3, 12, 12, 2],
    [4, 20, 21, 1]
  ]) {
    intervals.set([begin, finish], 4 * row);
    positions.set([begin, 0, finish, 0], 8 * row);
    status[2 * row] = flags;
  }
  memory.write(output.intervals, intervals);
  memory.write(output.positions, positions);
  memory.write(output.status, status);
  const nodes = createRestrictionCompactionNodes(graph, contributor.id, contributor.props);
  executeCommandNodesOnCPU(nodes, memory);
  // The monotone cursor crosses offset 0 immediately, so all source rows are in its track 1.
  // Streaming union keeps the first begin (5), rejects a contained earlier end (3), and the
  // equal final end (12) takes source row 3's closed flag and endpoint coordinates.
  expect(Array.from(memory.readUint32(output.compactSummary))).toEqual([2, 2, 0]);
  expect(Array.from(memory.readFloat32(output.compactIntervals).slice(0, 4))).toEqual([
    5, 12, 20, 21
  ]);
  expect(Array.from(memory.readUint32(output.compactSources).slice(0, 4))).toEqual([0, 3, 4, 4]);
  expect(Array.from(memory.readUint32(output.compactStatus).slice(0, 2))).toEqual([65, 1]);
  memory.write(output.status, new Uint32Array(264));
  executeCommandNodesOnCPU(nodes, memory);
  expect(Array.from(memory.readUint32(output.compactSummary))).toEqual([0, 0, 0]);
  expect(Array.from(memory.readFloat32(output.compactIntervals))).toEqual(new Array(12).fill(NaN));
  expect(Array.from(memory.readUint32(output.compactSources))).toEqual(
    new Array(12).fill(0xffffffff)
  );
  expect(Array.from(memory.readUint32(output.compactStatus))).toEqual(new Array(6).fill(0));
});

it('B3 clears geometry presence and loss when track ranges shrink between submissions', () => {
  const {graph, output, contributor} = createFixture(false, 6);
  const memory = createCPUGraphMemory();
  memory.write(contributor.props.positions, new Float32Array(264));
  memory.write(contributor.props.times, Float32Array.of(0, 1, 2));
  memory.write(contributor.props.parameters, Float32Array.of(0, 3, -1, -1, 1, 1, 0));
  const nodes = contributor.getCommandNodes(graph);
  for (const count of [3, 1, 0]) {
    memory.write(contributor.props.offsets, Uint32Array.of(0, count, count));
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(output.compactSummary))).toEqual([count, count, 0]);
    expect(Array.from(memory.readUint32(output.compactStatus).slice(count))).toEqual(
      new Array(6 - count).fill(0)
    );
  }
});

it('S4 plain compaction uses one presence scan without track or union passes', () => {
  const {graph, contributor} = createFixture(false, 6);
  const nodes = createRestrictionCompactionNodes(graph, contributor.id, contributor.props);
  expect(nodes.some(node => /compact-(track|source-tracks|end-max|group)/.test(node.id))).toBe(
    false
  );
  expect(nodes.filter(node => /compact-slot-scan/.test(node.id))).toHaveLength(1);
});
