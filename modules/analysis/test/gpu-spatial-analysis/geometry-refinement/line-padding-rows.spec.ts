// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Rows of a line position buffer before `lineOffsets[0]` and from `lineOffsets[last]` on belong to
// no line: unwritten capacity or a transient the graph reuses, holding anything. GPULineLoopRemoval
// and GPUOffsetCurveComponents node their lines through GPULineSplit, which read those rows until
// 2026-11-11. Each case runs once without padding and again with 2 leading and 3 trailing rows
// poisoned by NaN, by far finite values and by a value that puts phantom segments across the
// scene, and expects the same result; source-edge indices are rebased by the leading count.

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPULineLoopRemoval,
  GPUOffsetCurveComponents
} from '../../../src/gpu-spatial-analysis/geometry-refinement/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const LEADING = 2;
const TRAILING = 3;
/** NaN, far values, and 2, which puts phantom segments through the middle of the scene. */
const POISONS = [Number.NaN, 3e6, -3e6, 2];

const COMPONENTS = {uint32: 1, uint32x4: 4, float32x2: 2} as const;
type Format = keyof typeof COMPONENTS;

function createHarness(device: Device) {
  const graph = new GPUCommandGraph(device, {id: 'line-padding-rows'});
  const buffers = new Map<GraphDataView, Buffer>();
  let serial = 0;
  const view = <Value extends Format>(format: Value, length: number, values?: number[]) => {
    const buffer = values
      ? createInputBuffer(
          device,
          format === 'float32x2' ? Float32Array.from(values) : Uint32Array.from(values)
        )
      : createOutputBuffer(device, length * COMPONENTS[format]);
    const result = importGraphBuffer(graph, `view-${serial++}`, buffer, format, length);
    buffers.set(result, buffer);
    return result as GraphDataView<Value>;
  };
  const read = (value: GraphDataView, rows = value.length) =>
    value.format === 'float32x2'
      ? readFloat32(buffers.get(value)!, rows * COMPONENTS[value.format as Format])
      : readUint32(buffers.get(value)!, rows * COMPONENTS[value.format as Format]);
  const lines = (positions: number[], offsets: number[], leading: number, value: number) => ({
    kind: 'lines' as const,
    positions: view('float32x2', positions.length / 2 + (leading ? LEADING + TRAILING : 0), [
      ...new Array<number>(leading ? 2 * LEADING : 0).fill(value),
      ...positions,
      ...new Array<number>(leading ? 2 * TRAILING : 0).fill(value)
    ]),
    lineOffsets: view(
      'uint32',
      offsets.length,
      offsets.map(offset => offset + (leading ? LEADING : 0))
    )
  });
  return {graph, view, read, lines, destroy: () => buffers.forEach(buffer => buffer.destroy())};
}

/** A line with a figure-eight loop, and a straight line. */
const LOOP_POSITIONS = [0, 0, 4, 4, 0, 4, 4, 0, 6, 0, 8, 1, 9, 1];
const LOOP_OFFSETS = [0, 4, 7];

async function runLoopRemoval(padded: boolean, value: number) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device);
  const {graph, view, read, lines} = harness;
  const output = {
    positions: view('float32x2', 16),
    lineOffsets: view('uint32', LOOP_OFFSETS.length),
    sourceEdges: view('uint32', 16),
    status: view('uint32x4', 1)
  };
  graph.add(
    new GPULineLoopRemoval({
      lines: lines(LOOP_POSITIONS, LOOP_OFFSETS, padded ? 1 : 0, value),
      intersectionCapacity: 16,
      output
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled);
  const status = await read(output.status);
  const written = status[0];
  const result = {
    status,
    lineOffsets: await read(output.lineOffsets),
    positions: (await read(output.positions)).slice(0, written * 2),
    // Source edge starts name input vertices: rebase them to the unpadded numbering.
    sourceEdges: (await read(output.sourceEdges))
      .slice(0, written)
      .map(edge => edge - (padded ? LEADING : 0))
  };
  compiled.destroy();
  harness.destroy();
  return result;
}

it('GPULineLoopRemoval ignores position rows before and after every line', async () => {
  const reference = await runLoopRemoval(false, 0);
  expect(reference.status.slice(2)).toEqual([0, 0]);
  // The loop of line 0 is erased, so fewer vertices than the input line remain.
  expect(reference.status[0]).toBeLessThan(LOOP_POSITIONS.length / 2);
  for (const value of POISONS) {
    expect(await runLoopRemoval(true, value), `padding ${value}`).toEqual(reference);
  }
});

const ORIGINAL = [0, 0, 4, 0, 4, 4];
const RAW = [0, 0.5, 3.5, 0.5, 3.5, 4];

async function runOffsetComponents(padded: boolean, value: number) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device);
  const {graph, view, read, lines} = harness;
  const output = {
    positions: view('float32x2', 16),
    sourceEdges: view('uint32', 16),
    components: view('uint32x4', 4),
    status: view('uint32x4', 2)
  };
  graph.add(
    new GPUOffsetCurveComponents({
      original: lines(ORIGINAL, [0, 3], padded ? 1 : 0, value),
      raw: lines(RAW, [0, 3], padded ? 1 : 0, value),
      distance: 0.5,
      tolerance: 0.001,
      segmentCapacity: 16,
      intersectionCapacity: 16,
      output
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled);
  const status = await read(output.status);
  const result = {
    status,
    components: (await read(output.components)).slice(0, status[0] * 4),
    positions: (await read(output.positions)).slice(0, status[1] * 2),
    sourceEdges: (await read(output.sourceEdges))
      .slice(0, status[1])
      .map(edge => edge - (padded ? LEADING : 0))
  };
  compiled.destroy();
  harness.destroy();
  return result;
}

it('GPUOffsetCurveComponents ignores position rows before and after every line', async () => {
  const reference = await runOffsetComponents(false, 0);
  expect(reference.status.slice(4)).toEqual([0, 0, 0, 0]);
  expect(reference.status[0]).toBeGreaterThan(0);
  for (const value of POISONS) {
    expect(await runOffsetComponents(true, value), `padding ${value}`).toEqual(reference);
  }
});
