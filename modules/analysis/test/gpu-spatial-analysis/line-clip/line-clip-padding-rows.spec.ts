// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Rows of `lines.positions` before `lineOffsets[0]` and from `lineOffsets[last]` on belong to no
// line: unwritten capacity or a transient the graph reuses, holding anything. Before 2026-11-11
// GPULineClipByPolygon's segment table resolved a row before the first line to line 0, so each
// leading row formed a segment with the row after it, the last one with the first vertex of line
// 0; that phantom segment was classified by its midpoint and emitted as part of line 0. Each case
// runs once without padding and again with 2 leading and 3 trailing rows poisoned by NaN, by far
// finite values and by a value whose phantom segment has its midpoint inside the polygon, in both
// modes, and expects the same result.

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPULineClipByPolygon} from '../../../src/gpu-spatial-analysis/line-clip/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

/** A line across the unit square, and a line inside it. */
const LINES = [-1, 0.5, 2, 0.5, 0.25, 0.25, 0.75, 0.75];
const LINE_OFFSETS = [0, 2, 4];
const SQUARE = [0, 0, 1, 0, 1, 1, 0, 1, 0, 0];
/** NaN, far values, and 1.25: the segment (1.25, 1.25) to (-1, 0.5) has its midpoint inside. */
const POISONS = [Number.NaN, 3e6, -3e6, 1.25];
const LEADING = 2;
const TRAILING = 3;

const COMPONENTS = {uint32: 1, float32x2: 2} as const;
type Format = keyof typeof COMPONENTS;

async function runClip(
  mode: 'inside' | 'outside',
  leading: number,
  trailing: number,
  value: number,
  polygonLeading = 0,
  polygonTrailing = 0,
  polygonValue = 0
) {
  const testDevice = await getWebGPUTestDevice();
  requireTestDevice(testDevice);
  const device: Device = testDevice;
  const graph = new GPUCommandGraph(device, {id: 'line-clip-padding-rows'});
  const buffers = new Map<GraphDataView, Buffer>();
  let serial = 0;
  const view = <Value extends Format>(format: Value, length: number, values?: number[]) => {
    const buffer = values
      ? createInputBuffer(
          device,
          format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
        )
      : createOutputBuffer(device, length * COMPONENTS[format]);
    const result = importGraphBuffer(graph, `view-${serial++}`, buffer, format, length);
    buffers.set(result, buffer);
    return result as GraphDataView<Value>;
  };
  const read = (value: GraphDataView, rows = value.length) => {
    const buffer = buffers.get(value)!;
    const components = COMPONENTS[value.format as Format];
    return value.format === 'uint32'
      ? readUint32(buffer, rows * components)
      : readFloat32(buffer, rows * components);
  };
  const pieceCapacity = 16;
  const vertexCapacity = 64;
  const sourceIds = view('uint32', pieceCapacity);
  const offsets = view('uint32', pieceCapacity + 1);
  const positions = view('float32x2', vertexCapacity);
  const status = {
    count: view('uint32', 1),
    vertexCount: view('uint32', 1),
    overflow: view('uint32', 1),
    candidateOverflow: view('uint32', 1),
    requiredCount: view('uint32', 1),
    requiredVertexCount: view('uint32', 1)
  };
  const uncertainCount = view('uint32', 1);
  graph.add(
    new GPULineClipByPolygon({
      lines: {
        kind: 'lines',
        positions: view('float32x2', leading + LINES.length / 2 + trailing, [
          ...new Array<number>(2 * leading).fill(value),
          ...LINES,
          ...new Array<number>(2 * trailing).fill(value)
        ]),
        lineOffsets: view(
          'uint32',
          LINE_OFFSETS.length,
          LINE_OFFSETS.map(offset => offset + leading)
        )
      },
      polygons: {
        kind: 'polygons',
        positions: view('float32x2', polygonLeading + SQUARE.length / 2 + polygonTrailing, [
          ...new Array<number>(2 * polygonLeading).fill(polygonValue),
          ...SQUARE,
          ...new Array<number>(2 * polygonTrailing).fill(polygonValue)
        ]),
        featureOffsets: view('uint32', 2, [0, 1]),
        polygonOffsets: view('uint32', 2, [0, 1]),
        ringOffsets: view('uint32', 2, [polygonLeading, polygonLeading + SQUARE.length / 2])
      },
      mode,
      intersectionCapacity: 16,
      candidateCapacity: 64,
      output: {
        pieces: {geometry: {kind: 'lines', positions, lineOffsets: offsets}, sourceIds, status},
        status: {uncertainCount}
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const count = (await read(status.count))[0];
  const vertexCount = (await read(status.vertexCount))[0];
  const result = {
    count,
    vertexCount,
    overflow: await read(status.overflow),
    candidateOverflow: await read(status.candidateOverflow),
    requiredCount: await read(status.requiredCount),
    requiredVertexCount: await read(status.requiredVertexCount),
    uncertainCount: await read(uncertainCount),
    sourceIds: await read(sourceIds, count),
    offsets: await read(offsets, count + 1),
    positions: await read(positions, vertexCount)
  };
  compiled.destroy();
  buffers.forEach(buffer => buffer.destroy());
  return result;
}

for (const mode of ['inside', 'outside'] as const) {
  it(`GPULineClipByPolygon (${mode}) ignores rows before and after every line`, async () => {
    const reference = await runClip(mode, 0, 0, 0);
    expect(reference.overflow).toEqual([0]);
    // Inside: the crossing span of line 0 and all of line 1. Outside: both ends of line 0.
    expect(reference.count).toBe(2);
    for (const value of POISONS) {
      expect(await runClip(mode, LEADING, TRAILING, value), `padding ${value}`).toEqual(reference);
    }
  });
}

for (const mode of ['inside', 'outside'] as const) {
  it(`GPULineClipByPolygon (${mode}) ignores polygon rows before and after every ring`, async () => {
    const reference = await runClip(mode, 0, 0, 0);
    expect(reference.overflow).toEqual([0]);
    // Poison values include 0.5 and 1.25, which put phantom polygon edges across the lines.
    for (const value of [...POISONS, 0.5]) {
      expect(
        await runClip(mode, 0, 0, 0, LEADING, TRAILING, value),
        `polygon padding ${value}`
      ).toEqual(reference);
      expect(
        await runClip(mode, LEADING, TRAILING, value, LEADING, TRAILING, value),
        `both padding ${value}`
      ).toEqual(reference);
    }
  });
}
