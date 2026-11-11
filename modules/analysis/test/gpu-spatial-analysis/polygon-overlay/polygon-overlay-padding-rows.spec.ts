// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Rows of an operand past its last ring offset, and the single closing slot of an empty ring, are
// padding: no ring owns them and nothing promises what they hold (a transient the graph reuses
// holds another node's data; a fixed-capacity overlay result leaves them unwritten). The overlay
// must give the same result whatever they contain. Before 2026-11-11 `combine-positions` copied
// `positions[ringOffsets[last]]` into those rows and the noder measured its lattice frame over
// every row and flagged any non-finite row, so a NaN there set `overflow`, a far value coarsened
// the lattice, and a dissolve followed by a clip in one graph (building tessellation) depended on
// which other transients the compiler had placed in the same buffer.

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUPolygonOverlay,
  type GPUPolygonOverlayOperation
} from '../../../src/gpu-spatial-analysis/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

/** A square of side 0.6 at an offset that a coarser lattice would move. */
const LEFT_SQUARE = [0.3, 0.3, 0.9, 0.3, 0.9, 0.9, 0.3, 0.9];
const RIGHT_SQUARE = [0.6, 0.1, 1.3, 0.1, 1.3, 0.7, 0.6, 0.7];

type Padding = {
  /** Number of empty rings after the square. */
  emptyRings: number;
  /** Rows after the last ring offset. */
  tailRows: number;
  /** Value of every padding row. */
  value: number;
};

type OverlayResult = {
  count: number;
  overflow: number;
  candidateOverflow: number;
  boundaryOverflow: number;
  rings: number[];
  positions: number[];
};

async function runOverlay(
  operation: GPUPolygonOverlayOperation,
  leftPadding: Padding,
  rightPadding?: Padding
): Promise<OverlayResult> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: `overlay-padding-${operation}`});
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32x2' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array,
    format: Format,
    length: number
  ): GraphDataView<Format> => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, length) as GraphDataView<Format>;
  };
  const output = <Format extends 'float32x2' | 'float32x4' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const components = format === 'float32x4' ? 4 : format === 'float32x2' ? 2 : 1;
    const buffer = createOutputBuffer(device, length * components);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const operand = (prefix: string, square: number[], padding: Padding) => {
    const rows = square.length / 2 + padding.tailRows;
    const positions = new Float32Array(rows * 2).fill(padding.value);
    positions.set(square);
    const ringCount = 1 + padding.emptyRings;
    const ringOffsets = new Uint32Array(ringCount + 1).fill(square.length / 2);
    ringOffsets[0] = 0;
    return {
      kind: 'polygons' as const,
      positions: input(`${prefix}-positions`, positions, 'float32x2', rows),
      featureOffsets: input(`${prefix}-features`, Uint32Array.of(0, 1), 'uint32', 2),
      polygonOffsets: input(`${prefix}-polygons`, Uint32Array.of(0, ringCount), 'uint32', 2),
      ringOffsets: input(`${prefix}-rings`, ringOffsets, 'uint32', ringCount + 1)
    };
  };
  const scalar = (name: string) => output(name, 'uint32', 1);
  const positions = output('positions', 'float32x2', 64);
  const ringOffsets = output('rings', 'uint32', 9);
  const count = scalar('count');
  const overflow = scalar('overflow');
  const candidateOverflow = scalar('candidate-overflow');
  const boundaryOverflow = scalar('boundary-overflow');
  graph.add(
    new GPUPolygonOverlay({
      left: operand('left', LEFT_SQUARE, leftPadding),
      right: rightPadding ? operand('right', RIGHT_SQUARE, rightPadding) : undefined,
      operation,
      capacity: {intersections: 64, nodedSegments: 64},
      vertexTolerance: 1 / 1024,
      output: {
        geometry: {
          kind: 'polygons',
          positions: positions.view,
          featureOffsets: output('features', 'uint32', 9).view,
          polygonOffsets: output('polygons', 'uint32', 9).view,
          ringOffsets: ringOffsets.view
        },
        status: {
          count: count.view,
          overflow: overflow.view,
          candidateOverflow: candidateOverflow.view
        },
        boundary: {
          endpoints: output('boundary', 'float32x4', 64).view,
          operandIds: output('boundary-operands', 'uint32', 64).view,
          sourceFeatureIds: output('boundary-features', 'uint32', 64).view,
          status: {
            count: scalar('boundary-count').view,
            requiredCount: scalar('boundary-required').view,
            overflow: boundaryOverflow.view,
            candidateOverflow: scalar('boundary-candidate-overflow').view
          }
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const rings = await readUint32(ringOffsets.buffer, 9);
  const result: OverlayResult = {
    count: (await readUint32(count.buffer, 1))[0],
    overflow: (await readUint32(overflow.buffer, 1))[0],
    candidateOverflow: (await readUint32(candidateOverflow.buffer, 1))[0],
    boundaryOverflow: (await readUint32(boundaryOverflow.buffer, 1))[0],
    rings,
    positions: await readFloat32(positions.buffer, rings[8] * 2)
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

const NO_PADDING: Padding = {emptyRings: 0, tailRows: 0, value: 0};

it('GPUPolygonOverlay ignores non-finite padding rows past the last ring', async () => {
  const reference = await runOverlay('dissolve', NO_PADDING);
  expect(reference.overflow).toBe(0);
  expect(reference.count).toBe(1);
  const padded = await runOverlay('dissolve', {emptyRings: 2, tailRows: 3, value: Number.NaN});
  expect(padded.boundaryOverflow).toBe(0);
  expect(padded.overflow).toBe(0);
  expect(padded).toEqual(reference);
});

it('GPUPolygonOverlay keeps its lattice when padding rows hold far finite values', async () => {
  // A row at 3e6 would widen the lattice frame to a cell of 2^-1, which moves the square's corners.
  const reference = await runOverlay('dissolve', NO_PADDING);
  const padded = await runOverlay('dissolve', {emptyRings: 1, tailRows: 2, value: 3e6});
  expect(padded).toEqual(reference);
});

it('GPUPolygonOverlay intersection ignores padding rows of both operands', async () => {
  const reference = await runOverlay('intersection', NO_PADDING, NO_PADDING);
  expect(reference.overflow).toBe(0);
  expect(reference.count).toBe(1);
  for (const value of [Number.NaN, -3e6]) {
    const padded = await runOverlay(
      'intersection',
      {emptyRings: 2, tailRows: 1, value},
      {emptyRings: 1, tailRows: 2, value}
    );
    expect(padded).toEqual(reference);
  }
});
