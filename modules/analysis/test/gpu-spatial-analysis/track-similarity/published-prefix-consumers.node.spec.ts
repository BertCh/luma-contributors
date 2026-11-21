// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {afterEach, beforeEach, expect, it, vi} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  createLineLocateNode,
  createPointProjectionNode,
  LINE_LOCATE_RESULT_STRIDE,
  LINEAR_REFERENCING_RESULT_STRIDE
} from '../../../src/gpu-spatial-analysis/linear-referencing/linear-referencing-kernels';
import {
  createPieceRangeNode,
  LINE_PIECE_STRIDE
} from '../../../src/gpu-spatial-analysis/line-segmentize/line-chunk-kernels';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {createPublishedPrefixReviewFixture} from './published-prefix-review-fixture';
import * as chunkGroups from '../../../src/analysis-core/kernels/graph-chunk-groups';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

// The interpreter does not model the chunk-group parameter uniform. For these single-chunk
// fixtures, emit the producer's unchanged bindings/declarations/body with constant view offsets.
beforeEach(() => {
  vi.spyOn(chunkGroups, 'createGraphChunkGroupKernelNode').mockImplementation((graph, props) =>
    createWGSLKernelNode(graph, props)
  );
});
afterEach(() => vi.restoreAllMocks());

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const view = <Format extends 'float32x2' | 'float32' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `consumer-review-${sequence++}`, format, length);
  const positions = view('float32x2', 193),
    vertexMeasures = view('float32', 193),
    lineOffsets = view('uint32', 2);
  const fixture = createPublishedPrefixReviewFixture();
  const memory = createCPUGraphMemory();
  memory.write(positions, fixture.positions);
  memory.write(vertexMeasures, fixture.measures);
  memory.write(lineOffsets, Uint32Array.of(0, 193));
  return {device, graph, view, positions, vertexMeasures, lineOffsets, fixture, memory};
}

it.each([
  'distance',
  'fraction'
] as const)('B2 locates the original moving endpoint in %s mode', measureMode => {
  const fixture = createFixture();
  const {device, graph, view, positions, vertexMeasures, lineOffsets, memory} = fixture;
  try {
    const eventLineIndices = view('uint32', 1),
      eventMeasures = view('float32', 1);
    const results = view('uint32', LINE_LOCATE_RESULT_STRIDE);
    const node = createLineLocateNode(graph, {
      id: `locate-${measureMode}`,
      operation: 'test',
      positions,
      lineOffsets,
      vertexMeasures,
      eventLineIndices,
      eventMeasures,
      measureMode,
      results
    });
    memory.write(eventLineIndices, Uint32Array.of(0));
    memory.write(
      eventMeasures,
      Float32Array.of(measureMode === 'fraction' ? 1 : fixture.fixture.total)
    );
    executeCommandNodesOnCPU([node], memory);
    expect(Array.from(memory.readUint32(results)).slice(5)).toEqual([0]);
    expect(Array.from(memory.readFloat32(results)).slice(0, 2)).toEqual(fixture.fixture.endpoint);
  } finally {
    device.destroy();
  }
});

it('B2 uses a common first-row cut and preserves the terminal chunk row', () => {
  const fixture = createFixture();
  const {device, graph, view, lineOffsets, vertexMeasures, memory} = fixture;
  try {
    const parameters = view('float32', 4),
      pieceCounts = view('uint32', 1),
      pieceStarts = view('uint32', 1);
    const pieces = view('uint32', 2 * LINE_PIECE_STRIDE),
      vertexCounts = view('uint32', 2),
      pieceTotal = view('uint32', 1);
    const node = createPieceRangeNode(graph, {
      id: 'ranges',
      operation: 'test',
      mode: 'chunk',
      lineOffsets,
      rowMeasures: vertexMeasures,
      parameters,
      pieceCounts,
      pieceStarts,
      rowCount: 193,
      pieces,
      vertexCounts,
      pieceTotal
    });
    memory.write(parameters, Float32Array.of(1024, 0, 3.4e38, 0));
    memory.write(pieceCounts, Uint32Array.of(2));
    memory.write(pieceStarts, Uint32Array.of(0));
    executeCommandNodesOnCPU([node], memory);
    const rows = memory.readUint32(pieces);
    expect(Array.from(memory.readUint32(pieceTotal))).toEqual([2]);
    // Canonical cut is original row 64 at measure 1024. Row 65 is the first interior row
    // of chunk 2; its end excludes row 192 so emission can append that original final vertex.
    expect([rows[5], rows[LINE_PIECE_STRIDE + 4], rows[LINE_PIECE_STRIDE + 5]]).toEqual([
      64, 65, 192
    ]);
    expect(Array.from(memory.readUint32(vertexCounts))).toEqual([65, 129]);
  } finally {
    device.destroy();
  }
});

it('B2 projection publishes the stored endpoint measure at fraction one', () => {
  const fixture = createFixture();
  const {device, graph, view, positions, lineOffsets, vertexMeasures, memory} = fixture;
  try {
    const points = view('float32x2', 1),
      nearestSegments = view('uint32', 1),
      results = view('uint32', LINEAR_REFERENCING_RESULT_STRIDE);
    const node = createPointProjectionNode(graph, {
      id: 'project',
      operation: 'test',
      points,
      positions,
      lineOffsets,
      nearestSegments,
      vertexMeasures,
      results
    });
    // Row 129 is the end of segment 128, selected by the lowest-row tie rule.
    memory.write(points, fixture.fixture.positions.slice(2 * 129, 2 * 130));
    memory.write(nearestSegments, Uint32Array.of(128));
    executeCommandNodesOnCPU([node], memory);
    const result = memory.readFloat32(results);
    expect(Array.from(result).slice(2, 3)).toEqual([1]);
    expect(Array.from(result).slice(3, 5)).toEqual([1024, 65 * 2 ** -20]);
    expect(Array.from(result).slice(6, 7)).toEqual([1024.0001220703125]);
  } finally {
    device.destroy();
  }
});
