// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Rows of `lines.positions` before `lineOffsets[0]` and from `lineOffsets[last]` on belong to no
// line: unwritten capacity or a transient the graph reuses, holding anything. Results must not
// depend on them. Before 2026-11-11 GPULineSplit's non-finite check read every row, so a NaN in a
// padding row set `overflow` in the split and in every class that nodes lines through it. Each case
// runs once without padding and again with 2 leading and 3 trailing rows poisoned by NaN and by
// far finite values, and expects the same result. Input edge indices are global vertex indices, so
// they are compared relative to the first line.

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPULineSplit,
  GPU_LINE_SPLIT_SELF_CONTACT
} from '../../../src/gpu-spatial-analysis/line-split/gpu-line-split';
import {GPULineNoding} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-line-noding';
import {GPUPolygonize} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-polygonize';
import {GPUNetworkNoding} from '../../../src/gpu-network/network-noding/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createTopologyOutput} from '../geometry-refinement/topology-output';

/** Two crossing lines and one line that touches neither. */
const LINES = [0, 0, 2, 2, 0, 2, 2, 0, 3, 0, 3, 1, 4, 1];
const LINE_OFFSETS = [0, 2, 4, 7];
const CLOSED_SQUARE = [0.25, 0.25, 0.75, 0.25, 0.75, 0.75, 0.25, 0.75, 0.25, 0.25];
const NONE = 0xffffffff;
/** NaN, and values far outside the data. */
const POISONS = [Number.NaN, 3e6, -3e6];
const LEADING = 2;
const TRAILING = 3;

const COMPONENTS = {uint32: 1, float32: 1, float32x2: 2, float32x4: 4} as const;
type Format = keyof typeof COMPONENTS;

function createHarness(device: Device) {
  const graph = new GPUCommandGraph(device, {id: 'line-split-padding-rows'});
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
    const components = COMPONENTS[value.format as Format];
    const buffer = buffers.get(value)!;
    return value.format.startsWith('float')
      ? readFloat32(buffer, rows * components)
      : readUint32(buffer, rows * components);
  };
  const run = () => {
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    return () => {
      compiled.destroy();
      buffers.forEach(buffer => buffer.destroy());
    };
  };
  return {graph, view, read, run};
}

async function getDevice() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return device;
}

/** `lines` input with `leading` and `trailing` rows of `value` around `coordinates`. */
function createLines(
  view: ReturnType<typeof createHarness>['view'],
  coordinates: number[],
  offsets: number[],
  leading: number,
  trailing: number,
  value: number
) {
  const rows = coordinates.length / 2;
  return {
    kind: 'lines' as const,
    positions: view('float32x2', leading + rows + trailing, [
      ...new Array<number>(2 * leading).fill(value),
      ...coordinates,
      ...new Array<number>(2 * trailing).fill(value)
    ]),
    lineOffsets: view(
      'uint32',
      offsets.length,
      offsets.map(offset => offset + leading)
    )
  };
}

/** Removes the leading-row shift from an input edge index, keeping flag bits and NONE. */
function rebaseEdge(edge: number, leading: number, flagBits = 0) {
  if (edge === NONE) {
    return edge;
  }
  const flags = edge & flagBits;
  return (((edge & ~flagBits) >>> 0) - leading + flags) >>> 0;
}

async function runSplit(leading: number, trailing: number, value: number) {
  const harness = createHarness(await getDevice());
  const {view} = harness;
  const pieceCapacity = 16;
  const vertexCapacity = 64;
  const status = {
    count: view('uint32', 1),
    vertexCount: view('uint32', 1),
    overflow: view('uint32', 1),
    candidateOverflow: view('uint32', 1),
    requiredCount: view('uint32', 1),
    requiredVertexCount: view('uint32', 1)
  };
  const sourceIds = view('uint32', pieceCapacity);
  const offsets = view('uint32', pieceCapacity + 1);
  const positions = view('float32x2', vertexCapacity);
  const vertexSourceEdges = view('uint32', vertexCapacity);
  harness.graph.add(
    new GPULineSplit({
      lines: createLines(view, LINES, LINE_OFFSETS, leading, trailing, value),
      intersectionCapacity: 16,
      output: {
        pieces: {
          geometry: {kind: 'lines', positions, lineOffsets: offsets},
          sourceIds,
          vertexSourceEdges,
          status
        }
      }
    })
  );
  const destroy = harness.run();
  const count = (await harness.read(status.count))[0];
  const vertexCount = (await harness.read(status.vertexCount))[0];
  const pieceOffsets = await harness.read(offsets, count + 1);
  const edges = await harness.read(vertexSourceEdges, vertexCount);
  // The edge of the last vertex of each piece is unspecified.
  const lastVertices = new Set(Array.from(pieceOffsets.slice(1), end => end - 1));
  const result = {
    count,
    vertexCount,
    overflow: await harness.read(status.overflow),
    candidateOverflow: await harness.read(status.candidateOverflow),
    requiredCount: await harness.read(status.requiredCount),
    requiredVertexCount: await harness.read(status.requiredVertexCount),
    sourceIds: await harness.read(sourceIds, count),
    offsets: pieceOffsets,
    positions: await harness.read(positions, vertexCount),
    sourceEdges: Array.from(edges, (edge, row) =>
      lastVertices.has(row) ? null : rebaseEdge(edge, leading, GPU_LINE_SPLIT_SELF_CONTACT)
    )
  };
  destroy();
  return result;
}

async function runLineNoding(leading: number, trailing: number, value: number) {
  const harness = createHarness(await getDevice());
  const {view} = harness;
  const capacity = 16;
  const status = {
    count: view('uint32', 1),
    requiredCount: view('uint32', 1),
    overflow: view('uint32', 1),
    candidateOverflow: view('uint32', 1)
  };
  const output = {
    endpoints: view('float32x4', capacity),
    sourceFeatureIds: view('uint32', capacity),
    sourceRingIds: view('uint32', capacity),
    sourceEdgeIds: view('uint32', capacity),
    sourceStartParameters: view('float32', capacity),
    sourceEndParameters: view('float32', capacity),
    status
  };
  harness.graph.add(
    new GPULineNoding({
      lines: createLines(view, LINES, LINE_OFFSETS, leading, trailing, value),
      intersectionCapacity: 16,
      output
    })
  );
  const destroy = harness.run();
  const count = (await harness.read(status.count))[0];
  const result = {
    count,
    requiredCount: await harness.read(status.requiredCount),
    overflow: await harness.read(status.overflow),
    candidateOverflow: await harness.read(status.candidateOverflow),
    endpoints: await harness.read(output.endpoints, count),
    features: await harness.read(output.sourceFeatureIds, count),
    rings: await harness.read(output.sourceRingIds, count),
    edges: Array.from(await harness.read(output.sourceEdgeIds, count), edge =>
      rebaseEdge(edge, leading)
    ),
    starts: await harness.read(output.sourceStartParameters, count),
    ends: await harness.read(output.sourceEndParameters, count)
  };
  destroy();
  return result;
}

async function runNetworkNoding(leading: number, trailing: number, value: number) {
  const harness = createHarness(await getDevice());
  const {view} = harness;
  const pieceCapacity = 16;
  const nodeCapacity = 16;
  const output = {
    pieces: {
      geometry: {
        kind: 'lines' as const,
        positions: view('float32x2', 64),
        lineOffsets: view('uint32', pieceCapacity + 1)
      },
      sourceIds: view('uint32', pieceCapacity),
      status: {count: view('uint32', 1), overflow: view('uint32', 1)}
    },
    nodes: {
      positions: view('float32x2', nodeCapacity),
      status: {count: view('uint32', 1), requiredCount: view('uint32', 1)}
    },
    edges: {
      fromNodes: view('uint32', pieceCapacity),
      toNodes: view('uint32', pieceCapacity),
      lengths: view('float32', pieceCapacity)
    },
    topology: {
      offsets: view('uint32', nodeCapacity + 1),
      neighbors: view('uint32', 2 * pieceCapacity),
      weights: view('float32', 2 * pieceCapacity),
      edgeIds: view('uint32', 2 * pieceCapacity)
    },
    status: {overflow: view('uint32', 1)}
  };
  harness.graph.add(
    new GPUNetworkNoding({
      lines: createLines(view, LINES, LINE_OFFSETS, leading, trailing, value),
      intersectionCapacity: 16,
      tolerance: view('float32', 1, [0]),
      output
    })
  );
  const destroy = harness.run();
  const pieceCount = (await harness.read(output.pieces.status.count))[0];
  const nodeCount = (await harness.read(output.nodes.status.count))[0];
  const offsets = await harness.read(output.topology.offsets, nodeCount + 1);
  const adjacency = offsets[nodeCount];
  const pieceOffsets = await harness.read(output.pieces.geometry.lineOffsets, pieceCount + 1);
  const result = {
    overflow: await harness.read(output.status.overflow),
    pieceOverflow: await harness.read(output.pieces.status.overflow),
    pieceCount,
    pieceIds: await harness.read(output.pieces.sourceIds, pieceCount),
    pieceOffsets,
    piecePositions: await harness.read(output.pieces.geometry.positions, pieceOffsets[pieceCount]),
    nodeCount,
    requiredNodeCount: await harness.read(output.nodes.status.requiredCount),
    nodes: await harness.read(output.nodes.positions, nodeCount),
    from: await harness.read(output.edges.fromNodes, pieceCount),
    to: await harness.read(output.edges.toNodes, pieceCount),
    lengths: await harness.read(output.edges.lengths, pieceCount),
    offsets,
    neighbors: await harness.read(output.topology.neighbors, adjacency),
    weights: await harness.read(output.topology.weights, adjacency),
    edgeIds: await harness.read(output.topology.edgeIds, adjacency)
  };
  destroy();
  return result;
}

async function runPolygonize(leading: number, trailing: number, value: number) {
  const harness = createHarness(await getDevice());
  const {view} = harness;
  const output = createTopologyOutput(view);
  harness.graph.add(
    new GPUPolygonize({
      lines: createLines(view, CLOSED_SQUARE, [0, 5], leading, trailing, value),
      intersectionCapacity: 16,
      precision: {vertexTolerance: 1 / 64, storage: 'float32', predicates: 'exact-or-uncertain'},
      output
    })
  );
  const destroy = harness.run();
  const {polygons, status} = output;
  const rings = await harness.read(polygons.ringOffsets);
  const result = {
    count: await harness.read(status.count),
    overflow: await harness.read(status.overflow!),
    nodedOverflow: await harness.read(output.nodedSegments.status.overflow!),
    rings,
    positions: await harness.read(polygons.positions, rings[rings.length - 1])
  };
  destroy();
  return result;
}

it('GPULineSplit ignores rows before and after every line', async () => {
  const reference = await runSplit(0, 0, 0);
  expect(reference.overflow).toEqual([0]);
  expect(reference.count).toBe(5);
  for (const value of POISONS) {
    expect(await runSplit(LEADING, TRAILING, value), `padding ${value}`).toEqual(reference);
  }
});

it('GPULineNoding ignores rows before and after every line', async () => {
  const reference = await runLineNoding(0, 0, 0);
  expect(reference.overflow).toEqual([0]);
  expect(reference.count).toBe(6);
  for (const value of POISONS) {
    expect(await runLineNoding(LEADING, TRAILING, value), `padding ${value}`).toEqual(reference);
  }
});

it('GPUNetworkNoding ignores rows before and after every line', async () => {
  const reference = await runNetworkNoding(0, 0, 0);
  expect(reference.overflow).toEqual([0]);
  expect(reference.pieceCount).toBe(5);
  for (const value of POISONS) {
    expect(await runNetworkNoding(LEADING, TRAILING, value), `padding ${value}`).toEqual(reference);
  }
});

it('GPUPolygonize without snap rounding ignores rows before and after its lines', async () => {
  const reference = await runPolygonize(0, 0, 0);
  expect(reference.overflow).toEqual([0]);
  expect(reference.count).toEqual([1]);
  for (const value of POISONS) {
    expect(await runPolygonize(LEADING, TRAILING, value), `padding ${value}`).toEqual(reference);
  }
});
