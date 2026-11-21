// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {GPUTrackSimilarityOperation} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity-operation';
import {CPUTrackSimilarityOperation} from '../../../src/gpu-spatial-analysis/track-similarity/cpu-track-similarity-operation';
import {
  createMultipartNodingCase,
  createMultipartSplitCase,
  createMultipartValidityCase,
  createMultipartBufferCase,
  createMultipartSharedPathsCase,
  createMultipartClipCase,
  createMultipartTrackCase,
  createMultipartTrackSearchCase,
  createEmptyMultipartCase,
  createMultipartTrackOperationCase,
  MULTIPART_TRACK_EXPECTED,
  createMultipartPolygonizeCase,
  getMultipartNestedView,
  MULTIPART_NODED_EXPECTED
} from './multipart-line-forms-fixtures';

it('multipart track GPU and CPU operations each match independent set-metric literals', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUTrackSimilarityOperation, CPUTrackSimilarityOperation]) {
    const {inputs, props} = createMultipartTrackOperationCase();
    const result = await runOperationGraph(device, {
      inputs,
      operations: [new Operation(props)],
      outputs: Object.keys(MULTIPART_TRACK_EXPECTED)
    });
    for (const [port, values] of Object.entries(MULTIPART_TRACK_EXPECTED))
      for (const [row, expected] of values.entries()) {
        const actual = result[port][row];
        if (Number.isNaN(expected)) expect(actual).toBeNaN();
        else {
          expect(Number.isFinite(actual)).toBe(true);
          expect(Math.abs(actual - expected)).toBeLessThanOrEqual(2e-5);
        }
      }
  }
});

it.each([
  ['shared paths', createMultipartSharedPathsCase],
  [
    'single-part shared paths with caller IDs',
    (graph: GPUCommandGraph, allocate: Parameters<typeof createMultipartSharedPathsCase>[1]) =>
      createMultipartSharedPathsCase(graph, allocate, false)
  ],
  ['empty multipoints', createEmptyMultipartCase],
  [
    'empty multipoint predicates in isolation',
    (graph: GPUCommandGraph, allocate: Parameters<typeof createEmptyMultipartCase>[1]) =>
      createEmptyMultipartCase(graph, allocate, 'points', true)
  ],
  [
    'empty multiline predicates in isolation',
    (graph: GPUCommandGraph, allocate: Parameters<typeof createEmptyMultipartCase>[1]) =>
      createEmptyMultipartCase(graph, allocate, 'lines', true)
  ],
  [
    'empty multilines',
    (graph: GPUCommandGraph, allocate: Parameters<typeof createEmptyMultipartCase>[1]) =>
      createEmptyMultipartCase(graph, allocate, 'lines')
  ],
  ['track metrics', createMultipartTrackCase],
  ['track search Hausdorff', createMultipartTrackSearchCase],
  [
    'track search SSPD',
    (graph: GPUCommandGraph, allocate: Parameters<typeof createMultipartTrackSearchCase>[1]) =>
      createMultipartTrackSearchCase(graph, allocate, 'symmetric-segment-path')
  ],
  ['inside clip', createMultipartClipCase],
  [
    'outside clip',
    (graph: GPUCommandGraph, allocate: Parameters<typeof createMultipartClipCase>[1]) =>
      createMultipartClipCase(graph, allocate, 'outside')
  ]
] as const)('multipart %s matches every oracle port', async (_name, createCase) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers = new Map<GraphDataView, Buffer>();
  const fixture = createCase(graph, (name, format, length, values) => {
    const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
    const data = format.startsWith('float')
      ? Float32Array.from(values ?? new Array(length * width).fill(Number.NaN))
      : Uint32Array.from(values ?? new Array(length * width).fill(99));
    const buffer =
      values && data.length > 0
        ? createInputBuffer(device, data)
        : // A zero-row GraphDataView still needs one full physical row, not one scalar.
          createOutputBuffer(device, Math.max(length, 1) * width);
    if (!values) buffer.write(data);
    const view = importGraphBuffer(graph, name, buffer, format, length);
    buffers.set(view, buffer);
    return view;
  });
  for (const node of fixture.nodes) graph.add(node);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    for (const [path, expected] of Object.entries(fixture.expected)) {
      const view = getMultipartNestedView(fixture.output, path);
      const actual = view.format.startsWith('float')
        ? await readFloat32(buffers.get(view)!, expected.length)
        : await readUint32(buffers.get(view)!, expected.length);
      if (_name.includes('track')) {
        // At most a dozen f32 distances/sums on coordinates <=10; absolute 2e-5.
        for (const [row, value] of expected.entries())
          if (Number.isNaN(value)) expect(actual[row], path).toBeNaN();
          else {
            expect(Number.isFinite(actual[row]), path).toBe(true);
            expect(Math.abs(actual[row] - value), path).toBeLessThanOrEqual(2e-5);
          }
      } else expect(actual, path).toEqual(expected);
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers.values()) buffer.destroy();
  }
});

it('multipart buffer surface dissolves strokes and preserves logical boundary owners', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers = new Map<GraphDataView, Buffer>();
  const fixture = createMultipartBufferCase(graph, (name, format, length, values) => {
    const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
    const data = format.startsWith('float')
      ? Float32Array.from(values ?? new Array(length * width).fill(Number.NaN))
      : Uint32Array.from(values ?? new Array(length * width).fill(99));
    const buffer = values
      ? createInputBuffer(device, data)
      : createOutputBuffer(device, length * width);
    if (!values) buffer.write(data);
    const view = importGraphBuffer(graph, name, buffer, format, length);
    buffers.set(view, buffer);
    return view;
  });
  graph.add(fixture.contributor);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const expected: Record<string, number[]> = {
      'geometry.featureOffsets': [0, 1, 2, 2, 2],
      'geometry.polygonOffsets': [0, 1, 2, 2, 2],
      'geometry.ringOffsets': [0, 4, 8, 8, 8],
      'status.count': [2],
      // These count assembled output rings (shells + holes), not polygonize face cycles.
      // Shapely gives two disconnected rectangles, each with one shell and no holes.
      'status.ringCount': [2],
      'status.requiredRingCount': [2],
      'status.overflow': [0],
      'status.candidateOverflow': [0],
      'status.uncertainCount': [0],
      'status.nodedSegmentRequiredCount': [8],
      'boundary.operandIds': [0, 0, 0, 0, 0, 0, 0, 0],
      'boundary.sourceFeatureIds': [17, 17, 17, 17, 17, 17, 17, 17],
      'boundary.sourceFeatureRows': [1, 1, 1, 1, 1, 1, 1, 1],
      'boundary.status.count': [8],
      'boundary.status.requiredCount': [8],
      'boundary.status.overflow': [0],
      'boundary.status.candidateOverflow': [0],
      'featureSources.sourceRows': [1, 1],
      'featureSources.operandIds': [0, 0],
      'featureSources.mixed': [0, 0]
    };
    for (const [path, values] of Object.entries(expected))
      expect(
        await readUint32(buffers.get(getMultipartNestedView(fixture.output, path))!, values.length),
        path
      ).toEqual(values);
    // Shapely buffer(...,.5,cap_style='flat'): two rectangles, area 2 each. Compare actual
    // vertex sets and boundary edge sets, independently of the assembly's chosen starting vertex.
    const coordinates = await readFloat32(buffers.get(fixture.output.geometry.positions)!, 16);
    const pointKeys = (values: number[]) =>
      Array.from(
        {length: values.length / 2},
        (_, row) => `${values[row * 2]},${values[row * 2 + 1]}`
      ).sort();
    expect(coordinates.every(Number.isFinite)).toBe(true);
    expect(pointKeys(coordinates)).toEqual(
      pointKeys([0, -0.5, 2, -0.5, 2, 0.5, 0, 0.5, 5, -0.5, 7, -0.5, 7, 0.5, 5, 0.5])
    );
    for (let ring = 0; ring < 2; ring++) {
      let doubledArea = 0;
      for (let vertex = 0; vertex < 4; vertex++) {
        const current = (ring * 4 + vertex) * 2;
        const next = (ring * 4 + ((vertex + 1) % 4)) * 2;
        doubledArea +=
          coordinates[current] * coordinates[next + 1] -
          coordinates[next] * coordinates[current + 1];
      }
      expect(doubledArea).toBe(4);
    }
    const edges = await readFloat32(buffers.get(fixture.output.boundary.endpoints)!, 32);
    const edgeKeys = (values: number[]) =>
      Array.from({length: values.length / 4}, (_, row) =>
        pointKeys(values.slice(row * 4, row * 4 + 4)).join('|')
      ).sort();
    expect(edges.every(Number.isFinite)).toBe(true);
    expect(edgeKeys(edges)).toEqual(
      edgeKeys([
        0, -0.5, 2, -0.5, 2, -0.5, 2, 0.5, 2, 0.5, 0, 0.5, 0, 0.5, 0, -0.5, 5, -0.5, 7, -0.5, 7,
        -0.5, 7, 0.5, 7, 0.5, 5, 0.5, 5, 0.5, 5, -0.5
      ])
    );
  } finally {
    compiled.destroy();
    for (const buffer of buffers.values()) buffer.destroy();
  }
});

it.each([
  'points',
  'lines'
] as const)('multipart %s validity matches independent Shapely flags', async kind => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers = new Map<GraphDataView, Buffer>();
  const fixture = createMultipartValidityCase(
    graph,
    (name, format, length, values) => {
      const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
      const data = format.startsWith('float')
        ? Float32Array.from(values ?? new Array(length * width).fill(Number.NaN))
        : Uint32Array.from(values ?? new Array(length * width).fill(99));
      const buffer = values
        ? createInputBuffer(device, data)
        : createOutputBuffer(device, length * width);
      if (!values) buffer.write(data);
      const view = importGraphBuffer(graph, name, buffer, format, length);
      buffers.set(view, buffer);
      return view;
    },
    kind
  );
  for (const node of fixture.nodes) graph.add(node);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    for (const [path, expected] of Object.entries(fixture.expected))
      expect(
        await readUint32(
          buffers.get(getMultipartNestedView(fixture.output, path))!,
          expected.length
        ),
        path
      ).toEqual(expected);
  } finally {
    compiled.destroy();
    for (const buffer of buffers.values()) buffer.destroy();
  }
});

it.each([5, 3])('multipart split matches Shapely with capacity=%s', async capacity => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers = new Map<GraphDataView, Buffer>();
  const fixture = createMultipartSplitCase(
    graph,
    (name, format, length, values) => {
      const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
      const data = format.startsWith('float')
        ? Float32Array.from(values ?? new Array(length * width).fill(Number.NaN))
        : Uint32Array.from(values ?? new Array(length * width).fill(99));
      const buffer = values
        ? createInputBuffer(device, data)
        : createOutputBuffer(device, length * width);
      if (!values) buffer.write(data);
      const view = importGraphBuffer(graph, name, buffer, format, length);
      buffers.set(view, buffer);
      return view;
    },
    capacity
  );
  for (const node of fixture.nodes) graph.add(node);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    for (const [path, expected] of Object.entries(fixture.expected)) {
      const view = getMultipartNestedView(fixture.output, path);
      const actual = view.format.startsWith('float')
        ? await readFloat32(buffers.get(view)!, expected.length)
        : await readUint32(buffers.get(view)!, expected.length);
      expect(actual, path).toEqual(expected);
    }
    expect(
      (
        await readUint32(buffers.get(fixture.output.pieces.vertexSourceEdges)!, capacity * 2)
      ).filter((_, row) => row % 2 === 0)
    ).toEqual([0, 0, 2, 2, 4].slice(0, capacity));
  } finally {
    compiled.destroy();
    for (const buffer of buffers.values()) buffer.destroy();
  }
});

it.each([
  {grouped: false, snapped: false},
  {grouped: true, snapped: false},
  {grouped: false, snapped: true},
  {grouped: true, snapped: true}
])('multipart polygonize matches Shapely with groups=$grouped snap=$snapped', async ({
  grouped,
  snapped
}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers = new Map<GraphDataView, Buffer>();
  const fixture = createMultipartPolygonizeCase(
    graph,
    (name, format, length, values) => {
      const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
      const data = format.startsWith('float')
        ? Float32Array.from(values ?? new Array(length * width).fill(Number.NaN))
        : Uint32Array.from(values ?? new Array(length * width).fill(99));
      const buffer = values
        ? createInputBuffer(device, data)
        : createOutputBuffer(device, length * width);
      if (!values) buffer.write(data);
      const view = importGraphBuffer(graph, name, buffer, format, length);
      buffers.set(view, buffer);
      return view;
    },
    grouped,
    snapped
  );
  for (const node of fixture.nodes) graph.add(node);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    for (const [path, expected] of Object.entries(fixture.expected)) {
      const view = getMultipartNestedView(fixture.output, path);
      const actual = view.format.startsWith('float')
        ? await readFloat32(buffers.get(view)!, expected.length)
        : await readUint32(buffers.get(view)!, expected.length);
      expect(actual, path).toEqual(expected);
    }
    if (snapped) expect(await readFloat32(buffers.get(fixture.lattice!)!, 4)).toEqual([0, 0, 1, 0]);
  } finally {
    compiled.destroy();
    for (const buffer of buffers.values()) buffer.destroy();
  }
});

it.each([
  {sourceIds: true, snapped: false},
  {sourceIds: false, snapped: false},
  {sourceIds: true, snapped: true},
  {sourceIds: false, snapped: true}
])('multipart noding matches Shapely with identities=$sourceIds snap=$snapped', async ({
  sourceIds,
  snapped
}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers = new Map<GraphDataView, Buffer>();
  const fixture = createMultipartNodingCase(
    graph,
    (name, format, length, values) => {
      const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
      const data = format.startsWith('float')
        ? Float32Array.from(values ?? new Array(length * width).fill(Number.NaN))
        : Uint32Array.from(values ?? new Array(length * width).fill(99));
      const buffer = values
        ? createInputBuffer(device, data)
        : createOutputBuffer(device, length * width);
      if (!values) buffer.write(data);
      const view = importGraphBuffer(graph, name, buffer, format, length);
      buffers.set(view, buffer);
      return view;
    },
    sourceIds,
    snapped
  );
  for (const node of fixture.nodes) graph.add(node);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    if (snapped) {
      expect(await readFloat32(buffers.get(fixture.lattice!)!, 4)).toEqual([0, 0, 1, 0]);
      expect(await readFloat32(buffers.get(fixture.latticeEndpoints!)!, 20)).toEqual(
        MULTIPART_NODED_EXPECTED.endpoints
      );
    }
    for (const [name, expected] of Object.entries(MULTIPART_NODED_EXPECTED)) {
      const view = fixture.output[name as keyof typeof MULTIPART_NODED_EXPECTED];
      const actual = view.format.startsWith('float')
        ? await readFloat32(buffers.get(view)!, expected.length)
        : await readUint32(buffers.get(view)!, expected.length);
      expect(actual, name).toEqual(
        name === 'sourceFeatureIds' && !sourceIds ? [1, 1, 1, 1, 2] : expected
      );
    }
    for (const [name, view] of Object.entries(fixture.output.status))
      expect(await readUint32(buffers.get(view)!, 1), name).toEqual([
        name === 'count' || name === 'requiredCount' ? 5 : 0
      ]);
  } finally {
    compiled.destroy();
    for (const buffer of buffers.values()) buffer.destroy();
  }
});
