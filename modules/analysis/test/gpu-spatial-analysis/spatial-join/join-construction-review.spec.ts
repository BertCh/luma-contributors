// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createSpatialPredicateJoinOperation,
  createNearestFeatureJoinOperation
} from '../../../src/gpu-spatial-analysis/spatial-join/feature-join-operation';
import {formatGPUSpatialRelate} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-relate-types';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createMultipartSegmentCase} from './multipart-consumer-contracts-fixture';
import {getMultipartNestedView} from '../geometry-topology/multipart-line-forms-fixtures';

it('predicate GPU operation writes inherited destinations captured before caller mutation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const destinations = {
    leftRows: 'leftRows',
    rightRows: 'rightRows',
    relate: 'relate',
    count: 'count',
    requiredCount: 'requiredCount',
    overflow: 'overflow',
    candidateOverflow: 'candidateOverflow',
    candidateTruncated: 'candidateTruncated'
  };
  const outputs = Object.values(destinations);
  let patternReads = 0;
  const operation = createSpatialPredicateJoinOperation({
    left: {kind: 'points', positions: {value: 'left', length: 1}},
    right: {kind: 'points', positions: {value: 'right', length: 1}},
    predicate: 'relate',
    get pattern() {
      return ++patternReads === 1 ? ['T********'] : ['F********'];
    },
    candidateCapacity: 1,
    pairCapacity: 1,
    output: Object.create(destinations) as typeof destinations
  });
  expect(patternReads).toBe(1);
  destinations.count = 'replacement';
  const actual = await runOperationGraph(device, {
    inputs: {
      left: {format: 'float32x2', values: Float32Array.of(1, 2)},
      right: {format: 'float32x2', values: Float32Array.of(1, 2)}
    },
    operations: [operation],
    outputs
  });
  // Equal singleton points give exactly one pair and DE-9IM 0FFFFFFF2, by definition.
  for (const name of [
    'leftRows',
    'rightRows',
    'overflow',
    'candidateOverflow',
    'candidateTruncated'
  ])
    expect(Array.from(actual[name]), name).toEqual([0]);
  for (const name of ['count', 'requiredCount'])
    expect(Array.from(actual[name]), name).toEqual([1]);
  expect(formatGPUSpatialRelate(actual['relate'][0])).toBe('0FFFFFFF2');
});

it('nearest GPU operation writes inherited destinations captured before caller mutation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const destinations = {
    neighborRows: 'rows',
    neighborIds: 'ids',
    neighborCounts: 'counts',
    neighborDistances: 'distances',
    overflow: 'overflow',
    candidateOverflow: 'candidateOverflow'
  };
  const outputs = Object.values(destinations);
  const operation = createNearestFeatureJoinOperation({
    queries: {kind: 'points', positions: {value: 'queries', length: 1}},
    features: {
      kind: 'points',
      positions: {value: 'features', length: 2},
      sourceIds: {value: 'sourceIds', length: 2}
    },
    output: Object.create(destinations) as typeof destinations
  });
  destinations.neighborRows = 'replacement';
  const actual = await runOperationGraph(device, {
    inputs: {
      queries: {format: 'float32x2', values: Float32Array.of(0, 0)},
      features: {format: 'float32x2', values: Float32Array.of(6, 8, 3, 4)},
      sourceIds: {format: 'uint32', values: Uint32Array.of(77, 42)}
    },
    operations: [operation],
    outputs
  });
  // Pythagoras: row 0 is distance 10; row 1 is distance 5 and has caller ID 42.
  expect(
    Object.fromEntries(Object.entries(actual).map(([name, values]) => [name, Array.from(values)]))
  ).toEqual({
    rows: [1],
    ids: [42],
    counts: [1],
    distances: [5],
    overflow: [0],
    candidateOverflow: [0]
  });
});

it('multipart segment intersection retains logical feature rows on both sides', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers = new Map<GraphDataView, Buffer>();
  const fixture = createMultipartSegmentCase(graph, (name, format, length, values) => {
    const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
    const data = format.startsWith('float')
      ? Float32Array.from(values ?? Array(length * width).fill(NaN))
      : Uint32Array.from(values ?? Array(length * width).fill(99));
    const buffer = values
      ? createInputBuffer(device, data)
      : createOutputBuffer(device, length * width);
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
      expect(Array.from(actual), path).toEqual(expected);
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers.values()) buffer.destroy();
  }
});
