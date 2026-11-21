// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPULineChunk,
  getGPULineChunkParameterValues
} from '../../../src/gpu-spatial-analysis/line-segmentize/gpu-line-chunk';
import {GPULineLocate} from '../../../src/gpu-spatial-analysis/linear-referencing/gpu-line-locate';
import {
  createPointProjectionNode,
  LINEAR_REFERENCING_RESULT_STRIDE
} from '../../../src/gpu-spatial-analysis/linear-referencing/linear-referencing-kernels';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createPublishedPrefixReviewFixture} from './published-prefix-review-fixture';

it('B2 preserves moving endpoint, measures and a common cut through public chunk and locate consumers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createParityGraph(device);
  const expected = createPublishedPrefixReviewFixture();
  const {graph, input, output} = fixture;
  const positions = input('positions', Array.from(expected.positions), 'float32x2').view;
  const lineOffsets = input('offsets', [0, 193], 'uint32').view;
  const chunkPositions = output('chunk-positions', 194, 'float32x2');
  const chunkMeasures = output('chunk-measures', 194, 'float32');
  const chunkOffsets = output('chunk-offsets', 3, 'uint32');
  const chunkCount = output('chunk-count', 1, 'uint32');
  const chunkOverflow = output('chunk-overflow', 1, 'uint32');
  const located = output('located', 1, 'float32x2');
  const locateStatus = output('locate-status', 1, 'uint32');
  const measures = output('vertex-measures', 193, 'float32');
  let compiled: ReturnType<typeof graph.compile> | undefined;
  try {
    graph.add(
      new GPULineChunk({
        positions,
        lineOffsets,
        mode: 'chunk',
        parameters: input(
          'chunk-parameters',
          Array.from(getGPULineChunkParameterValues({chunkLength: 1024})),
          'float32'
        ).view,
        output: {
          positions: chunkPositions.view,
          measures: chunkMeasures.view,
          lineOffsets: chunkOffsets.view,
          status: {count: chunkCount.view, overflow: chunkOverflow.view}
        }
      })
    );
    graph.add(
      new GPULineLocate({
        positions,
        lineOffsets,
        measureMode: 'fraction',
        eventLineIndices: input('event-lines', [0], 'uint32').view,
        eventMeasures: input('event-measures', [1], 'float32').view,
        output: {
          positions: located.view,
          statuses: locateStatus.view,
          vertexMeasures: measures.view
        }
      })
    );
    compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    expect(Array.from(await readUint32(chunkCount.buffer, 1))).toEqual([2]);
    expect(Array.from(await readUint32(chunkOverflow.buffer, 1))).toEqual([0]);
    expect(Array.from(await readUint32(chunkOffsets.buffer, 3))).toEqual([0, 65, 194]);
    expect(Array.from(await readUint32(locateStatus.buffer, 1))).toEqual([0]);
    expect(Array.from(await readFloat32(located.buffer, 2))).toEqual(expected.endpoint);
    const published = await readFloat32(measures.buffer, 193);
    expect(Array.from(published.slice(191))).toEqual([expected.total, expected.total]);
    const chunks = await readFloat32(chunkPositions.buffer, 194 * 2);
    // Both chunks use the first row at the shared cut (row 64), then preserve all later rows.
    expect(Array.from(chunks)).toEqual([
      ...expected.positions.slice(0, 65 * 2),
      ...expected.positions.slice(64 * 2)
    ]);
    expect(Array.from(chunks.slice(64 * 2, 66 * 2))).toEqual([1024, 0, 1024, 0]);
    expect(Array.from(chunks.slice(-2))).toEqual(expected.endpoint);
    expect(Array.from(await readFloat32(chunkMeasures.buffer, 194))).toEqual([
      ...expected.measures.slice(0, 65),
      ...expected.measures.slice(64)
    ]);
  } finally {
    try {
      compiled?.destroy();
    } finally {
      fixture.destroy();
    }
  }
});

it('B2 uses the published end measure for a planar projection endpoint', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createParityGraph(device);
  const expected = createPublishedPrefixReviewFixture();
  const {graph, input, output} = fixture;
  const results = output('results', LINEAR_REFERENCING_RESULT_STRIDE, 'uint32');
  let compiled: ReturnType<typeof graph.compile> | undefined;
  try {
    graph.add(
      createPointProjectionNode(graph, {
        id: 'projection-endpoint',
        operation: 'test',
        positions: input('positions', Array.from(expected.positions), 'float32x2').view,
        lineOffsets: input('offsets', [0, 193], 'uint32').view,
        vertexMeasures: input('measures', Array.from(expected.measures), 'float32').view,
        points: input('query', [1024, 65 * 2 ** -20], 'float32x2').view,
        nearestSegments: input('nearest', [128], 'uint32').view,
        results: results.view
      })
    );
    compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const result = await readFloat32(results.buffer, LINEAR_REFERENCING_RESULT_STRIDE);
    expect(Array.from(result.slice(2, 5))).toEqual([1, 1024, 65 * 2 ** -20]);
    expect(Array.from(result.slice(6, 7))).toEqual([expected.total]);
  } finally {
    try {
      compiled?.destroy();
    } finally {
      fixture.destroy();
    }
  }
});
