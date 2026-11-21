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
  readUint32,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {getMultipartNestedView} from '../geometry-topology/multipart-line-forms-fixtures';
import {
  createAlignedRelateReviewCase,
  createPaddedSplitReviewCase,
  createEmptyTrackReviewCase
} from './multipart-review-fixtures';
import {createMeasureReviewCase} from './multipart-review-fixtures';
import {createEmptyPointRelateReviewCase} from './multipart-review-fixtures';
it.each([
  false,
  true
])('review2 R1 aligned empty multipoints and invalid suppression invalid=%s', async invalid => {
  await runReviewCase((graph, allocate) =>
    createEmptyPointRelateReviewCase(graph, allocate, invalid)
  );
});
import {createExtremeBoundsReviewCase} from './multipart-review-fixtures';
it.each([
  ['points', false],
  ['lines', false],
  ['lines', true]
] as const)('review2 R4 %s cooperative=%s preserves both FLT_MAX signs, extreme rows and group bounds', async (kind, cooperative) => {
  await runReviewCase((graph, allocate) =>
    createExtremeBoundsReviewCase(graph, allocate, kind, cooperative)
  );
});
import {createCollapsedClearanceReviewCase} from './multipart-review-fixtures';
import {createZeroAreaReviewCase} from './multipart-review-fixtures';
import {
  createIdentityLoopReviewCase,
  createInheritedMeasureReviewCase
} from './multipart-review-fixtures';
import {GPULineLoopRemovalOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-line-loop-removal-operation';
import {CPULineLoopRemovalOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-line-loop-removal-operation';
it('R7 contributor publishes ungrouped identity hierarchy', async () => {
  await runReviewCase(createIdentityLoopReviewCase);
});
it('R8 GPU measures captures inherited getters once and measures both parts', async () => {
  await runReviewCase(createInheritedMeasureReviewCase);
});
it('R7 both loop operations publish every output with output-only identity grouping', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPULineLoopRemovalOperation, CPULineLoopRemovalOperation]) {
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: Float32Array.of(0, 0, 2, 0)},
        parts: {format: 'uint32', values: Uint32Array.of(0, 2)}
      },
      operations: [
        new Operation({
          lines: {positions: 'positions', lineOffsets: 'parts'},
          lineCount: 1,
          vertexCapacity: 2,
          intersectionCapacity: 4,
          output: {
            positions: 'outPositions',
            lineOffsets: 'outParts',
            featureOffsets: 'outFeatures',
            sourceEdges: 'outEdges',
            status: 'status'
          }
        })
      ],
      outputs: ['outPositions', 'outParts', 'outFeatures', 'outEdges', 'status']
    });
    expect(Array.from(result['outPositions'])).toEqual([0, 0, 2, 0]);
    expect(Array.from(result['outParts'])).toEqual([0, 2]);
    expect(Array.from(result['outFeatures'])).toEqual([0, 1]);
    expect(Array.from(result['outEdges'])).toEqual([0, 0]);
    expect(Array.from(result['status'])).toEqual([2, 2, 0, 0]);
  }
});
it.each([
  'points',
  'lines'
] as const)('R9 %s zero areas and group totals match poisoned-buffer literals', async kind => {
  await runReviewCase((graph, allocate) => createZeroAreaReviewCase(graph, allocate, kind));
});
it.each([
  false,
  true
])('R6 collapsed-part clearance and winning row single=%s match GEOS', async single => {
  await runReviewCase((graph, allocate) =>
    createCollapsedClearanceReviewCase(graph, allocate, single)
  );
});
it.each([
  false,
  true
])('review2 R6a sorted=%s clearance tie uses the lowest original partner row', async spatialSort => {
  await runReviewCase((graph, allocate) =>
    createCollapsedClearanceReviewCase(graph, allocate, false, true, spatialSort)
  );
});
it.each([
  ['bounds-points', false, 4],
  ['bounds-lines', false, 4],
  ['bounds-lines', true, 4],
  ['collapsed', false, 4],
  ['collapsed', false, 5],
  ['collapsed', true, 4],
  ['collapsed', true, 5]
] as const)('R4/R5 measures %s cooperative=%s repeats=%s match GEOS', async (mode, cooperative, repeats) => {
  await runReviewCase((graph, allocate) =>
    createMeasureReviewCase(graph, allocate, mode, cooperative, repeats)
  );
});
import type {MultipartTestAllocate} from '../geometry-topology/multipart-line-forms-fixtures';
import {
  computeTrackSimilarityOnCPU,
  CPUTrackSimilarityOperation
} from '../../../src/gpu-spatial-analysis/track-similarity/cpu-track-similarity-operation';
import {computeLineLoopRemovalOnCPU} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-line-loop-removal-operation';
import {GPUTrackSimilarityOperation} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
async function runReviewCase(
  createCase: (
    graph: GPUCommandGraph,
    allocate: MultipartTestAllocate
  ) => {nodes: readonly unknown[]; output: object; expected: Record<string, number[]>}
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers = new Map<GraphDataView, Buffer>();
  const fixture = createCase(graph, (name, format, length, values) => {
    const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
    const data = format.startsWith('float')
      ? Float32Array.from(values ?? new Array(length * width).fill(12345))
      : Uint32Array.from(values ?? new Array(length * width).fill(99));
    const buffer =
      values && data.length
        ? createInputBuffer(device, data)
        : createOutputBuffer(device, Math.max(length, 1) * width);
    if (!values) buffer.write(data);
    const view = importGraphBuffer(graph, name, buffer, format, length);
    buffers.set(view, buffer);
    return view;
  });
  for (const node of fixture.nodes) graph.add(node as Parameters<typeof graph.add>[0]);
  const compiled = graph.compile();
  const result: Record<string, number[]> = {};
  try {
    submitGraph(device, compiled, undefined);
    for (const [path, expected] of Object.entries(fixture.expected)) {
      const view = getMultipartNestedView(fixture.output, path);
      result[path] = view.format.startsWith('float')
        ? await readFloat32(buffers.get(view)!, expected.length)
        : await readUint32(buffers.get(view)!, expected.length);
      expect(result[path], path).toEqual(expected);
    }
    return result;
  } finally {
    compiled.destroy();
    for (const buffer of buffers.values()) buffer.destroy();
  }
}
it.each([
  0, 1, 2, 3, 4, 5
])('R1 aligned topology variant %s matches independent Shapely matrices and predicates', async variant => {
  await runReviewCase((graph, allocate) => createAlignedRelateReviewCase(graph, allocate, variant));
});
it('R2 GPU split capacity padding is accepted by both CPU consumers', async () => {
  const result = await runReviewCase(createPaddedSplitReviewCase);
  const lines = {
    positions: Float32Array.from(result['pieces.geometry.positions']),
    lineOffsets: Uint32Array.from(result['pieces.geometry.lineOffsets']),
    featureOffsets: Uint32Array.from(result['pieces.geometry.featureOffsets'])
  };
  expect(
    Array.from(
      computeTrackSimilarityOnCPU(
        {linesA: lines, pairA: Uint32Array.of(0), pairB: Uint32Array.of(0)},
        {measures: ['hausdorff']}
      ).hausdorff
    )
  ).toEqual([0]);
  const erased = computeLineLoopRemovalOnCPU(lines, {vertexCapacity: 8});
  expect(Array.from(erased.positions.subarray(0, 4))).toEqual([0, 0, 2, 0]);
  expect(Array.from(erased.status)).toEqual([2, 2, 0, 0]);
});
it('R3 all-empty GPU contributors publish every documented sentinel', async () => {
  await runReviewCase(createEmptyTrackReviewCase);
});
it('R3 both track operation forms publish emptyTrack and NaNs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUTrackSimilarityOperation, CPUTrackSimilarityOperation]) {
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: new Float32Array(0)},
        parts: {format: 'uint32', values: Uint32Array.of(0)},
        features: {format: 'uint32', values: Uint32Array.of(0, 0, 0)},
        a: {format: 'uint32', values: Uint32Array.of(0)},
        b: {format: 'uint32', values: Uint32Array.of(1)}
      },
      operations: [
        new Operation({
          linesA: {positions: 'positions', lineOffsets: 'parts', featureOffsets: 'features'},
          pairA: 'a',
          pairB: 'b',
          pairCount: 1,
          output: {
            hausdorff: 'h',
            maximumDistance: 'maximum',
            symmetricSegmentPathDistance: 'path',
            pairStatus: 'status'
          }
        })
      ],
      outputs: ['h', 'maximum', 'path', 'status']
    });
    expect(Array.from(result['status'])).toEqual([1]);
    for (const name of ['h', 'maximum', 'path']) expect(result[name][0]).toBeNaN();
  }
});
