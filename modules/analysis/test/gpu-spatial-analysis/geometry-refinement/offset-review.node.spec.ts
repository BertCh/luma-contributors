// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {computeOffsetCurveComponentsOnCPU} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-offset-curve-components-operation';
import {GPUOffsetCurveComponents} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-offset-curve-components';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {OFFSET_REVIEW_CASES} from './offset-review-fixtures';

useCPUExecutorTestHooks();
it.each(OFFSET_REVIEW_CASES)('CPU offset review: $name', fixture => {
  const count = fixture.raw.length / 2;
  const result = computeOffsetCurveComponentsOnCPU(
    {
      original: {
        positions: Float32Array.from(fixture.original),
        lineOffsets: Uint32Array.of(0, count)
      },
      raw: {positions: Float32Array.from(fixture.raw), lineOffsets: Uint32Array.of(0, count)}
    },
    {distance: fixture.distance, tolerance: 0, vertexCapacity: 32, componentCapacity: 8}
  );
  const expectedCount = fixture.expected.length / 2;
  expect(Array.from(result.status)).toEqual([
    Number(expectedCount > 0),
    expectedCount,
    Number(expectedCount > 0),
    expectedCount,
    0,
    0,
    0,
    0
  ]);
  expect(Array.from(result.positions.slice(0, expectedCount * 2))).toEqual(fixture.expected);
});

it.each(OFFSET_REVIEW_CASES)('interpreted offset review: $name', fixture => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const count = fixture.raw.length / 2;
  const original = {
    positions: createTransientView(graph, 'original', 'float32x2', count),
    lineOffsets: createTransientView(graph, 'original-offsets', 'uint32', 2)
  };
  const raw = {
    positions: createTransientView(graph, 'raw', 'float32x2', count),
    lineOffsets: createTransientView(graph, 'raw-offsets', 'uint32', 2)
  };
  memory.write(original.positions, Float32Array.from(fixture.original));
  memory.write(raw.positions, Float32Array.from(fixture.raw));
  memory.write(original.lineOffsets, Uint32Array.of(0, count));
  memory.write(raw.lineOffsets, Uint32Array.of(0, count));
  const output = {
    positions: createTransientView(graph, 'positions', 'float32x2', 32),
    components: createTransientView(graph, 'components', 'uint32x4', 8),
    sourceEdges: createTransientView(graph, 'sources', 'uint32', 32),
    status: createTransientView(graph, 'status', 'uint32x4', 2)
  };
  try {
    executeCommandNodesOnCPU(
      new GPUOffsetCurveComponents({
        original,
        raw,
        output,
        distance: fixture.distance,
        tolerance: 0,
        intersectionCapacity: 8,
        segmentCapacity: 8
      }).getCommandNodes(graph),
      memory
    );
    const expectedCount = fixture.expected.length / 2;
    expect(Array.from(memory.readUint32(output.status))).toEqual([
      Number(expectedCount > 0),
      expectedCount,
      Number(expectedCount > 0),
      expectedCount,
      0,
      0,
      0,
      0
    ]);
    expect(
      Array.from(memory.readFloat32(output.positions).slice(0, fixture.expected.length))
    ).toEqual(fixture.expected);
  } finally {
    device.destroy();
  }
});
