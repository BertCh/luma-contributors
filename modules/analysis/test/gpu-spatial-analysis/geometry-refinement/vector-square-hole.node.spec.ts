// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUVectorStraightSkeleton} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-vector-straight-skeleton';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {computeVectorStraightSkeletonOnCPU} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-vector-straight-skeleton-operation';
import {
  SQUARE_HOLE_INPUT,
  SQUARE_HOLE_SEGMENTS,
  SQUARE_HOLE_SOURCE_EDGES
} from './vector-square-hole-fixture';

useCPUExecutorTestHooks();
it('interpreted square-with-hole kernel pins all twelve provenance rows', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const positions = createTransientView(graph, 'positions', 'float32x2', 8);
  const ringOffsets = createTransientView(graph, 'rings', 'uint32', 3);
  const polygonOffsets = createTransientView(graph, 'polygons', 'uint32', 2);
  const featureOffsets = createTransientView(graph, 'features', 'uint32', 2);
  memory.write(positions, SQUARE_HOLE_INPUT.positions);
  memory.write(ringOffsets, SQUARE_HOLE_INPUT.ringOffsets);
  memory.write(polygonOffsets, SQUARE_HOLE_INPUT.polygonOffsets);
  memory.write(featureOffsets, SQUARE_HOLE_INPUT.featureOffsets);
  const output = {
    segments: createTransientView(graph, 'segments', 'float32x4', 32),
    sourceEdges: createTransientView(graph, 'edges', 'uint32x2', 32),
    offsetPositions: createTransientView(graph, 'offset', 'float32x2', 40),
    offsetLinks: createTransientView(graph, 'links', 'uint32x2', 40),
    status: createTransientView(graph, 'status', 'uint32x4', 1)
  };
  try {
    executeCommandNodesOnCPU(
      new GPUVectorStraightSkeleton({
        positions,
        ringOffsets,
        polygonOffsets,
        featureOffsets,
        maximumVertices: 8,
        maximumEvents: 16,
        segmentCapacity: 32,
        offsetDistance: 0.25,
        output
      }).getCommandNodes(graph),
      memory
    );
    expect(Array.from(memory.readUint32(output.status))).toEqual([12, 12, 8, 0]);
    expect(Array.from(memory.readFloat32(output.segments).slice(0, 48))).toEqual(
      SQUARE_HOLE_SEGMENTS
    );
    expect(Array.from(memory.readUint32(output.sourceEdges).slice(0, 24))).toEqual(
      SQUARE_HOLE_SOURCE_EDGES
    );
  } finally {
    device.destroy();
  }
});

it('square-with-hole skeleton pins geometry, status and source edges for all twelve live arcs', () => {
  const result = computeVectorStraightSkeletonOnCPU(SQUARE_HOLE_INPUT, {
    maximumVertices: 8,
    maximumEvents: 16,
    segmentCapacity: 32,
    offsetDistance: 0.25
  });
  expect(Array.from(result.status)).toEqual([12, 12, 8, 0]);
  expect(Array.from(result.segments.slice(0, 48))).toEqual(SQUARE_HOLE_SEGMENTS);
  expect(Array.from(result.sourceEdges.slice(0, 24))).toEqual(SQUARE_HOLE_SOURCE_EDGES);
});
