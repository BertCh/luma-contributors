// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSubdividePolygons} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-subdivide-polygons';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('subdivision reports an empty batch despite resident position capacity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const positions = createInputBuffer(device, Float32Array.of(1, 1, 2, 2));
  const ringOffsets = createInputBuffer(device, Uint32Array.of(0));
  const polygonOffsets = createInputBuffer(device, Uint32Array.of(0));
  const featureOffsets = createInputBuffer(device, Uint32Array.of(0));
  const outputPositions = createOutputBuffer(device, 8);
  const pieces = createOutputBuffer(device, 4);
  const status = createOutputBuffer(device, 4);
  graph.add(
    new GPUSubdividePolygons({
      maximumVertices: 4,
      positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', 2),
      ringOffsets: importGraphBuffer(graph, 'rings', ringOffsets, 'uint32', 1),
      polygonOffsets: importGraphBuffer(graph, 'polygons', polygonOffsets, 'uint32', 1),
      featureOffsets: importGraphBuffer(graph, 'features', featureOffsets, 'uint32', 1),
      output: {
        positions: importGraphBuffer(graph, 'output-positions', outputPositions, 'float32x2', 4),
        pieces: importGraphBuffer(graph, 'pieces', pieces, 'uint32x4', 1),
        status: importGraphBuffer(graph, 'status', status, 'uint32x4', 1)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    // No polygon means no written/required piece, capacity loss, or invalid polygon.
    expect(Array.from(await readUint32(status, 4))).toEqual([0, 0, 0, 0]);
  } finally {
    compiled.destroy();
    for (const buffer of [
      positions,
      ringOffsets,
      polygonOffsets,
      featureOffsets,
      outputPositions,
      pieces,
      status
    ])
      buffer.destroy();
  }
});
