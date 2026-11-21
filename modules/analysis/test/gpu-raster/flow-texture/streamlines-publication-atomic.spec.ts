// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUStreamlines} from '../../../src/raster-analysis/flow-texture/gpu-streamlines';
import {
  getGPUStreamlinesParameterValues,
  getGPUStreamlinesUint32ParameterValues
} from '../../../src/raster-analysis/flow-texture/streamlines-parameters';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('streamline candidate loss clears on a completed resubmission after simultaneous undecided seeds', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const fieldBuffer = createInputBuffer(device, Float32Array.of(1, 0));
  const parameterBuffer = createInputBuffer(
    device,
    getGPUStreamlinesParameterValues({
      fieldPlacement: {origin: [0, 0], cellSize: [1, 1]},
      gridPlacement: {origin: [0, 0], cellSize: [1, 1]},
      stepLength: 0
    })
  );
  const wordBuffer = createInputBuffer(
    device,
    getGPUStreamlinesUint32ParameterValues({seed: 1, minimumPoints: 3})
  );
  const buffers = [fieldBuffer, parameterBuffer, wordBuffer];
  const word = (id: string, length = 1) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, id, buffer, 'uint32', length)};
  };
  const count = word('count');
  const overflow = word('overflow');
  const candidateOverflow = word('candidateOverflow');
  const iterationLimitReached = word('iterationLimitReached');
  const positionsBuffer = createOutputBuffer(device, 24);
  buffers.push(positionsBuffer);
  graph.add(
    new GPUStreamlines({
      velocities: importGraphBuffer(graph, 'field', fieldBuffer, 'float32x2', 1),
      parameters: importGraphBuffer(graph, 'parameters', parameterBuffer, 'float32', 12),
      uint32Parameters: importGraphBuffer(graph, 'words', wordBuffer, 'uint32', 4),
      fieldWidth: 1,
      fieldHeight: 1,
      gridWidth: 1,
      gridHeight: 1,
      seedColumns: 4,
      seedRows: 1,
      stepsPerDirection: 1,
      maximumIterations: 1,
      output: {
        lines: {
          ids: word('ids', 4).view,
          status: {
            count: count.view,
            overflow: overflow.view,
            candidateOverflow: candidateOverflow.view
          }
        },
        lineOffsets: word('offsets', 5).view,
        positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', 12),
        positionCount: word('positionCount').view,
        status: {iterationLimitReached: iterationLimitReached.view}
      }
    })
  );
  const compiled = graph.compile();
  try {
    // Four valid seeds in one occupancy cell: round 1 accepts the priority winner, leaving
    // three undecided invocations to publish candidate loss to the same storage word.
    for (const velocity of [1, 0, 1]) {
      fieldBuffer.write(Float32Array.of(velocity, 0));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(count.buffer, 1)).toEqual([velocity]);
      expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
      expect(await readUint32(candidateOverflow.buffer, 1)).toEqual([velocity]);
      expect(await readUint32(iterationLimitReached.buffer, 1)).toEqual([velocity]);
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
