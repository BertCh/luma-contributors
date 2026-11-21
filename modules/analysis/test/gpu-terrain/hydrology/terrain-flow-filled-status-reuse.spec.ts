// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  getGPUTerrainFlowParameterValues,
  GPUTerrainFlow
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('filled-only flow rewrites poisoned flat-status words on every submission (RT-122)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeGPURasterMetadata({width: 1, height: 1});
  const elevation = createInputBuffer(device, Float32Array.of(7));
  const parameters = createInputBuffer(device, getGPUTerrainFlowParameterValues({}, grid));
  const filled = createOutputBuffer(device, 1);
  const converged = createOutputBuffer(device, 1);
  const iterations = createOutputBuffer(device, 1);
  const graph = new GPUCommandGraph(device);
  graph.add(
    new GPUTerrainFlow({
      grid,
      elevation: makeGPURasterBand(
        importGraphBuffer(graph, 'elevation', elevation, 'float32', 1),
        grid
      ),
      parameters: importGraphBuffer(graph, 'parameters', parameters, 'float32', 8),
      fillDepressions: true,
      resolveFlats: true,
      maximumFillIterations: 2,
      maximumFlatIterations: 2,
      output: {
        filledElevation: importGraphBuffer(graph, 'filled', filled, 'float32', 1),
        status: {
          flatsConverged: importGraphBuffer(graph, 'converged', converged, 'uint32', 1),
          flatsIterationCount: importGraphBuffer(graph, 'iterations', iterations, 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  try {
    for (const height of [7, 11]) {
      elevation.write(Float32Array.of(height));
      converged.write(Uint32Array.of(12345));
      iterations.write(Uint32Array.of(12345));
      submitGraph(device, compiled, undefined);
      expect(await readFloat32(filled, 1)).toEqual([height]);
      expect(await readUint32(converged, 1)).toEqual([1]);
      // A lone cell has no neighbour or active flat edge. Each of the lower, higher and maximum
      // relaxations has an unchanged first round: the sum of their executed counts is 1+1+1.
      expect(await readUint32(iterations, 1)).toEqual([3]);
    }
  } finally {
    compiled.destroy();
    for (const buffer of [elevation, parameters, filled, converged, iterations]) buffer.destroy();
  }
});
