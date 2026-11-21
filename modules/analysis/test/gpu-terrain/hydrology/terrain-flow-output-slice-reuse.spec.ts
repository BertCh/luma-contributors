// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../../src/raster-analysis';
import {
  GPUTerrainFlow,
  GPU_TERRAIN_FLOW_STREAM_UNRESOLVED,
  getGPUTerrainFlowParameterValues
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('WRT-09 clears only the requested second word after overflow and clean reuse', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeGPURasterMetadata({width: 1, height: 1});
  const elevation = createInputBuffer(device, Float32Array.of(5));
  const parameters = createInputBuffer(
    device,
    getGPUTerrainFlowParameterValues({streamThreshold: 2}, grid)
  );
  const destination = createOutputBuffer(device, 2);
  const graph = new GPUCommandGraph(device);
  const backing = importGraphBuffer(graph, 'destination', destination, 'uint32', 2);
  const second = graph.createDataView(backing.buffer, {format: 'uint32', byteOffset: 4, length: 1});
  graph.add(
    new GPUTerrainFlow({
      grid,
      elevation: makeGPURasterBand(
        importGraphBuffer(graph, 'elevation', elevation, 'float32', 1),
        grid
      ),
      parameters: importGraphBuffer(graph, 'parameters', parameters, 'float32', 8),
      accumulationUnits: 'area',
      maximumAccumulationIterations: 1,
      output: {streams: second}
    })
  );
  const compiled = graph.compile();
  try {
    destination.write(Uint32Array.of(17, 99));
    for (const overflow of [true, false, true, false]) {
      // A one-cell weight is its area: 10^20 * 10^20 overflows f32; 1 * 1 is exactly 1.
      parameters.write(
        getGPUTerrainFlowParameterValues(
          {streamThreshold: 2},
          {...grid, affine: overflow ? [1e20, 0, 0, 0, -1e20, 0] : [1, 0, 0, 0, -1, 0]}
        )
      );
      submitGraph(device, compiled, undefined);
      expect(await readUint32(destination, 2)).toEqual([
        17,
        overflow ? GPU_TERRAIN_FLOW_STREAM_UNRESOLVED : 0
      ]);
    }
  } finally {
    compiled.destroy();
    for (const buffer of [elevation, parameters, destination]) buffer.destroy();
  }
});
