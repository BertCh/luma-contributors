// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPURasterBufferToTexture, GPURasterTextureToBuffer} from '@luma.gl/graph/gpu-raster';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUPointDensity} from '../../../src/gpu-spatial-analysis/point-density/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('publishes the hand-summed two-cell density into an independently poisoned texture', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Cell 0 contains weights 1 + 3, cell 1 contains weight 6. No smoothing or normalization.
  for (const withTexture of [true, false]) {
    const graph = new GPUCommandGraph(device);
    const positionsBuffer = createInputBuffer(
      device,
      new Float32Array([0.25, 0.5, 0.75, 0.5, 1.5, 0.5])
    );
    const weightsBuffer = createInputBuffer(device, new Float32Array([1, 3, 6]));
    const valuesBuffer = createOutputBuffer(device, 2);
    const readbackBuffer = createOutputBuffer(device, 2);
    const validityBuffer = createOutputBuffer(device, 2);
    const poisonBuffer = createInputBuffer(device, new Float32Array([91, 92]));
    valuesBuffer.write(new Float32Array([81, 82]));
    readbackBuffer.write(new Float32Array([71, 72]));
    const texture = withTexture
      ? device.createTexture({
          format: 'r32float',
          width: 2,
          height: 1,
          usage: Texture.STORAGE | Texture.SAMPLE | Texture.COPY_SRC | Texture.COPY_DST
        })
      : undefined;
    const textureView = texture
      ? graph.createTextureView(
          graph.importTexture(
            {id: 'texture', format: 'r32float', width: 2, height: 1, usage: texture.props.usage},
            texture
          ),
          {mipLevelCount: 1}
        )
      : undefined;
    if (textureView)
      new GPURasterBufferToTexture({
        id: 'poison-texture',
        input: {
          id: 'poison',
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: importGraphBuffer(graph, 'poison', poisonBuffer, 'float32', 2)
          }
        },
        output: textureView
      }).addToGraph(graph);
    graph.add(
      new GPUPointDensity({
        positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', 3),
        weights: importGraphBuffer(graph, 'weights', weightsBuffer, 'float32', 3),
        bounds: [0, 0, 2, 1],
        gridSize: [2, 1],
        statistic: 'sum',
        output: {
          values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', 2),
          texture: textureView
        }
      })
    );
    if (textureView)
      new GPURasterTextureToBuffer({
        id: 'read-texture',
        input: {id: 'density', format: 'float32', storage: {kind: 'texture', view: textureView}},
        output: importGraphBuffer(graph, 'readback', readbackBuffer, 'float32', 2),
        outputValidity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', 2)
      }).addToGraph(graph);
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readFloat32(valuesBuffer, 2)).toEqual([4, 6]);
      if (textureView) expect(await readFloat32(readbackBuffer, 2)).toEqual([4, 6]);
    } finally {
      compiled.destroy();
      for (const buffer of [
        positionsBuffer,
        weightsBuffer,
        valuesBuffer,
        readbackBuffer,
        validityBuffer,
        poisonBuffer
      ])
        buffer.destroy();
      texture?.destroy();
    }
  }
});
