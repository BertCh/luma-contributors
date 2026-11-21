// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPURasterTextureToBuffer} from '@luma.gl/graph/gpu-raster';
import {expect, it} from 'vitest';
import {GPUAdjacencyMatrix} from '../../../src/gpu-network/adjacency-matrix/gpu-adjacency-matrix';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOracleContext} from './network-output-oracle-test-utils';

it('matrix scalar clears preserve adjacent padding on empty input (WRT-08)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const context = createOracleContext(device);
  try {
    const {input, output, graph} = context;
    const maximumCount = input('maximum-count', 'uint32', [99, 0x12345678]);
    const maximumWeightSum = input('maximum-weight', 'uint32', [99, 0x12345678]);
    graph.add(
      new GPUAdjacencyMatrix({
        topology: {
          offsets: input('offsets', 'uint32', [0, 0, 0]),
          neighbors: input('neighbors', 'uint32', [0xffffffff]),
          weights: input('weights', 'float32', [0])
        },
        directed: true,
        resolution: 1,
        output: {counts: output('counts', 'uint32'), maximumCount, maximumWeightSum}
      })
    );
    context.run();
    expect(Array.from(await context.read(maximumCount))).toEqual([0, 0x12345678]);
    expect(Array.from(await context.read(maximumWeightSum))).toEqual([0, 0x12345678]);
  } finally {
    context.destroy();
  }
});

it('weighted maximum and texture share transient bins without public sums (WRT-12)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const context = createOracleContext(device);
  const texture = device.createTexture({
    format: 'r32float',
    width: 1,
    height: 1,
    usage: Texture.STORAGE | Texture.SAMPLE | Texture.COPY_SRC | Texture.COPY_DST
  });
  try {
    const {input, output, graph} = context;
    const textureView = graph.createTextureView(
      graph.importTexture(
        {
          id: 'weighted-texture',
          format: 'r32float',
          width: 1,
          height: 1,
          usage: texture.props.usage
        },
        texture
      ),
      {mipLevelCount: 1}
    );
    const counts = output('counts', 'uint32');
    const maximumWeightSum = output('maximum', 'uint32');
    const texel = output('texel', 'float32');
    graph.add(
      new GPUAdjacencyMatrix({
        topology: {
          offsets: input('offsets', 'uint32', [0, 1, 1]),
          neighbors: input('neighbors', 'uint32', [1]),
          weights: input('weights', 'float32', [3])
        },
        directed: true,
        resolution: 1,
        weightScale: 1,
        output: {counts, maximumWeightSum, texture: textureView, textureStatistic: 'weightSum'}
      })
    );
    new GPURasterTextureToBuffer({
      id: 'read-weighted-texture',
      input: {
        id: 'weighted-band',
        format: 'float32',
        storage: {kind: 'texture', view: textureView}
      },
      output: texel,
      outputValidity: output('validity', 'uint32')
    }).addToGraph(graph);
    context.run();
    // One directed edge with weight 3 in the single bin: no normalization or scale change.
    expect(Array.from(await context.read(counts))).toEqual([1]);
    expect(Array.from(await context.read(maximumWeightSum))).toEqual([3]);
    expect(Array.from(await context.read(texel))).toEqual([3]);
  } finally {
    context.destroy();
    texture.destroy();
  }
});
