// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPURasterTextureToBuffer} from '@luma.gl/graph/gpu-raster';
import {expect, it} from 'vitest';
import {
  GPUAdjacencyMatrix,
  GPUAdjacencyMatrixOrder
} from '../../../src/gpu-network/adjacency-matrix/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOracleContext} from './network-output-oracle-test-utils';

it('adjacency texture contains literal directed pixels with other optional outputs absent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Two parallel 0->1 arcs and one 1->0 arc. Row-major pixels are [0,2,1,0].
  for (const withTexture of [true, false]) {
    const context = createOracleContext(device);
    const {graph, input, output, read} = context;
    const counts = output('counts', 'uint32', 4);
    const pixels = output('pixels', 'float32', 4);
    const validity = output('validity', 'uint32', 4);
    const texture = withTexture
      ? device.createTexture({
          format: 'r32float',
          width: 2,
          height: 2,
          data: new Float32Array([913, 913, 913, 913]),
          usage: Texture.STORAGE | Texture.SAMPLE | Texture.COPY_SRC | Texture.COPY_DST
        })
      : undefined;
    const textureView =
      texture &&
      graph.createTextureView(
        graph.importTexture(
          {
            id: 'texture',
            format: 'r32float',
            width: 2,
            height: 2,
            usage: texture.props.usage
          },
          texture
        ),
        {mipLevelCount: 1}
      );
    graph.add(
      new GPUAdjacencyMatrix({
        topology: {
          offsets: input('offsets', 'uint32', [0, 2, 3]),
          neighbors: input('neighbors', 'uint32', [1, 1, 0])
        },
        directed: true,
        resolution: 2,
        output: {counts, texture: textureView}
      })
    );
    if (textureView)
      new GPURasterTextureToBuffer({
        input: {id: 'pixels', format: 'float32', storage: {kind: 'texture', view: textureView}},
        output: pixels,
        outputValidity: validity
      }).addToGraph(graph);
    try {
      context.run();
      expect(await read(counts)).toEqual([0, 2, 1, 0]);
      if (withTexture) {
        expect(await read(pixels)).toEqual([0, 2, 1, 0]);
        expect(await read(validity)).toEqual([1, 1, 1, 1]);
      }
    } finally {
      context.destroy();
      texture?.destroy();
    }
  }
});

it('adjacency order is the inverse of a hand-sorted node list', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const context = createOracleContext(device);
  const {graph, input, output, read} = context;
  const order = output('order', 'uint32', 3);
  // Sorting (group,tie,node) gives nodes [1,2,0], hence order[node]=[2,0,1].
  // A three-cycle also distinguishes the sorted list from its inverse permutation.
  graph.add(
    new GPUAdjacencyMatrixOrder({
      groups: input('groups', 'uint32', [2, 1, 1]),
      tieKeys: input('ties', 'uint32', [0, 1, 2]),
      output: {order}
    })
  );
  try {
    context.run();
    expect(await read(order)).toEqual([2, 0, 1]);
  } finally {
    context.destroy();
  }
});
