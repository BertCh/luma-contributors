// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUTerrainDerivatives,
  type GPUTerrainDerivativesOutput
} from '../../../src/gpu-terrain/terrain-analysis';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createBand} from '../terrain-test-utils';

it('TYP-11 accepts each derivative output on its own and rejects an empty selection', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = {
      grid: makeGPURasterMetadata({width: 1, height: 1}),
      elevation: createBand(graph, 'elevation', 1),
      parameters: createTransientView(graph, 'parameters', 'float32', 8)
    };
    const hillshadeTexture = graph.createTextureView(
      graph.createTransientTexture({
        id: 'hillshade-texture',
        format: 'r32float',
        width: 1,
        height: 1,
        usage: Texture.STORAGE
      })
    );
    const outputs: GPUTerrainDerivativesOutput[] = [
      {slope: createTransientView(graph, 'slope', 'float32', 1)},
      {aspect: createTransientView(graph, 'aspect', 'float32', 1)},
      {hillshade: createTransientView(graph, 'hillshade', 'float32', 1)},
      {outputValidity: createTransientView(graph, 'validity', 'uint32', 1)},
      {hillshadeTexture}
    ];
    for (const output of outputs) {
      expect(() => new GPUTerrainDerivatives({...props, output})).not.toThrow();
    }
    expect(() => {
      // @ts-expect-error At least one output must have a writer.
      new GPUTerrainDerivatives({...props, output: {}});
    }).toThrow(/^terrain-derivatives requires at least one output$/);
  } finally {
    device.destroy();
  }
});
