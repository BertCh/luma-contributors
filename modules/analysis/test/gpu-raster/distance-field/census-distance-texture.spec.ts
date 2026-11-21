// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPURasterTextureToBuffer} from '@luma.gl/graph/gpu-raster';
import {it} from 'vitest';
import {GPUDistanceField} from '../../../src/raster-analysis/distance-field/gpu-distance-field';
import {getGPUDistanceFieldParameterValues} from '../../../src/raster-analysis/distance-field/distance-field-parameters';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createCensusRasterHarness, expectCensusValues} from '../census-raster-oracle-utils';

for (const withTexture of [false, true]) {
  it(`census distance texture matches an independent one-seed field (present=${withTexture})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createCensusRasterHarness(device);
    const grid = makeGPURasterMetadata({width: 3, height: 1});
    const texture = withTexture
      ? device.createTexture({
          width: 3,
          height: 1,
          format: 'r32float',
          usage: Texture.STORAGE | Texture.SAMPLE | Texture.COPY_SRC | Texture.COPY_DST
        })
      : undefined;
    try {
      const textureView = texture
        ? harness.graph.createTextureView(
            harness.graph.importTexture(
              {
                id: 'census-distance',
                width: 3,
                height: 1,
                format: 'r32float',
                usage: texture.props.usage
              },
              texture
            )
          )
        : undefined;
      const distance = new GPUDistanceField({
        grid,
        seedMask: harness.input('uint32', [0, 7, 0]),
        parameters: harness.input(
          'float32',
          Array.from(getGPUDistanceFieldParameterValues({}, grid))
        ),
        output: {
          distances: harness.output('distances', 'float32', 3),
          ...(textureView ? {texture: textureView} : {})
        }
      });
      harness.graph.add(distance);
      if (textureView)
        new GPURasterTextureToBuffer({
          input: {id: 'distance', format: 'float32', storage: {kind: 'texture', view: textureView}},
          output: harness.output('texture', 'float32', 3),
          outputValidity: harness.output('textureValidity', 'uint32', 3)
        }).addToGraph(harness.graph);
      const result = await harness.run();
      // One seed at column 1: Euclidean distance is |column-1|, independently of allocation.
      // Unit cells and these distances are exactly representable in float32.
      expectCensusValues(result['distances'], [1, 0, 1]);
      if (withTexture) {
        expectCensusValues(result['texture'], [1, 0, 1]);
        expectCensusValues(result['textureValidity'], [1, 1, 1]);
      }
    } finally {
      texture?.destroy();
    }
  });
}
