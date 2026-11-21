// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUTextureShading,
  getGPUTextureShadingParameterValues
} from '../../../src/gpu-terrain/terrain-illumination/gpu-texture-shading';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('RT-136 aligned decimated and off-phase undecimated tiles match independent convolution samples', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Independent scipy.ndimage reference on a 129x129 field: convolve1d in x then y, mode='nearest',
  // sigma increments sqrt(4,12,48), radius ceil(3*sigma/scale), normalized exp(-x²/(2*sigma²)).
  // Blur finite-value numerator and validity denominator separately; decimate both with [::2,::2]
  // before level 2, then map_coordinates(order=1) numerator and denominator separately and divide.
  // Weights [1,.6,.2]. Field is float32-rounded before the float64 reference convolution.
  // At (52,54), decimated: -0.48355205839001336; at (53,55), undecimated: -0.7265054466738525.
  for (const missingRegion of [false, true]) {
    for (const downsampleLevels of [true, false]) {
      const origin = downsampleLevels ? ([4, 6] as const) : ([5, 7] as const);
      const width = 97;
      const elevation = Float32Array.from({length: width * width}, (_, index) => {
        const column = origin[0] + (index % width);
        const row = origin[1] + Math.floor(index / width);
        return (column === 54 && row === 55) ||
          (missingRegion && column >= 54 && column < 62 && row >= 53 && row < 62)
          ? NaN
          : 0.01 * column * column +
              0.03 * row +
              Math.sin(0.23 * column + 0.31 * row) +
              0.2 * Math.cos(0.17 * column - 0.11 * row);
      });
      const graph = new GPUCommandGraph(device);
      const input = createInputBuffer(device, elevation);
      const parameters = createInputBuffer(
        device,
        getGPUTextureShadingParameterValues({levelWeights: [1, 0.6, 0.2]})
      );
      const shade = createOutputBuffer(device, elevation.length);
      const validity = createOutputBuffer(device, elevation.length);
      const stage = new GPUTextureShading({
        grid: makeGPURasterMetadata({width, height: width}),
        levelCount: 3,
        baseSigma: 2,
        downsampleLevels,
        elevation: {
          id: 'elevation',
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: importGraphBuffer(graph, 'elevation', input, 'float32', elevation.length)
          }
        },
        parameters: importGraphBuffer(graph, 'parameters', parameters, 'float32', 12),
        output: {
          textureShade: importGraphBuffer(graph, 'shade', shade, 'float32', elevation.length),
          outputValidity: importGraphBuffer(graph, 'validity', validity, 'uint32', elevation.length)
        }
      });
      if (downsampleLevels) {
        expect(() => stage.getCommandNodesForTile(graph, [5, 6])).toThrow(/expandedOrigin/);
        expect(() => stage.getCommandNodesForTile(graph, [4, 7])).toThrow(/expandedOrigin/);
      }
      graph.add({getCommandNodes: target => stage.getCommandNodesForTile(target, origin)});
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const values = await readFloat32(shade, elevation.length);
        const valid = await readUint32(validity, elevation.length);
        const interior = 48 * width + 48;
        expect(Number.isFinite(values[interior])).toBe(true);
        expect(values[interior]).toBeCloseTo(
          missingRegion
            ? downsampleLevels
              ? 0.2250032513562708
              : 0.31458781054459234
            : downsampleLevels
              ? -0.48355205839001336
              : -0.7265054466738525,
          3
        );
        expect(valid[interior]).toBe(1);
        if (downsampleLevels) {
          const odd = (55 - origin[1]) * width + 53 - origin[0];
          expect(Number.isFinite(values[odd])).toBe(true);
          expect(valid[odd]).toBe(1);
          // Independent SciPy n/d reconstruction. With the asymmetric 8x9 missing rectangle,
          // interpolating quotients instead gives 0.3068379604691294, over 4 tolerances away.
          expect(values[odd]).toBeCloseTo(
            missingRegion ? 0.30923205042102775 : -0.72978599868228,
            3
          );
        }
        const missing = (55 - origin[1]) * width + 54 - origin[0];
        expect(valid[missing]).toBe(0);
        expect(Number.isNaN(values[missing])).toBe(true);
      } finally {
        compiled.destroy();
        for (const buffer of [input, parameters, shade, validity]) buffer.destroy();
      }
    }
  }
});
