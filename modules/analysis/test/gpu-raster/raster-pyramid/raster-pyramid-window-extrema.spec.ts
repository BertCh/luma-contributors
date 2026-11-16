// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('builds odd-sized pyramid batches and partial final stages against direct pixel-window extrema', async () => {
  const {
    GPURasterExtremaPyramid,
    getGPURasterExtremaPyramidLayout,
    GPU_RASTER_EXTREMA_PYRAMID_EMPTY_MAXIMUM,
    GPU_RASTER_EXTREMA_PYRAMID_EMPTY_MINIMUM
  } = await import('../../../src/raster-analysis/raster-pyramid/gpu-raster-extrema-pyramid');
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = {
    width: 257,
    height: 9,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const layout = getGPURasterExtremaPyramidLayout(grid, {firstBlockSize: 1, footprint: 'cell'});
  const samples = Float32Array.from(
    {length: grid.width * grid.height},
    (_, cell) => ((cell * 17) % 101) - 50
  );
  const validityValues = Uint32Array.from(samples, (_, cell) => Number(cell % 13 !== 0));
  const values = createInputBuffer(device, samples);
  const validity = createInputBuffer(device, validityValues);
  const combined = createOutputBuffer(device, layout.length * 2);
  const graph = new GPUCommandGraph(device);
  graph.add(
    new GPURasterExtremaPyramid({
      grid,
      firstBlockSize: 1,
      footprint: 'cell',
      input: {
        id: 'source',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'values', values, 'float32', samples.length)
        },
        validity: importGraphBuffer(graph, 'validity', validity, 'uint32', samples.length)
      },
      output: {
        combined: importGraphBuffer(graph, 'combined', combined, 'float32', layout.length * 2)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const actual = await readFloat32(combined, layout.length * 2);
    for (const level of layout.levels) {
      for (let row = 0; row < level.height; row++) {
        for (let column = 0; column < level.width; column++) {
          let maximum = GPU_RASTER_EXTREMA_PYRAMID_EMPTY_MAXIMUM;
          let minimum = GPU_RASTER_EXTREMA_PYRAMID_EMPTY_MINIMUM;
          for (
            let sourceRow = row * level.blockSize;
            sourceRow < Math.min(grid.height, (row + 1) * level.blockSize);
            sourceRow++
          ) {
            for (
              let sourceColumn = column * level.blockSize;
              sourceColumn < Math.min(grid.width, (column + 1) * level.blockSize);
              sourceColumn++
            ) {
              const cell = sourceRow * grid.width + sourceColumn;
              if (validityValues[cell]) {
                maximum = Math.max(maximum, samples[cell]);
                minimum = Math.min(minimum, samples[cell]);
              }
            }
          }
          const index = level.offset + row * level.width + column;
          expect(actual[index]).toBe(Math.fround(maximum));
          expect(actual[layout.length + index]).toBe(Math.fround(minimum));
        }
      }
    }
  } finally {
    compiled.destroy();
    for (const buffer of [values, validity, combined]) buffer.destroy();
  }
});
