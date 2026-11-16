// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('equalizes a raster and reflected population beyond the old 2^24 envelopes', async () => {
  const {GPURasterContrastEqualization} = await import(
    '../../../src/raster-analysis/raster-contrast-equalization/gpu-raster-contrast-equalization'
  );
  const {getGPURasterContrastEqualizationParameterValues} = await import(
    '../../../src/raster-analysis/raster-contrast-equalization/raster-contrast-equalization-parameters'
  );
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 4097;
  const height = 4096;
  const length = width * height;
  const graph = new GPUCommandGraph(device);
  // WebGPU initializes the input to zero: no cell-proportional CPU fixture/readback arrays.
  const source = createOutputBuffer(device, length);
  const values = createOutputBuffer(device, length);
  const validity = createOutputBuffer(device, length);
  const approximation = createOutputBuffer(device, 1);
  const parameters = createInputBuffer(
    device,
    getGPURasterContrastEqualizationParameterValues({clipLimit: 0})
  );
  let compiled: ReturnType<typeof graph.compile> | undefined;
  try {
    graph.add(
      new GPURasterContrastEqualization({
        grid: {width, height, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
        input: {
          id: 'constant',
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: importGraphBuffer(graph, 'constant', source, 'float32', length)
          }
        },
        tileWidth: 256,
        tileHeight: 256,
        binCount: 2,
        parameters: importGraphBuffer(graph, 'parameters', parameters, 'float32', 8),
        output: {
          values: importGraphBuffer(graph, 'values', values, 'float32', length),
          validity: importGraphBuffer(graph, 'validity', validity, 'uint32', length),
          status: {
            approximation: importGraphBuffer(graph, 'approximation', approximation, 'uint32', 1)
          }
        }
      })
    );
    compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    // Every reflected tile is constant: its occupied bin's CDF is exactly one, independently
    // of tile shape and interpolation. Include row/tile boundaries and the last reflected tile.
    for (const cell of [0, 255, 256, width - 1, width, Math.floor(length / 2), length - 1]) {
      const bytes = await values.readAsync(cell * 4, 4);
      expect(new Float32Array(bytes.buffer, bytes.byteOffset, 1)[0]).toBe(1);
      const flags = await validity.readAsync(cell * 4, 4);
      expect(new Uint32Array(flags.buffer, flags.byteOffset, 1)[0]).toBe(1);
    }
    expect(await readUint32(approximation, 1)).toEqual([1]);
  } finally {
    compiled?.destroy();
    for (const buffer of [source, values, validity, approximation, parameters]) buffer.destroy();
  }
}, 120000);
