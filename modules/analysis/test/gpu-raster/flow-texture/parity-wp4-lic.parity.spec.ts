// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {decodeSavedLICSettings} from './saved-flow-parameters';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {
  GPULineIntegralConvolution,
  getGPULineIntegralConvolutionParameterValues,
  getGPULineIntegralConvolutionUint32ParameterValues
} from '../../../src/raster-analysis/flow-texture/index';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {licCases} from '../../oracles/parity-wp4-round2-fixtures';
import {compareRasterParity} from '../../oracles/parity-wp4-compare';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const scene of licCases) {
  it(`GPULineIntegralConvolution independent parity: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device, {id: `parity-wp4-${scene.name}`});
    const buffers: Buffer[] = [];
    const trackBuffer = (buffer: Buffer) => {
      buffers.push(buffer);
      return buffer;
    };
    const length = scene.width * scene.height;
    const valuesBuffer = trackBuffer(createOutputBuffer(device, length));
    const speedsBuffer = trackBuffer(createOutputBuffer(device, length));
    const parameters = getGPULineIntegralConvolutionParameterValues(
      decodeSavedLICSettings(scene.settings)
    );
    const words = getGPULineIntegralConvolutionUint32ParameterValues(scene.settings);
    const fieldBuffer = trackBuffer(createInputBuffer(device, new Float32Array(scene.field)));
    graph.add(
      new GPULineIntegralConvolution({
        velocities: importGraphBuffer(
          graph,
          'field',
          fieldBuffer,
          'float32x2',
          scene.fieldWidth * scene.fieldHeight
        ),
        fieldWidth: scene.fieldWidth,
        fieldHeight: scene.fieldHeight,
        grid: makeGPURasterMetadata({width: scene.width, height: scene.height}),
        stepCount: scene.stepCount,
        parameters: importGraphBuffer(
          graph,
          'parameters',
          trackBuffer(createInputBuffer(device, parameters)),
          'float32',
          parameters.length
        ),
        uint32Parameters: importGraphBuffer(
          graph,
          'words',
          trackBuffer(createInputBuffer(device, words)),
          'uint32',
          words.length
        ),
        output: {
          values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', length),
          speeds: importGraphBuffer(graph, 'speeds', speedsBuffer, 'float32', length)
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      compareRasterParity(
        'GPULineIntegralConvolution',
        scene.name,
        scene.reference,
        'all-lic-values-and-invalid-mask',
        await readFloat32(valuesBuffer, length),
        scene.values,
        'L'
      );
      compareRasterParity(
        'GPULineIntegralConvolution',
        scene.name,
        scene.reference,
        'all-centre-speeds-and-invalid-mask',
        await readFloat32(speedsBuffer, length),
        scene.speeds,
        'L'
      );
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}
