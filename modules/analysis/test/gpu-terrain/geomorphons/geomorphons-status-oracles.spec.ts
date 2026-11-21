// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUGeomorphons,
  getGPUGeomorphonsParameterValues
} from '../../../src/gpu-terrain/geomorphons/gpu-geomorphons';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

it('geomorphon validity writes the entire border and clears every previously valid cell', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeGPURasterMetadata({width: 5, height: 5, cellSize: 1});
  const graph = new GPUCommandGraph(device);
  const elevationBuffer = createInputBuffer(device, new Float32Array(25));
  const validityBuffer = createInputBuffer(
    device,
    Uint32Array.from([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
  );
  const outputBuffer = createOutputBuffer(device, 25);
  outputBuffer.write(new Uint32Array(25).fill(17));
  const parameters = new GPUParameterBuffer(device, {
    id: 'geomorphon-validity-parameters',
    format: 'float32',
    length: 8,
    values: getGPUGeomorphonsParameterValues({}, grid)
  });
  graph.add(
    new GPUGeomorphons({
      grid,
      searchRadius: 2,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', 25)
        },
        validity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', 25)
      },
      parameters: parameters.importToGraph(graph),
      output: {
        outputValidity: importGraphBuffer(graph, 'output-validity', outputBuffer, 'uint32', 25)
      }
    })
  );
  const compiled = graph.compile();
  try {
    // By definition skipRadius=0 excludes the outermost ring; only the centre is masked.
    // All remaining finite cells are valid, including those next to the masked centre.
    submitGraph(device, compiled, undefined);
    expect(await readUint32(outputBuffer, 25)).toEqual([
      0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0
    ]);
    // A non-finite scale invalidates the whole operation. This must clear the eight ones.
    parameters.write(getGPUGeomorphonsParameterValues({zFactor: NaN}, grid));
    submitGraph(device, compiled, undefined);
    expect(await readUint32(outputBuffer, 25)).toEqual([
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]);
  } finally {
    compiled.destroy();
    parameters.destroy();
    for (const buffer of [elevationBuffer, validityBuffer, outputBuffer]) buffer.destroy();
  }
});
