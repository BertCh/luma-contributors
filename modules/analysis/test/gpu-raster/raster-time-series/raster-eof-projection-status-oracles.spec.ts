// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPURasterEOFProjection} from '../../../src/raster-analysis/raster-time-series/gpu-raster-eof-projection';
import {getGPURasterSnapshotCovarianceParameterValues} from '../../../src/raster-analysis/raster-time-series/gpu-raster-snapshot-covariance';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('overwrites every poisoned EOF status word from independent cell eligibility', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const values = createInputBuffer(device, Float32Array.of(1, 3, 1, NaN, 1, 9, 1, 3, 1, 3, 1, 3));
  const weights = createInputBuffer(device, Float32Array.of(1, 1, 1, 0, 1, 0.5));
  const mask = createInputBuffer(device, Uint32Array.of(1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1));
  const parameters = createInputBuffer(
    device,
    getGPURasterSnapshotCovarianceParameterValues({maximumValueMagnitude: 8}, 6)
  );
  const temporalModes = createInputBuffer(device, Float32Array.of(-1, 1));
  const singularValues = createInputBuffer(device, Float32Array.of(1));
  const eofs = createOutputBuffer(device, 6);
  const status = createOutputBuffer(device, 6);
  status.write(new Uint32Array(6).fill(0xdeadbeef));
  graph.add(
    new GPURasterEOFProjection({
      cellCount: 6,
      sliceCount: 2,
      modeCount: 1,
      values: importGraphBuffer(graph, 'values', values, 'float32', 12),
      weights: importGraphBuffer(graph, 'weights', weights, 'float32', 6),
      mask: importGraphBuffer(graph, 'mask', mask, 'uint32', 12),
      parameters: importGraphBuffer(graph, 'parameters', parameters, 'float32', 2),
      temporalModes: importGraphBuffer(graph, 'modes', temporalModes, 'float32', 2),
      singularValues: importGraphBuffer(graph, 'singular', singularValues, 'float32', 1),
      output: {
        eofs: importGraphBuffer(graph, 'eofs', eofs, 'float32', 6),
        status: importGraphBuffer(graph, 'status', status, 'uint32', 6)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    // Cells 0/5 have complete bounded samples and weights in (0,1].
    // Cells 1/2/3/4 fail respectively NaN, |9| > 8, zero weight, and a masked sample.
    // Compare every complete word: filtering only status==1 leaves wrong success codes unseen.
    expect(await readUint32(status, 6)).toEqual([0, 1, 1, 1, 1, 0]);
  } finally {
    compiled.destroy();
    for (const buffer of [
      values,
      weights,
      mask,
      parameters,
      temporalModes,
      singularValues,
      eofs,
      status
    ])
      buffer.destroy();
  }
});
