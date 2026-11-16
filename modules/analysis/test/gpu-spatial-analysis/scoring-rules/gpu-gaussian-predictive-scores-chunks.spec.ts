// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUGaussianPredictiveScores} from '../../../src/gpu-spatial-analysis/scoring-rules/gpu-gaussian-predictive-scores';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('Gaussian scores match packed rows exactly and independent closed-form CRPS values', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const inputs = {
    means: new Float32Array([0, 1, 2, 3, 4, 5, 6]),
    variances: new Float32Array([0, 1, 4, 9, 1, -1, NaN]),
    observations: new Float32Array([2, 1, 2, 3, 6, 9, 8])
  };
  const outputs: {buffer: Buffer; prefix: number; length: number}[][] = [];
  for (const [routeIndex, lengths] of [[7], [1, 4, 2]].entries()) {
    const outputBuffers: {buffer: Buffer; prefix: number; length: number}[] = [];
    const rows = (name: string, values?: Float32Array) => {
      let rowBase = 0;
      const chunks = lengths.map((length, chunkIndex) => {
        const prefix = chunkIndex + (values ? 1 : 2);
        const buffer = values
          ? (() => {
              const padded = new Float32Array(prefix + length);
              padded.set(values.subarray(rowBase, rowBase + length), prefix);
              return createInputBuffer(device, padded);
            })()
          : createOutputBuffer(device, prefix + length);
        buffers.push(buffer);
        const backing = importGraphBuffer(
          graph,
          `${name}-${routeIndex}-${chunkIndex}`,
          buffer,
          'float32',
          prefix + length
        );
        rowBase += length;
        if (!values) outputBuffers.push({buffer, prefix, length});
        return graph.createDataView(backing.buffer, {
          format: 'float32',
          length,
          byteOffset: prefix * 4
        });
      });
      return routeIndex === 0 ? chunks[0] : createVectorView(name, 'float32', chunks);
    };
    graph.add(
      new GPUGaussianPredictiveScores({
        id: `scores-${routeIndex}`,
        means: rows('means', inputs.means),
        variances: rows('variances', inputs.variances),
        observations: rows('observations', inputs.observations),
        output: {crps: rows('crps')}
      })
    );
    outputs.push(outputBuffers);
  }
  const compiled = graph.compile();
  try {
    for (let encoding = 0; encoding < 2; encoding++) {
      submitGraph(device, compiled, undefined);
      const results: number[][] = [];
      for (const chunks of outputs) {
        const values = [];
        for (const {buffer, prefix, length} of chunks)
          values.push(...(await readFloat32(buffer, prefix + length)).slice(prefix));
        results.push(values);
      }
      expect(results[1]).toEqual(results[0]);
      // At zero error, CRPS = sd * (sqrt(2) - 1) / sqrt(pi).
      // At zero variance it is absolute error. z=2 uses erf(sqrt(2)) in float64.
      const centered = (Math.sqrt(2) - 1) / Math.sqrt(Math.PI);
      const expected = [2, centered, 2 * centered, 3 * centered, 1.452791821685903, NaN, NaN];
      for (let row = 0; row < 7; row++) {
        if (Number.isNaN(expected[row])) expect(results[0][row]).toBeNaN();
        else expect(Math.abs(results[0][row] - expected[row])).toBeLessThan(2e-6);
      }
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
