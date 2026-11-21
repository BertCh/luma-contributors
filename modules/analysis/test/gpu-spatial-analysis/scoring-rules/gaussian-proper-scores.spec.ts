// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUGaussianPredictiveScores} from '../../../src/gpu-spatial-analysis/scoring-rules/gpu-gaussian-predictive-scores';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  createGaussianProperScoresInput,
  GAUSSIAN_BRIER_TAIL_EXPECTED,
  GAUSSIAN_BRIER_TAIL_RELATIVE_TOLERANCE,
  GAUSSIAN_PROPER_SCORES_EXPECTED
} from './gaussian-proper-scores-fixture';

const SCORES = ['crps', 'logScore', 'brierScore', 'intervalScore'] as const;

it('GPU Brier scores stay finite when threshold standardization would overflow float32', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = GAUSSIAN_BRIER_TAIL_EXPECTED;
  expectRows(await runBrierScores(device, fixture), fixture.expected);
});

async function runBrierScores(
  device: Device,
  input: {means: readonly number[]; variances: readonly number[]; observations: readonly number[]}
) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const rows = (name: string, values?: readonly number[]) => {
    const buffer = values
      ? createInputBuffer(device, Float32Array.from(values))
      : createOutputBuffer(device, input.means.length);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, 'float32', input.means.length);
  };
  const crps = rows('crps');
  const brierScore = rows('brier');
  graph.add(
    new GPUGaussianPredictiveScores({
      means: rows('means', input.means),
      variances: rows('variances', input.variances),
      observations: rows('observations', input.observations),
      brierThreshold: 0,
      output: {crps, brierScore}
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return await readFloat32(buffers[1], input.means.length);
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}
function expectRows(actual: ArrayLike<number>, expected: readonly (number | null)[]) {
  expect(actual.length).toBe(expected.length);
  expected.forEach((value, row) => {
    if (value === null) expect(actual[row]).toBeNaN();
    else {
      expect(Number.isFinite(actual[row]), `row ${row}`).toBe(true);
      // Float32 erf/log/normalization and rounded interval multiplier, including penalties.
      expect(Math.abs(actual[row] - value), `row ${row}`).toBeLessThanOrEqual(
        8e-6 * Math.max(1, Math.abs(value))
      );
    }
  });
}

it('all Gaussian score ports match SciPy for packed and prefixed chunked rows, including rescaling', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scaled of [false, true]) {
    for (const chunked of [false, true]) {
      const input = createGaussianProperScoresInput(scaled);
      const expected = scaled
        ? GAUSSIAN_PROPER_SCORES_EXPECTED.scaled
        : GAUSSIAN_PROPER_SCORES_EXPECTED.base;
      const graph = new GPUCommandGraph(device);
      const buffers: Buffer[] = [];
      const outputChunks: Record<string, {buffer: Buffer; prefix: number; length: number}[]> = {};
      const rows = (name: string, values?: Float32Array) => {
        let base = 0;
        const lengths = chunked ? [3, 4, 5] : [12];
        const chunks = lengths.map((length, chunkIndex) => {
          const prefix = chunkIndex + 1;
          const padded = new Float32Array(prefix + length).fill(-987);
          if (values) padded.set(values.subarray(base, base + length), prefix);
          const buffer = values
            ? createInputBuffer(device, padded)
            : createOutputBuffer(device, prefix + length);
          if (!values) {
            buffer.write(padded);
            outputChunks[name] ??= [];
            outputChunks[name].push({buffer, prefix, length});
          }
          buffers.push(buffer);
          base += length;
          const backing = importGraphBuffer(
            graph,
            `${name}-${chunkIndex}`,
            buffer,
            'float32',
            prefix + length
          );
          return graph.createDataView(backing.buffer, {
            format: 'float32',
            length,
            byteOffset: prefix * 4
          });
        });
        return chunked ? createVectorView(name, 'float32', chunks) : chunks[0];
      };
      graph.add(
        new GPUGaussianPredictiveScores({
          means: rows('means', input.means),
          variances: rows('variances', input.variances),
          observations: rows('observations', input.observations),
          brierThreshold: scaled ? 18 : 1,
          intervalAlpha: 0.1,
          output: {
            crps: rows('crps'),
            logScore: rows('logScore'),
            brierScore: rows('brierScore'),
            intervalScore: rows('intervalScore')
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        for (const name of SCORES) {
          const actual: number[] = [];
          for (const {buffer, prefix, length} of outputChunks[name]) {
            const data = await readFloat32(buffer, prefix + length);
            expect(data.slice(0, prefix)).toEqual(new Array(prefix).fill(-987));
            actual.push(...data.slice(prefix));
          }
          expectRows(actual, expected[name]);
        }
      } finally {
        compiled.destroy();
        for (const buffer of buffers) buffer.destroy();
      }
    }
  }
}, 120000);

it('R7 GPU Brier scores retain representable tails at both signs of six and eight deviations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const means = [-6, 6, -8, 8];
  const actual = await runBrierScores(device, {
    means,
    observations: means,
    variances: [1, 1, 1, 1]
  });
  // SciPy norm.sf(abs(z))**2. Near z=8 the erfc exponent is about -33:
  // one f32 exponent ULP is 2^-18, contributing ~7.63e-6 relative error
  // after squaring. Budget two such ULPs plus argument/polynomial/exp
  // rounding (2e-5 total). This relative bound cannot admit a zero score.
  const expected = [
    9.733551813410284e-19, 9.733551813410284e-19, 3.8700350466643384e-31, 3.8700350466643384e-31
  ];
  for (const [row, value] of expected.entries()) {
    expect(Number.isFinite(actual[row])).toBe(true);
    expect(Math.abs(actual[row] / value - 1), `row ${row}`).toBeLessThanOrEqual(
      GAUSSIAN_BRIER_TAIL_RELATIVE_TOLERANCE
    );
  }
});
