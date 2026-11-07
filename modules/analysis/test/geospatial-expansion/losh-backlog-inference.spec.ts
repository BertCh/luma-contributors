// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPULOSHResidualPermutationTest} from '../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {getGPUPermutationParameterValues} from '../../src/gpu-spatial-analysis/permutation-inference/index';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import reference from '../oracles/fixtures/losh-backlog-inference.json';

async function harness() {
  const candidateDevice = await getWebGPUTestDevice();
  requireTestDevice(candidateDevice);
  const device = candidateDevice;
  const graph = new GPUCommandGraph(device);
  let next = 0;
  const buffers: ReturnType<typeof createOutputBuffer>[] = [];
  function input(
    values: number[],
    format: 'uint32'
  ): ReturnType<typeof graph.createDataView<'uint32'>>;
  function input(
    values: number[],
    format: 'float32'
  ): ReturnType<typeof graph.createDataView<'float32'>>;
  function input(values: number[], format: 'uint32' | 'float32') {
    const buffer = createInputBuffer(
      device,
      format === 'float32' ? Float32Array.from(values) : Uint32Array.from(values)
    );
    buffers.push(buffer);
    return importGraphBuffer(graph, `input-${next++}`, buffer, format, values.length);
  }
  function output<Format extends 'uint32' | 'float32' | 'uint32x2'>(
    length: number,
    format: Format
  ) {
    const buffer = createOutputBuffer(device, length * (format === 'uint32x2' ? 2 : 1));
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, `output-${next++}`, buffer, format, length)};
  }
  return {
    device,
    graph,
    input,
    output,
    run() {
      const compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      return () => {
        compiled.destroy();
        buffers.forEach(buffer => buffer.destroy());
      };
    }
  };
}

for (const alternative of ['greater', 'lesser', 'two-sided'] as const) {
  it(`LOSH residual ${alternative} tail agrees with independent exhaustive null and Bonferroni`, async () => {
    const setup = await harness();
    const {graph, input, output} = setup;
    const rows = reference.values.length;
    const statistic = output(rows, 'float32');
    const normalizedResiduals = output(rows, 'float32');
    const status = output(rows, 'uint32');
    const pseudoPValues = output(rows, 'float32');
    const adjustedPValues = output(rows, 'float32');
    const exceedances = output(rows, 'uint32');
    const overflow = output(1, 'uint32');
    graph.add(
      new GPULOSHResidualPermutationTest({
        values: input(reference.values, 'float32'),
        weights: {
          offsets: input([0, 2, 4, 6, 8, 10], 'uint32'),
          neighbors: input(reference.neighbors.flat(), 'uint32'),
          weights: input(Array(10).fill(1), 'float32')
        },
        parameters: input(
          Array.from(getGPUPermutationParameterValues({seed: 417, permutationCount: 4095})),
          'uint32'
        ),
        maximumPermutations: 4095,
        alternative,
        multipleTesting: 'bonferroni',
        output: {
          statistic: statistic.view,
          status: status.view,
          normalizedResiduals: normalizedResiduals.view,
          inference: {
            pseudoPValues: pseudoPValues.view,
            adjustedPValues: adjustedPValues.view,
            exceedances: exceedances.view,
            status: {overflow: overflow.view}
          }
        }
      })
    );
    const cleanup = setup.run();
    try {
      expect(Array.from(await readUint32(overflow.buffer))).toEqual([0]);
      expect(Array.from(await readUint32(status.buffer))).toEqual(Array(rows).fill(0));
      const actualStatistic = await readFloat32(statistic.buffer);
      const actualResiduals = await readFloat32(normalizedResiduals.buffer);
      const actualPValues = await readFloat32(pseudoPValues.buffer);
      const actualAdjusted = await readFloat32(adjustedPValues.buffer);
      for (let row = 0; row < rows; row++) {
        expect(actualStatistic[row]).toBeCloseTo(reference.statistic[row], 5);
        expect(actualResiduals[row]).toBeCloseTo(reference.normalizedResiduals[row], 5);
        // Five standard deviations for two doubled-tail Monte Carlo estimates at P=4095.
        expect(
          Math.abs(actualPValues[row] - reference.probabilities[row][alternative])
        ).toBeLessThan(0.08);
        expect(actualAdjusted[row]).toBeCloseTo(Math.min(1, rows * actualPValues[row]), 6);
      }
    } finally {
      cleanup();
    }
  });
}

for (const failure of ['constant', 'nonfinite', 'upstream', 'count', 'capacity'] as const) {
  it(`LOSH residual inference handles ${failure} without plausible p-values`, async () => {
    const setup = await harness();
    const {graph, input, output} = setup;
    const statistic = output(5, 'float32');
    const status = output(5, 'uint32');
    const pseudoPValues = output(5, 'float32');
    const exceedances = output(5, 'uint32');
    const overflow = output(1, 'uint32');
    const parameters = Array.from(getGPUPermutationParameterValues({seed: 1, permutationCount: 9}));
    if (failure === 'count') parameters[2] = 0;
    graph.add(
      new GPULOSHResidualPermutationTest({
        values: input(
          failure === 'constant'
            ? Array(5).fill(1)
            : failure === 'nonfinite'
              ? Array(5).fill(NaN)
              : reference.values,
          'float32'
        ),
        weights: {
          offsets: input([0, 2, 4, 6, 8, 10], 'uint32'),
          neighbors: input(reference.neighbors.flat(), 'uint32'),
          weights: input(Array(10).fill(1), 'float32'),
          status: {overflow: input([failure === 'upstream' ? 1 : 0], 'uint32')}
        },
        parameters: input(parameters, 'uint32'),
        maximumPermutations: 9,
        maximumNeighbors: failure === 'capacity' ? 1 : 2,
        alternative: 'greater',
        output: {
          statistic: statistic.view,
          status: status.view,
          inference: {
            pseudoPValues: pseudoPValues.view,
            exceedances: exceedances.view,
            status: {overflow: overflow.view}
          }
        }
      })
    );
    const cleanup = setup.run();
    try {
      for (const value of await readFloat32(pseudoPValues.buffer)) expect(value).toBeNaN();
      for (const value of await readUint32(exceedances.buffer)) expect(value).toBe(0xffffffff);
      expect((await readUint32(overflow.buffer))[0]).toBe(
        ['upstream', 'count', 'capacity'].includes(failure) ? 1 : 0
      );
    } finally {
      cleanup();
    }
  });
}
