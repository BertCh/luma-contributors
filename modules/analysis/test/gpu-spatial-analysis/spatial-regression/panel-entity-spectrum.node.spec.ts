// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeSpatialPanelErrorMaximumLikelihoodOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-error-maximum-likelihood-operation';

for (const periodCount of [8, 513]) {
  it(`SSR-118 repeats the entity spectrum through all likelihood derivatives T=${periodCount}`, () => {
    const rows = 2 * periodCount;
    const predictors = Float32Array.from({length: rows}, (_, row) => ((7 * row) % 19) - 9);
    const input = {
      predictors,
      response: predictors.map((value, row) => 1 + 2 * value + ((row * row + 3 * row) % 11) - 5),
      weights: {
        offsets: Uint32Array.of(0, 1, 2),
        neighbors: Uint32Array.of(1, 0),
        weights: Float32Array.of(1, 1)
      }
    };
    // The swap matrix has eigenvalues -1,+1; I_T tensor W repeats each T times.
    const options = {
      periodCount,
      predictorCount: 1,
      maximumPower: 8,
      logDeterminant: 'exact' as const
    };
    const short = computeSpatialPanelErrorMaximumLikelihoodOnCPU(
      {...input, eigenvalues: Float32Array.of(-1, 1)},
      options
    );
    const expanded = computeSpatialPanelErrorMaximumLikelihoodOnCPU(
      {
        ...input,
        eigenvalues: Float32Array.from({length: rows}, (_, index) => (index % 2 ? 1 : -1))
      },
      options
    );
    expect(Array.from(short.status)).toEqual([0]);
    // Independent NumPy least squares of filtered within data and SciPy bounded optimization;
    // logL=-NT/2*(log(2*pi*RSS/NT)+1)+T*log(1-lambda^2), scratch entity.py.
    const expected =
      periodCount === 8 ? [2.18674899, -0.24101298659907666] : [2.00667205, -0.04408877896547473];
    expect(short.table[4]).toBeCloseTo(expected[0], 5);
    expect(short.table[8]).toBeCloseTo(expected[1], 5);
    for (const index of [4, 5, 8, 9]) {
      expect(Number.isFinite(short.table[index])).toBe(true);
      expect(short.table[index]).toBeCloseTo(expanded.table[index], 5);
    }
    expect(() =>
      computeSpatialPanelErrorMaximumLikelihoodOnCPU(
        {...input, eigenvalues: Float32Array.of(-1, 0, 1)},
        options
      )
    ).toThrow(/eigenvalues/);
  });
}

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPUSpatialPanelErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

it('SSR-118 nested GPU graph bounds the representation and repeats determinant and both traces', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `spectrum-${serial++}`, format, length);
    const props = {
      periodCount: 513,
      predictorCount: 1,
      logDeterminant: 'exact' as const,
      eigenvalues: view('float32', 2),
      predictors: view('float32', 1026),
      response: view('float32', 1026),
      weights: {
        offsets: view('uint32', 3),
        neighbors: view('uint32', 2),
        weights: view('float32', 2)
      },
      output: {
        table: view('float32', 12),
        summary: view('float32', 22),
        status: {code: view('uint32', 1)}
      }
    };
    const nodes = new GPUSpatialPanelErrorMaximumLikelihood(props).getCommandNodes(graph);
    const search =
      getWGSLKernelNodeRecord(
        graph,
        nodes.find(node => node.id.endsWith('-error-lambda-search'))?.id ?? ''
      ) ??
      nodes
        .map(node => getWGSLKernelNodeRecord(graph, node.id))
        .find(record => record?.source.includes('fn getTraceTwo'))!;
    expect(search.source.match(/return total \* 513\.0;/g)).toHaveLength(3);
    const scalar = nodes
      .map(node => getWGSLKernelNodeRecord(graph, node.id))
      .find(record => record?.source.includes('var scalarRow'))!;
    expect(scalar.source).toContain('eigenvaluesOffset + index % 2u');
    expect(
      () =>
        new GPUSpatialPanelErrorMaximumLikelihood({...props, eigenvalues: view('float32', 1026)})
    ).toThrow(/1024/);
    expect(
      () => new GPUSpatialPanelErrorMaximumLikelihood({...props, eigenvalues: view('float32', 3)})
    ).toThrow(/eigenvalues|1024/);
  } finally {
    device.destroy();
  }
});

it('SSR-118 repeated scalar spectra retain the certified flat-profile convention', () => {
  const predictors = Float32Array.from({length: 16}, (_, row) => ((7 * row) % 19) - 9);
  const result = computeSpatialPanelErrorMaximumLikelihoodOnCPU(
    {
      predictors,
      response: predictors.map((value, row) => 1 + 2 * value + ((row * row + 3 * row) % 11) - 5),
      weights: {
        offsets: Uint32Array.of(0, 1, 2),
        neighbors: Uint32Array.of(0, 1),
        weights: Float32Array.of(0.25, 0.25)
      },
      eigenvalues: Float32Array.of(0.25, 0.25)
    },
    {periodCount: 8, predictorCount: 1, maximumPower: 8, logDeterminant: 'exact'}
  );
  // W=.25 I: every profile has the same likelihood. Select the lower bound -.99/.25;
  // log|I-lambda W|=16*log(1.99), with no identifiable lambda standard error.
  expect(Array.from(result.status)).toEqual([0]);
  expect(result.table[8]).toBeCloseTo(-3.96, 5);
  expect(result.table[9]).toBeNaN();
  expect(result.summary[19]).toBe(0);
  expect(result.summary[20]).toBe(1);
  expect(result.summary[21]).toBeCloseTo(16 * Math.log(1.99), 5);
});
