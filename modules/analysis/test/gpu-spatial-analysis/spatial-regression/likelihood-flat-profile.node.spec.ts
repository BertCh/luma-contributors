// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterAll, expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUSpatialErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood';
import {GPUSpatialLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  computeSpatialErrorMaximumLikelihoodOnCPU,
  CPUSpatialErrorMaximumLikelihoodOperation
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-error-maximum-likelihood-operation';
import {hasSpatialErrorScalarWeights} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-error-scalar-profile';
import {
  fixtures,
  getFlatProfileInput,
  getFlatProfileProps
} from './likelihood-flat-profile-test-utils';
import {expectRegressionRowsClose} from './regression-operation-test-utils';

const maximumDifferences: Record<string, number> = {};
it('review 2 R6 rejects eigenvalues aliased by lag power-trace output', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `alias-${serial++}`, format, length);
    const eigenvalues = view('float32', 8);
    expect(
      () =>
        new GPUSpatialLagMaximumLikelihood({
          predictorCount: 1,
          maximumPower: 8,
          logDeterminant: 'exact',
          eigenvalues,
          weights: {
            offsets: view('uint32', 9),
            neighbors: view('uint32', 8),
            weights: view('float32', 8)
          },
          predictors: view('float32', 8),
          response: view('float32', 8),
          output: {
            table: view('float32', 12),
            summary: view('float32', 22),
            status: {code: view('uint32', 1)},
            powerTraces: eigenvalues,
            powerSums: view('float32', 8),
            infinityNorm: view('float32', 1)
          }
        })
    ).toThrow(/outputs must not share buffers with inputs/);
  } finally {
    device.destroy();
  }
});
afterAll(() =>
  console.info('flat-profile maximum differences from rounded dense literals', maximumDifferences)
);

for (const fixture of fixtures) {
  it(`flat error profile has no spatial inference: Q=${fixture.maximumPower}, c=${fixture.scalarWeight}, transformed=${fixture.transformed}`, () => {
    const actual = computeSpatialErrorMaximumLikelihoodOnCPU(
      {
        weights: {
          offsets: Uint32Array.from({length: 9}, (_, row) => row),
          neighbors: Uint32Array.from({length: 8}, (_, row) => row),
          weights: new Float32Array(8).fill(fixture.scalarWeight)
        },
        eigenvalues: new Float32Array(8).fill(fixture.scalarWeight),
        predictors: Float32Array.from(fixture.predictors),
        response: Float32Array.from(fixture.response)
      },
      {predictorCount: 1, maximumPower: fixture.maximumPower, logDeterminant: 'exact'}
    );
    expect(actual.table[9], `unidentifiable spatial SE=${actual.table[9]}`).toBeNaN();
    for (const port of Object.keys(fixture.expected) as (keyof typeof actual)[]) {
      maximumDifferences[port] = Math.max(
        maximumDifferences[port] ?? 0,
        ...fixture.expected[port].map((value, row) =>
          value === null ? 0 : Math.abs(actual[port][row] - Math.fround(value))
        )
      );
      expectRegressionRowsClose(
        actual[port],
        Float32Array.from(fixture.expected[port], value => value ?? NaN),
        3e-6
      );
    }
  });
  it(`CPU operation retains every flat-profile oracle port: Q=${fixture.maximumPower}, c=${fixture.scalarWeight}`, async () => {
    const input = getFlatProfileInput(fixture);
    const operation = new CPUSpatialErrorMaximumLikelihoodOperation(getFlatProfileProps(fixture));
    const actual = await operation.execute({
      inputs: Object.fromEntries(
        Object.entries(input).map(([port, value]) => [
          port,
          {
            format: value.format,
            length: value.values.length,
            chunks: [{values: value.values, length: value.values.length}],
            pack: () => value.values
          }
        ])
      )
    });
    for (const port of Object.keys(fixture.expected) as (keyof typeof fixture.expected)[]) {
      expectRegressionRowsClose(
        actual[port] as Float32Array,
        Float32Array.from(fixture.expected[port], value => value ?? NaN),
        3e-6
      );
    }
  });
}

it('certifies scalar weights structurally, retaining tiny off-diagonal links and ignoring sentinels', () => {
  const scalar = {
    offsets: Uint32Array.of(0, 1, 2),
    neighbors: Uint32Array.of(0, 1),
    weights: Float32Array.of(3, 3)
  };
  expect(hasSpatialErrorScalarWeights(scalar, 2, 3)).toBe(true);
  expect(hasSpatialErrorScalarWeights({...scalar, weights: Float32Array.of(3, 4)}, 2, 3)).toBe(
    false
  );
  // Both eigenvalues round to 3 in float32, but W is not 3I. No near-zero tolerance is valid.
  expect(
    hasSpatialErrorScalarWeights(
      {
        offsets: Uint32Array.of(0, 2, 3),
        neighbors: Uint32Array.of(0, 1, 1),
        weights: Float32Array.of(3, 2 ** -40, 3)
      },
      2,
      3
    )
  ).toBe(false);
  expect(
    hasSpatialErrorScalarWeights(
      {
        offsets: Uint32Array.of(0, 2, 3),
        neighbors: Uint32Array.of(0, 99, 1),
        weights: Float32Array.of(3, NaN, 3)
      },
      2,
      3
    )
  ).toBe(true);
  expect(hasSpatialErrorScalarWeights(scalar, 2, NaN)).toBe(false);
});

it('declares the exact-mode structural certificate for each chunk within its resource budget', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const Contributor of [GPUSpatialErrorMaximumLikelihood, GPUSpatialLagMaximumLikelihood])
      for (const exact of [false, true]) {
        const graph = new GPUCommandGraph(device);
        let serial = 0;
        const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
          createTransientView(graph, `view-${serial++}`, format, length);
        const contributor = new Contributor({
          id: 'flat-profile',
          predictorCount: 1,
          maximumPower: 3,
          logDeterminant: exact ? 'exact' : 'series',
          eigenvalues: exact ? view('float32', 8) : undefined,
          predictors: view('float32', 8),
          response: view('float32', 8),
          weights: {
            rowCount: 8,
            chunks: [0, 3].map((rowStart, chunkIndex) => {
              const rowCount = chunkIndex === 0 ? 3 : 5;
              return {
                rowStart,
                rowCount,
                weights: {
                  offsets: view('uint32', rowCount + 1),
                  neighbors: view('uint32', rowCount),
                  weights: view('float32', rowCount)
                }
              };
            })
          },
          output: {
            table: view('float32', 12),
            summary: view('float32', 22),
            status: {code: view('uint32', 1)}
          }
        });
        const nodes = contributor.getCommandNodes(graph);
        expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
        const kernels = getGPUCommandNodeKernelSources(nodes);
        const certificates = kernels.filter(kernel =>
          kernel.id.includes('scalar-profile-certificate')
        );
        expect(certificates.length).toBe(exact ? 2 : 0);
        if (exact) {
          expect(certificates[1].source).toContain('WEIGHTS_ROW_START');
          expect(certificates[1].source).toContain(
            `eigenvalues[eigenvaluesOffset + ${Contributor === GPUSpatialErrorMaximumLikelihood ? 'index % 8u' : 'index'}] == scalarWeight`
          );
        }
      }
  } finally {
    device.destroy();
  }
});
