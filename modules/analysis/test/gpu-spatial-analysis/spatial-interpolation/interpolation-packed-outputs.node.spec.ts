// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterAll, expect, it} from 'vitest';
import {GPUKriging} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-kriging';
import {GPUFocalStatistics} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-focal-statistics';
import {GPULocalRBF} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf';
import {GPULocalKriging} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {GPUConditionalGaussianSimulation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation';
import {GPUVecchiaLogLikelihood} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-log-likelihood';
import {
  GPUVecchiaSpaceTimeLogLikelihood,
  GPU_VECCHIA_SPACE_TIME_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-space-time-log-likelihood';
import {GPUVecchiaFisherScoring} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-fisher-scoring';
import {GPU_VECCHIA_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-parameters';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const device = createNullWebGPUDevice();
afterAll(() => device.destroy());

function createFixture() {
  const graph = new GPUCommandGraph(device);
  const backing = createTransientView(graph, 'packed-outputs', 'uint32', 512);
  const output = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    format: Format,
    length: number,
    byteOffset: number
  ) => graph.createDataView(backing.buffer, {format, length, byteOffset});
  const input = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  return {graph, output, input};
}

it('accepts aligned co-written RBF, kriging and cooperative simulation outputs in one buffer', () => {
  for (const mode of ['rbf', 'kriging', 'simulation'] as const) {
    const {graph, output, input} = createFixture();
    const positions = input('positions', 'float32x2', 3);
    const observations = input('observations', 'float32x4', 3);
    const queries = input('queries', 'float32x2', 1);
    if (mode === 'rbf') {
      expect(() =>
        graph.add(
          new GPULocalRBF({
            positions,
            queries,
            values: input('values', 'float32', 3),
            neighborCount: 3,
            output: {values: output('float32x4', 1, 0), status: output('uint32', 1, 256)}
          })
        )
      ).not.toThrow();
    } else if (mode === 'kriging') {
      expect(() =>
        graph.add(
          new GPULocalKriging({
            positions,
            observations,
            supports: queries,
            mode: 'ordinary',
            covariance: {model: 'exponential', sill: 1, range: 2},
            output: {values: output('float32x4', 1, 0), status: output('uint32', 1, 256)}
          })
        )
      ).not.toThrow();
    } else {
      const contributor = new GPUConditionalGaussianSimulation({
        positions,
        observations,
        queries,
        covariance: {model: 'exponential', sill: 1, range: 2},
        seed: 1,
        replicateCount: 1,
        output: {
          values: output('float32', 1, 0),
          moments: input('moments', 'float32x2', 1),
          status: output('uint32', 1, 512)
        }
      });
      try {
        expect(() => graph.add(contributor)).not.toThrow();
      } finally {
        contributor.destroy();
      }
    }
  }
});

it('accepts focal moments and extremes sharing aligned prefixes across separate passes', () => {
  const {graph, output, input} = createFixture();
  const props = {
    values: input('values', 'float32', 4),
    parameters: input('params', 'float32', 4),
    maximumRadius: 1,
    grid: makeGPURasterMetadata({width: 2, height: 2}),
    output: {mean: output('float32', 4, 0), min: output('float32', 4, 16)}
  };
  expect(() => graph.add(new GPUFocalStatistics(props))).not.toThrow();
  expect(
    () => new GPUFocalStatistics({...props, output: {...props.output, sum: props.output.min}})
  ).toThrow(/must not overlap/);
});

for (const mode of ['spatial', 'space-time', 'fisher'] as const) {
  it(`accepts packed ${mode} Vecchia outputs and separate-pass aligned prefixes`, () => {
    const {graph, output, input} = createFixture();
    const siteBacking = createTransientView(graph, 'site-outputs', 'uint32', 256);
    const siteOutput = <Format extends 'float32' | 'uint32'>(
      format: Format,
      length: number,
      byteOffset: number
    ) => graph.createDataView(siteBacking.buffer, {format, length, byteOffset});
    const parameters = input(
      'params',
      'float32',
      mode === 'space-time' ? GPU_VECCHIA_SPACE_TIME_PARAMETER_LENGTH : GPU_VECCHIA_PARAMETER_LENGTH
    );
    const props = {
      positions: input('positions', 'float32x2', 2),
      observations: input('observations', 'float32x4', 2),
      neighborhoods: input('neighbors', 'uint32', 2),
      neighborhoodOffsets: input('offsets', 'uint32', 3),
      neighborCount: 1,
      parameters,
      output: {
        conditionals: output('float32x4', 2, 0),
        conditionalStatus: output('uint32', 2, 256),
        siteTerms: siteOutput('float32', 2, 0),
        siteStatus: siteOutput('uint32', 2, 256),
        logLikelihood: output('float32', 2, 32),
        status: siteOutput('uint32', 3, 512)
      }
    };
    const contributor =
      mode === 'fisher'
        ? new GPUVecchiaFisherScoring({
            ...props,
            output: {
              siteTerms: siteOutput('float32', 20, 0),
              siteStatus: siteOutput('uint32', 2, 256),
              totals: output('float32', 20, 0),
              status: siteOutput('uint32', 3, 512)
            }
          })
        : mode === 'space-time'
          ? new GPUVecchiaSpaceTimeLogLikelihood({
              ...props,
              times: input('times', 'float32', 2),
              temporalContext: {
                unit: 'second',
                epoch: 0n,
                encoding: 'float32-relative',
                axis: {kind: 'irregular'}
              }
            })
          : new GPUVecchiaLogLikelihood(props);
    expect(() => graph.add(contributor)).not.toThrow();
  });
}

it('accepts aligned raster kriging outputs and adjacent separate-pass predictions in one buffer', () => {
  const {graph, output, input} = createFixture();
  expect(() =>
    graph.add(
      new GPUKriging({
        positions: input('positions', 'float32x2', 2),
        values: input('samples', 'float32', 2),
        parameters: input('params', 'float32', 16),
        grid: makeGPURasterMetadata({width: 2, height: 1}),
        indexGridSize: [2, 1],
        indexBounds: [-1, -1, 1, 1],
        output: {
          values: output('float32', 2, 0),
          variance: output('float32', 2, 256),
          crossValidationPrediction: output('float32', 2, 8),
          crossValidationVariance: output('float32', 2, 512),
          status: {
            invalidCount: output('uint32', 1, 768),
            uncertainCount: output('uint32', 1, 1024)
          }
        }
      })
    )
  ).not.toThrow();
});

it('rejects disjoint read/write storage views of one physical buffer in a dispatch', () => {
  const {output, input} = createFixture();
  const positions = input('positions', 'float32x2', 2);
  const observations = input('observations', 'float32x4', 2);
  const queries = input('queries', 'float32x2', 1);
  expect(
    () =>
      new GPUConditionalGaussianSimulation({
        positions,
        observations,
        queries,
        covariance: {model: 'exponential', sill: 1, range: 2},
        seed: 1,
        replicateCount: 1,
        output: {
          values: output('float32', 1, 0),
          moments: output('float32x2', 1, 256),
          status: output('uint32', 1, 512)
        }
      })
  ).toThrow(/storage-write.*storage-read.*physical buffer/);
});

it('permits retained blocked values/moments sharing a buffer across separate dispatches', () => {
  const {graph, output, input} = createFixture();
  const props = {
    positions: input('positions', 'float32x2', 1),
    observations: input('observations', 'float32x4', 1),
    queries: input('queries', 'float32x2', 129),
    covariance: {model: 'exponential' as const, sill: 1, range: 2},
    seed: 1,
    reusePreparation: true,
    replicateCount: 1,
    output: {
      values: output('float32', 129, 0),
      moments: output('float32x2', 129, 516),
      status: input('status', 'uint32', 1)
    }
  };
  const contributor = new GPUConditionalGaussianSimulation(props);
  try {
    expect(() => graph.add(contributor)).not.toThrow();
  } finally {
    contributor.destroy();
  }
  // The blocked finalization reads status and writes values, so even disjoint bytes are invalid.
  expect(
    () =>
      new GPUConditionalGaussianSimulation({
        ...props,
        output: {...props.output, status: output('uint32', 1, 1792)}
      })
  ).toThrow(/storage-write.*storage-read.*physical buffer/);
});

it('permits retained cooperative moments/values/status sharing co-written aligned ranges', () => {
  const {graph, output, input} = createFixture();
  const contributor = new GPUConditionalGaussianSimulation({
    positions: input('positions', 'float32x2', 2),
    observations: input('observations', 'float32x4', 2),
    queries: input('queries', 'float32x2', 1),
    covariance: {model: 'exponential', sill: 1, range: 2},
    seed: 1,
    reusePreparation: true,
    replicateCount: 1,
    output: {
      values: output('float32', 1, 0),
      moments: output('float32x2', 1, 256),
      status: output('uint32', 1, 512)
    }
  });
  try {
    expect(() => graph.add(contributor)).not.toThrow();
  } finally {
    contributor.destroy();
  }
});
