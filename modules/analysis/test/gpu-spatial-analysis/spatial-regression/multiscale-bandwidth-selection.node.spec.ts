// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {computeMultiscaleGeographicallyWeightedRegressionOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-multiscale-geographically-weighted-regression-operation';
import {
  GPUMultiscaleGeographicallyWeightedRegression,
  type GPUMultiscaleGeographicallyWeightedRegressionProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression';
import {getDefaultGWRGridSize} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-grid-index';
import {selectGoldenSectionBandwidth} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-golden-section-nodes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {expectRegressionRowsClose} from './regression-operation-test-utils';
import fixtures from './multiscale-bandwidth-selection-fixtures.json';

for (const fixture of fixtures) {
  it(`matches NumPy ${fixture.kernel}/${fixture.criterion}/${fixture.searchMethod}/inference=${fixture.inference}`, () => {
    const result = computeMultiscaleGeographicallyWeightedRegressionOnCPU(
      {
        positions: Float32Array.from(fixture.positions),
        predictors: Float32Array.from(fixture.predictors),
        response: Float32Array.from(fixture.response),
        parameters: Float32Array.from(fixture.parameters),
        mask: Uint32Array.from(fixture.mask),
        bandwidthCandidates: Float32Array.from(fixture.bandwidthCandidates)
      },
      {
        predictorCount: 2,
        kernel: fixture.kernel as 'gaussian' | 'bisquare',
        bandwidthCandidateCount: fixture.bandwidthCandidateCount,
        bandwidthSelection: {
          criterion: fixture.criterion as 'AICc' | 'leave-one-out',
          searchMethod: fixture.searchMethod as 'finite-grid' | 'golden-section',
          maximumSearchIterations: fixture.searchMethod === 'golden-section' ? 12 : undefined
        },
        inference: fixture.inference ? {} : undefined
      }
    );
    expect(Array.from(result.codes)).toEqual(fixture.mask.map(keep => (keep ? 0 : 1)));
    for (const port of [
      'coefficients',
      'fitted',
      'selectedBandwidths',
      'iterationCount'
    ] as const) {
      expectRegressionRowsClose(
        result[port],
        Float32Array.from(fixture.expected[port], value => value ?? NaN),
        3e-5
      );
    }
    if (fixture.inference) {
      expectRegressionRowsClose(
        result.inference!.summary,
        fixture.expected.inferenceSummary!,
        3e-5
      );
      expectRegressionRowsClose(
        result.inference!.effectiveParameters,
        fixture.expected.inferenceEffectiveParameters!,
        3e-5
      );
      expectRegressionRowsClose(
        result.inference!.standardErrors,
        Float32Array.from(fixture.expected.inferenceStandardErrors!, value => value ?? NaN),
        3e-5
      );
      expect(result.inference!.summary[0]).toBeCloseTo(
        result.inference!.effectiveParameters.reduce((sum, value) => sum + value, 0),
        5
      );
    }
  });
}

function withContributor<T>(
  rowCount: number,
  overrides: Partial<GPUMultiscaleGeographicallyWeightedRegressionProps>,
  inspect: (contributor: GPUMultiscaleGeographicallyWeightedRegression, graph: GPUCommandGraph) => T
): T {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const vector = (name: string, length: number) =>
      createTransientView(graph, name, 'float32', length);
    const contributor = new GPUMultiscaleGeographicallyWeightedRegression({
      positions: createTransientView(graph, 'positions', 'float32x2', rowCount),
      predictors: vector('predictors', rowCount * 2),
      response: vector('response', rowCount),
      parameters: vector('parameters', 4),
      predictorCount: 2,
      kernel: 'bisquare',
      maximumIterations: 2,
      output: {
        coefficients: vector('coefficients', rowCount * 3),
        fitted: vector('fitted', rowCount),
        status: {
          codes: createTransientView(graph, 'codes', 'uint32', rowCount),
          iterationCount: createTransientView(graph, 'iterationCount', 'uint32', 1)
        }
      },
      ...overrides
    });
    return inspect(contributor, graph);
  } finally {
    device.destroy();
  }
}

it('shares the GWR default grid, honors overrides, and preserves Gaussian/great-circle fallbacks', () => {
  expect(getDefaultGWRGridSize(1023)).toBeUndefined();
  expect(getDefaultGWRGridSize(1024)).toEqual([12, 12]);
  expect(getDefaultGWRGridSize(2 ** 30)).toEqual([1024, 1024]);
  for (const rowCount of [1023, 1024, 70000])
    withContributor(rowCount, {}, contributor =>
      expect(contributor.props.indexGridSize).toEqual(getDefaultGWRGridSize(rowCount))
    );
  for (const overrides of [
    {indexGridSize: false as const},
    {kernel: 'gaussian' as const},
    {
      spatialContext: {
        coordinateSpace: 'longitude-latitude' as const,
        metric: 'great-circle' as const
      }
    }
  ])
    withContributor(1024, overrides, contributor =>
      expect(contributor.props.indexGridSize).toBeUndefined()
    );
  withContributor(1024, {kernel: 'gaussian', gaussianCutoff: 5}, contributor =>
    expect(contributor.props.indexGridSize).toEqual([12, 12])
  );
  withContributor(1024, {indexGridSize: [7, 9]}, contributor =>
    expect(contributor.props.indexGridSize).toEqual([7, 9])
  );
  expect(() => withContributor(70000, {indexGridSize: false}, () => {})).toThrow();
});

it('bounds actual command nodes for finite-grid and golden selection and rejects oversized search graphs', () => {
  for (const searchMethod of ['finite-grid', 'golden-section'] as const) {
    withContributor(1024, {}, (contributor, graph) => {
      const candidates = createTransientView(graph, 'candidates', 'float32', 6);
      const selected = new GPUMultiscaleGeographicallyWeightedRegression({
        ...contributor.props,
        bandwidthCandidates: candidates,
        bandwidthCandidateCount: 2,
        bandwidthSelection: {
          criterion: 'AICc',
          searchMethod,
          maximumSearchIterations: searchMethod === 'golden-section' ? 4 : undefined
        }
      });
      const nodes = selected.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(selected.estimatedCommandNodeCount);
      const sources = getGPUCommandNodeKernelSources(nodes);
      for (const source of sources) {
        try {
          expect(new WgslReflect(source.source).storage.length, source.id).toBeLessThanOrEqual(8);
        } catch (error) {
          throw new Error(
            `${source.id}: ${error}\n${source.source
              .split('\n')
              .map((line, index) => `${index + 1}: ${line}`)
              .join('\n')}`
          );
        }
      }
      const evaluation = sources.find(
        source => source.id.includes('-selection-') && source.id.endsWith('-evaluate')
      )!;
      expect(evaluation.source).toContain('getGridRange');
      expect(evaluation.source).toContain('value * value / denominator');
    });
  }
  withContributor(10, {}, (contributor, graph) => {
    const candidates = createTransientView(graph, 'candidates', 'float32', 6);
    for (const bandwidthSelection of [
      {criterion: 'invalid' as never},
      {searchMethod: 'invalid' as never},
      {searchMethod: 'golden-section' as const, maximumSearchIterations: 0},
      {searchMethod: 'golden-section' as const, maximumSearchIterations: 65},
      {maximumSearchIterations: 3}
    ])
      expect(
        () =>
          new GPUMultiscaleGeographicallyWeightedRegression({
            ...contributor.props,
            bandwidthCandidates: candidates,
            bandwidthCandidateCount: 2,
            bandwidthSelection
          })
      ).toThrow();
    expect(
      () =>
        new GPUMultiscaleGeographicallyWeightedRegression({
          ...contributor.props,
          maximumIterations: 128,
          bandwidthCandidates: candidates,
          bandwidthCandidateCount: 2,
          bandwidthSelection: {searchMethod: 'golden-section', maximumSearchIterations: 64}
        })
    ).toThrow(/command node count/);
    expect(
      () =>
        new GPUMultiscaleGeographicallyWeightedRegression({
          ...contributor.props,
          bandwidthSelection: {criterion: 'AICc'}
        })
    ).toThrow(/bandwidthCandidates/);
    expect(
      () =>
        new GPUMultiscaleGeographicallyWeightedRegression({
          ...contributor.props,
          bandwidthCandidates: candidates,
          bandwidthCandidateCount: 1,
          bandwidthSelection: {searchMethod: 'golden-section'}
        })
    ).toThrow(/must be 2/);
  });
});

it('the command-node allowance covers default indexing at the maximum row count', () => {
  withContributor(1048576, {maximumIterations: 1}, (contributor, graph) => {
    expect(contributor.getCommandNodes(graph).length).toBeLessThanOrEqual(
      contributor.estimatedCommandNodeCount
    );
  });
});

it('golden ties choose the lower interior, invalid bounds fail, and the iteration budget bounds evaluations', () => {
  let evaluations = 0;
  const selected = selectGoldenSectionBandwidth(
    1,
    9,
    () => {
      evaluations++;
      return 5;
    },
    12
  );
  expect(selected).toBe(1 + 0.38197 * 8);
  expect(evaluations).toBe(3);
  expect(selectGoldenSectionBandwidth(-1, 2, () => 1, 12)).toBeNaN();
  expect(selectGoldenSectionBandwidth(2, 1, () => 1, 12)).toBeNaN();
  expect(selectGoldenSectionBandwidth(1, 2, () => NaN, 12)).toBeNaN();
  evaluations = 0;
  selectGoldenSectionBandwidth(
    1,
    9,
    bandwidth => {
      evaluations++;
      return bandwidth;
    },
    3
  );
  expect(evaluations).toBe(5);
});

it('AICc rejects saturated scalar fits, handles exclusions and validates continuous bounds', () => {
  const count = 8;
  const input = {
    positions: Float32Array.from({length: count * 2}, (_, index) => (index % 2 ? 0 : 100 * index)),
    predictors: Float32Array.from({length: count}, (_, row) => (row % 2 ? -1 : 1)),
    response: Float32Array.of(1, 2, 4, 3, 2, 1, 5, 3),
    parameters: Float32Array.of(1e-3, 2, 2),
    bandwidthCandidates: Float32Array.of(2, 10, 2, 20)
  };
  const options = {
    predictorCount: 1,
    kernel: 'bisquare' as const,
    bandwidthCandidateCount: 2,
    bandwidthSelection: {criterion: 'AICc' as const}
  };
  const saturated = computeMultiscaleGeographicallyWeightedRegressionOnCPU(input, options);
  expect(Array.from(saturated.codes)).toEqual(new Array(count).fill(2));
  expect(saturated.selectedBandwidths.every(Number.isNaN)).toBe(true);
  const excluded = computeMultiscaleGeographicallyWeightedRegressionOnCPU(
    {...input, mask: new Uint32Array(count)},
    options
  );
  expect(Array.from(excluded.codes)).toEqual(new Array(count).fill(1));
  const invalid = computeMultiscaleGeographicallyWeightedRegressionOnCPU(
    {...input, bandwidthCandidates: Float32Array.of(-1, 10, 20, 2)},
    {...options, bandwidthSelection: {criterion: 'AICc', searchMethod: 'golden-section'}}
  );
  expect(Array.from(invalid.codes)).toEqual(new Array(count).fill(2));
  expect(invalid.selectedBandwidths.every(Number.isNaN)).toBe(true);
  for (const searchMethod of ['finite-grid', 'golden-section'] as const) {
    const tied = computeMultiscaleGeographicallyWeightedRegressionOnCPU(
      {...input, positions: new Float32Array(count * 2)},
      {...options, bandwidthSelection: {criterion: 'AICc', searchMethod}}
    );
    expect(Array.from(tied.selectedBandwidths)).toEqual(
      Array.from(
        Float32Array.from(
          searchMethod === 'finite-grid' ? [2, 2] : [2 + 0.38197 * 8, 2 + 0.38197 * 18]
        )
      )
    );
  }
});

it('divides the scalar coefficient before multiplying a large predictor in selection residuals', () => {
  const predictor = Float32Array.from({length: 8}, (_, row) => (row - 3.5) * 1e15);
  const response = Float32Array.from(
    predictor,
    (value, row) => value * 1e-5 + (row % 2 ? 1e9 : -1e9)
  );
  let numerator = 0;
  let denominator = 0;
  for (let row = 0; row < 8; row++) {
    numerator = Math.fround(numerator + Math.fround(predictor[row] * response[row]));
    denominator = Math.fround(denominator + Math.fround(predictor[row] * predictor[row]));
  }
  const value = predictor[0];
  const productFirst = Math.fround(
    response[0] - Math.fround(Math.fround(value * numerator) / denominator)
  );
  const quotientFirst = Math.fround(
    response[0] - Math.fround(value * Math.fround(numerator / denominator))
  );
  expect(Number.isFinite(productFirst)).toBe(false);
  expect(Number.isFinite(quotientFirst)).toBe(true);
  expect(quotientFirst).toBeCloseTo(-6.66667e8, -4);
  const reference = computeMultiscaleGeographicallyWeightedRegressionOnCPU(
    {
      positions: new Float32Array(16),
      predictors: predictor,
      response,
      parameters: Float32Array.of(1e-3, 2, 2),
      bandwidthCandidates: Float32Array.of(2, 10, 2, 10)
    },
    {
      predictorCount: 1,
      kernel: 'bisquare',
      bandwidthCandidateCount: 2,
      bandwidthSelection: {criterion: 'AICc'},
      inference: {}
    }
  );
  expect(Array.from(reference.codes)).toEqual(new Array(8).fill(0));
  expect(Array.from(reference.selectedBandwidths)).toEqual([2, 2]);
  expect(reference.inference!.status[0]).toBe(0);
  expect(reference.inference!.standardErrors.every(Number.isFinite)).toBe(true);
  withContributor(8, {}, (contributor, graph) => {
    const selected = new GPUMultiscaleGeographicallyWeightedRegression({
      ...contributor.props,
      bandwidthCandidates: createTransientView(graph, 'large-candidates', 'float32', 6),
      bandwidthCandidateCount: 2,
      bandwidthSelection: {criterion: 'AICc'}
    });
    const source = getGPUCommandNodeKernelSources(selected.getCommandNodes(graph)).find(entry =>
      entry.id.endsWith('-pass-0-evaluate')
    )!.source;
    expect(source).toContain('let coefficient = numerator / denominator;');
    expect(source).toContain(' - value * coefficient;');
  });
});

for (const searchMethod of ['finite-grid', 'golden-section'] as const) {
  it(`gates every pass-zero selection stage in 128 backfit sweeps: ${searchMethod}`, () => {
    withContributor(1024, {}, (contributor, graph) => {
      const selected = new GPUMultiscaleGeographicallyWeightedRegression({
        ...contributor.props,
        predictorCount: 1,
        predictors: createTransientView(graph, 'one-predictor', 'float32', 1024),
        parameters: createTransientView(graph, 'one-parameters', 'float32', 3),
        mask: createTransientView(graph, 'all-excluded', 'uint32', 1024),
        maximumIterations: 128,
        bandwidthCandidates: createTransientView(graph, 'gated-candidates', 'float32', 4),
        bandwidthCandidateCount: 2,
        bandwidthSelection: {
          criterion: 'AICc',
          searchMethod,
          maximumSearchIterations: searchMethod === 'golden-section' ? 12 : undefined
        },
        output: {
          ...contributor.props.output,
          coefficients: createTransientView(graph, 'one-coefficients', 'float32', 2048)
        }
      });
      const nodes = selected.getCommandNodes(graph);
      const passZero = nodes.filter(node =>
        /-selection-\d+-\d+-pass-0-(evaluate|tiles|merge|score)$/.test(node.id)
      );
      expect(passZero.length).toBe(128 * 2 * 4);
      for (const node of passZero)
        expect((node as {condition?: unknown}).condition, node.id).toMatchObject({
          source: 'gpu',
          mode: 'indirect'
        });
      expect(nodes.length).toBeLessThanOrEqual(selected.estimatedCommandNodeCount);
    });
  });
}

it('probes a feasible upper bound when both golden-section interiors have invalid AICc', () => {
  // At the two initial interiors (< 1.5), only the focal row contributes: trace = 8.
  // At 2.2, only immediate neighbours join. Endpoints have one neighbour; six rows have two.
  const adjacentWeight = (1 - (1.5 / 2.2) ** 2) ** 2;
  const endpointTrace = 2 / (1 + adjacentWeight) + 6 / (1 + 2 * adjacentWeight);
  expect(endpointTrace).toBeCloseTo(5.369835, 5);
  expect(8 - endpointTrace - 2).toBeGreaterThan(0);
  const positions = Float32Array.from({length: 16}, (_, index) =>
    index % 2 ? 0 : (1.5 * index) / 2
  );
  const input = {
    positions,
    predictors: Float32Array.of(1, -1, 1, -1, 1, -1, 1, -1),
    response: Float32Array.of(1, 2, 4, 3, 2, 1, 5, 3),
    parameters: Float32Array.of(1e-3, 2.2, 2.2),
    bandwidthCandidates: Float32Array.of(0.1, 2.2, 0.1, 2.2)
  };
  const result = computeMultiscaleGeographicallyWeightedRegressionOnCPU(input, {
    predictorCount: 1,
    kernel: 'bisquare',
    bandwidthCandidateCount: 2,
    bandwidthSelection: {
      criterion: 'AICc',
      searchMethod: 'golden-section',
      maximumSearchIterations: 12
    }
  });
  expect(Array.from(result.codes)).toEqual(new Array(8).fill(0));
  expect(result.coefficients.every(Number.isFinite)).toBe(true);
  expect(result.fitted.every(Number.isFinite)).toBe(true);
  for (const bandwidth of result.selectedBandwidths) {
    expect(bandwidth).toBeGreaterThan(1.5);
    expect(bandwidth).toBeLessThanOrEqual(Math.fround(2.2));
  }
  expect(
    selectGoldenSectionBandwidth(
      0.1,
      2.2,
      bandwidth => (bandwidth <= 1.5 ? NaN : 2.2 - bandwidth),
      12
    )
  ).toBe(2.2);
});

it('AICc scales residual squares before summing when RSS overflows but variance is finite', () => {
  const response = Float32Array.from([1, -1, -1, 1, 1, -1, -1, 1], sign => sign * 1e19);
  const predictors = Float32Array.of(1, -1, 1, -1, 1, -1, 1, -1);
  let unscaledTotal = 0;
  let scaledTotal = 0;
  for (const residual of response) {
    const square = Math.fround(residual * residual);
    expect(Number.isFinite(square)).toBe(true);
    unscaledTotal = Math.fround(unscaledTotal + square);
    const scaledResidual = Math.fround(residual * 0.25);
    scaledTotal = Math.fround(scaledTotal + Math.fround(scaledResidual * scaledResidual));
  }
  expect(unscaledTotal).toBe(Infinity);
  const variance = Math.fround(scaledTotal * (16 / 8));
  expect(Number.isFinite(variance)).toBe(true);
  // Coincident sites give each scalar smoother trace 1. Both partial-response fits are zero.
  const expectedScore = 8 * Math.log(variance) + 8 * Math.log(2 * Math.PI) + (8 * 9) / 5;
  expect(expectedScore).toBeCloseTo(729.089, 3);
  const reference = computeMultiscaleGeographicallyWeightedRegressionOnCPU(
    {
      positions: new Float32Array(16),
      predictors,
      response,
      parameters: Float32Array.of(1e-3, 2, 2),
      bandwidthCandidates: Float32Array.of(2, 10, 2, 10)
    },
    {
      predictorCount: 1,
      kernel: 'bisquare',
      bandwidthCandidateCount: 2,
      bandwidthSelection: {criterion: 'AICc'}
    }
  );
  expect(Array.from(reference.codes)).toEqual(new Array(8).fill(0));
  expect(Array.from(reference.selectedBandwidths)).toEqual([2, 2]);
  expect(reference.coefficients.every(value => value === 0)).toBe(true);
  withContributor(8, {}, (contributor, graph) => {
    const selected = new GPUMultiscaleGeographicallyWeightedRegression({
      ...contributor.props,
      bandwidthCandidates: createTransientView(graph, 'scaled-candidates', 'float32', 6),
      bandwidthCandidateCount: 2,
      bandwidthSelection: {criterion: 'AICc'}
    });
    const sources = getGPUCommandNodeKernelSources(selected.getCommandNodes(graph));
    const evaluate = sources.find(source => source.id.endsWith('-pass-0-evaluate'))!.source;
    const score = sources.find(source => source.id.endsWith('-pass-0-score'))!.source;
    expect(evaluate).toContain('let scaledResidual = residual * f32(0.25);');
    expect(evaluate).toContain('scores[slot] = scaledResidual * scaledResidual;');
    expect(score).toContain('let variance = residualSquares * (f32(16) / count);');
  });
});
