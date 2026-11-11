// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeMultiscaleGeographicallyWeightedRegressionOnCPU,
  CPUMultiscaleGeographicallyWeightedRegressionOperation
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-multiscale-geographically-weighted-regression-operation';
import {
  GPUMultiscaleGeographicallyWeightedRegressionOperation,
  type MultiscaleGeographicallyWeightedRegressionOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression-operation';
import fixtures from './multiscale-geographically-weighted-regression-operation-fixtures.json';
import {
  constructRegressionOperationContributor,
  expectRegressionRowsClose
} from './regression-operation-test-utils';

function getInput(fixture = fixtures[0]) {
  return {
    positions: Float32Array.from(fixture.positions),
    predictors: Float32Array.from(fixture.predictors),
    response: Float32Array.from(fixture.response),
    parameters: Float32Array.from(fixture.parameters),
    mask: Uint32Array.from(fixture.mask),
    noiseVariance: Float32Array.from(fixture.noiseVariance)
  };
}

for (const fixture of fixtures) {
  it(`matches the independent NumPy simultaneous block solve and conditional variances: ${fixture.kernel}`, () => {
    const result = computeMultiscaleGeographicallyWeightedRegressionOnCPU(getInput(fixture), {
      predictorCount: fixture.predictorCount,
      kernel: fixture.kernel as 'gaussian' | 'bisquare',
      maximumIterations: 1
    });
    expect(Array.from(result.codes)).toEqual(fixture.mask.map(keep => (keep ? 0 : 1)));
    expect(result.iterationCount[0]).toBeGreaterThan(1);
    for (const port of ['coefficients', 'fitted', 'conditionalStandardErrors'] as const)
      expectRegressionRowsClose(
        result[port],
        Float32Array.from(fixture.expected[port], value => value ?? NaN),
        2e-6
      );
    expectRegressionRowsClose(result.selectedBandwidths, fixture.parameters.slice(1), 0);
  });
  for (const varianceConvention of ['residual-df', 'smoother-df'] as const) {
    it(`matches the independent coupled response Jacobian and full inference: ${fixture.kernel}/${varianceConvention}`, () => {
      const result = computeMultiscaleGeographicallyWeightedRegressionOnCPU(getInput(fixture), {
        predictorCount: 2,
        kernel: fixture.kernel as 'gaussian' | 'bisquare',
        inference: {varianceConvention}
      });
      expect(result.inference!.status[0]).toBe(0);
      expect(Array.from(result.codes)).toEqual(fixture.mask.map(keep => (keep ? 0 : 1)));
      for (const port of ['coefficients', 'fitted'] as const)
        expectRegressionRowsClose(
          result[port],
          Float32Array.from(fixture.expected[port], value => value ?? NaN),
          2e-6
        );
      for (const port of ['standardErrors', 'effectiveParameters', 'summary'] as const)
        expectRegressionRowsClose(
          result.inference![port],
          Float32Array.from(
            fixture.expected.inference[varianceConvention][port],
            value => value ?? NaN
          ),
          2e-6
        );
    });
  }
}

it('rejects unusable CV candidates and preserves strict bisquare support', () => {
  const input = getInput();
  const candidates = Float32Array.of(0.01, 3.5, 0.01, 2, 0.01, 2.75);
  const result = computeMultiscaleGeographicallyWeightedRegressionOnCPU(
    {...input, bandwidthCandidates: candidates},
    {predictorCount: 2, kernel: 'bisquare', bandwidthCandidateCount: 2}
  );
  expect(Array.from(result.selectedBandwidths)).toEqual([3.5, 2, 2.75]);
  expect(Array.from(result.codes)).toEqual(Array.from(input.mask, keep => (keep ? 0 : 1)));
  const exactBoundary = {
    positions: Float32Array.of(0, 0, 1, 0, 2, 0),
    predictors: Float32Array.of(1, -1, 1),
    response: Float32Array.of(2, 4, 8),
    parameters: Float32Array.of(1e-5, 1, 1)
  };
  const boundary = computeMultiscaleGeographicallyWeightedRegressionOnCPU(exactBoundary, {
    predictorCount: 1,
    kernel: 'bisquare'
  });
  // A bandwidth exactly equal to neighbor spacing excludes that neighbor: every fitted
  // response is its own observation after a complete sweep.
  expect(Array.from(boundary.codes)).toEqual([0, 0, 0]);
  expectRegressionRowsClose(boundary.fitted, exactBoundary.response, 0);
});

it('ranks distinct usable Gaussian bandwidths using the hand-derived four-point CV scores', () => {
  const input = {
    positions: Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0),
    predictors: Float32Array.of(1, -1, 1, -1),
    response: Float32Array.of(1, 2, 3, 4),
    parameters: Float32Array.of(2, 10, 10),
    bandwidthCandidates: Float32Array.of(10, 1, 10, 1),
    noiseVariance: Float32Array.of(1)
  };
  const result = computeMultiscaleGeographicallyWeightedRegressionOnCPU(input, {
    predictorCount: 1,
    kernel: 'gaussian',
    bandwidthCandidateCount: 2
  });
  // Global OLS is [2.5,-0.5]. Intercept partial responses are [1.5,1.5,3.5,3.5].
  // Writing t=exp(-1/(2*h^2)), its LOO score is
  // 8*(((t^4+t^9)/(t+t^4+t^9))^2 + ((t+t^4)/(2*t+t^4))^2):
  // 7.02840555 at h=10, 2.72422093 at h=1. Both candidates have four finite scores.
  const narrow = Math.exp(-0.5);
  const firstIntercept =
    1.5 + (2 * (narrow ** 4 + narrow ** 9)) / (1 + narrow + narrow ** 4 + narrow ** 9);
  const secondIntercept = 1.5 + (2 * (narrow + narrow ** 4)) / (1 + 2 * narrow + narrow ** 4);
  const firstPartial = firstIntercept - 1;
  const secondPartial = secondIntercept - 2;
  // Slope partial responses are [-a,-b,b,a]. Its LOO score divided by (a+b)^2 is
  // 2*(((t+t^4)/(t+t^4+t^9))^2 + ((t+t^4)/(2*t+t^4))^2):
  // 1.79262369 at h=10, 2.54682582 at h=1, so this component chooses the first candidate.
  const wide = Math.exp(-1 / 200);
  const firstSlope =
    (secondPartial * (wide + wide ** 4) - firstPartial * (1 + wide ** 9)) /
    (1 + wide + wide ** 4 + wide ** 9);
  const secondSlope =
    (secondPartial * (1 + wide) - firstPartial * (wide + wide ** 4)) / (1 + 2 * wide + wide ** 4);
  const intercepts = [firstIntercept, secondIntercept, 5 - secondIntercept, 5 - firstIntercept];
  const slopes = [firstSlope, secondSlope, secondSlope, firstSlope];
  expect(Array.from(result.selectedBandwidths)).toEqual([1, 10]);
  expect(Array.from(result.codes)).toEqual([0, 0, 0, 0]);
  // Tolerance 2 deliberately ends after the analytically derived first sweep; no GPU cap is used.
  expect(result.iterationCount[0]).toBe(1);
  expectRegressionRowsClose(
    result.coefficients,
    Float32Array.from(intercepts.flatMap((value, row) => [value, slopes[row]])),
    2e-7
  );
  expectRegressionRowsClose(
    result.fitted,
    Float32Array.from(intercepts, (value, row) => value + input.predictors[row] * slopes[row]),
    2e-7
  );
});

for (const kernel of ['gaussian', 'bisquare'] as const) {
  it(`observes first-candidate ties between distinct usable ${kernel} bandwidths`, () => {
    // Coincident positions give every pair weight 1 at either positive bandwidth. Both
    // components have the exact same LOO score 64/9 for both candidates. The bandwidth
    // values differ, so picking the last tied candidate is observable in the output.
    const result = computeMultiscaleGeographicallyWeightedRegressionOnCPU(
      {
        positions: new Float32Array(8),
        predictors: Float32Array.of(1, -1, 1, -1),
        response: Float32Array.of(1, 2, 3, 4),
        parameters: Float32Array.of(1e-6, 10, 2),
        bandwidthCandidates: Float32Array.of(10, 1, 2, 20),
        noiseVariance: Float32Array.of(1)
      },
      {predictorCount: 1, kernel, bandwidthCandidateCount: 2}
    );
    expect(Array.from(result.selectedBandwidths)).toEqual([10, 2]);
    expect(Array.from(result.codes)).toEqual([0, 0, 0, 0]);
    expect(result.iterationCount[0]).toBe(1);
    expectRegressionRowsClose(result.coefficients, [2.5, -0.5, 2.5, -0.5, 2.5, -0.5, 2.5, -0.5], 0);
    expectRegressionRowsClose(result.fitted, [2, 3, 2, 3], 0);
    expectRegressionRowsClose(result.conditionalStandardErrors, new Float32Array(8).fill(0.5), 0);
  });
}

it('uses the geodesic seam branch: -180 and 180 represent the same location', () => {
  const input = {
    positions: Float32Array.of(-180, 0, 180, 0, 179, 0, -179, 0),
    predictors: Float32Array.of(1, -1, 2, -2),
    response: Float32Array.of(1, 2, 4, 3),
    parameters: Float32Array.of(1e-8, 200000, 300000)
  };
  const result = computeMultiscaleGeographicallyWeightedRegressionOnCPU(input, {
    predictorCount: 1,
    kernel: 'gaussian',
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
  });
  expect(Array.from(result.codes)).toEqual([0, 0, 0, 0]);
  expectRegressionRowsClose(
    result.coefficients.subarray(0, 2),
    result.coefficients.subarray(2, 4),
    0
  );
  const equivalent = computeMultiscaleGeographicallyWeightedRegressionOnCPU(
    {...input, positions: Float32Array.of(180, 0, 180, 0, 179, 0, -179, 0)},
    {
      predictorCount: 1,
      kernel: 'gaussian',
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
    }
  );
  expectRegressionRowsClose(result.coefficients, equivalent.coefficients, 0);
});

it('keeps failure precedence, exclusion, selected bandwidths and NaN layouts', () => {
  const input = getInput();
  const excluded = computeMultiscaleGeographicallyWeightedRegressionOnCPU(
    {...input, mask: new Uint32Array(12)},
    {predictorCount: 2}
  );
  expect(Array.from(excluded.codes)).toEqual(new Array(12).fill(1));
  expect(excluded.iterationCount[0]).toBe(1);
  const singular = computeMultiscaleGeographicallyWeightedRegressionOnCPU(
    {...input, predictors: new Float32Array(24)},
    {predictorCount: 2}
  );
  expect(Array.from(singular.codes)).toEqual(fixtures[0].mask.map(keep => (keep ? 2 : 1)));
  const invalid = computeMultiscaleGeographicallyWeightedRegressionOnCPU(
    {...input, parameters: Float32Array.of(0, -1, -1, -1)},
    {predictorCount: 2}
  );
  expect(Array.from(invalid.codes)).toEqual(fixtures[0].mask.map(keep => (keep ? 4 : 1)));
  for (const result of [excluded, singular, invalid])
    for (const port of ['coefficients', 'fitted', 'conditionalStandardErrors'] as const)
      expect(result[port].every(Number.isNaN)).toBe(true);
});

it('retains inference failure layouts and the zero-variance saturated fit', () => {
  const input = getInput();
  const failure = computeMultiscaleGeographicallyWeightedRegressionOnCPU(
    {...input, predictors: new Float32Array(24)},
    {predictorCount: 2, inference: {}}
  );
  expect(failure.inference!.status[0]).toBe(1);
  expect(failure.inference!.summary.every(Number.isNaN)).toBe(true);
  const excluded = computeMultiscaleGeographicallyWeightedRegressionOnCPU(
    {...input, mask: new Uint32Array(12)},
    {predictorCount: 2, inference: {}}
  );
  expect(excluded.inference!.status[0]).toBe(2);
  expect(Array.from(excluded.inference!.summary)).toEqual([0, 0, NaN, 0]);
  for (const result of [failure, excluded]) {
    expect(result.inference!.standardErrors.every(Number.isNaN)).toBe(true);
    expect(result.inference!.effectiveParameters.every(Number.isNaN)).toBe(true);
  }
  const zero = computeMultiscaleGeographicallyWeightedRegressionOnCPU(
    {...input, response: new Float32Array(12)},
    {predictorCount: 2, inference: {}}
  );
  expect(zero.inference!.status[0]).toBe(0);
  expect(zero.inference!.summary[2]).toBe(0);
  for (let row = 0; row < 12; row++)
    if (input.mask[row] !== 0)
      for (let column = 0; column < 3; column++)
        expect(zero.inference!.standardErrors[row * 3 + column]).toBe(0);
});

it('validates array lengths and construction-time settings of both classes', () => {
  const input = getInput();
  for (const changes of [
    {positions: new Float32Array(1)},
    {predictors: new Float32Array(1)},
    {mask: new Uint32Array(1)},
    {parameters: new Float32Array(1)},
    {noiseVariance: new Float32Array(2)},
    {bandwidthCandidates: new Float32Array(2)}
  ])
    expect(() =>
      computeMultiscaleGeographicallyWeightedRegressionOnCPU(
        {...input, ...changes},
        {predictorCount: 2}
      )
    ).toThrow(new RegExp(Object.keys(changes)[0]));
  const props: MultiscaleGeographicallyWeightedRegressionOperationProps = {
    id: 'multiscale-test',
    rowCount: 12,
    predictorCount: 2,
    positions: 'positions',
    predictors: 'predictors',
    response: 'response',
    parameters: 'parameters',
    mask: 'mask',
    noiseVariance: 'noiseVariance',
    output: {
      coefficients: 'coefficients',
      fitted: 'fitted',
      status: {codes: 'codes', iterationCount: 'iterationCount'},
      selectedBandwidths: 'selectedBandwidths',
      conditionalStandardErrors: 'conditionalStandardErrors'
    }
  };
  const shapes = {
    positions: {format: 'float32x2' as const, length: 12},
    predictors: {format: 'float32' as const, length: 24},
    response: {format: 'float32' as const, length: 12},
    parameters: {format: 'float32' as const, length: 4},
    mask: {format: 'uint32' as const, length: 12},
    noiseVariance: {format: 'float32' as const, length: 1}
  };
  for (const changes of [
    {predictorCount: 0},
    {maximumIterations: 0},
    {kernel: 'invalid' as never},
    {gaussianCutoff: 0},
    {bandwidthCandidateCount: 0}
  ]) {
    expect(
      () => new CPUMultiscaleGeographicallyWeightedRegressionOperation({...props, ...changes})
    ).toThrow(/multiscale-test/);
    expect(() =>
      constructRegressionOperationContributor(
        new GPUMultiscaleGeographicallyWeightedRegressionOperation({...props, ...changes}),
        shapes
      )
    ).toThrow(/multiscale-test/);
  }
  for (const Operation of [
    GPUMultiscaleGeographicallyWeightedRegressionOperation,
    CPUMultiscaleGeographicallyWeightedRegressionOperation
  ])
    expect(() => new Operation({...props, output: {...props.output, fitted: 'response'}})).toThrow(
      /multiscale-test/
    );
  expect(
    new CPUMultiscaleGeographicallyWeightedRegressionOperation(props).contract.outputs
  ).toEqual(new GPUMultiscaleGeographicallyWeightedRegressionOperation(props).contract.outputs);
  expect(() =>
    constructRegressionOperationContributor(
      new GPUMultiscaleGeographicallyWeightedRegressionOperation(props),
      shapes
    )
  ).not.toThrow();
  const inferenceProps = {
    ...props,
    inference: {
      output: {
        standardErrors: 'inferenceStandardErrors',
        effectiveParameters: 'inferenceEffectiveParameters',
        summary: 'inferenceSummary',
        status: 'inferenceStatus'
      }
    }
  };
  expect(
    new CPUMultiscaleGeographicallyWeightedRegressionOperation(inferenceProps).contract.outputs
  ).toEqual(
    new GPUMultiscaleGeographicallyWeightedRegressionOperation(inferenceProps).contract.outputs
  );
  expect(() =>
    constructRegressionOperationContributor(
      new GPUMultiscaleGeographicallyWeightedRegressionOperation(inferenceProps),
      shapes
    )
  ).not.toThrow();
  const invalidInference = {
    ...inferenceProps,
    inference: {...inferenceProps.inference, varianceConvention: 'invalid' as never}
  };
  expect(
    () => new CPUMultiscaleGeographicallyWeightedRegressionOperation(invalidInference)
  ).toThrow(/multiscale-test/);
  expect(() =>
    constructRegressionOperationContributor(
      new GPUMultiscaleGeographicallyWeightedRegressionOperation(invalidInference),
      shapes
    )
  ).toThrow(/multiscale-test/);
});
