// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeRipleyEnvelopeOnCPU,
  CPURipleyEnvelopeOperation
} from '../../../src/dataframe-analysis/pair-statistics/cpu-ripley-envelope-operation';
import {
  GPURipleyEnvelopeOperation,
  type RipleyEnvelopeOperationProps
} from '../../../src/dataframe-analysis/pair-statistics/gpu-ripley-envelope-operation';
import {getGPURipleyParameterValues} from '../../../src/dataframe-analysis/pair-statistics/ripley-parameters';
import {
  getPhilox4x32,
  getPhiloxUnitFloat,
  PHILOX_PURPOSES
} from '../../../src/analysis-core/kernels/philox-random';
import {computeRipleyEnvelopeOnCPU as computeEnvelopeOracle} from './ripley-envelope-oracle';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {computeRipleyOnCPU} from './ripley-oracle';

const PROPS: RipleyEnvelopeOperationProps = {
  id: 'ripley-operation-test',
  positions: 'positions',
  parameters: 'parameters',
  rowCount: 4,
  gridSize: [4, 4],
  radiusCount: 3,
  simulationCount: 6,
  output: {lower: 'lower'}
};
const POSITIONS = Float32Array.of(0.1, 0.2, 0.8, 0.3, 0.4, 0.6, 0.7, 0.9);

it('exercises a moderate Monte Carlo probability on the observed spatial pattern', () => {
  const positions = Float32Array.of(0.3, 0.2, 0.8, 0.3, 0.4, 0.6, 0.7, 0.9, 0.2, 0.8, 0.6, 0.1);
  const mask = Uint32Array.of(1, 1, 0, 1, 1, 1);
  const frame = {
    bounds: [0, 0, 1, 1] as const,
    maximumDistance: 0.5,
    edgeCorrection: 'isotropic' as const
  };
  const observedK = Float32Array.from(computeRipleyOnCPU({positions, mask}, frame, 4).k);
  const result = computeRipleyEnvelopeOnCPU(
    {positions, mask, observedK, parameters: getGPURipleyParameterValues({...frame, seed: 31})},
    {...PROPS, radiusCount: 4}
  );
  expect(result.statistics[4]).toBeGreaterThan(0.2);
  expect(result.statistics[4]).toBeLessThan(0.95);
});

it('matches the closed form when every simulated ordered pair lies within the radius', () => {
  const parameters = getGPURipleyParameterValues({
    bounds: [0, 0, 2, 3],
    maximumDistance: 4,
    edgeCorrection: 'none',
    seed: 7
  });
  const result = computeRipleyEnvelopeOnCPU(
    {positions: POSITIONS, parameters, observedK: Float32Array.of(6)},
    {...PROPS, radiusCount: 1, statistic: 'k'}
  );
  // Every pattern has K=A=6, so all curves have identical |sqrt(6/pi)-4|.
  const deviation = 4 - Math.sqrt(6 / Math.PI);
  expect([...result.lower]).toEqual([6]);
  expect([...result.upper]).toEqual([6]);
  expect(result.globalLower[0]).toBe(Math.fround(6));
  expect(result.globalUpper[0]).toBe(Math.fround(Math.PI * (4 + deviation) ** 2));
  expect([...result.theoretical]).toEqual([Math.fround(16 * Math.PI)]);
  expect([...result.statistics]).toEqual([
    4,
    6,
    Math.fround(deviation),
    Math.fround(deviation),
    1,
    1
  ]);
  expect([...result.unplacedCount]).toEqual([0]);
});

it('ignores oversized observedK capacity while retaining the required prefix', () => {
  const parameters = getGPURipleyParameterValues({
    bounds: [0, 0, 1, 1],
    maximumDistance: 2,
    edgeCorrection: 'none'
  });
  const input = {
    positions: Float32Array.of(0, 0, 1, 1),
    parameters,
    observedK: Float32Array.of(1, 99)
  };
  const props: RipleyEnvelopeOperationProps = {
    ...PROPS,
    rowCount: 2,
    radiusCount: 1,
    simulationCount: 2,
    statistic: 'k',
    observedK: 'observed'
  };
  const result = computeRipleyEnvelopeOnCPU(input, props);
  // The unit square has area 1 and diameter sqrt(2) < 2. Both directed pairs are
  // counted in every two-point simulation: K = 1/(2*1) * 2 = 1. Observed K is 1,
  // so all deviations are 2 - sqrt(1/pi), and (1 + 2)/(2 + 1) = 1.
  const deviation = 2 - Math.sqrt(1 / Math.PI);
  expect([...result.lower]).toEqual([1]);
  expect([...result.upper]).toEqual([1]);
  expect([...result.globalLower]).toEqual([1]);
  expect([...result.globalUpper]).toEqual([Math.fround(Math.PI * (2 + deviation) ** 2)]);
  expect([...result.theoretical]).toEqual([Math.fround(4 * Math.PI)]);
  expect([...result.statistics]).toEqual([
    2,
    2,
    Math.fround(deviation),
    Math.fround(deviation),
    1,
    1
  ]);
  expect([...result.unplacedCount]).toEqual([0]);
  expect(result).toEqual(
    computeRipleyEnvelopeOnCPU({...input, observedK: Float32Array.of(1)}, props)
  );
  expect(() => new CPURipleyEnvelopeOperation(props)).not.toThrow();
  expect(() =>
    constructNetworkOperationContributor(new GPURipleyEnvelopeOperation(props), {
      positions: {format: 'float32x2', length: 2},
      parameters: {format: 'float32', length: parameters.length},
      observedK: {format: 'float32', length: 2}
    })
  ).not.toThrow();
  expect(() =>
    computeRipleyEnvelopeOnCPU({...input, observedK: new Float32Array(0)}, props)
  ).toThrow(/observedK/);
});

it('agrees with independent circle-angle quadrature, including edge weights and simulated order statistics', () => {
  const maximumDistance = 0.5;
  const radii = [maximumDistance / 3, (2 * maximumDistance) / 3, maximumDistance];
  const parameters = getGPURipleyParameterValues({
    bounds: [0, 0, 1, 1],
    maximumDistance,
    edgeCorrection: 'isotropic',
    seed: 31
  });
  const angularCount = 65536;
  const cosine = Array.from({length: angularCount}, (_, index) =>
    Math.cos((2 * Math.PI * (index + 0.5)) / angularCount)
  );
  const sine = Array.from({length: angularCount}, (_, index) =>
    Math.sin((2 * Math.PI * (index + 0.5)) / angularCount)
  );
  const curves = Array.from({length: 6}, (_, simulation) => {
    const points = Array.from({length: 4}, (_, row) =>
      getPhilox4x32([row, simulation, 0, 0], [31, PHILOX_PURPOSES.ripleyEnvelopeSimulation.id])
        .slice(0, 2)
        .map(getPhiloxUnitFloat)
    );
    const curve = [0, 0, 0];
    for (let focus = 0; focus < 4; focus++)
      for (let other = 0; other < 4; other++)
        if (focus !== other) {
          const distance = Math.hypot(
            points[focus][0] - points[other][0],
            points[focus][1] - points[other][1]
          );
          if (distance > maximumDistance) continue;
          let inside = 0;
          for (let angle = 0; angle < angularCount; angle++) {
            const x = points[focus][0] + distance * cosine[angle];
            const y = points[focus][1] + distance * sine[angle];
            if (x >= 0 && x <= 1 && y >= 0 && y <= 1) inside++;
          }
          const weight = Math.min(100, angularCount / inside);
          radii.forEach((radius, index) => {
            if (distance <= radius) curve[index] += weight / 12;
          });
        }
    return curve;
  });
  const observedK = Float32Array.of(0.2, 0.3, 0.8);
  for (const statistic of ['k', 'l', 'l-minus-r'] as const) {
    const result = computeRipleyEnvelopeOnCPU(
      {positions: POSITIONS, parameters, observedK},
      {...PROPS, statistic}
    );
    const expected = computeEnvelopeOracle(curves, [...observedK], radii, 1, statistic);
    for (const port of ['lower', 'upper', 'globalLower', 'globalUpper', 'theoretical'] as const)
      result[port].forEach((value, index) =>
        expect(Math.abs(value - expected[port][index])).toBeLessThan(0.0002)
      );
    expect(Math.abs(result.statistics[3] - expected.critical)).toBeLessThan(0.0001);
    expect(result.statistics[4]).toBe(Math.fround(expected.pValue));
  }
});

it('honors masks and inclusion, and keeps theoretical values for undefined simulations', () => {
  const parameters = getGPURipleyParameterValues({
    bounds: [0, 0, 1, 1],
    maximumDistance: 1,
    edgeCorrection: 'border'
  });
  const result = computeRipleyEnvelopeOnCPU(
    {
      positions: Float32Array.of(0, 0, NaN, 0, 2, 2, 0.5, 0.5),
      parameters,
      mask: Uint32Array.of(1, 1, 1, 0)
    },
    PROPS
  );
  expect([...result.statistics].slice(0, 2)).toEqual([1, 0]);
  expect(result.lower.every(Number.isNaN)).toBe(true);
  expect(result.upper.every(Number.isNaN)).toBe(true);
  expect(result.globalLower.every(Number.isNaN)).toBe(true);
  expect([...result.theoretical]).toEqual([0, 0, 0]);
});

it('validates arrays and both CPU and deferred GPU constructions', () => {
  const parameters = getGPURipleyParameterValues({bounds: [0, 0, 1, 1], maximumDistance: 1});
  expect(() =>
    computeRipleyEnvelopeOnCPU({positions: new Float32Array(3), parameters}, PROPS)
  ).toThrow(/positions/);
  expect(() =>
    computeRipleyEnvelopeOnCPU({positions: POSITIONS, parameters, mask: new Uint32Array(3)}, PROPS)
  ).toThrow(/mask/);
  expect(() =>
    computeRipleyEnvelopeOnCPU(
      {positions: POSITIONS, parameters, observedK: new Float32Array(2)},
      PROPS
    )
  ).toThrow(/observedK/);
  expect(() =>
    computeRipleyEnvelopeOnCPU({positions: POSITIONS, parameters: new Float32Array(2)}, PROPS)
  ).toThrow(/parameters/);
  expect(() => new CPURipleyEnvelopeOperation({...PROPS, simulationCount: 1})).toThrow(
    /ripley-operation-test.*simulationCount/
  );
  expect(() => new GPURipleyEnvelopeOperation({...PROPS, radiusCount: 0})).toThrow(
    /ripley-operation-test.*length/
  );
  const shapes = {
    positions: {format: 'float32x2' as const, length: 4},
    parameters: {format: 'float32' as const, length: 8}
  };
  expect(() =>
    constructNetworkOperationContributor(
      new GPURipleyEnvelopeOperation({...PROPS, simulationCount: 1}),
      shapes
    )
  ).toThrow(/simulationCount/);
  expect(() =>
    constructNetworkOperationContributor(
      new GPURipleyEnvelopeOperation({...PROPS, radiusCount: 257}),
      shapes
    )
  ).toThrow(/radiusCount/);
});
