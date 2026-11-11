// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeSpaceTimeDiggleTestOnCPU,
  CPUSpaceTimeDiggleTestOperation
} from '../../../src/dataframe-analysis/pair-statistics/cpu-space-time-diggle-test-operation';
import {
  GPUSpaceTimeDiggleTestOperation,
  type SpaceTimeDiggleTestOperationProps
} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-diggle-test-operation';
import {getGPUSpaceTimeKFunctionParameterValues} from '../../../src/dataframe-analysis/pair-statistics/space-time-k-function-nodes';
import {
  getGPUSpaceTimeDiggleRandomParameterValues,
  GPU_SPACE_TIME_DIGGLE_SUMMARY
} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-diggle-test';
import {fillShufflePermutation} from '../../../src/analysis-core/kernels/permutation-shuffle';
import {PHILOX_PURPOSES} from '../../../src/analysis-core/kernels/philox-random';
import {SPLANCS_FIXTURE} from './space-time-diggle-splancs-fixture';
import {getSpaceTimeKReference} from './space-time-k-function-reference';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';

const PROPS: SpaceTimeDiggleTestOperationProps = {
  id: 'diggle-operation-test',
  positions: 'positions',
  times: 'times',
  parameters: 'parameters',
  randomParameters: 'random',
  rowCount: 2,
  gridSize: [4, 4],
  distanceClassCount: 2,
  timeClassCount: 2,
  permutationCount: 7,
  output: {d: 'd', statistics: 'statistics', summary: 'summary'}
};

it('matches the independent committed NumPy splancs port on all whole-day grids and labellings', () => {
  const scene = SPLANCS_FIXTURE.integerTimes;
  for (const entry of scene.cases)
    for (let labelling = 0; labelling < scene.labellingTimes.length; labelling++) {
      const expected = entry.labellings[labelling];
      const parameters = getGPUSpaceTimeKFunctionParameterValues({
        bounds: SPLANCS_FIXTURE.bounds,
        timeWindow: scene.timeLimits,
        maximumDistance: scene.maximumDistance,
        maximumTimeLag: entry.maximumTimeLag
      });
      const result = computeSpaceTimeDiggleTestOnCPU(
        {
          positions: Float32Array.from(scene.positions),
          times: Float32Array.from(scene.labellingTimes[labelling]),
          parameters,
          randomParameters: getGPUSpaceTimeDiggleRandomParameterValues(19)
        },
        {
          distanceClassCount: scene.distanceClassCount,
          timeClassCount: entry.timeClassCount,
          permutationCount: 1
        }
      );
      const scale = Math.max(...expected.k.map(Math.abs));
      for (const port of ['k', 'd', 'spatialK', 'temporalK'] as const)
        result[port].forEach((value, index) =>
          expect(Math.abs(value - expected[port][index]), `${port} ${index}`).toBeLessThanOrEqual(
            2e-6 * scale
          )
        );
      expect(Math.abs(result.statistics[0] - expected.statistic)).toBeLessThanOrEqual(
        2e-6 * scale * result.d.length
      );
      expect([...result.status]).toEqual([0]);
    }
});

it('matches the hand-computed temporal edge tie and open largest lag', () => {
  const tie = SPLANCS_FIXTURE.temporalTie;
  const input = {
    positions: Float32Array.from(tie.positions),
    times: Float32Array.from(tie.times),
    parameters: getGPUSpaceTimeKFunctionParameterValues({
      bounds: SPLANCS_FIXTURE.bounds,
      timeWindow: tie.timeLimits,
      maximumDistance: tie.distanceBreaks[0],
      maximumTimeLag: tie.timeBreaks[0]
    }),
    randomParameters: getGPUSpaceTimeDiggleRandomParameterValues(1)
  };
  const options = {distanceClassCount: 1, timeClassCount: 1, permutationCount: 1};
  const result = computeSpaceTimeDiggleTestOnCPU(input, options);
  // t=3,6 in [0,10]: directed temporal weights 2 and 1, K_t=15, K_s=80, K=1200.
  expect([...result.temporalK]).toEqual([15]);
  expect([...result.spatialK]).toEqual([80]);
  expect([...result.k]).toEqual([1200]);
  expect([...result.d]).toEqual([0]);
  expect(result.permutationStandardDeviation.every(Number.isNaN)).toBe(true);
  const atMaximum = getGPUSpaceTimeKFunctionParameterValues({
    bounds: SPLANCS_FIXTURE.bounds,
    timeWindow: tie.timeLimits,
    maximumDistance: tie.distanceBreaks[0],
    maximumTimeLag: 3
  });
  expect([
    ...computeSpaceTimeDiggleTestOnCPU({...input, parameters: atMaximum}, options).temporalK
  ]).toEqual([0]);
});

it('matches independently evaluated relabellings and their moments over the masked row domain', () => {
  const scene = {
    positions: Float32Array.of(1, 1, 3, 2, 6, 1, 2, 6, 7, 6, 8, 3),
    times: Float32Array.of(0, 2, 5, 8, 13, 20),
    mask: Uint32Array.of(1, 0, 1, 1, 1, 1)
  };
  const settings = {
    bounds: [0, 0, 10, 8] as const,
    timeWindow: [0, 20] as const,
    maximumDistance: 7,
    maximumTimeLag: 11
  };
  const options = {
    distanceClassCount: 3,
    timeClassCount: 2,
    permutationCount: 19,
    edgeCorrection: 'none' as const
  };
  const referenceOptions = {...settings, ...options, tieRule: 'splancs' as const};
  const referenceScene = {
    positions: [...scene.positions],
    times: [...scene.times],
    mask: [...scene.mask]
  };
  const observed = getSpaceTimeKReference({scene: referenceScene, ...referenceOptions});
  const key = getGPUSpaceTimeDiggleRandomParameterValues(2 ** 40 + 21);
  const result = computeSpaceTimeDiggleTestOnCPU(
    {
      ...scene,
      parameters: getGPUSpaceTimeKFunctionParameterValues(settings),
      randomParameters: key
    },
    options
  );
  const residuals: number[][] = [];
  for (let slot = 1; slot <= options.permutationCount; slot++) {
    const order = new Uint32Array(observed.included.length);
    fillShufflePermutation(
      order,
      order.length,
      [key[0], key[1]],
      slot,
      PHILOX_PURPOSES.spaceTimeDiggleShuffle.id
    );
    const times = [...scene.times];
    observed.included.forEach((row, rank) => {
      times[row] = scene.times[observed.included[order[rank]]];
    });
    residuals.push(
      getSpaceTimeKReference({...referenceOptions, scene: {...referenceScene, times}}).d
    );
  }
  const statistics = [observed.d, ...residuals].map(curve =>
    curve.reduce((sum, value) => sum + value, 0)
  );
  result.statistics.forEach((value, slot) =>
    expect(Math.abs(value - statistics[slot])).toBeLessThan(1e-4)
  );
  result.permutationMean.forEach((value, bin) => {
    const mean = residuals.reduce((sum, curve) => sum + curve[bin], 0) / options.permutationCount;
    const deviation = Math.sqrt(
      residuals.reduce((sum, curve) => sum + (curve[bin] - mean) ** 2, 0) /
        (options.permutationCount - 1)
    );
    expect(Math.abs(value - mean)).toBeLessThan(1e-4);
    expect(Math.abs(result.permutationStandardDeviation[bin] - deviation)).toBeLessThan(1e-4);
  });
  const mean =
    statistics.slice(1).reduce((sum, value) => sum + value, 0) / options.permutationCount;
  const variance =
    statistics.slice(1).reduce((sum, value) => sum + (value - mean) ** 2, 0) /
    (options.permutationCount - 1);
  const greater = statistics.slice(1).filter(value => value >= statistics[0]).length;
  const lesser = statistics.slice(1).filter(value => value <= statistics[0]).length;
  const greaterProbability = (greater + 1) / (options.permutationCount + 1);
  const lesserProbability = (lesser + 1) / (options.permutationCount + 1);
  const summary = [
    statistics[0],
    mean,
    variance,
    greater,
    lesser,
    greaterProbability,
    lesserProbability,
    Math.min(1, 2 * Math.min(greaterProbability, lesserProbability)),
    (statistics[0] - mean) / Math.sqrt(variance),
    result.d.length,
    5
  ];
  result.summary.forEach((value, index) =>
    expect(Math.abs(value - summary[index])).toBeLessThanOrEqual(
      2e-6 * Math.max(1, Math.abs(summary[index]))
    )
  );
  expect(result.summary[GPU_SPACE_TIME_DIGGLE_SUMMARY.pseudoPGreater]).toBeGreaterThan(0.1);
  expect(result.summary[GPU_SPACE_TIME_DIGGLE_SUMMARY.pseudoPGreater]).toBeLessThan(0.95);
});

it('returns undefined status and NaN inference for excluded events', () => {
  const result = computeSpaceTimeDiggleTestOnCPU(
    {
      positions: Float32Array.of(1, 1, 2, 2),
      times: Float32Array.of(0, 1),
      mask: new Uint32Array(2),
      parameters: getGPUSpaceTimeKFunctionParameterValues({
        bounds: [0, 0, 10, 8],
        timeWindow: [0, 20],
        maximumDistance: 5,
        maximumTimeLag: 10
      }),
      randomParameters: getGPUSpaceTimeDiggleRandomParameterValues(1)
    },
    PROPS
  );
  expect([...result.status]).toEqual([1]);
  for (const port of [
    'd',
    'statistics',
    'k',
    'spatialK',
    'temporalK',
    'permutationMean',
    'permutationStandardDeviation'
  ] as const)
    expect(result[port].every(Number.isNaN)).toBe(true);
  expect([...result.summary].slice(0, 9).every(Number.isNaN)).toBe(true);
  expect([...result.summary].slice(9)).toEqual([0, 0]);
});

it('validates arrays and CPU/deferred GPU constructions', () => {
  const input = {
    positions: Float32Array.of(1, 1, 2, 2),
    times: Float32Array.of(0, 1),
    parameters: new Float32Array(11),
    randomParameters: new Uint32Array(2)
  };
  expect(() =>
    computeSpaceTimeDiggleTestOnCPU({...input, times: new Float32Array(1)}, PROPS)
  ).toThrow(/times/);
  expect(() =>
    computeSpaceTimeDiggleTestOnCPU({...input, mask: new Uint32Array(1)}, PROPS)
  ).toThrow(/mask/);
  expect(() =>
    computeSpaceTimeDiggleTestOnCPU({...input, parameters: new Float32Array(3)}, PROPS)
  ).toThrow(/parameters/);
  expect(() =>
    computeSpaceTimeDiggleTestOnCPU({...input, randomParameters: new Uint32Array(1)}, PROPS)
  ).toThrow(/randomParameters/);
  expect(() => new CPUSpaceTimeDiggleTestOperation({...PROPS, permutationCount: 0})).toThrow(
    /diggle-operation-test.*permutationCount/
  );
  expect(() => new GPUSpaceTimeDiggleTestOperation({...PROPS, distanceClassCount: 0})).toThrow(
    /diggle-operation-test.*length/
  );
  const shapes = {
    positions: {format: 'float32x2' as const, length: 2},
    times: {format: 'float32' as const, length: 2},
    parameters: {format: 'float32' as const, length: 11},
    randomParameters: {format: 'uint32' as const, length: 2}
  };
  expect(() =>
    constructNetworkOperationContributor(
      new GPUSpaceTimeDiggleTestOperation({...PROPS, permutationCount: 0}),
      shapes
    )
  ).toThrow(/permutationCount/);
  expect(() =>
    constructNetworkOperationContributor(
      new GPUSpaceTimeDiggleTestOperation({...PROPS, timeClassCount: 65}),
      shapes
    )
  ).toThrow(/timeClassCount/);
});
