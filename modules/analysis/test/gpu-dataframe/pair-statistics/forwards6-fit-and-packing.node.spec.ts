// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  fitVariogramModelWithDiagnostics,
  fitSpaceTimeVariogramModel
} from '@luma-contributors/analysis/dataframe-analysis';
import {fitSpaceTimeVariogramModel as fitSpatialAlias} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {computeVariogramModelFitOnCPU} from '../../../src/dataframe-analysis/pair-statistics/cpu-variogram-model-fit-operation';
import {getGPUSpaceTimeKFunctionParameterValues} from '../../../src/dataframe-analysis/pair-statistics/space-time-k-function-nodes';

const constantBins = {distances: [1, 2, 3], semivariances: [0, 0, 0], pairCounts: [4, 3, 2]};

it('B2 constant fields retain the exact zero Cressie objective in public fitters', () => {
  // Every observed and fitted ordinate is zero: sum(weight * (0 - 0)^2) = 0.
  // The range is unidentified; no unique range or convergence status is asserted.
  const spatial = fitVariogramModelWithDiagnostics(constantBins, {model: 'spherical'}).model;
  expect([spatial.nugget, spatial.sill, spatial.residual]).toEqual([0, 0, 0]);
  for (const fit of [fitSpaceTimeVariogramModel, fitSpatialAlias]) {
    const result = fit(
      {...constantBins, timeLags: [1, 2, 3]},
      {type: 'metric', structure: 'spherical'}
    );
    expect(result.model.type).toBe('metric');
    if (result.model.type !== 'metric') throw new Error('expected metric model');
    expect([result.model.joint.nugget, result.model.joint.sill, result.residual]).toEqual([
      0, 0, 0
    ]);
  }
});

it('BC3 CPU fitting publishes a usable zero field instead of numerical failure', () => {
  const result = computeVariogramModelFitOnCPU(constantBins, {models: ['spherical']});
  expect(Array.from(result.parameters).every(Number.isFinite)).toBe(true);
  expect([result.parameters[0], result.parameters[1], result.parameters[3]]).toEqual([0, 0, 0]);
  expect([0, 3]).toContain(result.status[0]);
});

it('B3 public space-time fitting rejects unknown weighting with accepted choices', () => {
  expect(() =>
    fitSpaceTimeVariogramModel(
      {...constantBins, semivariances: [1, 2, 3], timeLags: [1, 2, 3]},
      {type: 'metric', structure: 'spherical', weighting: 'paers' as 'pairs'}
    )
  ).toThrow(/fitSpaceTimeVariogramModel.*weighting.*cressie.*pairs.*none.*paers/);
});

it('N3 space-time K validates the captured bounds once before writing a target', () => {
  let reads = 0;
  const target = new Float32Array(11).fill(93);
  expect(() =>
    getGPUSpaceTimeKFunctionParameterValues(
      {
        get bounds(): [number, number, number, number] {
          reads++;
          return reads === 1 ? [0, 0, 2 ** 70, 2 ** 70] : [0, 0, 1, 1];
        },
        timeWindow: [0, 1],
        maximumDistance: 1,
        maximumTimeLag: 1
      },
      target
    )
  ).toThrow(/area.*finite/);
  expect(reads).toBe(1);
  expect(Array.from(target)).toEqual(new Array(11).fill(93));
});
