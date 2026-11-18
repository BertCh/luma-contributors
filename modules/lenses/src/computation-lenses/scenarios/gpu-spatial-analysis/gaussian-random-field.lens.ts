// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {defineLensScenario, TABLE_LENS_RUNGS} from '../../define-lens-scenarios';
import type {RasterLensSession} from '../gpu-raster/raster-lens-harness';

const SCENARIO_ID = 'gpu-spatial-analysis.gaussian-random-field';
const DATASET = {
  id: 'synthetic-spectral-field',
  licence: 'MIT',
  source: 'Seeded exponential Gaussian field on a synthetic unit-spaced planar lattice'
};

/** Unit-spaced synthetic planar grid used by the executable lens. */
export const GAUSSIAN_RANDOM_FIELD_GRID: GPURasterMetadata = {
  width: 4,
  height: 4,
  affine: [1, 0, 0, 0, 1, 0],
  pixelInterpretation: 'area'
};

/** Builds the contributor and actual GPU captures for this synthetic example. */
export async function buildGaussianRandomFieldScenario(session: RasterLensSession): Promise<void> {
  const [{runExample}, {getGPUGaussianRandomFieldParameterValues}] = await Promise.all([
    import('./gaussian-random-field.example'),
    import('@luma-contributors/analysis/gpu-spatial-analysis')
  ]);
  await runExample(session.graph, {
    grid: GAUSSIAN_RANDOM_FIELD_GRID,
    method: 'spectral',
    modeCount: 64,
    parameters: session.input(
      'parameters',
      'float32',
      getGPUGaussianRandomFieldParameterValues({
        grid: GAUSSIAN_RANDOM_FIELD_GRID,
        variogram: {model: 'exponential', sill: 1, range: 3},
        seed: 347,
        realization: 0,
        mean: 2
      }),
      false
    ),
    output: {
      values: session.output('values', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'Seed 347 realization of mean 2, sill 1, exponential range 3'
      }),
      validity: session.output('validity', 'uint32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'Finite usable field samples'
      }),
      status: {
        approximation: session.output(
          'approximation',
          'uint32',
          {
            role: 'intermediate',
            keySpace: 'status',
            meaning: 'One because finite spectral modes approximate the covariance'
          },
          1
        )
      },
      clippedEigenvalueMass: session.output(
        'clippedEigenvalueMass',
        'float32',
        {
          role: 'intermediate',
          keySpace: 'status',
          meaning: 'Zero for spectral modes; no FFT spectrum was clipped'
        },
        1
      )
    }
  });
}

/** Executes the graph; records and capture sidecars come only from GPU readback. */
export async function run(device: Device) {
  const [{runRasterLens}, {getSpatialOperatorEntries}] = await Promise.all([
    import('../gpu-raster/raster-lens-harness'),
    import('../../../gpu-spatial-analysis/computation-pack/pack')
  ]);
  const formula = getSpatialOperatorEntries()['GPUGaussianRandomField'].formula;
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-spatial-analysis',
    label: 'GPUGaussianRandomField: synthetic-spectral-field',
    grid: GAUSSIAN_RANDOM_FIELD_GRID,
    coordinateModel: 'projected',
    unit: 'arbitrary planar units',
    dataset: {id: DATASET.id, version: '1'},
    formulas: 'tex' in formula ? [formula] : [],
    keySpaces: [{id: 'status', label: 'run status', size: 1, indexSpace: '1d'}],
    choices: [
      {stage: 'operator', name: 'method', value: 'spectral', sweepable: false},
      {stage: 'operator', name: 'modeCount', value: 64, sweepable: true},
      {stage: 'inference', name: 'seed', value: 347, sweepable: true},
      {stage: 'inference', name: 'realization', value: 0, sweepable: true}
    ],
    build: buildGaussianRandomFieldScenario
  });
}

defineLensScenario({
  id: SCENARIO_ID,
  element: 'GPUGaussianRandomField',
  title: 'GPUGaussianRandomField: synthetic-spectral-field',
  rungs: TABLE_LENS_RUNGS,
  dataset: DATASET,
  load: async () => ({run})
});
