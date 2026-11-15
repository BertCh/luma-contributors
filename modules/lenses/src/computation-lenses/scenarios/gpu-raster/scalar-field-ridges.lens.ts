// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {defineLensScenario, TABLE_LENS_RUNGS} from '../../define-lens-scenarios';
import type {RasterLensSession} from './raster-lens-harness';

const SCENARIO_ID = 'gpu-raster.scalar-field-ridges';
const DATASET = {
  id: 'quadratic-scalar-ridge',
  licence: 'MIT',
  source: 'Synthetic 7 by 7 field: 20 - (column - 2.5)^2, with cell 40 masked'
};

/** Unit-spaced planar cells; the maximum lies exactly between columns 2 and 3. */
export const SCALAR_RIDGE_GRID: GPURasterMetadata = {
  width: 7,
  height: 7,
  affine: [1, 0, 0, 0, 1, 0],
  pixelInterpretation: 'area'
};

/** Builds the same real contributor for the executable lens and its node/headless checks. */
export async function buildScalarFieldRidgesScenario(session: RasterLensSession): Promise<void> {
  const [{runExample}, {makeGPURasterBand}] = await Promise.all([
    import('./scalar-field-ridges.example'),
    import('@luma-contributors/analysis/raster-analysis')
  ]);
  const values = Float32Array.from({length: 49}, (_, index) => 20 - ((index % 7) - 2.5) ** 2);
  const validity = new Uint32Array(49).fill(1);
  validity[40] = 0;
  const input = makeGPURasterBand(
    session.input('scalarFieldValues', 'float32', values, {
      role: 'input',
      geometry: 'raster',
      meaning: 'quadratic field, maximum between columns 2 and 3'
    }),
    SCALAR_RIDGE_GRID,
    {
      validity: session.input('scalarFieldValidity', 'uint32', validity, {
        role: 'context',
        geometry: 'raster',
        meaning: 'input validity: cell 40 is missing'
      })
    }
  );
  session.track(
    await runExample(session.device, session.graph, SCALAR_RIDGE_GRID, input, {
      ridges: session.output('ridgeMask', 'uint32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'ridge cells: column 3 in interior rows 1 through 5',
        highlight: {key: 24, label: 'half-cell tie selects column 3'}
      }),
      validity: session.output('ridgeValidity', 'uint32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'finite interior stencils without a missing sample'
      }),
      strength: session.output('ridgeStrength', 'float32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'downward normal curvature: 2 at every valid stencil'
      })
    })
  );
}

/** Captures a half-cell tie and a missing sample without fabricating GPU records. */
export async function run(device: Device) {
  const [{runRasterLens}, {getRasterOperatorEntries}] = await Promise.all([
    import('./raster-lens-harness'),
    import('../../../gpu-raster/computation-pack')
  ]);
  const formula = getRasterOperatorEntries()['GPUScalarFieldRidges'].formula;
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'Height ridges of a quadratic scalar field',
    grid: SCALAR_RIDGE_GRID,
    coordinateModel: 'projected',
    unit: 'm',
    dataset: {id: DATASET.id, version: '1'},
    formulas: 'tex' in formula ? [formula] : [],
    choices: [
      {stage: 'operator', name: 'minimumValue', value: 0, sweepable: true},
      {stage: 'operator', name: 'minimumCurvature', value: 1, sweepable: true},
      {stage: 'support', name: 'cellSizeMode', value: 'uniform', sweepable: false}
    ],
    build: buildScalarFieldRidgesScenario
  });
}

defineLensScenario({
  id: SCENARIO_ID,
  element: 'GPUScalarFieldRidges',
  title: 'Scalar field ridges: half-cell ties and missing stencils',
  rungs: TABLE_LENS_RUNGS,
  dataset: DATASET,
  load: async () => ({run})
});
