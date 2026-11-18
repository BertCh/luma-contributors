// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {defineLensScenario, TABLE_LENS_RUNGS} from '../../define-lens-scenarios';
import type {RasterLensSession} from './raster-lens-harness';

const SCENARIO_ID = 'gpu-raster.raster-contrast-equalization';
const DATASET = {
  id: 'synthetic-contrast-gradient',
  licence: 'MIT',
  source: 'Synthetic 4 by 4 reflectance gradient with one missing source cell'
};

/** Unit-spaced synthetic planar grid used by the executable lens. */
export const RASTER_CONTRAST_EQUALIZATION_GRID: GPURasterMetadata = {
  width: 4,
  height: 4,
  affine: [1, 0, 0, 0, 1, 0],
  pixelInterpretation: 'area'
};

/** Builds the contributor and actual GPU captures for this synthetic example. */
export async function buildRasterContrastEqualizationScenario(
  session: RasterLensSession
): Promise<void> {
  const [{runExample}, {makeGPURasterBand, getGPURasterContrastEqualizationParameterValues}] =
    await Promise.all([
      import('./raster-contrast-equalization.example'),
      import('@luma-contributors/analysis/raster-analysis')
    ]);
  const values = session.input(
    'inputValues',
    'float32',
    Array.from({length: 16}, (_, index) => index / 15),
    {
      role: 'input',
      geometry: 'raster',
      meaning: 'Synthetic reflectance gradient in [0,1]'
    }
  );
  const validity = session.input('inputValidity', 'uint32', [0, ...new Array(15).fill(1)], {
    role: 'context',
    geometry: 'raster',
    meaning: 'Cell zero is missing and excluded from tile histograms'
  });
  await runExample(session.graph, {
    grid: RASTER_CONTRAST_EQUALIZATION_GRID,
    input: makeGPURasterBand(values, RASTER_CONTRAST_EQUALIZATION_GRID, {validity}),
    mode: 'clahe',
    binCount: 16,
    tileWidth: 2,
    tileHeight: 2,
    parameters: session.input(
      'parameters',
      'float32',
      getGPURasterContrastEqualizationParameterValues({clipLimit: 0.25}),
      false
    ),
    output: {
      values: session.output('equalizedValues', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'Normalized CLAHE mapping with reflected two-cell tiles'
      }),
      validity: session.output('equalizedValidity', 'uint32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'Source validity preserved by contrast mapping'
      }),
      status: {
        approximation: session.output(
          'approximation',
          'uint32',
          {
            role: 'intermediate',
            keySpace: 'status',
            meaning: 'One because histogram bins quantize the contrast mapping'
          },
          1
        )
      }
    }
  });
}

/** Executes the graph; records and capture sidecars come only from GPU readback. */
export async function run(device: Device) {
  const [{runRasterLens}, {getRasterOperatorEntries}] = await Promise.all([
    import('./raster-lens-harness'),
    import('../../../gpu-raster/computation-pack/pack')
  ]);
  const formula = getRasterOperatorEntries()['GPURasterContrastEqualization'].formula;
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPURasterContrastEqualization: synthetic-contrast-gradient',
    grid: RASTER_CONTRAST_EQUALIZATION_GRID,
    coordinateModel: 'projected',
    unit: 'arbitrary planar units',
    dataset: {id: DATASET.id, version: '1'},
    formulas: 'tex' in formula ? [formula] : [],
    keySpaces: [{id: 'status', label: 'run status', size: 1, indexSpace: '1d'}],
    choices: [
      {stage: 'operator', name: 'mode', value: 'clahe', sweepable: false},
      {stage: 'operator', name: 'clipLimit', value: 0.25, sweepable: true},
      {stage: 'neighbourhood', name: 'tileWidth', value: 2, sweepable: true},
      {stage: 'neighbourhood', name: 'tileHeight', value: 2, sweepable: true}
    ],
    build: buildRasterContrastEqualizationScenario
  });
}

defineLensScenario({
  id: SCENARIO_ID,
  element: 'GPURasterContrastEqualization',
  title: 'GPURasterContrastEqualization: synthetic-contrast-gradient',
  rungs: TABLE_LENS_RUNGS,
  dataset: DATASET,
  load: async () => ({run})
});
