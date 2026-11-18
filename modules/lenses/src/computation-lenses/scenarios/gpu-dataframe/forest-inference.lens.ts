// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {defineLensScenario, TABLE_LENS_RUNGS} from '../../define-lens-scenarios';
import type {RasterLensSession} from '../gpu-raster/raster-lens-harness';

const SCENARIO_ID = 'gpu-dataframe.forest-inference';
const DATASET = {
  id: 'synthetic-forest-split',
  licence: 'MIT',
  source:
    'One synthetic classification stump, a tied leaf, a missing feature and a masked padding row'
};

/** Unit-spaced synthetic planar grid used by the executable lens. */
export const FOREST_INFERENCE_GRID: GPURasterMetadata = {
  width: 5,
  height: 1,
  affine: [1, 0, 0, 0, 1, 0],
  pixelInterpretation: 'area'
};

/** Builds the contributor and actual GPU captures for this synthetic example. */
export async function buildForestInferenceScenario(session: RasterLensSession): Promise<void> {
  const [{runExample}, {readScikitLearnForestModel}, {makeGPURasterBand}] = await Promise.all([
    import('./forest-inference.example'),
    import('@luma-contributors/analysis/dataframe-analysis'),
    import('@luma-contributors/analysis/raster-analysis')
  ]);
  const model = readScikitLearnForestModel({
    format: 'sklearn-forest',
    version: 1,
    estimator: 'RandomForestClassifier',
    n_features_in: 1,
    classes: ['first', 'second'],
    trees: [
      {
        children_left: [1, -1, -1],
        children_right: [2, -1, -1],
        feature: [0, -2, -2],
        threshold: [0, -2, -2],
        missing_go_to_left: [1, 0, 0],
        value: [
          [1, 1],
          [1, 1],
          [0, 2]
        ]
      }
    ]
  });
  const features = session.input('features', 'float32', [-1, 0, 1, NaN, 2], {
    role: 'input',
    geometry: 'raster',
    meaning: 'Synthetic feature; NaN follows the left missing branch'
  });
  const mask = session.input('mask', 'uint32', [1, 1, 1, 1, 0], {
    role: 'context',
    geometry: 'raster',
    meaning: 'Last cell is a masked padding row'
  });
  await runExample(session.graph, {
    model,
    nodes: session.input('nodes', 'uint32', model.nodes, false),
    trees: session.input('trees', 'uint32', model.trees, false),
    leafValues: session.input('leafValues', 'float32', model.leafValues, false),
    parameters: session.input('parameters', 'float32', model.parameters, false),
    rowCount: 5,
    rasterStack: {
      grid: FOREST_INFERENCE_GRID,
      bands: [makeGPURasterBand(features, FOREST_INFERENCE_GRID)]
    },
    mask,
    output: {
      predictions: session.output('predictions', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'First class wins the left-leaf tie; right leaf predicts second class'
      }),
      scores: session.output(
        'scores',
        'float32',
        {
          role: 'intermediate',
          keySpace: 'class-score',
          meaning: 'Row-major class probabilities, with NaN for padding'
        },
        10
      ),
      validity: session.output('validity', 'uint32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'Zero for the masked padding row'
      }),
      leafIndices: session.output('leafIndices', 'uint32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'Original stump leaf id, or 0xffffffff for padding'
      })
    }
  });
}

/** Executes the graph; records and capture sidecars come only from GPU readback. */
export async function run(device: Device) {
  const [{runRasterLens}, {getDataFrameOperatorEntries}] = await Promise.all([
    import('../gpu-raster/raster-lens-harness'),
    import('../../../gpu-dataframe/computation-pack/pack')
  ]);
  const formula = getDataFrameOperatorEntries()['GPUForestInference'].formula;
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-dataframe',
    label: 'GPUForestInference: synthetic-forest-split',
    grid: FOREST_INFERENCE_GRID,
    coordinateModel: 'projected',
    unit: 'arbitrary planar units',
    dataset: {id: DATASET.id, version: '1'},
    formulas: 'tex' in formula ? [formula] : [],
    keySpaces: [
      {id: 'class-score', label: 'row-major class probability', size: 10, indexSpace: '1d'}
    ],
    choices: [
      {
        stage: 'representation',
        name: 'featureLayout',
        value: 'disabled: rasterStack input',
        sweepable: false
      },
      {stage: 'classify', name: 'task', value: 'classification', sweepable: false},
      {stage: 'classify', name: 'transform', value: 'identity', sweepable: false}
    ],
    build: buildForestInferenceScenario
  });
}

defineLensScenario({
  id: SCENARIO_ID,
  element: 'GPUForestInference',
  title: 'GPUForestInference: synthetic-forest-split',
  rungs: TABLE_LENS_RUNGS,
  dataset: DATASET,
  load: async () => ({run})
});
