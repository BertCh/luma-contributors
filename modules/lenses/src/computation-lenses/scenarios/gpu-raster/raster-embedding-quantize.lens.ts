// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {defineLensScenario, TABLE_LENS_RUNGS} from '../../define-lens-scenarios';
import type {RasterLensSession} from './raster-lens-harness';

const SCENARIO_ID = 'gpu-raster.raster-embedding-quantize';
const DATASET = {
  id: 'synthetic-embedding-codec',
  licence: 'MIT',
  source:
    'Three synthetic three-component vectors: finite, masked, and nonfinite; nine components leave three unused final bytes'
};

/** Unit-spaced synthetic planar grid used by the executable lens. */
export const RASTER_EMBEDDING_QUANTIZE_GRID: GPURasterMetadata = {
  width: 3,
  height: 1,
  affine: [1, 0, 0, 0, 1, 0],
  pixelInterpretation: 'area'
};

/** Builds the contributor and actual GPU captures for this synthetic example. */
export async function buildRasterEmbeddingQuantizeScenario(
  session: RasterLensSession
): Promise<void> {
  const {runExample} = await import('./raster-embedding-quantize.example');
  const vectors = session.input('vectors', 'float32', [0.25, 0, -2, 1, -1, 0.5, NaN, 0, 1], {
    role: 'input',
    keySpace: 'component',
    meaning:
      'Pixel-major components; -2 saturates and a nonfinite component invalidates the third pixel'
  });
  const mask = session.input('mask', 'uint32', [1, 0, 1], {
    role: 'context',
    geometry: 'raster',
    meaning: 'Second embedding pixel is masked'
  });
  await runExample(session.graph, {
    pixelCount: 3,
    dimensions: 3,
    vectors,
    mask,
    output: {
      quantized: session.output(
        'quantized',
        'uint32',
        {
          role: 'output',
          keySpace: 'word',
          meaning: 'Four signed bytes per word, little-endian; unused final bytes zero'
        },
        3
      ),
      validity: session.output('validity', 'uint32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'Only the first finite, unmasked pixel is valid'
      })
    }
  });
}

/** Executes the graph; records and capture sidecars come only from GPU readback. */
export async function run(device: Device) {
  const [{runRasterLens}, {getRasterOperatorEntries}] = await Promise.all([
    import('./raster-lens-harness'),
    import('../../../gpu-raster/computation-pack/pack')
  ]);
  const formula = getRasterOperatorEntries()['GPURasterEmbeddingQuantize'].formula;
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPURasterEmbeddingQuantize: synthetic-embedding-codec',
    grid: RASTER_EMBEDDING_QUANTIZE_GRID,
    coordinateModel: 'projected',
    unit: 'arbitrary planar units',
    dataset: {id: DATASET.id, version: '1'},
    formulas: 'tex' in formula ? [formula] : [],
    keySpaces: [
      {id: 'component', label: 'pixel-major component', size: 9, indexSpace: '1d'},
      {id: 'word', label: 'packed word', size: 3, indexSpace: '1d'}
    ],
    choices: [{stage: 'representation', name: 'dimensions', value: 3, sweepable: false}],
    build: buildRasterEmbeddingQuantizeScenario
  });
}

defineLensScenario({
  id: SCENARIO_ID,
  element: 'GPURasterEmbeddingQuantize',
  title: 'GPURasterEmbeddingQuantize: synthetic-embedding-codec',
  rungs: TABLE_LENS_RUNGS,
  dataset: DATASET,
  load: async () => ({run})
});
