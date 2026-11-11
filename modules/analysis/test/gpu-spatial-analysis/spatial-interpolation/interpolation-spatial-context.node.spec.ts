// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {GPUConditionalGaussianSimulation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation';
import {GPULocalKriging} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {GPULocalRBF} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf';
import {GPUNaturalNeighbor} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-natural-neighbor';
import {GPUVecchiaFisherScoring} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-fisher-scoring';
import {GPUVecchiaLogLikelihood} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-log-likelihood';

// The context is resolved before any other prop is read, so these props carry only the context.
const CONTRIBUTORS: [string, (spatialContext: GPUSpatialContext) => unknown][] = [
  ['local-kriging', spatialContext => new GPULocalKriging({spatialContext} as never)],
  ['local-rbf', spatialContext => new GPULocalRBF({spatialContext} as never)],
  ['natural-neighbor', spatialContext => new GPUNaturalNeighbor({spatialContext} as never)],
  [
    'conditional-gaussian-simulation',
    spatialContext => new GPUConditionalGaussianSimulation({spatialContext} as never)
  ],
  [
    'vecchia-fisher-scoring',
    spatialContext => new GPUVecchiaFisherScoring({spatialContext} as never)
  ],
  [
    'vecchia-log-likelihood',
    spatialContext => new GPUVecchiaLogLikelihood({spatialContext} as never)
  ]
];

for (const [id, create] of CONTRIBUTORS) {
  it(`${id} takes a declared spatialContext and accepts only native planar distances`, () => {
    expect(() => create({coordinateSpace: 'longitude-latitude', metric: 'great-circle'})).toThrow(
      new RegExp(`^${id} does not support longitude-latitude coordinates`)
    );
    expect(() => create({coordinateSpace: 'planar', metric: 'none'})).toThrow(
      new RegExp(`^${id} planar positions need the native metric`)
    );
    // A planar native context passes the context check; the empty props then fail elsewhere.
    expect(() => create({coordinateSpace: 'planar', metric: 'native'})).not.toThrow(
      /does not support|native metric/
    );
  });
}
