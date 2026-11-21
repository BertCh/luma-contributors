// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {GPUConditionalGaussianSimulation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation';
import {GPULocalKriging} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {GPULocalRBF} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf';
import {GPUNaturalNeighbor} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-natural-neighbor';
import {GPUVecchiaFisherScoring} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-fisher-scoring';
import {GPUVecchiaLogLikelihood} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-log-likelihood';

const covariance = {model: 'exponential' as const, sill: 1, range: 3, nugget: 0.1};

// Valid graph ports isolate the spatial contract from earlier port-shape validation.
function createViews(graph: GPUCommandGraph) {
  return {
    positions: createTransientView(graph, 'positions', 'float32x2', 1),
    observations: createTransientView(graph, 'observations', 'float32x4', 1),
    values: createTransientView(graph, 'values', 'float32', 1),
    queries: createTransientView(graph, 'queries', 'float32x2', 1),
    parameters: createTransientView(graph, 'parameters', 'float32', 8),
    neighborhoodOffsets: createTransientView(graph, 'offsets', 'uint32', 2),
    neighborhoods: createTransientView(graph, 'neighbors', 'uint32', 1),
    predictionOutput: {
      values: createTransientView(graph, 'predictions', 'float32x4', 1),
      status: createTransientView(graph, 'prediction-status', 'uint32', 1)
    },
    simulationOutput: {
      values: createTransientView(graph, 'draws', 'float32', 1),
      moments: createTransientView(graph, 'moments', 'float32x2', 1),
      status: createTransientView(graph, 'draw-status', 'uint32', 1)
    },
    likelihoodOutput: {
      conditionals: createTransientView(graph, 'conditionals', 'float32x4', 1),
      conditionalStatus: createTransientView(graph, 'conditional-status', 'uint32', 1),
      siteTerms: createTransientView(graph, 'likelihood-terms', 'float32', 1),
      siteStatus: createTransientView(graph, 'likelihood-sites', 'uint32', 1),
      logLikelihood: createTransientView(graph, 'likelihood', 'float32', 2),
      status: createTransientView(graph, 'likelihood-status', 'uint32', 3)
    },
    fisherOutput: {
      siteTerms: createTransientView(graph, 'fisher-terms', 'float32', 10),
      siteStatus: createTransientView(graph, 'fisher-sites', 'uint32', 1),
      totals: createTransientView(graph, 'fisher-totals', 'float32', 20),
      status: createTransientView(graph, 'fisher-status', 'uint32', 3)
    }
  };
}

const CONTRIBUTORS: [
  string,
  (spatialContext: GPUSpatialContext, views: ReturnType<typeof createViews>) => unknown
][] = [
  [
    'local-kriging',
    (spatialContext, views) =>
      new GPULocalKriging({
        spatialContext,
        positions: views.positions,
        observations: views.observations,
        supports: views.queries,
        covariance,
        mode: 'simple',
        neighborCount: 1,
        output: views.predictionOutput
      })
  ],
  [
    'local-rbf',
    (spatialContext, views) =>
      new GPULocalRBF({
        spatialContext,
        positions: views.positions,
        values: views.values,
        queries: views.queries,
        kernel: 'gaussian',
        degree: 0,
        neighborCount: 1,
        output: views.predictionOutput
      })
  ],
  [
    'natural-neighbor',
    (spatialContext, views) =>
      new GPUNaturalNeighbor({
        spatialContext,
        positions: views.positions,
        values: views.values,
        queries: views.queries,
        output: views.predictionOutput
      })
  ],
  [
    'conditional-gaussian-simulation',
    (spatialContext, views) =>
      new GPUConditionalGaussianSimulation({
        spatialContext,
        positions: views.positions,
        observations: views.observations,
        queries: views.queries,
        covariance,
        seed: 1,
        replicateCount: 1,
        output: views.simulationOutput
      })
  ],
  [
    'vecchia-fisher-scoring',
    (spatialContext, views) =>
      new GPUVecchiaFisherScoring({
        spatialContext,
        positions: views.positions,
        observations: views.observations,
        parameters: views.parameters,
        neighborhoodOffsets: views.neighborhoodOffsets,
        neighborhoods: views.neighborhoods,
        neighborCount: 1,
        output: views.fisherOutput
      })
  ],
  [
    'vecchia-log-likelihood',
    (spatialContext, views) =>
      new GPUVecchiaLogLikelihood({
        spatialContext,
        positions: views.positions,
        observations: views.observations,
        parameters: views.parameters,
        neighborhoodOffsets: views.neighborhoodOffsets,
        neighborhoods: views.neighborhoods,
        neighborCount: 1,
        output: views.likelihoodOutput
      })
  ]
];

for (const [id, create] of CONTRIBUTORS) {
  it(`${id} takes a declared spatialContext and accepts only native planar distances`, () => {
    const device = createNullWebGPUDevice();
    const construct = (spatialContext: GPUSpatialContext) =>
      create(spatialContext, createViews(new GPUCommandGraph(device)));
    try {
      expect(() =>
        construct({coordinateSpace: 'longitude-latitude', metric: 'great-circle'})
      ).toThrow(new RegExp(`^${id} does not support longitude-latitude coordinates`));
      expect(() => construct({coordinateSpace: 'planar', metric: 'none'})).toThrow(
        new RegExp(`^${id} planar positions need the native metric`)
      );
      expect(() => construct({coordinateSpace: 'planar', metric: 'native'})).not.toThrow();
    } finally {
      device.destroy();
    }
  });
}
