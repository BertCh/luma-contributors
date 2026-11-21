// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {addHotSpotAnalysisRecipe} from '../../../src/gpu-spatial-analysis/recipes/index';
import {GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {GPUParameterBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {computeHotSpotOracle} from '../spatial-autocorrelation/spatial-autocorrelation-oracle';
import {ESDA_GI_STAR_LATTICE} from './hot-spot-esda-fixture';
import {computeLocalPermutationOracle} from '../permutation-inference/permutation-oracle';
import {GPU_PERMUTATION_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/permutation-inference/index';

it('the Gi* oracle reproduces esda G_Local(star=True) including a cold spot of zeros', () => {
  const fixture = ESDA_GI_STAR_LATTICE;
  const oracle = computeHotSpotOracle({
    weights: {
      offsets: Uint32Array.from(fixture.offsets),
      neighbors: Uint32Array.from(fixture.neighbors),
      weights: new Float32Array(fixture.neighbors.length).fill(1)
    },
    values: Float64Array.from(fixture.values) as unknown as Float32Array
  });
  for (const [row, expected] of fixture.zScores.entries()) {
    expect(oracle.zScores[row], `row ${row}`).toBeCloseTo(expected, 6);
  }
  expect(Math.min(...oracle.zScores)).toBeLessThan(-2.576);
});

function createProps(graph: GPUCommandGraph, overrides: Record<string, unknown> = {}) {
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length);
  return {
    source: {
      kind: 'points' as const,
      positions: view('positions', 'float32x2', 10),
      family: 'quadbin' as const,
      resolution: 6,
      studyArea: {
        kind: 'polygons' as const,
        positions: view('polygon', 'float32x2', 4),
        featureOffsets: view('features', 'uint32', 2),
        polygonOffsets: view('polygons', 'uint32', 2),
        ringOffsets: view('rings', 'uint32', 2),
        candidateCapacity: 64
      },
      cellCapacity: 32,
      ...overrides
    },
    parameters: new GPUParameterBuffer(graph.device, {
      id: 'hot-spot-parameters',
      format: 'float32',
      length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
    })
  };
}

it('addHotSpotAnalysisRecipe validates the study-area points source', () => {
  const graph = () => new GPUCommandGraph(createNullWebGPUDevice(), {id: 'hot-spot-validation'});
  const g1 = graph();
  expect(() => addHotSpotAnalysisRecipe(g1, {...createProps(g1), analyze: 'count'})).toThrow(
    /quadbin/
  );
  const g2 = graph();
  expect(() =>
    addHotSpotAnalysisRecipe(g2, {...createProps(g2), multipleTesting: 'sidak' as never})
  ).toThrow();
  const g3 = graph();
  expect(() => addHotSpotAnalysisRecipe(g3, {...createProps(g3), minimumNeighbors: 9})).toThrow(
    /exceeds the 8 neighbors/
  );
  const g4 = graph();
  expect(() => addHotSpotAnalysisRecipe(g4, {...createProps(g4, {neighborRings: 0})})).toThrow(
    /neighborRings/
  );
});

it('addHotSpotAnalysisRecipe states its inference and exposes the guard and study stages', () => {
  const g = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'hot-spot-metadata'});
  const result = addHotSpotAnalysisRecipe(g, createProps(g));
  expect(result.inference).toEqual({
    alternative: 'two-sided',
    permutationCount: 0,
    multipleTesting: 'benjamini-hochberg',
    testedRowsDefinition:
      'included rows with at least 8 active neighbors (islands excluded) and a defined z-score'
  });
  const stages = result.status.stages.map(stage => stage.stage);
  expect(stages).toEqual([
    'study-area',
    'study-table',
    'point-aggregation',
    'weights',
    'minimum-neighbors'
  ]);
  expect(result.status.stages.at(-1)!.status.invalidCount).toBeDefined();
  expect(result.weights.information?.provenance?.criterion).toBe('disk');
  expect(result.rowCount).toBe(32);
});

it('addHotSpotAnalysisRecipe accepts only a great-circle distance band over cell centers', () => {
  const graph = () => new GPUCommandGraph(createNullWebGPUDevice(), {id: 'hot-spot-band'});
  const band = (g: GPUCommandGraph, spatialContext?: unknown) => ({
    spatialContext,
    neighborSearchParameters: createTransientView(g, 'band-parameters', 'float32', 16),
    gridSize: [8, 8] as const,
    neighborCapacity: 256
  });
  // Omitted (planar) context: a radius in degrees is refused.
  const g1 = graph();
  expect(() =>
    addHotSpotAnalysisRecipe(g1, createProps(g1, {distanceBand: band(g1)}) as never)
  ).toThrow(/planar/);
  // Longitude/latitude with a planar-degree metric is refused too.
  const g2 = graph();
  expect(() =>
    addHotSpotAnalysisRecipe(
      g2,
      createProps(g2, {
        distanceBand: band(g2, {coordinateSpace: 'longitude-latitude', metric: 'native'})
      }) as never
    )
  ).toThrow(/metric/);
  // Exclusive with grid rings.
  const g3 = graph();
  expect(() =>
    addHotSpotAnalysisRecipe(
      g3,
      createProps(g3, {
        neighborRings: 2,
        distanceBand: band(g3, {coordinateSpace: 'longitude-latitude', metric: 'great-circle'})
      }) as never
    )
  ).toThrow(/exclusive/);
});

it('addHotSpotAnalysisRecipe labels distance-band weights binary or row', () => {
  const context = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
  const build = (rowStandardize?: boolean) => {
    const g = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'hot-spot-band-transform'});
    return addHotSpotAnalysisRecipe(
      g,
      createProps(g, {
        distanceBand: {
          spatialContext: context,
          neighborSearchParameters: createTransientView(g, 'band-parameters', 'float32', 16),
          gridSize: [8, 8] as const,
          neighborCapacity: 256,
          rowStandardize
        }
      }) as never
    );
  };
  expect(build().weights.information?.transform).toBe('binary');
  expect(build(false).weights.information?.transform).toBe('binary');
  const row = build(true);
  expect(row.weights.information?.transform).toBe('row');
  expect(row.weights.information?.provenance?.criterion).toBe('distance-band');
  expect(row.weights.status?.overflow).toBeDefined();
  expect((row.intermediates['weights'] as typeof row.weights).information?.transform).toBe('row');
});

it('the hot-spot distance band rejects a Lambert planar frame for geographic cell centers', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'hot-spot-lambert'});
  expect(() =>
    addHotSpotAnalysisRecipe(
      graph,
      createProps(graph, {
        distanceBand: {
          spatialContext: {
            coordinateSpace: 'planar',
            metric: 'native',
            units: 'meters',
            frame: {id: 'equal-area', kind: 'local-lambert-equal-area'}
          },
          neighborSearchParameters: createTransientView(graph, 'band-parameters', 'float32', 16),
          gridSize: [8, 8],
          neighborCapacity: 256
        }
      }) as never
    )
  ).toThrow(/distanceBand does not support planar coordinates/);
});

it('reports the distinct analytic and permutation testing families on a 5 by 5 queen lattice', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const offsets = [0];
  const neighbors: number[] = [];
  for (let row = 0; row < 25; row++) {
    for (let neighbor = 0; neighbor < 25; neighbor++) {
      if (
        neighbor !== row &&
        Math.abs((neighbor % 5) - (row % 5)) <= 1 &&
        Math.abs(Math.floor(neighbor / 5) - Math.floor(row / 5)) <= 1
      ) {
        neighbors.push(neighbor);
      }
    }
    offsets.push(neighbors.length);
  }
  const values = Float32Array.from({length: 25}, (_, row) => row + 1);
  const weights = {
    offsets: Uint32Array.from(offsets),
    neighbors: Uint32Array.from(neighbors),
    weights: new Float32Array(neighbors.length).fill(1)
  };
  const analytic = computeHotSpotOracle({weights, values, minimumNeighbors: 8});
  const permutation = computeLocalPermutationOracle({
    weights,
    values,
    statistic: 'localGStar',
    seed: 4,
    permutationCount: 199,
    significanceLevel: 0.05,
    maximumNeighbors: 32,
    multipleTesting: 'benjamini-hochberg'
  });
  expect(analytic.zScores.filter(Number.isFinite)).toHaveLength(9);
  expect(permutation.pseudoPValues.filter(Number.isFinite)).toHaveLength(25);
  expect(permutation.pseudoPValues[6]).toBeCloseTo(0.01, 6);
  expect(permutation.adjustedPValues[6]).toBeCloseTo(1 / 18, 6);
  expect(permutation.significant[6]).toBe(0);
  const parameters = new GPUParameterBuffer(device, {
    id: 'analytic-parameters',
    format: 'float32',
    length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
  });
  const permutationParameters = new GPUParameterBuffer(device, {
    id: 'permutation-parameters',
    format: 'uint32',
    length: GPU_PERMUTATION_PARAMETER_LENGTH
  });
  try {
    const result = addHotSpotAnalysisRecipe(graph, {
      source: {
        kind: 'lattice',
        values: createTransientView(graph, 'values', 'float32', 25),
        width: 5,
        height: 5,
        neighborCapacity: neighbors.length
      },
      parameters,
      settings: {seed: 4, permutationCount: 199},
      permutation: {
        parameters: permutationParameters,
        maximumPermutations: 199,
        multipleTesting: 'benjamini-hochberg'
      }
    });
    expect(result.inference.testedRowsDefinition).toBe(
      'included rows with at least 8 active neighbors (islands excluded) and a defined z-score'
    );
    expect(result.permutationInference?.testedRowsDefinition).toContain('1 to 32');
    result.setSettings({seed: 5, permutationCount: 19});
    expect(result.permutationInference?.testedRowsDefinition).toContain('1 to 32');
  } finally {
    parameters.destroy();
    permutationParameters.destroy();
    device.destroy();
  }
});

it('R6 describes the effective analytic minimum and excluded islands when minimumNeighbors is zero', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const parameters = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
  });
  const offsets = [0];
  const neighbors: number[] = [];
  for (let row = 0; row < 25; row++) {
    for (let neighbor = 0; neighbor < 25; neighbor++) {
      if (
        neighbor !== row &&
        Math.abs((neighbor % 5) - (row % 5)) <= 1 &&
        Math.abs(Math.floor(neighbor / 5) - Math.floor(row / 5)) <= 1
      )
        neighbors.push(neighbor);
    }
    offsets.push(neighbors.length);
  }
  const mask = new Uint32Array(25).fill(1);
  for (const row of [1, 5, 6]) mask[row] = 0;
  const oracle = computeHotSpotOracle({
    weights: {
      offsets: Uint32Array.from(offsets),
      neighbors: Uint32Array.from(neighbors),
      weights: new Float32Array(neighbors.length).fill(1)
    },
    values: Float32Array.from({length: 25}, (_, row) => row + 1),
    mask,
    minimumNeighbors: 1
  });
  expect(oracle.zScores[0]).toBeNaN();
  expect(oracle.zScores.filter(Number.isFinite)).toHaveLength(21);
  try {
    const result = addHotSpotAnalysisRecipe(graph, {
      source: {
        kind: 'lattice',
        values: createTransientView(graph, 'values', 'float32', 25),
        mask: createTransientView(graph, 'mask', 'uint32', 25),
        width: 5,
        height: 5,
        neighborCapacity: 200
      },
      parameters,
      minimumNeighbors: 0
    });
    expect(result.inference.testedRowsDefinition).toBe(
      'included rows with at least 1 active neighbor (islands excluded) and a defined z-score'
    );
    result.setSettings({significanceLevel: 0.1});
    expect(result.inference.testedRowsDefinition).toContain('at least 1 active neighbor');
  } finally {
    parameters.destroy();
    device.destroy();
  }
});

it('weighted points retain their exact sum-loss word on the stage and joined table', () => {
  for (const family of ['quadbin', 'h3'] as const) {
    for (const supplied of [false, true]) {
      const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'sum-loss'});
      const values = createTransientView(graph, 'weights', 'float32', 10);
      const sumOverflow = supplied
        ? createTransientView(graph, 'sum-overflow', 'uint32', 1)
        : undefined;
      const recipe = addHotSpotAnalysisRecipe(graph, {
        ...createProps(graph, {family, values}),
        scratch: {pointTable: {status: {sumOverflow}}}
      });
      const stage = recipe.status.stages.find(entry => entry.stage === 'point-aggregation');
      expect(stage?.status.sumOverflow).toBeDefined();
      expect(recipe.table?.status.sumOverflow).toBe(stage?.status.sumOverflow);
      if (supplied) {
        expect(stage?.status.sumOverflow).toBe(sumOverflow);
      }
    }
  }
});
