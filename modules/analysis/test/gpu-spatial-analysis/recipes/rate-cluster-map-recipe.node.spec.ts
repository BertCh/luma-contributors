// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, test, vi} from 'vitest';
import {
  GPU_PERMUTATION_PARAMETER_LENGTH,
  GPULocalPermutationTest
} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {
  addRateClusterMapRecipe,
  type GPURateClusterMapRecipeProps
} from '../../../src/gpu-spatial-analysis/recipes/rate-cluster-map-recipe';
import {
  getGPUSpatialAutocorrelationParameterValues,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPULocalMoran
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {GPUEmpiricalBayesRates} from '../../../src/gpu-spatial-analysis/rate-smoothing/index';
import {GPUParameterBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {computeContiguityWeightsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-contiguity-weights-operation';
import {flattenPolygons, type OraclePolygons} from '../spatial-weights/spatial-weights-oracle';

const ROWS = 4;

test('F13 pair-only loss is separate from final neighbor capacity on three squares', () => {
  const layout = flattenPolygons(
    [0, 1, 2].map(x => [
      [
        [x, 0],
        [x + 1, 0],
        [x + 1, 1],
        [x, 1]
      ]
    ]) as OraclePolygons
  );
  const input = {...layout, featureOffsets: Uint32Array.of(0, 1, 2, 3)};
  const runs = [1, 32].map(pairCapacity =>
    computeContiguityWeightsOnCPU(input, {criterion: 'queen', capacity: 16, pairCapacity})
  );
  expect(Array.from(runs[0].overflow)).toEqual([0]);
  expect(Array.from(runs[0].candidateOverflow)).toEqual([1]);
  expect(runs[0].requiredCount[0]).toBeLessThanOrEqual(4);
  expect(Array.from(runs[1].overflow)).toEqual([0]);
  expect(Array.from(runs[1].candidateOverflow)).toEqual([0]);
  expect(Array.from(runs[1].requiredCount)).toEqual([4]);
  const finalLoss = computeContiguityWeightsOnCPU(input, {
    criterion: 'queen',
    capacity: 3,
    pairCapacity: 32
  });
  expect(Array.from(finalLoss.overflow)).toEqual([1]);
  expect(Array.from(finalLoss.candidateOverflow)).toEqual([0]);
  expect(Array.from(finalLoss.requiredCount)).toEqual([4]);
});

test('F13 preserves caller candidate loss and publishes it in the weights stage', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const candidateOverflow = createTransientView(graph, 'candidate-overflow', 'uint32', 1);
  const props = createProps(graph);
  try {
    const result = addRateClusterMapRecipe(graph, {
      ...props,
      scratch: {weights: {status: {candidateOverflow}}}
    });
    expect(result.weights.status.candidateOverflow).toBe(candidateOverflow);
    expect(
      result.status.stages.find(stage => stage.stage === 'weights')?.status.candidateOverflow
    ).toBe(candidateOverflow);
  } finally {
    props.parameters.destroy();
    graph.device.destroy();
  }
});

function createParameterBuffer<Format extends 'float32' | 'uint32'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
): GPUParameterBuffer<Format> {
  return new GPUParameterBuffer(graph.device, {id: `test-${format}-parameters`, format, length});
}

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPURateClusterMapRecipeProps> = {}
): GPURateClusterMapRecipeProps {
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, `test-${id}`, format, length);
  return {
    events: view('events', 'float32', ROWS),
    populations: view('populations', 'float32', ROWS),
    polygons: {
      kind: 'polygons',
      positions: view('positions', 'float32x2', 16),
      ringOffsets: view('ring-offsets', 'uint32', ROWS + 1),
      polygonOffsets: view('polygon-offsets', 'uint32', ROWS + 1),
      featureOffsets: view('feature-offsets', 'uint32', ROWS + 1)
    },
    neighborCapacity: 16,
    parameters: createParameterBuffer(
      graph,
      'float32',
      GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
    ),
    palette: view('palette', 'uint32', 5),
    ...overrides
  };
}

describe('addRateClusterMapRecipe inference metadata', () => {
  test('permutation metadata is the permutation node getInferenceMetadata, not a copy', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'rate-cluster-permutation'});
    const result = addRateClusterMapRecipe(
      graph,
      createProps(graph, {
        multipleTesting: 'holm',
        settings: {significanceLevel: 0.01, permutationCount: 49, seed: 7},
        permutation: {
          parameters: createParameterBuffer(graph, 'uint32', GPU_PERMUTATION_PARAMETER_LENGTH),
          maximumPermutations: 99,
          alternative: 'greater'
        }
      })
    );
    const node = result.contributors.find(
      contributor => contributor instanceof GPULocalPermutationTest
    ) as GPULocalPermutationTest;
    expect(node).toBeDefined();
    const expected = node.getInferenceMetadata({
      seed: 7,
      permutationCount: 49,
      significanceLevel: 0.01
    });
    expect(result.inference).toEqual(expected);
    expect(result.inference).toMatchObject({
      alternative: 'greater',
      permutationCount: 49,
      seed: 7,
      multipleTesting: 'holm',
      significanceLevel: 0.01
    });
  });

  test('permutation metadata defaults: two-sided, maximumPermutations, seed 0', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'rate-cluster-defaults'});
    const result = addRateClusterMapRecipe(
      graph,
      createProps(graph, {
        permutation: {
          parameters: createParameterBuffer(graph, 'uint32', GPU_PERMUTATION_PARAMETER_LENGTH),
          maximumPermutations: 99
        }
      })
    );
    expect(result.inference).toMatchObject({
      alternative: 'two-sided',
      permutationCount: 99,
      multipleTesting: 'benjamini-hochberg',
      seed: 0
    });
  });

  test('a permutation count above maximumPermutations is refused', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'rate-cluster-too-many'});
    expect(() =>
      addRateClusterMapRecipe(
        graph,
        createProps(graph, {
          settings: {permutationCount: 100, seed: 1},
          permutation: {
            parameters: createParameterBuffer(graph, 'uint32', GPU_PERMUTATION_PARAMETER_LENGTH),
            maximumPermutations: 99
          }
        })
      )
    ).toThrow();
  });

  test('analytic metadata is the local Moran getInferenceMetadata', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'rate-cluster-analytic'});
    const result = addRateClusterMapRecipe(
      graph,
      createProps(graph, {
        multipleTesting: 'benjamini-yekutieli',
        settings: {significanceLevel: 0.1}
      })
    );
    const node = result.contributors.find(
      contributor => contributor instanceof GPULocalMoran
    ) as GPULocalMoran;
    expect(result.inference).toEqual(node.getInferenceMetadata({significanceLevel: 0.1}));
    expect(result.inference).toMatchObject({
      alternative: 'two-sided',
      permutationCount: 0,
      multipleTesting: 'benjamini-yekutieli',
      significanceLevel: 0.1
    });
  });
});

describe('addRateClusterMapRecipe settings (one source of truth, rule 19)', () => {
  function createBuffers(graph: GPUCommandGraph) {
    const parameters = new GPUParameterBuffer(graph.device, {
      id: 'rate-cluster-test-parameters',
      format: 'float32',
      length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
    });
    const permutationParameters = new GPUParameterBuffer(graph.device, {
      id: 'rate-cluster-test-permutation-parameters',
      format: 'uint32',
      length: GPU_PERMUTATION_PARAMETER_LENGTH
    });
    const writes: {parameters: number[][]; permutation: number[][]} = {
      parameters: [],
      permutation: []
    };
    vi.spyOn(parameters, 'write').mockImplementation(values => {
      writes.parameters.push(Array.from(values));
    });
    vi.spyOn(permutationParameters, 'write').mockImplementation(values => {
      writes.permutation.push(Array.from(values));
    });
    return {parameters, permutationParameters, writes};
  }

  test('packs the settings into its parameter buffers and describes the same settings', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'rate-cluster-settings'});
    const {parameters, permutationParameters, writes} = createBuffers(graph);
    const settings = {significanceLevel: 0.01, seed: 7, permutationCount: 49};
    const result = addRateClusterMapRecipe(
      graph,
      createProps(graph, {
        parameters,
        settings,
        permutation: {parameters: permutationParameters, maximumPermutations: 99}
      })
    );
    const node = result.contributors.find(
      contributor => contributor instanceof GPULocalPermutationTest
    ) as GPULocalPermutationTest;
    expect(result.inference).toEqual(node.getInferenceMetadata(settings));
    expect(writes.permutation).toEqual([Array.from(node.getParameterValues(settings))]);
    expect(writes.parameters).toEqual([
      Array.from(getGPUSpatialAutocorrelationParameterValues({significanceLevel: 0.01}))
    ]);

    // Per-frame change: both buffers and the metadata follow one object.
    const next = {significanceLevel: 0.05, seed: 11, permutationCount: 99};
    result.setSettings!(next);
    expect(result.inference).toEqual(node.getInferenceMetadata(next));
    expect(writes.permutation.at(-1)).toEqual(Array.from(node.getParameterValues(next)));
    // Invalid settings write nothing and keep the metadata.
    expect(() => result.setSettings!({permutationCount: 100})).toThrow();
    expect(writes.permutation.length).toBe(2);
    expect(result.inference.permutationCount).toBe(99);
  });

  test('accepts uint32 counts, converting them with one node', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'rate-cluster-uint32'});
    const added: string[] = [];
    // The empirical-Bayes stage converts uint32 columns itself, one node each.
    const getCommandNodes = GPUEmpiricalBayesRates.prototype.getCommandNodes;
    vi.spyOn(GPUEmpiricalBayesRates.prototype, 'getCommandNodes').mockImplementation(function (
      this: GPUEmpiricalBayesRates,
      target
    ) {
      const nodes = getCommandNodes.call(this, target);
      // Preparation repeats host expansion; only the caller graph executes these conversions.
      if (target === graph) added.push(...nodes.map(node => node.id));
      return nodes;
    });
    addRateClusterMapRecipe(
      graph,
      createProps(graph, {
        events: createTransientView(graph, 'uint32-events', 'uint32', ROWS),
        populations: createTransientView(graph, 'uint32-populations', 'uint32', ROWS)
      })
    );
    expect(added.filter(id => id.endsWith('-to-float32'))).toEqual([
      'rate-cluster-map-empirical-bayes-events-to-float32',
      'rate-cluster-map-empirical-bayes-populations-to-float32'
    ]);
  });
});
