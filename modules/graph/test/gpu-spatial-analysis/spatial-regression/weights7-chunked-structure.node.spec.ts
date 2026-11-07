// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPULocalCollinearity} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-local-collinearity';
import {GPUSpatialErrorGM} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm';
import {GPUSpatialErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood';
import {GPUSpatialLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood';
import {GPUSpatialWeightsTraces} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {createWeights7TransientChunks} from '../spatial-weights/weights7-chunk-harness';

const ROW_COUNT = 80;
const PREDICTOR_COUNT = 2;
const MAXIMUM_POWER = 6;
const TILE_ROW_COUNT = 16;
const CHUNK_ROWS = [40, 25, 15];
// Tiles never span chunks: ceil(40 / 16) + ceil(25 / 16) + ceil(15 / 16) = 3 + 2 + 1.
const CHUNK_TILE_BASES = [0, 3, 5];
const CHUNKED_TILE_COUNT = 6;
const WHOLE_TILE_COUNT = 5;

function createContext(chunkRowCounts: number[] | undefined) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'weights7-structure'});
  const chunked = createWeights7TransientChunks(graph, chunkRowCounts ?? [ROW_COUNT]);
  const weights = chunkRowCounts ? chunked : chunked.chunks[0].weights;
  const view = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const regressionProps = {
    weights,
    predictors: view('predictors', 'float32', ROW_COUNT * PREDICTOR_COUNT),
    response: view('response', 'float32', ROW_COUNT),
    predictorCount: PREDICTOR_COUNT,
    maximumPower: MAXIMUM_POWER,
    tileRowCount: TILE_ROW_COUNT,
    traceMode: 'exact' as const
  };
  const regressionOutput = () => ({
    table: view('table', 'float32', (PREDICTOR_COUNT + 2) * 4),
    summary: view('summary', 'float32', 32),
    status: {code: view('status', 'uint32', 1)}
  });
  return {graph, weights, view, regressionProps, regressionOutput};
}

function getIds(nodes: readonly unknown[]) {
  const kernels = getGPUCommandNodeKernelSources(nodes);
  return {
    kernels,
    ids: kernels.map(kernel => kernel.id)
  };
}

it('lays traces tiles out per chunk and dispatches the weights kernels once per chunk', () => {
  const {graph, weights, view} = createContext(CHUNK_ROWS);
  const nodes = new GPUSpatialWeightsTraces({
    weights,
    maximumPower: MAXIMUM_POWER,
    mode: 'stochastic',
    probeCount: 8,
    tileRowCount: TILE_ROW_COUNT,
    output: {
      powerTraces: view('powerTraces', 'float32', MAXIMUM_POWER),
      powerSums: view('powerSums', 'float32', MAXIMUM_POWER)
    }
  }).getCommandNodes(graph);
  const {kernels, ids} = getIds(nodes);
  for (const family of ['norm-tiles', 'scale-weights', 'sums-step-first', 'sums-step-2']) {
    expect(
      ids.filter(id => id.includes(family) && /chunk\d/.test(id)),
      family
    ).toHaveLength(CHUNK_ROWS.length);
  }
  expect(ids.filter(id => id.includes('norm-merge'))).toHaveLength(1);
  const normTiles = kernels.filter(kernel => kernel.id.includes('norm-tiles'));
  normTiles.forEach((kernel, index) => {
    expect(kernel.source).toContain(`const TILE_BASE: u32 = ${CHUNK_TILE_BASES[index]}u;`);
    expect(kernel.source).toContain(`const TILE_COUNT: u32 = ${CHUNKED_TILE_COUNT}u;`);
    expect(kernel.source).toContain(`const WEIGHTS_ROW_START: u32 = ${[0, 40, 65][index]}u;`);
  });
  expect(
    kernels.find(kernel => kernel.id.includes('norm-merge'))?.source,
    'merge loops over every chunk tile'
  ).toContain(`const TILE_COUNT: u32 = ${CHUNKED_TILE_COUNT}u;`);
  for (const {id, source} of kernels) {
    expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
  }
});

it('keeps the whole-CSR traces graph unsuffixed with the unchunked tile count', () => {
  const {graph, weights, view} = createContext(undefined);
  const nodes = new GPUSpatialWeightsTraces({
    weights,
    maximumPower: MAXIMUM_POWER,
    mode: 'exact',
    tileRowCount: TILE_ROW_COUNT,
    output: {
      powerTraces: view('powerTraces', 'float32', MAXIMUM_POWER),
      powerSums: view('powerSums', 'float32', MAXIMUM_POWER)
    }
  }).getCommandNodes(graph);
  const {kernels, ids} = getIds(nodes);
  expect(ids.filter(id => id.includes('chunk'))).toHaveLength(0);
  expect(ids.filter(id => id.includes('norm-tiles'))).toHaveLength(1);
  for (const {id, source} of kernels) {
    expect(source, id).not.toContain('WEIGHTS_ROW_START');
    expect(source, id).not.toContain('TILE_BASE');
  }
  expect(kernels.find(kernel => kernel.id.includes('norm-merge'))?.source).toContain(
    `const TILE_COUNT: u32 = ${WHOLE_TILE_COUNT}u;`
  );
});

it('rejects Gram traces and the homoskedastic GM estimator for chunked weights', () => {
  const {graph, weights, view} = createContext(CHUNK_ROWS);
  expect(() =>
    new GPUSpatialWeightsTraces({
      weights,
      maximumPower: MAXIMUM_POWER,
      includeGramTraces: true,
      output: {
        powerTraces: view('powerTraces', 'float32', MAXIMUM_POWER),
        powerSums: view('powerSums', 'float32', MAXIMUM_POWER),
        gramTraces: view('gram', 'float32', 4)
      }
    }).getCommandNodes(graph)
  ).toThrow();
  const context = createContext(CHUNK_ROWS);
  expect(
    () =>
      new GPUSpatialErrorGM({
        ...context.regressionProps,
        estimator: 'homoskedastic',
        output: {...context.regressionOutput(), summary: context.view('s', 'float32', 13)}
      })
  ).toThrow();
});

for (const [name, create] of [
  [
    'spatial lag ML',
    (context: ReturnType<typeof createContext>) =>
      new GPUSpatialLagMaximumLikelihood({
        ...context.regressionProps,
        output: context.regressionOutput()
      })
  ],
  [
    'spatial error ML',
    (context: ReturnType<typeof createContext>) =>
      new GPUSpatialErrorMaximumLikelihood({
        ...context.regressionProps,
        output: context.regressionOutput()
      })
  ],
  [
    'Kelejian-Prucha error GM',
    (context: ReturnType<typeof createContext>) =>
      new GPUSpatialErrorGM({
        ...context.regressionProps,
        estimator: 'kelejian-prucha',
        output: {...context.regressionOutput(), summary: context.view('s', 'float32', 13)}
      })
  ]
] as const) {
  it(`${name} dispatches its weights kernels once per chunk and the whole CSR once`, () => {
    const chunkedContext = createContext(CHUNK_ROWS);
    const {ids, kernels} = getIds(create(chunkedContext).getCommandNodes(chunkedContext.graph));
    const chunkedFamilies = new Set(
      ids.filter(id => /-chunk\d$/.test(id)).map(id => id.replace(/-chunk\d$/, ''))
    );
    expect(chunkedFamilies.size).toBeGreaterThan(3);
    for (const family of chunkedFamilies) {
      expect(
        ids.filter(id => id.startsWith(`${family}-chunk`)),
        family
      ).toHaveLength(CHUNK_ROWS.length);
      expect(ids, `${family} has no unsuffixed twin`).not.toContain(family);
    }
    for (const {id, source} of kernels) {
      expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
    }

    const wholeContext = createContext(undefined);
    const whole = getIds(create(wholeContext).getCommandNodes(wholeContext.graph));
    expect(whole.ids.filter(id => /chunk\d/.test(id))).toHaveLength(0);
    for (const {id, source} of whole.kernels) {
      expect(source, id).not.toContain('WEIGHTS_ROW_START');
      expect(source, id).not.toContain('TILE_BASE');
    }
  });
}

it('dispatches local collinearity once per chunk', () => {
  const context = createContext(CHUNK_ROWS);
  const {graph, weights, view} = context;
  const nodes = new GPULocalCollinearity({
    weights,
    predictors: view('p', 'float32', ROW_COUNT * 3),
    predictorCount: 3,
    output: {
      varianceInflationFactors: view('vif', 'float32', ROW_COUNT * 3),
      conditionNumbers: view('condition', 'float32', ROW_COUNT),
      designConditionNumbers: view('design', 'float32', ROW_COUNT),
      varianceDecomposition: view('decomposition', 'float32', ROW_COUNT * 4),
      status: view('status', 'uint32', ROW_COUNT)
    }
  }).getCommandNodes(graph);
  const {ids} = getIds(nodes);
  expect(ids.filter(id => id.includes('-fit-chunk'))).toHaveLength(3);
  expect(ids.filter(id => id.includes('-variance-decomposition-chunk'))).toHaveLength(3);
});
