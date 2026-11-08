// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {getGPUPermutationParameterValues} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {GPUMultivariateLocalGearyPermutationTest} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {GPUSpatialEmpiricalBayesRates} from '../../../src/gpu-spatial-analysis/rate-smoothing/index';
import {
  addSpatialRateRecipe,
  GPUQuantileLISA
} from '../../../src/gpu-spatial-analysis/recipes/index';
import {
  planGPUSpatialWeightsChunks,
  type GPUChunkedSpatialWeights,
  type GPUSpatialWeights
} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createKnnWeights,
  createSeededRandom,
  type CPUSpatialWeights
} from '../permutation-inference/permutation-oracle';

/**
 * Parity of every chunk-aware weights consumer: the same dataset is run with one CSR and with row
 * chunks forced small by a tiny slot limit, and the outputs must agree. Per-row consumers are
 * exact (the slot order is identical); reductions are compared within f32 association tolerance.
 */

const ROW_COUNT = 180;
const NEIGHBOR_COUNT = 5;
const MAXIMUM_SLOTS_PER_CHUNK = 220;
/** Chunk plans: a few chunks, many chunks, and one row per chunk. */
const CHUNK_SLOT_LIMITS = [220, 60, 1];

function sliceWeights(weights: CPUSpatialWeights, rowStart: number, rowCount: number) {
  const begin = weights.offsets[rowStart];
  const end = weights.offsets[rowStart + rowCount];
  return {
    offsets: Uint32Array.from(
      weights.offsets.subarray(rowStart, rowStart + rowCount + 1),
      value => value - begin
    ),
    neighbors: weights.neighbors.slice(begin, Math.max(end, begin + 1)),
    weights: weights.weights.slice(begin, Math.max(end, begin + 1))
  };
}

type Scenario = {
  positions: Float64Array;
  values: Float32Array;
  secondValues: Float32Array;
  mask: Uint32Array;
  rowStandardized: boolean;
  weights: CPUSpatialWeights;
  chunkRowCounts: number[];
};

function createScenario(
  rowStandardized: boolean,
  maximumSlots = MAXIMUM_SLOTS_PER_CHUNK
): Scenario {
  const random = createSeededRandom(5);
  const positions = Float64Array.from({length: ROW_COUNT * 2}, () => random() * 100);
  const values = Float32Array.from(
    {length: ROW_COUNT},
    (_, index) => Math.sin(positions[index * 2] / 11) + random()
  );
  const secondValues = Float32Array.from({length: ROW_COUNT}, () => random());
  // Mask out about one row in nine so masked neighbors and islands are exercised.
  const mask = Uint32Array.from({length: ROW_COUNT}, (_, index) => (index % 9 === 4 ? 0 : 1));
  const weights = createKnnWeights(positions, NEIGHBOR_COUNT, rowStandardized);
  const counts = Array.from(
    {length: ROW_COUNT},
    (_, row) => weights.offsets[row + 1] - weights.offsets[row]
  );
  const plan = planGPUSpatialWeightsChunks(counts, maximumSlots);
  expect(plan.length).toBeGreaterThan(2);
  return {
    positions,
    values,
    secondValues,
    mask,
    rowStandardized,
    weights,
    chunkRowCounts: plan.map(chunk => chunk.rowCount)
  };
}

type Output = {
  name: string;
  kind: 'float32' | 'uint32';
  length: number;
  buffer: ReturnType<typeof createOutputBuffer>;
};
type Run = (context: {
  device: Device;
  graph: GPUCommandGraph;
  scenario: Scenario;
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  output: <Format extends 'float32' | 'uint32'>(
    name: string,
    kind: Format,
    length: number
  ) => GraphDataView<Format>;
  input: {
    (name: string, data: Float32Array): GraphDataView<'float32'>;
    (name: string, data: Uint32Array): GraphDataView<'uint32'>;
  };
}) => void;

async function runConsumer(
  device: Device,
  scenario: Scenario,
  isChunked: boolean,
  run: Run
): Promise<Record<string, number[]>> {
  const graph = new GPUCommandGraph(device, {id: 'chunked-consumer-parity'});
  const importCsr = (csr: CPUSpatialWeights, name: string, rowCount: number) => ({
    offsets: importGraphBuffer(
      graph,
      `${name}-offsets`,
      createInputBuffer(device, csr.offsets),
      'uint32',
      rowCount + 1
    ),
    neighbors: importGraphBuffer(
      graph,
      `${name}-neighbors`,
      createInputBuffer(device, csr.neighbors),
      'uint32',
      csr.neighbors.length
    ),
    weights: importGraphBuffer(
      graph,
      `${name}-weights`,
      createInputBuffer(device, csr.weights),
      'float32',
      csr.weights.length
    )
  });
  const information = {
    rowCount: ROW_COUNT,
    transform: scenario.rowStandardized ? ('row' as const) : ('binary' as const),
    symmetric: 'unknown' as const
  };
  let weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  if (isChunked) {
    let rowStart = 0;
    weights = {
      rowCount: ROW_COUNT,
      information,
      chunks: scenario.chunkRowCounts.map((rowCount, index) => {
        const chunk = {
          rowStart,
          rowCount,
          weights: importCsr(
            sliceWeights(scenario.weights, rowStart, rowCount),
            `chunk${index}`,
            rowCount
          )
        };
        rowStart += rowCount;
        return chunk;
      })
    };
  } else {
    weights = {...importCsr(scenario.weights, 'whole', ROW_COUNT), information};
  }
  const outputs: Output[] = [];
  run({
    device,
    graph,
    scenario,
    weights,
    output: (name, kind, length) => {
      const buffer = createOutputBuffer(device, length);
      outputs.push({name, kind, length, buffer});
      return importGraphBuffer(graph, name, buffer, kind, length);
    },
    input: ((name: string, data: Float32Array | Uint32Array) =>
      importGraphBuffer(
        graph,
        name,
        createInputBuffer(device, data),
        data instanceof Float32Array ? 'float32' : 'uint32',
        data.length
      )) as never
  });
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result: Record<string, number[]> = {};
  for (const output of outputs) {
    result[output.name] = Array.from(
      output.kind === 'float32'
        ? await readFloat32(output.buffer, output.length)
        : await readUint32(output.buffer, output.length)
    );
  }
  compiled.destroy();
  return result;
}

function expectNear(actual: number[], expected: number[], label: string, tolerance: number) {
  expect(actual.length, label).toBe(expected.length);
  actual.forEach((value, index) => {
    const reference = expected[index];
    if (Number.isNaN(reference)) {
      expect(Number.isNaN(value), `${label}[${index}] NaN`).toBe(true);
    } else {
      expect(Math.abs(value - reference), `${label}[${index}]`).toBeLessThanOrEqual(
        tolerance * Math.max(1, Math.abs(reference))
      );
    }
  });
}

async function expectParity(
  device: Device,
  scenario: Scenario,
  run: Run,
  tolerance: number,
  exactKeys: readonly string[] = []
) {
  const whole = await runConsumer(device, scenario, false, run);
  const chunked = await runConsumer(device, scenario, true, run);
  for (const key of Object.keys(whole)) {
    if (exactKeys.includes(key)) expect(chunked[key], key).toEqual(whole[key]);
    else expectNear(chunked[key], whole[key], key, tolerance);
  }
}

/** The same dataset under each chunk plan; the last plan has one row per chunk. */
const PLANS = CHUNK_SLOT_LIMITS.map(limit => createScenario(false, limit));

it('uses one-row chunks in the smallest plan', () => {
  expect(PLANS[PLANS.length - 1].chunkRowCounts.every(rowCount => rowCount === 1)).toBe(true);
});

function createRateInputs(scenario: Scenario) {
  const populations = Float32Array.from({length: ROW_COUNT}, (_, row) => 50 + ((row * 37) % 90));
  const events = Float32Array.from({length: ROW_COUNT}, (_, row) => (row * 13) % 11);
  // Exclude a non-positive population and a non-finite event so exclusion paths are exercised.
  populations[7] = 0;
  events[11] = Number.NaN;
  return {events, populations, mask: scenario.mask};
}

it('chunked spatial empirical-Bayes rates match the single CSR for every chunk plan', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scenario of PLANS) {
    const {events, populations, mask} = createRateInputs(scenario);
    await expectParity(
      device,
      scenario,
      ({graph, weights, output, input}) => {
        graph.add(
          new GPUSpatialEmpiricalBayesRates({
            events: input('events', events),
            populations: input('populations', populations),
            mask: input('mask', mask),
            weights,
            output: {
              spatialRates: output('spatialRates', 'float32', ROW_COUNT),
              smoothedRates: output('smoothedRates', 'float32', ROW_COUNT),
              priorMeans: output('priorMeans', 'float32', ROW_COUNT),
              priorVariances: output('priorVariances', 'float32', ROW_COUNT)
            }
          })
        );
      },
      1e-6
    );
  }
});

it('chunked spatial rate recipe matches the single CSR (islands and scale)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scenario of PLANS) {
    const {events, populations, mask} = createRateInputs(scenario);
    for (const islandPolicy of ['undefined', 'own-rate'] as const) {
      await expectParity(
        device,
        scenario,
        ({graph, weights, output, input}) => {
          addSpatialRateRecipe(graph, {
            events: input('events', events),
            populations: input('populations', populations),
            mask: input('mask', mask),
            weights,
            scale: 1000,
            islandPolicy,
            output: output('rates', 'float32', ROW_COUNT)
          });
        },
        1e-6
      );
    }
  }
});

it('chunked multivariate local Geary permutation test matches the single CSR', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const columns = 3;
  for (const [planIndex, scenario] of PLANS.entries()) {
    const interleaved = Float32Array.from({length: ROW_COUNT * columns}, (_, index) => {
      const row = Math.floor(index / columns);
      return scenario.values[row] * (1 + (index % columns)) + scenario.secondValues[row];
    });
    // The one-row-chunk plan compiles a kernel per chunk, so it runs fewer configurations.
    const configurations =
      planIndex === PLANS.length - 1
        ? ([['tiled', 7]] as const)
        : ([
            ['exact', undefined],
            ['tiled', 1],
            ['tiled', 7],
            ['tiled', 4096]
          ] as const);
    for (const [strategy, rowTileSize] of configurations) {
      await expectParity(
        device,
        scenario,
        ({graph, weights, output, input}) => {
          graph.add(
            new GPUMultivariateLocalGearyPermutationTest({
              weights,
              values: input('values', interleaved),
              variableCount: columns,
              mask: input('mask', scenario.mask),
              maximumPermutations: 49,
              seed: [11, 0],
              strategy,
              rowTileSize,
              permutationTileSize: 16,
              output: {
                statistic: output('statistic', 'float32', ROW_COUNT),
                pValues: output('pValues', 'float32', ROW_COUNT),
                classes: output('classes', 'uint32', ROW_COUNT)
              }
            })
          );
        },
        1e-6,
        ['classes', 'pValues']
      );
    }
  }
}, 300000);

it('chunked quantile LISA matches the single CSR', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scenario of PLANS) {
    await expectParity(
      device,
      scenario,
      ({graph, weights, output, input}) => {
        graph.add(
          new GPUQuantileLISA({
            values: input('values', scenario.values),
            mask: input('mask', scenario.mask),
            quantileParameters: input('quantileParameters', new Uint32Array([4, 1])),
            parameters: input(
              'parameters',
              getGPUPermutationParameterValues({
                seed: 314159,
                permutationCount: 49,
                significanceLevel: 0.1
              })
            ),
            maximumPermutations: 49,
            maximumNeighbors: 8,
            weights,
            output: {
              membership: output('membership', 'uint32', ROW_COUNT),
              boundaries: output('boundaries', 'float32', 2),
              validCount: output('validCount', 'uint32', 1),
              classes: output('classes', 'uint32', ROW_COUNT),
              invalidQuantile: output('invalidQuantile', 'uint32', 1),
              exceedances: output('exceedances', 'uint32', ROW_COUNT),
              pseudoPValues: output('pseudoPValues', 'float32', ROW_COUNT),
              observed: output('observed', 'float32', ROW_COUNT),
              significant: output('significant', 'uint32', ROW_COUNT),
              status: {overflow: output('overflow', 'uint32', 1)}
            }
          })
        );
      },
      1e-6,
      ['membership', 'boundaries', 'classes', 'exceedances', 'significant', 'overflow']
    );
  }
}, 300000);
