// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUPermutationParameterValues,
  GPUGlobalPermutationTest,
  GPU_GLOBAL_PERMUTATION_SUMMARY,
  GPU_PERMUTATION_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/permutation-inference';
import {
  getGPUSpatialAutocorrelationParameterValues,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH,
  GPUHotSpotAnalysis,
  GPULocalGeary,
  GPULocalMoran
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation';
import {
  GPUSpatialLag,
  GPUSpatialWeightsSummary,
  planGPUSpatialWeightsChunks,
  type GPUChunkedSpatialWeights,
  type GPUSpatialWeights
} from '../../../src/gpu-spatial-analysis/spatial-weights';
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

function createScenario(rowStandardized: boolean): Scenario {
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
  const plan = planGPUSpatialWeightsChunks(counts, MAXIMUM_SLOTS_PER_CHUNK);
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

const SCENARIOS = [false, true];

it('chunked Local Moran matches the single CSR (binary and row-standardized)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const rowStandardized of SCENARIOS) {
    await expectParity(
      device,
      createScenario(rowStandardized),
      ({graph, scenario, weights, output, input}) => {
        const parameters = new GPUParameterBuffer(graph.device, {
          id: 'parameters',
          format: 'float32',
          length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
          values: getGPUSpatialAutocorrelationParameterValues({significanceLevel: 0.05})
        });
        graph.add(
          new GPULocalMoran({
            weights,
            values: input('values', scenario.values),
            mask: input('mask', scenario.mask),
            parameters: parameters.importToGraph(graph),
            islands: 'exclude',
            output: {
              zScores: output('zScores', 'float32', ROW_COUNT),
              localI: output('localI', 'float32', ROW_COUNT),
              spatialLag: output('spatialLag', 'float32', ROW_COUNT),
              quadrants: output('quadrants', 'uint32', ROW_COUNT),
              pValues: output('pValues', 'float32', ROW_COUNT),
              neighborCounts: output('neighborCounts', 'uint32', ROW_COUNT),
              globalStatistics: output(
                'globalStatistics',
                'float32',
                GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH
              )
            }
          })
        );
      },
      1e-6,
      ['quadrants', 'neighborCounts']
    );
  }
});

it('chunked hotspot analysis (Gi*) matches the single CSR', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const rowStandardized of SCENARIOS) {
    await expectParity(
      device,
      createScenario(rowStandardized),
      ({graph, scenario, weights, output, input}) => {
        const parameters = new GPUParameterBuffer(graph.device, {
          id: 'parameters',
          format: 'float32',
          length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
          values: getGPUSpatialAutocorrelationParameterValues({significanceLevel: 0.05})
        });
        graph.add(
          new GPUHotSpotAnalysis({
            weights,
            values: input('values', scenario.values),
            mask: input('mask', scenario.mask),
            parameters: parameters.importToGraph(graph),
            output: {
              zScores: output('zScores', 'float32', ROW_COUNT),
              pValues: output('pValues', 'float32', ROW_COUNT),
              neighborCounts: output('neighborCounts', 'uint32', ROW_COUNT)
            }
          })
        );
      },
      1e-6,
      ['neighborCounts']
    );
  }
});

it('chunked Local Geary matches the single CSR', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  await expectParity(
    device,
    createScenario(false),
    ({graph, scenario, weights, output, input}) => {
      const parameters = new GPUParameterBuffer(graph.device, {
        id: 'parameters',
        format: 'float32',
        length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
        values: getGPUSpatialAutocorrelationParameterValues({significanceLevel: 0.05})
      });
      graph.add(
        new GPULocalGeary({
          weights,
          values: input('values', scenario.values),
          mask: input('mask', scenario.mask),
          parameters: parameters.importToGraph(graph),
          output: {statistic: output('statistic', 'float32', ROW_COUNT)}
        })
      );
    },
    1e-6
  );
});

it('chunked spatial lag matches the single CSR (masked, normalized, two columns)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const normalize of [false, true]) {
    await expectParity(
      device,
      createScenario(false),
      ({graph, scenario, weights, output, input}) => {
        const values = new Float32Array(ROW_COUNT * 2);
        scenario.values.forEach((value, row) => {
          values[row * 2] = value;
          values[row * 2 + 1] = scenario.secondValues[row];
        });
        graph.add(
          new GPUSpatialLag({
            weights,
            values: input('values', values),
            columnCount: 2,
            mask: input('mask', scenario.mask),
            normalize,
            output: output('lag', 'float32', ROW_COUNT * 2)
          })
        );
      },
      1e-6
    );
  }
});

it('chunked weights summary matches the single CSR (asymmetric kNN links)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  await expectParity(
    device,
    createScenario(false),
    ({weights, graph, output}) => {
      graph.add(
        new GPUSpatialWeightsSummary({
          weights,
          output: {
            statistics: output('statistics', 'float32', 3),
            counts: output('counts', 'uint32', 6),
            cardinality: output('cardinality', 'uint32', ROW_COUNT)
          }
        })
      );
    },
    1e-5,
    ['counts', 'cardinality']
  );
});

it('chunked global permutation test matches the single CSR', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const statistic of ['moran', 'geary'] as const) {
    const scenario = createScenario(true);
    const whole = await runConsumer(device, scenario, false, context =>
      runGlobal(context, statistic)
    );
    const chunked = await runConsumer(device, scenario, true, context =>
      runGlobal(context, statistic)
    );
    const R = GPU_GLOBAL_PERMUTATION_SUMMARY;
    // The block association differs per chunk, so compare within f32 tolerance; the counted
    // permutation total is exact and the p-value may differ by at most a few ties.
    expect(chunked['summary'][R.permutationCount]).toBe(whole['summary'][R.permutationCount]);
    expect(Math.abs(chunked['summary'][R.observed] - whole['summary'][R.observed])).toBeLessThan(
      1e-4
    );
    expect(
      Math.abs(chunked['summary'][R.pseudoPValue] - whole['summary'][R.pseudoPValue])
    ).toBeLessThan(0.03);
    expect(
      Math.abs(chunked['summary'][R.simulatedMean] - whole['summary'][R.simulatedMean])
    ).toBeLessThan(1e-3);
  }
});

function runGlobal(
  {graph, scenario, weights, output, input}: Parameters<Run>[0],
  statistic: 'moran' | 'geary'
) {
  const parameters = new GPUParameterBuffer(graph.device, {
    id: 'parameters',
    format: 'uint32',
    length: GPU_PERMUTATION_PARAMETER_LENGTH,
    values: getGPUPermutationParameterValues({permutationCount: 99, seed: 7})
  });
  graph.add(
    new GPUGlobalPermutationTest({
      weights,
      values: input('values', scenario.values),
      mask: input('mask', scenario.mask),
      statistic,
      maximumPermutations: 99,
      parameters: parameters.importToGraph(graph),
      output: {summary: output('summary', 'float32', GPU_GLOBAL_PERMUTATION_SUMMARY.length)}
    })
  );
}
