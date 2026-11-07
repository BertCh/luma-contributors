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
import {GPUQuantileLISA} from '../../../src/gpu-spatial-analysis/recipes/index';
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
  createSeededRandom,
  type CPUSpatialWeights
} from '../permutation-inference/permutation-oracle';

/**
 * Chunked consumers with REAL self slots: distance-band weights that include the focus row as its
 * own neighbor (`includeSelf`), unlike kNN weights, which have no self slot. The same data is run
 * with one CSR and with chunks and the outputs must agree.
 */

const ROW_COUNT = 150;
const RADIUS = 14;
const CHUNK_SLOT_LIMITS = [260, 70];

function createSelfSlotWeights(positions: Float64Array): CPUSpatialWeights {
  const offsets = new Uint32Array(ROW_COUNT + 1);
  const neighbors: number[] = [];
  for (let row = 0; row < ROW_COUNT; row++) {
    for (let other = 0; other < ROW_COUNT; other++) {
      const dx = positions[row * 2] - positions[other * 2];
      const dy = positions[row * 2 + 1] - positions[other * 2 + 1];
      // Ascending IDs within the row; `other === row` is the self slot (distance 0).
      if (Math.hypot(dx, dy) <= RADIUS) neighbors.push(other);
    }
    offsets[row + 1] = neighbors.length;
  }
  return {
    offsets,
    neighbors: Uint32Array.from(neighbors),
    weights: new Float32Array(neighbors.length).fill(1)
  };
}

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

function createScenario() {
  const random = createSeededRandom(31);
  const positions = Float64Array.from({length: ROW_COUNT * 2}, () => random() * 100);
  const values = Float32Array.from(
    {length: ROW_COUNT},
    (_, index) => Math.sin(positions[index * 2] / 9) + random()
  );
  const secondValues = Float32Array.from({length: ROW_COUNT}, () => random());
  const mask = Uint32Array.from({length: ROW_COUNT}, (_, index) => (index % 11 === 5 ? 0 : 1));
  const weights = createSelfSlotWeights(positions);
  const counts = Array.from(
    {length: ROW_COUNT},
    (_, row) => weights.offsets[row + 1] - weights.offsets[row]
  );
  return {values, secondValues, mask, weights, counts};
}

type Output = {
  name: string;
  kind: 'float32' | 'uint32';
  length: number;
  buffer: ReturnType<typeof createOutputBuffer>;
};

async function runConsumer(
  device: Device,
  scenario: ReturnType<typeof createScenario>,
  chunkRowCounts: readonly number[] | undefined,
  run: (context: {
    graph: GPUCommandGraph;
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
  }) => void
): Promise<Record<string, number[]>> {
  const graph = new GPUCommandGraph(device, {id: 'weights7-self-slot'});
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
    transform: 'binary' as const,
    symmetric: 'unknown' as const,
    hasSelfWeights: true
  };
  let weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  if (chunkRowCounts) {
    let rowStart = 0;
    weights = {
      rowCount: ROW_COUNT,
      information,
      chunks: chunkRowCounts.map((rowCount, index) => {
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
    graph,
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
    if (value === reference) return;
    if (Number.isNaN(reference)) {
      expect(Number.isNaN(value), `${label}[${index}] NaN`).toBe(true);
    } else {
      expect(Math.abs(value - reference), `${label}[${index}]`).toBeLessThanOrEqual(
        tolerance * Math.max(1, Math.abs(reference))
      );
    }
  });
}

it('the fixture has self slots in every row', () => {
  const scenario = createScenario();
  for (let row = 0; row < ROW_COUNT; row++) {
    const slots = scenario.weights.neighbors.subarray(
      scenario.weights.offsets[row],
      scenario.weights.offsets[row + 1]
    );
    expect(slots.includes(row)).toBe(true);
  }
  expect(Math.max(...scenario.counts)).toBeLessThanOrEqual(32);
});

it('chunked multivariate local Geary with self slots matches the single CSR', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scenario = createScenario();
  const columns = 3;
  const interleaved = Float32Array.from({length: ROW_COUNT * columns}, (_, index) => {
    const row = Math.floor(index / columns);
    return scenario.values[row] * (1 + (index % columns)) + scenario.secondValues[row];
  });
  const run =
    (strategy: 'exact' | 'tiled') =>
    ({graph, weights, output, input}: Parameters<Parameters<typeof runConsumer>[3]>[0]) => {
      graph.add(
        new GPUMultivariateLocalGearyPermutationTest({
          weights,
          values: input('values', interleaved),
          variableCount: columns,
          mask: input('mask', scenario.mask),
          maximumPermutations: 49,
          seed: [11, 0],
          strategy,
          rowTileSize: strategy === 'tiled' ? 7 : undefined,
          permutationTileSize: 16,
          output: {
            statistic: output('statistic', 'float32', ROW_COUNT),
            pValues: output('pValues', 'float32', ROW_COUNT),
            classes: output('classes', 'uint32', ROW_COUNT)
          }
        })
      );
    };
  for (const strategy of ['exact', 'tiled'] as const) {
    const whole = await runConsumer(device, scenario, undefined, run(strategy));
    for (const limit of CHUNK_SLOT_LIMITS) {
      const plan = planGPUSpatialWeightsChunks(scenario.counts, limit);
      expect(plan.length).toBeGreaterThan(1);
      const chunked = await runConsumer(
        device,
        scenario,
        plan.map(chunk => chunk.rowCount),
        run(strategy)
      );
      expectNear(chunked.statistic, whole.statistic, `${strategy} statistic`, 1e-6);
      expect(chunked.pValues, `${strategy} pValues`).toEqual(whole.pValues);
      expect(chunked.classes, `${strategy} classes`).toEqual(whole.classes);
    }
  }
}, 300000);

it('chunked quantile LISA with self slots matches the single CSR', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scenario = createScenario();
  const run = ({
    graph,
    weights,
    output,
    input
  }: Parameters<Parameters<typeof runConsumer>[3]>[0]) => {
    graph.add(
      new GPUQuantileLISA({
        values: input('values', scenario.values),
        mask: input('mask', scenario.mask),
        quantileParameters: input('quantileParameters', new Uint32Array([4, 1])),
        parameters: input(
          'parameters',
          getGPUPermutationParameterValues({
            seed: 271828,
            permutationCount: 49,
            significanceLevel: 0.1
          })
        ),
        maximumPermutations: 49,
        maximumNeighbors: 32,
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
  };
  const whole = await runConsumer(device, scenario, undefined, run);
  expect(whole.overflow[0]).toBe(0);
  expect(whole.validCount[0], JSON.stringify(whole.boundaries)).toBeGreaterThan(0);
  expect(new Set(whole.classes).size).toBeGreaterThan(1);
  for (const limit of CHUNK_SLOT_LIMITS) {
    const plan = planGPUSpatialWeightsChunks(scenario.counts, limit);
    expect(plan.length).toBeGreaterThan(1);
    const chunked = await runConsumer(
      device,
      scenario,
      plan.map(chunk => chunk.rowCount),
      run
    );
    for (const key of Object.keys(whole)) {
      if (['boundaries', 'observed'].includes(key)) expectNear(chunked[key], whole[key], key, 1e-6);
      else expect(chunked[key], key).toEqual(whole[key]);
    }
  }
}, 300000);
