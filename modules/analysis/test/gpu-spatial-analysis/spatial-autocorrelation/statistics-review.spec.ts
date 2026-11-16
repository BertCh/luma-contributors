// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPULatticeWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-lattice-weights';
import {GPULocalMoran} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-local-moran';
import {getGPUSpatialAutocorrelationParameterValues} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/spatial-autocorrelation-parameters';
import type {Buffer, Device} from '@luma.gl/core';
import {
  GPUEmergingHotSpots,
  getGPUEmergingHotSpotsParameterValues
} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/index';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  getSwapOrNotRounds,
  getSwapOrNotPermutationIndex,
  getPermutationShuffleWGSL,
  PERMUTATION_SHUFFLE_FULL_RANGE_ROUNDS
} from '../../../src/analysis-core/kernels/permutation-shuffle';
import {PERMUTATION_RANDOM_WGSL} from '../../../src/analysis-core/kernels/permutation-random';
import {PHILOX_PURPOSES} from '../../../src/analysis-core/kernels/philox-random';
import {SPATIAL_STATISTICS_COUNTS_WGSL} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-statistics-counts-wgsl';
import {createSpatialStatisticsCollectiveNode} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-statistics-collective-node';
import {
  GPULocalPermutationTest,
  GPUGlobalPermutationTest,
  getGPUPermutationParameterValues
} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {GPUMultipleTestingAdjustment} from '../../../src/gpu-spatial-analysis/multiple-testing/index';
import type {
  GPUChunkedSpatialWeights,
  GPUSpatialWeights
} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GPUEmpiricalBayesRates} from '../../../src/gpu-spatial-analysis/rate-smoothing/index';
import {GPUSpatialEntropy} from '../../../src/gpu-spatial-analysis/spatial-entropy/index';
import {
  GPUSegregation,
  getGPUSegregationLayout,
  getGPUSegregationParameterValues
} from '../../../src/gpu-spatial-analysis/segregation/index';
import {GraphRig} from '../areal-interpolation/areal-interpolation-harness';
import {computeSegregationOracle} from '../segregation/segregation-oracle';
import {computeEmpiricalBayesOracle} from '../rate-smoothing/empirical-bayes-oracle';
import {
  computeLocalPermutationOracle,
  computeGlobalPermutationOracle,
  createWeights,
  type CPUSpatialWeights
} from '../permutation-inference/permutation-oracle';

it('uint32 count arithmetic preserves odd counts above 2^24 through final float32 rounding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const counts = Uint32Array.of(
    2 ** 24 + 1,
    30_000_001,
    0xffffefff,
    3,
    0xffffffff,
    2 ** 24 + 1,
    0xffffffff,
    30_000_001,
    2 ** 24 + 1
  );
  const values = Float32Array.of(
    2 ** 24,
    1,
    1234567,
    8,
    -0.125,
    2 ** -120,
    2 ** -149,
    2 ** 100,
    -16777216
  );
  const countBuffer = createInputBuffer(device, counts);
  const valueBuffer = createInputBuffer(device, values);
  const output = createOutputBuffer(device, counts.length * 4);
  let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
  try {
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'exact-count-arithmetic',
        operation: 'review',
        invocationCount: counts.length,
        declarations: SPATIAL_STATISTICS_COUNTS_WGSL,
        bindings: [
          {
            name: 'counts',
            view: importGraphBuffer(graph, 'counts', countBuffer, 'uint32'),
            type: 'u32',
            access: 'read'
          },
          {
            name: 'values',
            view: importGraphBuffer(graph, 'values', valueBuffer, 'float32'),
            type: 'f32',
            access: 'read'
          },
          {
            name: 'result',
            view: importGraphBuffer(graph, 'result', output, 'float32'),
            type: 'f32',
            access: 'read_write'
          }
        ],
        body: `let count = counts[countsOffset + index]; let value = values[valuesOffset + index];
  result[resultOffset + 4u * index] = divideBySpatialCount(value, count);
  result[resultOffset + 4u * index + 1u] = multiplyBySpatialCount(value, count);
  result[resultOffset + 4u * index + 2u] = divideSpatialCounts(count - 1u, count);
  result[resultOffset + 4u * index + 3u] = divideSpatialCountByValue(count, value);`
      })
    );
    compiled = graph.compile();
    submitGraph(device, compiled);
    const actual = await readFloat32(output);
    for (let index = 0; index < counts.length; index++) {
      expect(actual[4 * index]).toBe(Math.fround(values[index] / counts[index]));
      expect(actual[4 * index + 1]).toBe(Math.fround(values[index] * counts[index]));
      expect(actual[4 * index + 2]).toBe(Math.fround((counts[index] - 1) / counts[index]));
      expect(actual[4 * index + 3]).toBe(Math.fround(counts[index] / values[index]));
    }
    expect(actual[0]).not.toBe(1);
  } finally {
    compiled?.destroy();
    output.destroy();
    countBuffer.destroy();
    valueBuffer.destroy();
  }
});

for (const route of ['inline', 'materialized'] as const) {
  it(`global ${route} shuffle samples match the CPU bijection at 2^24, 2^24+1 and 30M`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const counts = [2 ** 24, 2 ** 24 + 1, 30_000_000];
    const probeCount = 6;
    const graph = new GPUCommandGraph(device);
    const domains = createInputBuffer(device, Uint32Array.from(counts));
    const output = createOutputBuffer(device, counts.length * probeCount);
    let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
    try {
      graph.add(
        createSpatialStatisticsCollectiveNode(graph, {
          id: `shuffle-${route}`,
          operation: 'review',
          invocationCount: counts.length * 256,
          bindings: [
            {
              name: 'domains',
              view: importGraphBuffer(graph, 'domains', domains, 'uint32'),
              type: 'u32',
              access: 'read'
            },
            {
              name: 'result',
              view: importGraphBuffer(graph, 'result', output, 'uint32'),
              type: 'u32',
              access: 'read_write'
            }
          ],
          declarations: `${PERMUTATION_RANDOM_WGSL}\n${getPermutationShuffleWGSL(PHILOX_PURPOSES.permutationShuffle.id, PERMUTATION_SHUFFLE_FULL_RANGE_ROUNDS)}`,
          body: `let domain = index / 256u; let count = domains[domainsOffset + domain];
  loadSwapOrNotRounds(vec2<u32>(7u, 0u), 3u, count, localInvocationIndex, 256u);
  workgroupBarrier();
  if (localInvocationIndex < 6u) {
    let probe = array<u32, 6>(0u, 1u, 16777215u, 16777216u, count - 2u, count - 1u);
    let position = min(probe[localInvocationIndex], count - 1u);
    result[resultOffset + domain * 6u + localInvocationIndex] = getSwapOrNotPermutationIndex(position, count);
  }`
        })
      );
      compiled = graph.compile();
      submitGraph(device, compiled);
      const actual = await readUint32(output);
      for (const [domain, count] of counts.entries()) {
        const rounds = getSwapOrNotRounds([7, 0], 3, count, PHILOX_PURPOSES.permutationShuffle.id);
        const probes = [0, 1, 2 ** 24 - 1, Math.min(2 ** 24, count - 1), count - 2, count - 1];
        expect(actual.slice(domain * probeCount, (domain + 1) * probeCount)).toEqual(
          probes.map(position => getSwapOrNotPermutationIndex(position, count, rounds))
        );
      }
    } finally {
      compiled?.destroy();
      domains.destroy();
      output.destroy();
    }
  });
}

type Scene = {weights: CPUSpatialWeights; values: Float32Array; mask: Uint32Array};
async function runChunkedPermutation(
  device: Device,
  scene: Scene,
  sizes: readonly number[],
  islands: 'keep' | 'exclude',
  globalRoute?: 'inline' | 'materialized'
) {
  const rows = scene.values.length;
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const input = <Format extends 'uint32' | 'float32'>(
    values: Uint32Array | Float32Array,
    format: Format
  ) => {
    const padded =
      values instanceof Uint32Array
        ? new Uint32Array(values.length + 1)
        : new Float32Array(values.length + 1);
    padded.set(values, 1);
    const buffer = createInputBuffer(device, padded);
    buffers.push(buffer);
    const whole = importGraphBuffer(graph, `input-${serial++}`, buffer, format);
    return graph.createDataView(whole.buffer, {format, length: values.length, byteOffset: 4});
  };
  const output = <Format extends 'uint32' | 'float32'>(length: number, format: Format) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, `output-${serial++}`, buffer, format, length)};
  };
  let rowStart = 0;
  const chunks = sizes.map(rowCount => {
    const begin = scene.weights.offsets[rowStart];
    const end = scene.weights.offsets[rowStart + rowCount];
    const weights = {
      offsets: input(
        Uint32Array.from(
          scene.weights.offsets.subarray(rowStart, rowStart + rowCount + 1),
          value => value - begin
        ),
        'uint32'
      ),
      neighbors: input(scene.weights.neighbors.slice(begin, end || begin + 1), 'uint32'),
      weights: input(scene.weights.weights.slice(begin, end || begin + 1), 'float32')
    };
    // Empty CSR chunks still bind one dummy slot; no offsets refer to it.
    if (begin === end) {
      weights.neighbors = input(Uint32Array.of(0), 'uint32');
      weights.weights = input(Float32Array.of(0), 'float32');
    }
    const chunk = {rowStart, rowCount, weights};
    rowStart += rowCount;
    return chunk;
  });
  const information = {rowCount: rows, transform: 'row' as const, symmetric: 'unknown' as const};
  const weights: GPUSpatialWeights | GPUChunkedSpatialWeights =
    sizes.length === 1
      ? {...chunks[0].weights, information}
      : {rowCount: rows, chunks, information};
  const analysisCount = output(1, 'uint32');
  const islandCount = output(1, 'uint32');
  const overflow = output(1, 'uint32');
  const observed = output(rows, 'float32');
  const pseudoPValues = output(rows, 'float32');
  const exceedances = output(rows, 'uint32');
  const summary = output(12, 'float32');
  const parameters = input(
    getGPUPermutationParameterValues({seed: 7, permutationCount: 7}),
    'uint32'
  );
  const common = {
    values: input(scene.values, 'float32'),
    mask: input(scene.mask, 'uint32'),
    weights,
    parameters,
    maximumPermutations: 7,
    islands
  };
  let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
  try {
    if (globalRoute)
      graph.add(
        new GPUGlobalPermutationTest({
          ...common,
          statistic: 'moran',
          permutationBatchSize: globalRoute === 'materialized' ? 3 : undefined,
          output: {
            summary: summary.view,
            analysisCount: analysisCount.view,
            islandCount: islandCount.view
          }
        })
      );
    else
      graph.add(
        new GPULocalPermutationTest({
          ...common,
          statistic: 'localMoran',
          rowTileSize: 17,
          output: {
            observed: observed.view,
            pseudoPValues: pseudoPValues.view,
            exceedances: exceedances.view,
            analysisCount: analysisCount.view,
            islandCount: islandCount.view,
            status: {overflow: overflow.view}
          }
        })
      );
    compiled = graph.compile();
    submitGraph(device, compiled);
    return {
      observed: await readFloat32(observed.buffer),
      pseudoPValues: await readFloat32(pseudoPValues.buffer),
      exceedances: await readUint32(exceedances.buffer),
      summary: await readFloat32(summary.buffer),
      analysisCount: (await readUint32(analysisCount.buffer))[0],
      islandCount: (await readUint32(islandCount.buffer))[0]
    };
  } finally {
    compiled?.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
}

it('packed and unequal CSR chunks with masks, islands, row standardization, unaligned views and 17-row tiles match independent oracles', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 769;
  const scene = {
    values: Float32Array.from({length: rows}, (_, row) => (row % 17) - 8),
    mask: Uint32Array.from({length: rows}, (_, row) => Number(row % 11 !== 0)),
    weights: createWeights(
      Array.from({length: rows}, (_, row) =>
        row % 13 === 0
          ? []
          : [
              [(row + 1) % rows, Math.fround(1 / 3)] as const,
              [(row + 3) % rows, Math.fround(2 / 3)] as const
            ]
      )
    )
  };
  for (const islands of ['keep', 'exclude'] as const) {
    const settings = {
      ...scene,
      islands,
      seed: 7,
      permutationCount: 7,
      significanceLevel: 0.05,
      rowStandardized: true
    };
    const local = computeLocalPermutationOracle({
      ...settings,
      statistic: 'localMoran',
      maximumNeighbors: 32
    });
    const global = computeGlobalPermutationOracle({...settings, statistic: 'moran'});
    for (const sizes of [[rows], [1, 255, 256, 257]]) {
      const actual = await runChunkedPermutation(device, scene, sizes, islands);
      expect(actual.analysisCount).toBe(global.analysisCount);
      expect(actual.islandCount).toBe(local.islandCount);
      for (let row = 0; row < rows; row++) {
        if (Number.isNaN(local.observed[row])) expect(actual.observed[row]).toBeNaN();
        else expect(actual.observed[row]).toBeCloseTo(local.observed[row], 4);
        // Distinct GPU mean rounding can move a tail tie by one draw.
        if (local.exceedances[row] === 0xffffffff) expect(actual.exceedances[row]).toBe(0xffffffff);
        else
          expect(Math.abs(actual.exceedances[row] - local.exceedances[row])).toBeLessThanOrEqual(1);
      }
      for (const route of ['inline', 'materialized'] as const) {
        const result = await runChunkedPermutation(device, scene, sizes, islands, route);
        expect(result.analysisCount).toBe(global.analysisCount);
        expect(result.islandCount).toBe(global.islandCount);
        expect(result.summary[0]).toBeCloseTo(global.observed, 4);
        expect(result.summary[1]).toBeCloseTo(global.pseudoPValue, 5);
      }
    }
  }
});

for (const local of [true, false]) {
  it(`${local ? 'local' : 'global'} Getis-Ord publishes an exact odd negative-row diagnostic above 2^24`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rows = 2 ** 24 + 1;
    const graph = new GPUCommandGraph(device);
    const values = createTransientView(graph, 'negative-values', 'float32', rows);
    const offsets = createTransientView(graph, 'negative-offsets', 'uint32', rows + 1);
    const neighbors = createTransientView(graph, 'negative-neighbors', 'uint32', rows);
    const weights = createTransientView(graph, 'negative-weights', 'float32', rows);
    const invalid = createOutputBuffer(device, 1);
    const count = createOutputBuffer(device, 1);
    const parameters = new GPUParameterBuffer(device, {
      id: 'negative-parameters',
      format: 'uint32',
      length: 4,
      values: getGPUPermutationParameterValues({seed: 7, permutationCount: 1})
    });
    let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
    try {
      graph.add(
        createWGSLKernelNode(graph, {
          id: 'negative-population',
          operation: 'review',
          invocationCount: rows + 1,
          bindings: [
            {name: 'values', view: values, type: 'f32', access: 'read_write'},
            {name: 'offsets', view: offsets, type: 'u32', access: 'read_write'},
            {name: 'neighbors', view: neighbors, type: 'u32', access: 'read_write'},
            {name: 'weights', view: weights, type: 'f32', access: 'read_write'}
          ],
          declarations: `const ROWS: u32 = ${rows}u;`,
          body: `offsets[offsetsOffset + index] = index;
  if (index < ROWS) {values[valuesOffset + index] = -1.0; neighbors[neighborsOffset + index] = (index + 1u) % ROWS; weights[weightsOffset + index] = 1.0;}`
        })
      );
      const common = {
        values,
        weights: {offsets, neighbors, weights},
        parameters: parameters.importToGraph(graph),
        maximumPermutations: 1
      };
      const status = {invalidCount: importGraphBuffer(graph, 'invalid', invalid, 'uint32', 1)};
      const analysisCount = importGraphBuffer(graph, 'count', count, 'uint32', 1);
      if (local)
        graph.add(
          new GPULocalPermutationTest({
            ...common,
            statistic: 'localG',
            output: {
              analysisCount,
              exceedances: createTransientView(graph, 'exceedances', 'uint32', rows),
              pseudoPValues: createTransientView(graph, 'p-values', 'float32', rows),
              status: {...status, overflow: createTransientView(graph, 'overflow', 'uint32', 1)}
            }
          })
        );
      else
        graph.add(
          new GPUGlobalPermutationTest({
            ...common,
            statistic: 'getisOrdG',
            output: {
              analysisCount,
              status,
              summary: createTransientView(graph, 'summary', 'float32', 12)
            }
          })
        );
      compiled = graph.compile();
      submitGraph(device, compiled);
      expect(await readUint32(invalid)).toEqual([rows]);
      expect(await readUint32(count)).toEqual([rows]);
    } finally {
      compiled?.destroy();
      parameters.destroy();
      invalid.destroy();
      count.destroy();
    }
  }, 180_000);
}

it('a genuine large BH family covers ranks around 2^24 and publishes the exact tested count', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 2 ** 24 + 257;
  const probes = [0, 2 ** 24 - 2, 2 ** 24 - 1, 2 ** 24, rows - 1];
  const graph = new GPUCommandGraph(device);
  const pValues = createTransientView(graph, 'p-values', 'float32', rows);
  const adjusted = createTransientView(graph, 'adjusted', 'float32', rows);
  const output = createOutputBuffer(device, probes.length);
  const count = createOutputBuffer(device, 1);
  let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
  try {
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'rank-population',
        operation: 'review',
        invocationCount: rows,
        bindings: [{name: 'values', view: pValues, type: 'f32', access: 'read_write'}],
        body: `values[valuesOffset + index] = f32(index + 1u) / ${2 * rows}.0;`
      })
    );
    graph.add(
      new GPUMultipleTestingAdjustment({
        pValues,
        method: 'benjamini-hochberg',
        output: {
          adjustedPValues: adjusted,
          testedCount: importGraphBuffer(graph, 'count', count, 'uint32', 1)
        }
      })
    );
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'rank-samples',
        operation: 'review',
        invocationCount: probes.length,
        bindings: [
          {name: 'values', view: adjusted, type: 'f32', access: 'read'},
          {
            name: 'result',
            view: importGraphBuffer(graph, 'result', output, 'float32'),
            type: 'f32',
            access: 'read_write'
          }
        ],
        declarations: `const PROBES = array<u32, ${probes.length}>(${probes.map(row => `${row}u`).join(', ')});`,
        body: 'result[resultOffset + index] = values[valuesOffset + PROBES[index]];'
      })
    );
    compiled = graph.compile();
    submitGraph(device, compiled);
    expect(await readUint32(count)).toEqual([rows]);
    // p_(k)=k/(2n), hence min_{j>=k} p_(j)*n/j = 1/2, independently of the scan implementation.
    for (const value of await readFloat32(output)) expect(value).toBeCloseTo(0.5, 5);
  } finally {
    compiled?.destroy();
    output.destroy();
    count.destroy();
  }
}, 180_000);

it('empirical-Bayes exact counts and invalid status match independent included-row rules', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device);
  const events = Float32Array.of(2, NaN, 8, 4, Infinity, 6);
  const populations = Float32Array.of(10, 20, 0, 8, 30, 12);
  const mask = Uint32Array.of(1, 1, 1, 0, 0, 1);
  const count = rig.output('uint32', 1);
  const invalid = rig.output('uint32', 1);
  const summary = rig.output('float32', 8);
  try {
    rig.run(
      new GPUEmpiricalBayesRates({
        events: rig.input(events, 'float32'),
        populations: rig.input(populations, 'float32'),
        mask: rig.input(mask, 'uint32'),
        output: {count: count.view, status: {invalidCount: invalid.view}, summary: summary.view}
      })
    );
    expect(await count.readUint32()).toEqual([2]);
    expect(await invalid.readUint32()).toEqual([2]);
    const expected = computeEmpiricalBayesOracle({events, populations, mask});
    const actual = await summary.readFloat32();
    expect(actual[3]).toBeCloseTo(expected.pooledRate, 6);
    expect(actual[5]).toBeCloseTo(expected.priorVariance, 6);
  } finally {
    rig.destroy();
  }
});

it('empirical-Bayes publishes an odd included-row count above 2^24', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const rows = 2 ** 24 + 1;
  const events = createTransientView(graph, 'rate-events', 'float32', rows);
  const populations = createTransientView(graph, 'rate-populations', 'float32', rows);
  const output = createOutputBuffer(device, 1);
  let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
  try {
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'rate-population',
        operation: 'review',
        invocationCount: rows,
        bindings: [
          {name: 'events', view: events, type: 'f32', access: 'read_write'},
          {name: 'populations', view: populations, type: 'f32', access: 'read_write'}
        ],
        body: 'events[eventsOffset + index] = 1.0; populations[populationsOffset + index] = 2.0;'
      })
    );
    graph.add(
      new GPUEmpiricalBayesRates({
        events,
        populations,
        output: {includedCount: importGraphBuffer(graph, 'exact-rate-count', output, 'uint32', 1)}
      })
    );
    compiled = graph.compile();
    submitGraph(device, compiled);
    expect(await readUint32(output)).toEqual([rows]);
  } finally {
    compiled?.destroy();
    output.destroy();
  }
}, 180_000);

it('spatial entropy publishes exact integer counters and closed-form category entropies', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device);
  const summary = rig.output('float32', 8);
  const counts = rig.output('uint32', 2);
  const pairs = rig.output('uint32', 4);
  try {
    // A four-row cycle with alternating categories: four ordered cross-category links.
    rig.run(
      new GPUSpatialEntropy({
        categoryCount: 2,
        categories: rig.input(Uint32Array.of(0, 1, 0, 1), 'uint32'),
        weights: {
          offsets: rig.input(Uint32Array.of(0, 1, 2, 3, 4), 'uint32'),
          neighbors: rig.input(Uint32Array.of(1, 2, 3, 0), 'uint32'),
          weights: rig.input(Float32Array.of(1, 1, 1, 1), 'float32')
        },
        output: {summary: summary.view, counts: counts.view, pairCounts: pairs.view}
      })
    );
    expect(await counts.readUint32()).toEqual([4, 4]);
    expect(await pairs.readUint32()).toEqual([0, 2, 2, 0]);
    const actual = await summary.readFloat32();
    for (const field of [0, 1, 2, 3, 4]) expect(actual[field]).toBeCloseTo(Math.log(2), 6);
    expect(actual[5]).toBeCloseTo(1, 6);
  } finally {
    rig.destroy();
  }
});

it('segregation runtime parameters update self weights and Atkinson without rebuilding the graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device);
  const counts = Float32Array.of(9, 1, 2, 8, 6, 4);
  const weights = {offsets: [0, 1, 2, 3], neighbors: [1, 2, 0], weights: [1, 1, 1]};
  const parameters = new GPUParameterBuffer(device, {
    id: 'segregation-runtime',
    format: 'float32',
    length: 2,
    values: getGPUSegregationParameterValues()
  });
  const output = rig.output('float32', getGPUSegregationLayout(2).stride);
  try {
    rig.run(
      new GPUSegregation({
        groupCount: 2,
        groupCounts: rig.input(counts, 'float32'),
        scales: [
          {
            offsets: rig.input(Uint32Array.from(weights.offsets), 'uint32'),
            neighbors: rig.input(Uint32Array.from(weights.neighbors), 'uint32'),
            weights: rig.input(Float32Array.from(weights.weights), 'float32')
          }
        ],
        parameters: parameters.importToGraph(rig.graph),
        output: {indices: output.view}
      })
    );
    for (const [selfWeight, atkinsonB] of [
      [1, 0.5],
      [2, 0.25]
    ]) {
      parameters.write(
        getGPUSegregationParameterValues({selfWeight, atkinsonInequalityAversion: atkinsonB})
      );
      rig.resubmit();
      const actual = await output.readFloat32();
      const expected = computeSegregationOracle({
        unitCount: 3,
        groupCount: 2,
        counts,
        weights,
        selfWeight,
        atkinsonB
      }).indices;
      for (let field = 0; field < expected.length; field++)
        expect(actual[field]).toBeCloseTo(expected[field], 4);
    }
  } finally {
    rig.destroy();
    parameters.destroy();
  }
});

it('emerging hot spots retains an odd bin count above 2^24 through moments and sampled Gi* values', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const rows = 2 ** 24 + 5;
  const slices = 3;
  const cells = rows / slices;
  const values = createTransientView(graph, 'large-emerging-values', 'float32', rows);
  const giZScores = createTransientView(graph, 'large-emerging-gi', 'float32', rows);
  const count = createOutputBuffer(device, 1);
  const moments = createOutputBuffer(device, 4);
  const samples = createOutputBuffer(device, 5);
  const probes = [0, 1, 2 ** 24 - 1, 2 ** 24, rows - 1];
  const parameters = new GPUParameterBuffer(device, {
    id: 'large-emerging-parameters',
    format: 'float32',
    length: 8,
    values: getGPUEmergingHotSpotsParameterValues({
      radius: 0,
      temporalWindow: 0,
      multipleTesting: 'none'
    })
  });
  let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
  try {
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'large-emerging-population',
        operation: 'review',
        invocationCount: rows,
        bindings: [{name: 'values', view: values, type: 'f32', access: 'read_write'}],
        body: 'values[valuesOffset + index] = select(1.0, -1.0, (index & 1u) != 0u);'
      })
    );
    graph.add(
      new GPUEmergingHotSpots({
        values,
        gridWidth: cells,
        gridHeight: 1,
        sliceCount: slices,
        maximumRadius: 0,
        parameters: parameters.importToGraph(graph),
        multipleTesting: 'none',
        output: {
          giZScores,
          includedCount: importGraphBuffer(graph, 'large-emerging-count', count, 'uint32', 1),
          globalStatistics: importGraphBuffer(
            graph,
            'large-emerging-moments',
            moments,
            'float32',
            4
          ),
          trendZ: createTransientView(graph, 'large-trend-z', 'float32', cells),
          trendP: createTransientView(graph, 'large-trend-p', 'float32', cells),
          trendS: createTransientView(graph, 'large-trend-s', 'sint32', cells),
          category: createTransientView(graph, 'large-category', 'uint32', cells),
          hotSliceCount: createTransientView(graph, 'large-hot-slices', 'uint32', cells),
          coldSliceCount: createTransientView(graph, 'large-cold-slices', 'uint32', cells)
        }
      })
    );
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'large-emerging-samples',
        operation: 'review',
        invocationCount: probes.length,
        bindings: [
          {name: 'values', view: giZScores, type: 'f32', access: 'read'},
          {
            name: 'result',
            view: importGraphBuffer(graph, 'large-emerging-result', samples, 'float32'),
            type: 'f32',
            access: 'read_write'
          }
        ],
        declarations: `const PROBES = array<u32, ${probes.length}>(${probes.map(row => `${row}u`).join(', ')});`,
        body: 'result[resultOffset + index] = values[valuesOffset + PROBES[index]];'
      })
    );
    compiled = graph.compile();
    submitGraph(device, compiled);
    expect(await readUint32(count)).toEqual([rows]);
    const actual = await readFloat32(moments);
    expect(actual[0]).toBe(Math.fround(rows));
    expect(actual[1]).toBeCloseTo(1 / rows, 6);
    expect(actual[2]).toBeCloseTo(1 - (1 / rows) ** 2, 5);
    const observed = await readFloat32(samples);
    for (const [index, row] of probes.entries()) {
      // Radius/window zero gives one unit-weight bin, hence spread=1 and z=(x-mean)/sd.
      const expected = ((row % 2 ? -1 : 1) - 1 / rows) / Math.sqrt(1 - (1 / rows) ** 2);
      expect(observed[index]).toBeCloseTo(expected, 5);
    }
  } finally {
    compiled?.destroy();
    parameters.destroy();
    count.destroy();
    moments.destroy();
    samples.destroy();
  }
}, 180_000);

it('fixed moments accept legacy per-frame count updates and explicit extended odd counts', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device);
  const parameters = rig.output('float32', 8);
  const counts = rig.output('uint32', 2);
  const moments = rig.output('float32', 5);
  const localI = rig.output('float32', 4);
  const packed = getGPUSpatialAutocorrelationParameterValues({
    fixedMoments: {count: 100, mean: 0.5, variance: 0.25}
  });
  parameters.buffer.write(packed);
  try {
    rig.run(
      new GPULocalMoran({
        weights: {
          offsets: rig.input(Uint32Array.of(0, 1, 2, 3, 4), 'uint32'),
          neighbors: rig.input(Uint32Array.of(1, 2, 3, 0), 'uint32'),
          weights: rig.input(Float32Array.of(1, 1, 1, 1), 'float32')
        },
        values: rig.input(Float32Array.of(0, 1, 0, 1), 'float32'),
        parameters: parameters.view,
        output: {
          zScores: rig.output('float32', 4).view,
          localI: localI.view,
          globalStatistics: moments.view,
          globalCounts: counts.view
        }
      })
    );
    expect((await counts.readUint32())[0]).toBe(100);
    for (const count of [200, 2 ** 24 + 1, 30_000_001, 100]) {
      if (count === 200)
        packed[2] = count; // Existing hosts update only the documented legacy slot.
      else
        getGPUSpatialAutocorrelationParameterValues(
          {fixedMoments: {count, mean: 0.5, variance: 0.25}},
          packed
        );
      parameters.buffer.write(packed);
      rig.resubmit();
      expect((await counts.readUint32())[0]).toBe(count);
      expect((await moments.readFloat32())[0]).toBe(Math.fround(count));
      // Local Moran uses the sample correction (n - 1) / n with the fixed population variance.
      for (const value of await localI.readFloat32())
        expect(value).toBeCloseTo(-(count - 1) / count, 5);
    }
  } finally {
    rig.destroy();
  }
});

it('lattice compiles with one physical buffer containing disjoint writable edge and status ranges', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device);
  const packed = rig.output('uint32', 128);
  const offsets = rig.output('uint32', 4);
  try {
    rig.run(
      new GPULatticeWeights({
        width: 3,
        height: 1,
        criterion: 'rook',
        output: {
          offsets: offsets.view,
          neighbors: rig.graph.createDataView(packed.view.buffer, {
            format: 'uint32',
            length: 4,
            byteOffset: 4
          }),
          weights: rig.graph.createDataView(packed.view.buffer, {
            format: 'float32',
            length: 4,
            byteOffset: 256
          }),
          status: {
            overflow: rig.graph.createDataView(packed.view.buffer, {format: 'uint32', length: 1})
          }
        }
      })
    );
    expect(await offsets.readUint32()).toEqual([0, 1, 3, 4]);
    expect((await packed.readUint32()).slice(0, 5)).toEqual([0, 1, 0, 2, 1]);
    expect((await packed.readFloat32()).slice(64, 68)).toEqual([1, 1, 1, 1]);
  } finally {
    rig.destroy();
  }
});
