// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT,
  GPUGlobalSpatialStatistics
} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/index';
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
import {
  createKnnWeights,
  createSeededRandom,
  type CPUSpatialWeights
} from './global-spatial-statistics-oracle';

/**
 * Parity of chunked global statistics: the same dataset runs with one CSR and with row chunks of
 * several tile sizes (including 1-row chunks); results agree within f32 association tolerance.
 * kNN weights are asymmetric, so the (row chunk, target chunk) reverse lookup is exercised.
 */

const LAYOUT = GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT;
const NEIGHBOR_COUNT = 4;

type Scenario = {
  rowCount: number;
  values: Float32Array;
  secondValues: Float32Array;
  mask: Uint32Array;
  rowStandardized: boolean;
  weights: CPUSpatialWeights;
};

function createScenario(rowCount: number, rowStandardized: boolean): Scenario {
  const random = createSeededRandom(11 + rowCount);
  const positions = Float64Array.from({length: rowCount * 2}, () => random() * 100);
  const values = Float32Array.from(
    {length: rowCount},
    (_, index) => Math.sin(positions[index * 2] / 13) + random()
  );
  // Zeros make black/white rows for the join counts.
  for (let index = 0; index < rowCount; index += 3) {
    values[index] = 0;
  }
  const secondValues = Float32Array.from({length: rowCount}, () => random());
  // Mask out about one row in seven so masked neighbors and islands are exercised.
  const mask = Uint32Array.from({length: rowCount}, (_, index) => (index % 7 === 3 ? 0 : 1));
  return {
    rowCount,
    values,
    secondValues,
    mask,
    rowStandardized,
    weights: createKnnWeights(positions, NEIGHBOR_COUNT, rowStandardized)
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

function getTileRowCounts(rowCount: number, tileSize: number): number[] {
  const tiles: number[] = [];
  for (let rowStart = 0; rowStart < rowCount; rowStart += tileSize) {
    tiles.push(Math.min(tileSize, rowCount - rowStart));
  }
  return tiles;
}

async function runStatistics(
  device: Device,
  scenario: Scenario,
  tileSize: number | undefined,
  islands: 'exclude' | 'keep',
  symmetryStrategy: 'transpose' | 'pairwise' = 'pairwise'
): Promise<{summary: number[]; joinCounts: number[]}> {
  const {rowCount} = scenario;
  const graph = new GPUCommandGraph(device, {id: 'chunked-global-statistics'});
  const importCsr = (csr: CPUSpatialWeights, name: string, chunkRows: number) => ({
    offsets: importGraphBuffer(
      graph,
      `${name}-offsets`,
      createInputBuffer(device, csr.offsets),
      'uint32',
      chunkRows + 1
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
    rowCount,
    transform: scenario.rowStandardized ? ('row' as const) : ('binary' as const),
    symmetric: 'unknown' as const
  };
  let weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  if (tileSize === undefined) {
    weights = {...importCsr(scenario.weights, 'whole', rowCount), information};
  } else {
    let rowStart = 0;
    weights = {
      rowCount,
      information,
      chunks: getTileRowCounts(rowCount, tileSize).map((chunkRows, index) => {
        const chunk = {
          rowStart,
          rowCount: chunkRows,
          weights: importCsr(
            sliceWeights(scenario.weights, rowStart, chunkRows),
            `chunk${index}`,
            chunkRows
          )
        };
        rowStart += chunkRows;
        return chunk;
      })
    };
  }
  const summary = createOutputBuffer(device, LAYOUT.length);
  const joinCounts = createOutputBuffer(device, 3);
  graph.add(
    new GPUGlobalSpatialStatistics({
      weights,
      islands,
      symmetryStrategy,
      values: importGraphBuffer(
        graph,
        'values',
        createInputBuffer(device, scenario.values),
        'float32',
        rowCount
      ),
      secondValues: importGraphBuffer(
        graph,
        'second-values',
        createInputBuffer(device, scenario.secondValues),
        'float32',
        rowCount
      ),
      mask: importGraphBuffer(
        graph,
        'mask',
        createInputBuffer(device, scenario.mask),
        'uint32',
        rowCount
      ),
      statistics: ['moran', 'geary', 'getisOrdG', 'bivariateMoran', 'joinCount'],
      output: {
        summary: importGraphBuffer(graph, 'summary', summary, 'float32', LAYOUT.length),
        joinCounts: importGraphBuffer(graph, 'join-counts', joinCounts, 'uint32', 3)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    summary: Array.from(await readFloat32(summary, LAYOUT.length)),
    joinCounts: Array.from(await readUint32(joinCounts, 3))
  };
  compiled.destroy();
  return result;
}

function expectNear(actual: number[], expected: number[], label: string) {
  expect(actual.length, label).toBe(expected.length);
  actual.forEach((value, index) => {
    const reference = expected[index];
    if (Number.isNaN(reference)) {
      expect(Number.isNaN(value), `${label}[${index}] NaN`).toBe(true);
    } else {
      expect(Math.abs(value - reference), `${label}[${index}]`).toBeLessThanOrEqual(
        1e-4 * Math.max(1, Math.abs(reference))
      );
    }
  });
}

async function expectParity(
  device: Device,
  scenario: Scenario,
  tileSizes: number[],
  islands: 'exclude' | 'keep',
  symmetryStrategy: 'transpose' | 'pairwise' = 'pairwise'
) {
  const whole = await runStatistics(device, scenario, undefined, islands);
  for (const tileSize of tileSizes) {
    const label = `${symmetryStrategy} rows ${scenario.rowCount} tile ${tileSize} ${islands} standardized ${scenario.rowStandardized}`;
    const chunked = await runStatistics(device, scenario, tileSize, islands, symmetryStrategy);
    expectNear(chunked.summary, whole.summary, `${label} summary`);
    // Counts are integer reductions: exact.
    expect(chunked.joinCounts, `${label} joinCounts`).toEqual(whole.joinCounts);
  }
}

it('chunked global statistics match the single CSR (binary and row-standardized)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const rowStandardized of [false, true]) {
    for (const islands of ['exclude', 'keep'] as const) {
      await expectParity(device, createScenario(96, rowStandardized), [32, 17, 7], islands);
    }
  }
});

it('chunked global statistics match the single CSR with 1-row chunks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const rowStandardized of [false, true]) {
    await expectParity(device, createScenario(20, rowStandardized), [1, 3], 'exclude');
  }
});

it('chunked global statistics with the transpose symmetry strategy match the single CSR', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const rowStandardized of [false, true]) {
    for (const islands of ['exclude', 'keep'] as const) {
      await expectParity(
        device,
        createScenario(96, rowStandardized),
        [32, 17, 7],
        islands,
        'transpose'
      );
    }
    await expectParity(device, createScenario(20, rowStandardized), [1, 3], 'exclude', 'transpose');
  }
});
