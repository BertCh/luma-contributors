// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  calculateNetworkReference,
  calculatePairReference,
  calculateWatershedReference,
  createNetworkWorkload,
  createPairWorkload,
  createWatershedWorkload
} from './backlog-performance-fixtures';
import {createBacklogBenchmarkCases} from './backlog-performance-harness';

it('backlog pair reference matches an analytic translated weighted pair', () => {
  const result = calculatePairReference(
    {
      positions: Float32Array.from([0, 0, 3, 4]),
      marks: Float32Array.from([1, 2]),
      radiusCount: 1,
      maximumRadius: 5.375,
      updatedMaximumRadius: 4.375
    },
    5.375
  );
  const estimate = (4096 * 4096) / (2 * 61 * 60);
  expect(result.pairCounts).toEqual([2]);
  expect(result.status).toEqual([0]);
  expect(result.statistics[0]).toBeCloseTo(estimate, 8);
  expect(result.statistics[1]).toBeCloseTo(Math.sqrt(estimate / Math.PI), 8);
  expect(result.statistics[2]).toBeCloseTo(estimate / (Math.PI * 5.375 ** 2), 8);
  expect(result.statistics[3]).toBeCloseTo(8 / 9, 8);
});

it('backlog watershed heap preserves plateau FIFO ties and marker ownership', () => {
  const values = Float32Array.from([0, 1, 1, 0]);
  const workload = {
    width: 2,
    values,
    updatedValues: values,
    markers: Uint32Array.from([1, 0, 0, 2])
  };
  expect(calculateWatershedReference(workload, values)).toEqual({
    labels: [1, 1, 1, 2],
    converged: [1]
  });
  const plateau = new Float32Array(9);
  expect(
    calculateWatershedReference(
      {
        width: 3,
        values: plateau,
        updatedValues: plateau,
        markers: Uint32Array.from([1, 0, 2, 0, 0, 0, 0, 0, 3])
      },
      plateau
    ).labels
  ).toEqual([1, 1, 2, 1, 1, 2, 1, 3, 3]);
});

it('backlog SCC oracle handles cycles, a one-way bridge and an isolated vertex', () => {
  expect(
    calculateNetworkReference(Uint32Array.from([0, 1, 3, 4, 4]), Uint32Array.from([1, 0, 2, 2]))
  ).toEqual({labels: [0, 0, 2, 3]});
  for (const count of [16, 128]) {
    const workload = createNetworkWorkload(count);
    expect(calculateNetworkReference(workload.offsets, workload.neighbors).labels).toEqual(
      Array.from({length: count}, (_, vertex) => Math.floor(vertex / 8) * 8)
    );
    expect(calculateNetworkReference(workload.offsets, workload.updatedNeighbors).labels).toEqual(
      new Array(count).fill(0)
    );
  }
});

it('backlog workloads reproduce exactly and their updates change the checked results', () => {
  const pairs = createPairWorkload(64);
  expect(pairs).toEqual(createPairWorkload(64));
  expect(pairs.positions).not.toEqual(createPairWorkload(64, 7).positions);
  expect(calculatePairReference(pairs, pairs.maximumRadius).pairCounts).not.toEqual(
    calculatePairReference(pairs, pairs.updatedMaximumRadius).pairCounts
  );
  const raster = createWatershedWorkload(8);
  expect(raster).toEqual(createWatershedWorkload(8));
  expect(calculateWatershedReference(raster, raster.values).labels).not.toEqual(
    calculateWatershedReference(raster, raster.updatedValues).labels
  );
  expect(createNetworkWorkload(128)).toEqual(createNetworkWorkload(128));
  expect(createBacklogBenchmarkCases().map(benchmark => benchmark.name)).toEqual([
    'marked-pairs-64',
    'marked-pairs-512',
    'watershed-8',
    'watershed-24',
    'watershed-32',
    'watershed-33',
    'watershed-45',
    'watershed-46',
    'watershed-64',
    'strong-components-16',
    'strong-components-128'
  ]);
});
