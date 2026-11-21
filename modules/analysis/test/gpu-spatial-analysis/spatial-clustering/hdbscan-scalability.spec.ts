// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {runHDBSCAN} from './hierarchical-density-clustering-harness';
import {canonicalizeLabels} from './hierarchical-density-clustering-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

it('the default BVH core search and packed reference match hand-computed two-square clusters', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    const positions = new Float32Array([0, 0, 1, 0, 0, 1, 1, 1, 10, 0, 11, 0, 10, 1, 11, 1]);
    for (const coreDistanceSearch of ['bvh', 'neighbor-search'] as const) {
      const result = await runHDBSCAN(device, positions, {
        minimumClusterSize: 3,
        minimumSamples: 3,
        coreDistanceSearch
      });
      expect(canonicalizeLabels(result.labels)).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
      expect(result.clusterCount).toBe(2);
      for (const distance of result.coreDistances) expect(distance).toBeCloseTo(1, 6);
    }
  } finally {
    device.destroy();
  }
}, 120000);

it('the exact BVH core search matches exhaustive distances on duplicates, invalid rows and short inputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    const positions = new Float32Array([
      0,
      0,
      0,
      0,
      0.5,
      1,
      1.5,
      -0.2,
      3,
      4,
      4,
      2,
      10,
      1,
      10.1,
      0,
      10.2,
      0.1,
      11,
      -0.1,
      12,
      0,
      13,
      1,
      ...Array.from({length: 80}, (_, row) => [
        Math.sin(row * 1.7) * 20,
        Math.cos(row * 0.3) * 11
      ]).flat(),
      NaN,
      0,
      0,
      Infinity
    ]);
    const short = await runHDBSCAN(device, positions.slice(0, 8), {
      minimumClusterSize: 3,
      minimumSamples: 32
    });
    expect(short.coreDistances).toEqual([Infinity, Infinity, Infinity, Infinity]);
    const rows = positions.length / 2;
    for (const minimumSamples of [1, 2, 3, 7, 32]) {
      const result = await runHDBSCAN(device, positions, {
        minimumClusterSize: 3,
        minimumSamples
      });
      for (let row = 0; row < rows; row++) {
        const distances: number[] = [];
        if (Number.isFinite(positions[row * 2]) && Number.isFinite(positions[row * 2 + 1])) {
          for (let other = 0; other < rows; other++) {
            if (
              !Number.isFinite(positions[other * 2]) ||
              !Number.isFinite(positions[other * 2 + 1])
            )
              continue;
            distances.push(
              Math.hypot(
                positions[row * 2] - positions[other * 2],
                positions[row * 2 + 1] - positions[other * 2 + 1]
              )
            );
          }
        }
        distances.sort((first, second) => first - second);
        const expected = distances[minimumSamples - 1] ?? Infinity;
        if (Number.isFinite(expected)) expect(result.coreDistances[row]).toBeCloseTo(expected, 5);
        else expect(result.coreDistances[row]).toBe(Infinity);
      }
      const reference = await runHDBSCAN(device, positions, {
        minimumClusterSize: 3,
        minimumSamples,
        coreDistanceSearch: 'neighbor-search'
      });
      expect(result.coreDistances).toEqual(reference.coreDistances);
      expect(result.labels).toEqual(reference.labels);
      expect(result.labels.slice(-2)).toEqual([0xffffffff, 0xffffffff]);
    }
  } finally {
    device.destroy();
  }
}, 180000);

it('clusters four million Gaussian points into their two known separated blobs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    const rows = 4_000_000;
    const half = rows / 2;
    const positions = new Float32Array(rows * 2);
    // Deterministic Box-Muller pairs. The smallest uniform is 1/(2^32+1), so the
    // radius is below seven; centers 2000 apart have disjoint, known memberships.
    let randomState = 0x12345678;
    const random = () => {
      randomState ^= randomState << 13;
      randomState ^= randomState >>> 17;
      randomState ^= randomState << 5;
      return ((randomState >>> 0) + 1) / 4294967297;
    };
    for (let row = 0; row < rows; row++) {
      const radius = Math.sqrt(-2 * Math.log(random()));
      const angle = 2 * Math.PI * random();
      positions[row * 2] = (row < half ? -1000 : 1000) + radius * Math.cos(angle);
      positions[row * 2 + 1] = radius * Math.sin(angle);
    }
    // Large minimum clusters prevent selecting small fluctuations inside either Gaussian.
    const result = await runHDBSCAN(device, positions, {
      minimumClusterSize: 1_000_000,
      minimumSamples: 8
    });
    expect(result.clusterCount).toBe(2);
    const firstLabel = result.labels[0];
    const secondLabel = result.labels[half];
    expect(firstLabel).not.toBe(0xffffffff);
    expect(secondLabel).not.toBe(0xffffffff);
    expect(secondLabel).not.toBe(firstLabel);
    let mismatches = 0;
    let invalidCoreDistances = 0;
    for (let row = 0; row < rows; row++) {
      if (result.labels[row] !== (row < half ? firstLabel : secondLabel)) mismatches++;
      if (!Number.isFinite(result.coreDistances[row]) || result.coreDistances[row] < 0)
        invalidCoreDistances++;
    }
    expect(mismatches).toBe(0);
    expect(invalidCoreDistances).toBe(0);
  } finally {
    device.destroy();
  }
}, 900000);

it('overwrites poisoned status.count after changing positions on repeated encodings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    const squares = new Float32Array([0, 0, 1, 0, 0, 1, 1, 1, 10, 0, 11, 0, 10, 1, 11, 1]);
    const invalid = new Float32Array(squares.length).fill(NaN);
    const actual = await runHDBSCAN(device, squares, {
      minimumClusterSize: 3,
      minimumSamples: 3,
      statusPoison: 123,
      positionUpdates: [invalid, squares, invalid, squares]
    });
    expect(actual.clusterCount).toBe(2);
    expect(canonicalizeLabels(actual.labels)).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
    for (let encoding = 0; encoding < actual.positionUpdated.length; encoding++) {
      const result = actual.positionUpdated[encoding];
      const expectedCount = encoding % 2 === 0 ? 0 : 2;
      expect(result.clusterCount).toBe(expectedCount);
      expect(canonicalizeLabels(result.labels)).toEqual(
        expectedCount === 0 ? Array.from({length: 8}, () => -1) : [0, 0, 0, 0, 1, 1, 1, 1]
      );
    }
  } finally {
    device.destroy();
  }
}, 180000);

it('handles workgroup and power-of-two boundaries with invalid rows and valid tail points, through the exact cap', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    for (const rows of [255, 256, 257, 2_097_151, 2_097_152, 2_097_153, 4_194_304]) {
      // Only these eight points are finite. Their closed-form partition is independent of the
      // surrounding invalid rows, including the rows around 256, 2^21 and the final workgroup.
      const positions = new Float32Array(rows * 2).fill(NaN);
      positions.set([0, 0, 1, 0, 0, 1, 1, 1], 0);
      const secondRows = [rows - 8, rows - 6, rows - 3, rows - 1];
      const secondPoints = [
        [10, 0],
        [11, 0],
        [10, 1],
        [11, 1]
      ];
      for (let point = 0; point < secondRows.length; point++)
        positions.set(secondPoints[point], secondRows[point] * 2);
      const secondRowSet = new Set(secondRows);
      const actual = await runHDBSCAN(device, positions, {
        minimumClusterSize: 3,
        minimumSamples: 3
      });
      expect(actual.clusterCount, `rows=${rows}`).toBe(2);
      const firstLabel = actual.labels[0];
      const secondLabel = actual.labels[rows - 1];
      expect(firstLabel).not.toBe(0xffffffff);
      expect(secondLabel).not.toBe(0xffffffff);
      expect(secondLabel).not.toBe(firstLabel);
      let wrongLabels = 0;
      let wrongDistances = 0;
      for (let row = 0; row < rows; row++) {
        const finite = row < 4 || secondRowSet.has(row);
        const label = row < 4 ? firstLabel : secondRowSet.has(row) ? secondLabel : 0xffffffff;
        if (actual.labels[row] !== label) wrongLabels++;
        if (
          finite
            ? Math.abs(actual.coreDistances[row] - 1) > 1e-6
            : actual.coreDistances[row] !== Infinity
        )
          wrongDistances++;
      }
      expect(wrongLabels, `rows=${rows}`).toBe(0);
      expect(wrongDistances, `rows=${rows}`).toBe(0);
    }
  } finally {
    device.destroy();
  }
}, 900000);

it('finishes one million identical points at minimumSamples 32 with exact zero cores and no selected root', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    const actual = await runHDBSCAN(device, new Float32Array(1_000_000 * 2), {
      minimumClusterSize: 100_000,
      minimumSamples: 32
    });
    expect(actual.coreDistances.every(distance => distance === 0)).toBe(true);
    expect(actual.labels.every(label => label === 0xffffffff)).toBe(true);
    expect(actual.clusterCount).toBe(0);
  } finally {
    device.destroy();
  }
}, 900000);

it('preserves the two known memberships in increasing separated duplicate groups', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    for (const rows of [256, 4096, 65536]) {
      const positions = new Float32Array(rows * 2);
      const belongsToSecond = (row: number) => (rows === 256 ? row % 2 === 1 : row >= rows / 2);
      for (let row = 0; row < rows; row++) if (belongsToSecond(row)) positions[row * 2] = 10;
      const actual = await runHDBSCAN(device, positions, {
        minimumClusterSize: rows / 4,
        minimumSamples: 32
      });
      const firstLabel = actual.labels[0];
      const secondLabel = actual.labels[rows - 1];
      expect(actual.clusterCount).toBe(2);
      expect(firstLabel).not.toBe(0xffffffff);
      expect(secondLabel).not.toBe(0xffffffff);
      expect(secondLabel).not.toBe(firstLabel);
      expect(
        actual.labels.every(
          (label, row) => label === (belongsToSecond(row) ? secondLabel : firstLabel)
        )
      ).toBe(true);
      expect(actual.coreDistances.every(distance => distance === 0)).toBe(true);
    }
  } finally {
    device.destroy();
  }
}, 900000);
