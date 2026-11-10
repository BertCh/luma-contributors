// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GPUCommandGraph, GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUTrajectoryEncounters} from '../../../src/gpu-spatial-analysis/trajectory-encounters/index';
import {GPUTrajectoryRestriction} from '../../../src/gpu-spatial-analysis/trajectory-restriction/gpu-trajectory-restriction';
import {createOptimizationHarness} from '../../gpu-network/network-optimization/optimization-test-harness';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRandom} from '../trajectory-interpolation/trajectory-interpolation-fixtures';

type Interval = {start: number; end: number};

/**
 * Binary64 reference: every segment pair of two time-sorted tracks (nested loops, no merge) with
 * closed time and geometry, no gaps. Returns the within-distance interval of each overlapping pair
 * in (track A segment, track B segment) order.
 */
function computeWithinDistanceIntervals(
  timesA: number[],
  positionsA: number[],
  timesB: number[],
  positionsB: number[],
  windowStart: number,
  windowEnd: number,
  radius: number
): Interval[] {
  const intervals: Interval[] = [];
  for (let rowA = 0; rowA < timesA.length; rowA++) {
    const nextA = Math.min(rowA + 1, timesA.length - 1);
    for (let rowB = 0; rowB < timesB.length; rowB++) {
      const nextB = Math.min(rowB + 1, timesB.length - 1);
      const lower = Math.max(windowStart, timesA[rowA], timesB[rowB]);
      const upper = Math.min(windowEnd, timesA[nextA], timesB[nextB]);
      if (lower > upper) continue;
      const velocityOf = (times: number[], positions: number[], row: number, next: number) => {
        const duration = times[next] - times[row];
        return duration > 0
          ? [
              (positions[2 * next] - positions[2 * row]) / duration,
              (positions[2 * next + 1] - positions[2 * row + 1]) / duration
            ]
          : [0, 0];
      };
      const velocityA = velocityOf(timesA, positionsA, rowA, nextA);
      const velocityB = velocityOf(timesB, positionsB, rowB, nextB);
      const relative = [0, 1].map(
        axis =>
          positionsA[2 * rowA + axis] +
          velocityA[axis] * (lower - timesA[rowA]) -
          (positionsB[2 * rowB + axis] + velocityB[axis] * (lower - timesB[rowB]))
      );
      const velocity = [velocityA[0] - velocityB[0], velocityA[1] - velocityB[1]];
      const quadratic = velocity[0] ** 2 + velocity[1] ** 2;
      let begin = 0;
      let finish = upper - lower;
      if (quadratic === 0) {
        if (relative[0] ** 2 + relative[1] ** 2 > radius * radius) continue;
      } else {
        const center = -(relative[0] * velocity[0] + relative[1] * velocity[1]) / quadratic;
        const closest = [relative[0] + velocity[0] * center, relative[1] + velocity[1] * center];
        const remaining = radius * radius - (closest[0] ** 2 + closest[1] ** 2);
        if (remaining < 0) continue;
        const half = Math.sqrt(remaining / quadratic);
        begin = Math.max(begin, center - half);
        finish = Math.min(finish, center + half);
        if (begin > finish) continue;
      }
      intervals.push({start: lower + begin, end: lower + finish});
    }
  }
  return intervals;
}

/** Random walks with irregular sample times that are exact in f32 (multiples of 1/16). */
function createTracks(seed: number, trackCount: number, rowCount: number) {
  const random = createRandom(seed);
  const times: number[][] = [];
  const positions: number[][] = [];
  for (let track = 0; track < trackCount; track++) {
    let time = Math.floor(random() * 8) / 16;
    let x = Math.floor(random() * 64) / 16;
    let y = Math.floor(random() * 64) / 16;
    const trackTimes: number[] = [];
    const trackPositions: number[] = [];
    for (let row = 0; row < rowCount; row++) {
      trackTimes.push(time);
      trackPositions.push(x, y);
      time += (1 + Math.floor(random() * 24)) / 16;
      x += (Math.floor(random() * 33) - 16) / 16;
      y += (Math.floor(random() * 33) - 16) / 16;
    }
    times.push(trackTimes);
    positions.push(trackPositions);
  }
  return {times, positions};
}

function readIntervals(
  intervals: number[],
  status: number[],
  pairIndex: number,
  capacity: number
): Interval[] {
  const result: Interval[] = [];
  for (let slot = 0; slot < capacity; slot++) {
    const row = pairIndex * capacity + slot;
    if (Number.isNaN(intervals[2 * row])) break;
    expect(status[row] & 32, `pair ${pairIndex} slot ${slot} has support`).toBe(32);
    result.push({start: intervals[2 * row], end: intervals[2 * row + 1]});
  }
  return result;
}

it('GPUTrajectoryRestriction pairs from GPU views match a binary64 all-segment-pairs reference', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = 60;
  const {times, positions} = createTracks(7, 3, rowCount);
  const radius = 3;
  const capacity = 256;
  const pairs: [number, number][] = [
    [0, 1],
    [1, 2],
    [0, 2],
    [2, 2],
    [1, 0],
    [0, 1],
    [0, 2]
  ];
  // Row 5 repeats a pair beyond `count` (inactive), row 6 is out of range of the active count.
  const activeCount = 5;
  const expected = pairs
    .slice(0, activeCount)
    .map(([first, second]) =>
      computeWithinDistanceIntervals(
        times[first],
        positions[first],
        times[second],
        positions[second],
        0,
        1000,
        radius
      )
    );
  // Teeth: several pairs with many intervals, so a wrong merge cannot pass by emptiness.
  expect(expected.every(intervals => intervals.length > 5)).toBe(true);
  const run = async (staticPairs: boolean) => {
    const harness = createOptimizationHarness(device);
    const pairRows = staticPairs ? activeCount : pairs.length;
    const output = {
      intervals: harness.output('float32x2', pairRows * capacity),
      positions: harness.output('float32x4', pairRows * capacity),
      status: harness.output('uint32', pairRows * capacity),
      relationships: harness.output('uint32x4', pairRows)
    };
    const result = await harness.run(
      new GPUTrajectoryRestriction({
        positions: harness.input('float32x2', Float32Array.from(positions.flat())),
        times: harness.input('float32', Float32Array.from(times.flat())),
        offsets: harness.input(
          'uint32',
          Uint32Array.from([0, rowCount, 2 * rowCount, 3 * rowCount])
        ),
        parameters: harness.input('float32', Float32Array.of(0, 1000, radius, 0, 0, 0, 0)),
        ...(staticPairs
          ? {pairwiseTracks: pairs.slice(0, activeCount)}
          : {
              pairs: {
                first: harness.input('uint32', Uint32Array.from(pairs.map(pair => pair[0]))),
                second: harness.input('uint32', Uint32Array.from(pairs.map(pair => pair[1]))),
                count: harness.input('uint32', Uint32Array.of(activeCount))
              }
            }),
        intervalsPerPair: capacity,
        output
      } as ConstructorParameters<typeof GPUTrajectoryRestriction>[0])
    );
    return {
      intervals: result(output.intervals),
      status: result(output.status),
      relationships: result(output.relationships)
    };
  };
  const dynamic = await run(false);
  for (let pair = 0; pair < activeCount; pair++) {
    const actual = readIntervals(dynamic.intervals, dynamic.status, pair, capacity);
    expect(actual.length, `pair ${pair} interval count`).toBe(expected[pair].length);
    actual.forEach((interval, slot) => {
      expect(interval.start, `pair ${pair} slot ${slot} start`).toBeCloseTo(
        expected[pair][slot].start,
        3
      );
      expect(interval.end, `pair ${pair} slot ${slot} end`).toBeCloseTo(
        expected[pair][slot].end,
        3
      );
    });
    expect(dynamic.relationships.slice(4 * pair, 4 * pair + 2), `pair ${pair}`).toEqual([
      1,
      pair === 3 ? 1 : 0
    ]);
    expect(dynamic.relationships[4 * pair + 3]).toBe(0);
    expect(
      dynamic.status
        .slice(pair * capacity, pair * capacity + actual.length)
        .every(flags => (flags & 4) === 0)
    ).toBe(true);
  }
  // Inactive rows hold NaN intervals and unknown relationships.
  for (let pair = activeCount; pair < pairs.length; pair++) {
    expect(dynamic.relationships.slice(4 * pair, 4 * pair + 4)).toEqual([
      0xffffffff, 0xffffffff, 0, 0
    ]);
    expect(Number.isNaN(dynamic.intervals[2 * pair * capacity])).toBe(true);
  }
  // The compile-time pair list gives the same rows as the GPU view.
  const compiled = await run(true);
  expect(compiled.intervals).toEqual(dynamic.intervals.slice(0, activeCount * capacity * 2));
  expect(compiled.relationships).toEqual(dynamic.relationships.slice(0, activeCount * 4));
});

it('GPUTrajectoryRestriction rejects out-of-range pair tracks and counts invalid segments once', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    intervals: harness.output('float32x2', 8),
    positions: harness.output('float32x4', 8),
    status: harness.output('uint32', 8),
    relationships: harness.output('uint32x4', 2)
  };
  const result = await harness.run(
    new GPUTrajectoryRestriction({
      // Track 0: rows 0..2, the middle row has a NaN coordinate (segments 0 and 1 are invalid).
      positions: harness.input(
        'float32x2',
        Float32Array.of(0, 0, Number.NaN, 0, 2, 0, 0, 0, 2, 0, 4, 0)
      ),
      times: harness.input('float32', Float32Array.of(0, 1, 2, 0, 2, 4)),
      offsets: harness.input('uint32', Uint32Array.of(0, 3, 6)),
      parameters: harness.input('float32', Float32Array.of(0, 10, 1, 0, 0, 0, 0)),
      pairs: {
        first: harness.input('uint32', Uint32Array.of(0, 0)),
        second: harness.input('uint32', Uint32Array.of(1, 7))
      },
      intervalsPerPair: 4,
      output
    })
  );
  const relationships = result(output.relationships);
  const status = result(output.status);
  // Row 1 of track 0 is NaN: segments 0 (into the NaN) and 1 (out of it) are invalid, counted once each.
  expect(relationships[3]).toBe(2);
  // The singleton last row of track 0 meets track 1 at (2, 0): known support despite the invalid segments.
  expect(relationships.slice(0, 2)).toEqual([1, 0xffffffff]);
  expect(status[0] & 4).toBe(4);
  // The second pair names track 7 of 2.
  expect(relationships.slice(4, 8)).toEqual([0xffffffff, 0xffffffff, 0, 1]);
  expect(status[4] & 4).toBe(4);
});

it('GPUTrajectoryRestriction takes encounter pairs without a readback', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const trackCount = 30;
  const bucketCount = 6;
  const random = createRandom(11);
  const samples = new Float32Array(trackCount * bucketCount * 2);
  const bucketTimes = Array.from({length: bucketCount}, (_, bucket) => bucket * 4);
  for (let track = 0; track < trackCount; track++) {
    let x = 20 + random() * 60;
    let y = 20 + random() * 60;
    for (let bucket = 0; bucket < bucketCount; bucket++) {
      samples[2 * (track * bucketCount + bucket)] = x;
      samples[2 * (track * bucketCount + bucket) + 1] = y;
      x += (random() - 0.5) * 8;
      y += (random() - 0.5) * 8;
    }
  }
  const distance = 6;
  const pairCapacity = 64;
  const harness = createOptimizationHarness(device);
  const encounterIds = harness.output('uint32', pairCapacity);
  const encounterPartners = harness.output('uint32', pairCapacity);
  const encounterCount = harness.output('uint32', 1);
  const output = {
    intervals: harness.output('float32x2', pairCapacity),
    positions: harness.output('float32x4', pairCapacity),
    status: harness.output('uint32', pairCapacity),
    relationships: harness.output('uint32x4', pairCapacity)
  };
  const encounters = new GPUTrajectoryEncounters({
    samples: harness.input('float32x2', samples),
    trackCount,
    bucketCount,
    distance: harness.input('float32', Float32Array.of(distance)),
    cellSize: 10,
    bounds: [0, 0, 100, 100],
    hitCapacity: 4096,
    output: {
      pairs: {
        ids: encounterIds,
        partners: encounterPartners,
        status: {
          count: encounterCount,
          overflow: harness.output('uint32', 1),
          requiredCount: harness.output('uint32', 1)
        }
      },
      status: {candidateOverflow: harness.output('uint32', 1)}
    }
  });
  const restriction = new GPUTrajectoryRestriction({
    positions: harness.input('float32x2', samples),
    times: harness.input(
      'float32',
      Float32Array.from(
        {length: trackCount * bucketCount},
        (_, row) => bucketTimes[row % bucketCount]
      )
    ),
    offsets: harness.input(
      'uint32',
      Uint32Array.from({length: trackCount + 1}, (_, track) => track * bucketCount)
    ),
    parameters: harness.input('float32', Float32Array.of(0, 100, distance, 0, 0, 0, 0)),
    pairs: {first: encounterIds, second: encounterPartners, count: encounterCount},
    intervalsPerPair: 1,
    output
  } as ConstructorParameters<typeof GPUTrajectoryRestriction>[0]);
  const chain: GPUCommandNodeProducer = {
    id: 'encounters-then-restriction',
    getCommandNodes: <Parameters>(graph: GPUCommandGraph<Parameters>) => [
      ...encounters.getCommandNodes(graph),
      ...restriction.getCommandNodes(graph)
    ]
  } as GPUCommandNodeProducer;
  const result = await harness.run(chain);
  const count = result(encounterCount)[0];
  expect(count).toBeGreaterThan(3);
  expect(count).toBeLessThan(pairCapacity);
  const ids = result(encounterIds);
  const partners = result(encounterPartners);
  const relationships = result(output.relationships);
  for (let pair = 0; pair < count; pair++) {
    // An encounter at a bucket time is a within-distance instant of the continuous tracks.
    const expected = computeWithinDistanceIntervals(
      bucketTimes,
      Array.from(samples.slice(ids[pair] * bucketCount * 2, (ids[pair] + 1) * bucketCount * 2)),
      bucketTimes,
      Array.from(
        samples.slice(partners[pair] * bucketCount * 2, (partners[pair] + 1) * bucketCount * 2)
      ),
      0,
      100,
      distance
    );
    expect(expected.length, `pair ${pair}`).toBeGreaterThan(0);
    expect(relationships[4 * pair], `pair ${pair} ever`).toBe(1);
    expect(result(output.intervals)[2 * pair], `pair ${pair} first start`).toBeCloseTo(
      expected[0].start,
      3
    );
  }
  for (let pair = count; pair < pairCapacity; pair++) {
    expect(relationships.slice(4 * pair, 4 * pair + 2)).toEqual([0xffffffff, 0xffffffff]);
  }
});

it('GPUTrajectoryRestriction merges tracks of tens of thousands of rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = 30_000;
  const harness = createOptimizationHarness(device);
  const capacity = 2 * rowCount + 8;
  const output = {
    intervals: harness.output('float32x2', capacity),
    positions: harness.output('float32x4', capacity),
    status: harness.output('uint32', capacity),
    relationships: harness.output('uint32x4', 1)
  };
  // Both tracks move along x at speed 1, half a unit apart in time and 0.25 apart in y: always within 1.
  const timesA = Float32Array.from({length: rowCount}, (_, row) => row);
  const timesB = Float32Array.from({length: rowCount}, (_, row) => row + 0.5);
  const positions = new Float32Array(rowCount * 4);
  for (let row = 0; row < rowCount; row++) {
    positions.set([row, 0], 2 * row);
    positions.set([row + 0.5, 0.25], 2 * (rowCount + row));
  }
  const result = await harness.run(
    new GPUTrajectoryRestriction({
      positions: harness.input('float32x2', positions),
      times: harness.input('float32', Float32Array.of(...timesA, ...timesB)),
      offsets: harness.input('uint32', Uint32Array.of(0, rowCount, 2 * rowCount)),
      parameters: harness.input('float32', Float32Array.of(0, 1e6, 1, 0, 0, 0, 0)),
      pairs: {
        first: harness.input('uint32', Uint32Array.of(0)),
        second: harness.input('uint32', Uint32Array.of(1))
      },
      intervalsPerPair: capacity,
      output
    })
  );
  const intervals = result(output.intervals);
  const relationships = result(output.relationships);
  expect(relationships.slice(0, 2)).toEqual([1, 1]);
  expect(result(output.status).some(flags => (flags & 256) !== 0)).toBe(false);
  // Overlap chain: [0.5, 1], [1, 1.5], ... every half unit up to rowCount - 1, then the last rows.
  expect(intervals[0]).toBeCloseTo(0.5, 5);
  let written = 0;
  while (written < capacity && !Number.isNaN(intervals[2 * written])) written++;
  expect(written).toBeGreaterThan(2 * rowCount - 8);
  expect(intervals[2 * (written - 1) + 1]).toBeCloseTo(rowCount - 1, 3);
  // Consecutive intervals touch, so the merge visited every segment pair once and in order.
  for (let slot = 1; slot < written; slot++) {
    expect(intervals[2 * slot]).toBeGreaterThanOrEqual(intervals[2 * (slot - 1)]);
    expect(intervals[2 * slot]).toBeLessThanOrEqual(intervals[2 * (slot - 1) + 1] + 1e-3);
  }
});
