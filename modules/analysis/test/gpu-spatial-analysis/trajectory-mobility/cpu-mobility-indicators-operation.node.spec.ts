// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeMobilityIndicatorsOnCPU,
  CPUMobilityIndicatorsOperation
} from '../../../src/gpu-spatial-analysis/trajectory-mobility/cpu-mobility-indicators-operation';
import {GPUMobilityIndicatorsOperation} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-mobility-indicators-operation';
import scikitMobility from '../../oracles/fixtures/movement-scikit-mobility.json';
import {getMobilityFixture, getMobilityProps} from './mobility-indicators-operation-fixture';

it('requires distance-compatible contexts in both the pure function and CPU constructor', () => {
  const input = {
    timestamps: new Float32Array([0, 1]),
    staypoints: {
      tracks: new Uint32Array([0, 0]),
      centroids: new Float32Array([0, 0, 3, 4]),
      startRows: new Uint32Array([0, 1]),
      durations: new Float32Array([1, 1]),
      locationIds: new Uint32Array([0, 1]),
      count: Uint32Array.of(2)
    }
  };
  const props = {
    ...getMobilityProps(),
    trackCount: 1,
    rowCount: 2,
    staypointCapacity: 2,
    triplegs: undefined,
    output: {radiusOfGyration: 'radiusOfGyration', jumpLengths: 'jumpLengths'}
  };
  const native = {coordinateSpace: 'planar', metric: 'native'} as const;
  const result = computeMobilityIndicatorsOnCPU(input, {trackCount: 1, spatialContext: native});
  // Two points five units apart are each 2.5 units from their mean.
  expect(Array.from(result.radiusOfGyration)).toEqual([2.5]);
  expect(Array.from(result.jumpLengths)).toEqual([0, 5]);
  expect(
    () => new CPUMobilityIndicatorsOperation({...props, spatialContext: native})
  ).not.toThrow();
  for (const [spatialContext, expectedError] of [
    [{coordinateSpace: 'planar', metric: 'none'}, /planar positions need the native metric/],
    [
      {
        coordinateSpace: 'planar',
        metric: 'native',
        frame: {id: 'review-lambert', kind: 'local-lambert-equal-area'}
      },
      /measures distance/
    ]
  ] as const) {
    expect(() => computeMobilityIndicatorsOnCPU(input, {trackCount: 1, spatialContext})).toThrow(
      expectedError
    );
    expect(() => new CPUMobilityIndicatorsOperation({...props, spatialContext})).toThrow(
      expectedError
    );
  }
});

it('matches hand-computed measures, location tie order and endpoint-based tracking quality', () => {
  const result = computeMobilityIndicatorsOnCPU(getMobilityFixture(), {trackCount: 3});
  // Pairwise-distance identity: gyration^2 = sum_{i<j}(distance^2) / n^2.
  expect(Array.from(result.radiusOfGyration)).toEqual([2, Math.fround(Math.sqrt(50) / 3), 0]);
  expect(Array.from(result.jumpLengths)).toEqual([0, 4, 4, 4, 0, 5, 5, 0]);
  expect(Array.from(result.waitingTimes)).toEqual([0, 5, 7, 8, 0, 4, 6, 0]);
  expect(Array.from(result.randomEntropy)).toEqual([1, 1, 0]);
  expect(Array.from(result.uncorrelatedEntropy)).toEqual([1, Math.fround(Math.log2(3) - 2 / 3), 0]);
  // ABAB has lambda=[1,1,3,2]; AAB has lambda=[1,3,2], under scikit-mobility's
  // special first/last and end-reaching conventions.
  expect(Array.from(result.realEntropy)).toEqual([
    Math.fround(8 / 7),
    Math.fround(Math.log2(3) / 2),
    0
  ]);
  expect(Array.from(result.homeLocations)).toEqual([2, 9, 0xffffffff]);
  expect(Array.from(result.workLocations)).toEqual([5, 4, 0xffffffff]);
  expect(Array.from(result.trackingQuality)).toEqual([
    Math.fround(16 / 20),
    Math.fround(9 / 13),
    0
  ]);
  expect(Array.from(result.candidateOverflow)).toEqual([0]);
});

it('matches the committed scikit-mobility 1.3.1 fixture on three geographic tracks and Int64 times', () => {
  const {scenario} = scikitMobility;
  const capacity = scenario.locationIds.length;
  const tracks = new Uint32Array(capacity);
  const timestamps = new Uint32Array(capacity * 2);
  for (let track = 0; track + 1 < scenario.userOffsets.length; track++)
    tracks.fill(track, scenario.userOffsets[track], scenario.userOffsets[track + 1]);
  scenario.timesMs.forEach((time, row) => {
    const timestamp = BigInt(time);
    timestamps[row * 2] = Number(timestamp & 0xffffffffn);
    timestamps[row * 2 + 1] = Number(timestamp >> 32n);
  });
  const result = computeMobilityIndicatorsOnCPU(
    {
      timestamps,
      staypoints: {
        tracks,
        centroids: Float32Array.from(
          scenario.latLon.flatMap(([latitude, longitude]) => [longitude, latitude])
        ),
        startRows: Uint32Array.from({length: capacity}, (_, row) => row),
        durations: new Float32Array(capacity).fill(60),
        locationIds: Uint32Array.from(scenario.locationIds),
        count: Uint32Array.of(capacity)
      }
    },
    {
      trackCount: 3,
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        sphereRadius: 6371
      }
    }
  );
  for (let track = 0; track < 3; track++) {
    const key = String(track);
    for (const [port, expectedValues] of [
      ['randomEntropy', scikitMobility.randomEntropy],
      ['uncorrelatedEntropy', scikitMobility.uncorrelatedEntropy],
      ['realEntropy', scikitMobility.realEntropy]
    ] as const)
      expect(result[port][track]).toBe(
        Math.fround((expectedValues as Record<string, number>)[key])
      );
    // Fixture coordinates are float64; input coordinates are rounded float32. 1e-4 relative
    // tolerance covers that input quantization, while still rejecting degree-based distances.
    const expectedRadius = (scikitMobility.radiusOfGyrationKm as Record<string, number>)[key];
    expect(Math.abs(result.radiusOfGyration[track] - expectedRadius) / expectedRadius).toBeLessThan(
      1e-4
    );
    const begin = scenario.userOffsets[track];
    const end = scenario.userOffsets[track + 1];
    const jumps = (scikitMobility.jumpLengthsKm as Record<string, number[]>)[key];
    const waits = (scikitMobility.waitingTimesSeconds as Record<string, number[]>)[key];
    for (let row = begin + 1; row < end; row++) {
      const expected = jumps[row - begin - 1];
      if (expected === 0) expect(result.jumpLengths[row]).toBe(0);
      else expect(Math.abs(result.jumpLengths[row] - expected) / expected).toBeLessThan(1e-4);
      expect(result.waitingTimes[row]).toBe(waits[row - begin - 1] * 1000);
    }
  }
});

it('matches a substring-set entropy oracle for every binary sequence through length seven', () => {
  // Independent algorithm: enumerate prefix substrings and look up each candidate word in a set.
  // This uses no longest-common-prefix recurrence or matching loop from the implementation.
  for (let length = 2; length <= 7; length++) {
    for (let pattern = 0; pattern < 2 ** length; pattern++) {
      const sequence = Array.from({length}, (_, position) => (pattern >> position) & 1);
      let denominator = 3;
      for (let position = 1; position + 1 < length; position++) {
        let unseenLength = 1;
        for (; unseenLength < length - position; unseenLength++) {
          const words = new Set(
            Array.from({length: Math.max(0, position - unseenLength + 1)}, (_, start) =>
              sequence.slice(start, start + unseenLength).join(',')
            )
          );
          if (!words.has(sequence.slice(position, position + unseenLength).join(','))) break;
        }
        denominator += unseenLength >= length - position ? length - position + 1 : unseenLength;
      }
      const result = computeMobilityIndicatorsOnCPU(
        {
          timestamps: new Float32Array(length),
          staypoints: {
            tracks: new Uint32Array(length),
            centroids: new Float32Array(length * 2),
            startRows: Uint32Array.from({length}, (_, row) => row),
            durations: new Float32Array(length),
            locationIds: Uint32Array.from(sequence),
            count: Uint32Array.of(length)
          }
        },
        {trackCount: 1}
      );
      expect(result.realEntropy[0]).toBe(Math.fround((length * Math.log2(length)) / denominator));
    }
  }
});

it('computes beyond the default entropy cap and handles missing keys as the GPU does', () => {
  const length = 1025;
  const input = {
    timestamps: new Float32Array(length),
    staypoints: {
      tracks: new Uint32Array(length),
      centroids: new Float32Array(length * 2),
      startRows: new Uint32Array(length),
      durations: new Float32Array(length),
      locationIds: new Uint32Array(length),
      count: Uint32Array.of(length)
    }
  };
  const result = computeMobilityIndicatorsOnCPU(input, {trackCount: 1});
  expect(result.realEntropy[0]).toBeGreaterThan(0);
  expect(result.candidateOverflow[0]).toBe(0);
  const missing = computeMobilityIndicatorsOnCPU(
    {
      ...input,
      staypoints: {
        ...input.staypoints,
        count: Uint32Array.of(2),
        locationIds: Uint32Array.from({length}, (_, row) => (row === 0 ? 0xffffffff : 3))
      }
    },
    {trackCount: 1}
  );
  expect(missing.randomEntropy[0]).toBe(1);
  expect(missing.homeLocations[0]).toBe(3); // Equal durations: smaller location ID wins.
  expect(missing.workLocations[0]).toBe(0xffffffff);
});

it('checks malformed arrays and construction-time contracts in both operation classes', () => {
  const input = getMobilityFixture();
  expect(() =>
    computeMobilityIndicatorsOnCPU({...input, timestamps: new Uint32Array(3)}, {trackCount: 3})
  ).toThrow(/timestamps/);
  expect(() =>
    computeMobilityIndicatorsOnCPU(
      {...input, staypoints: {...input.staypoints, durations: new Float32Array(1)}},
      {trackCount: 3}
    )
  ).toThrow(/staypoints.durations/);
  expect(() =>
    computeMobilityIndicatorsOnCPU(
      {...input, staypoints: {...input.staypoints, startRows: new Uint32Array(8).fill(99)}},
      {trackCount: 3}
    )
  ).toThrow(/staypoints.startRows/);
  expect(() =>
    computeMobilityIndicatorsOnCPU(
      {...input, triplegs: {...input.triplegs, endRows: new Uint32Array(4)}},
      {trackCount: 3}
    )
  ).toThrow(/triplegs.endRows/);
  const props = getMobilityProps();
  for (const Operation of [GPUMobilityIndicatorsOperation, CPUMobilityIndicatorsOperation]) {
    expect(() => new Operation({...props, trackCount: 0})).toThrow(/^mobility-test /);
    expect(
      () => new Operation({...props, output: {...props.output, jumpLengths: 'centroids'}})
    ).toThrow(/^mobility-test /);
  }
  expect(() => new CPUMobilityIndicatorsOperation({...props, triplegs: undefined})).toThrow(
    /^mobility-test trackingQuality/
  );
  expect(() => new CPUMobilityIndicatorsOperation({...props, output: {}})).toThrow(
    /^mobility-test requires at least one output/
  );
  expect(
    () => new CPUMobilityIndicatorsOperation({...props, maximumEntropySequenceRows: 1e9})
  ).not.toThrow();
});
