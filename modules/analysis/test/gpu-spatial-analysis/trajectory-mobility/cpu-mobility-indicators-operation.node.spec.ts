// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import type {GPUCPUVector} from '@luma.gl/graph/gpu-core';

import {WgslReflect} from 'wgsl_reflect';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {getGPUCommandNodeKernelSources} from '../../utils/gpu-contributor-test-utils';
import {GPUMobilityIndicators} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-mobility-indicators';

it.each([
  true,
  false
])('declares location filtering kernels with location IDs supplied: %s', withIds => {
  const context = createKernelAuditContext('mobility-location-kernels');
  try {
    const {view} = context;
    const indicator = new GPUMobilityIndicators({
      trackCount: 2,
      timestamps: view('float32', 4),
      staypoints: {
        tracks: view('uint32', 4),
        centroids: view('float32x2', 4),
        startRows: view('uint32', 4),
        durations: view('float32', 4),
        locationIds: withIds ? view('uint32', 4) : undefined,
        status: {count: view('uint32', 1), locationIterationLimitReached: view('uint32', 1)}
      },
      output: {
        randomEntropy: view('float32', 2),
        uncorrelatedEntropy: view('float32', 2),
        realEntropy: view('float32', 2),
        homeLocations: view('uint32', 2),
        workLocations: view('uint32', 2),
        status: {excludedLocationCount: view('uint32', 1), candidateOverflow: view('uint32', 1)}
      }
    });
    const sources = getGPUCommandNodeKernelSources(indicator.getCommandNodes(context.graph));
    for (const suffix of [
      '-filter-locations',
      '-excluded-location-count',
      '-locations',
      '-real-entropy'
    ]) {
      const kernel = sources.find(source => source.id === `mobility-indicators${suffix}`);
      expect(kernel, suffix).toBeDefined();
      expect(() => new WgslReflect(kernel!.source), suffix).not.toThrow();
    }
  } finally {
    context.device.destroy();
  }
});

it('counts and excludes missing or unsettled location rows from every location indicator', () => {
  const staypoints = {
    tracks: Uint32Array.of(0, 0, 0, 0),
    centroids: Float32Array.of(0, 0, 2, 0, 4, 0, 6, 0),
    startRows: Uint32Array.of(0, 1, 2, 3),
    durations: Float32Array.of(1, 100, 4, 1),
    locationIds: Uint32Array.of(7, 0xffffffff, 9, 7),
    count: Uint32Array.of(4)
  };
  const timestamps = Float32Array.of(0, 1, 2, 3);
  const options = {trackCount: 1, maximumEntropySequenceRows: 3};
  const result = computeMobilityIndicatorsOnCPU({timestamps, staypoints}, options);
  // Retained sequence [7, 9, 7]: two keys with shares 2/3 and 1/3,
  // novelty lengths [1, 1, 2], and duration totals 2 and 4.
  expect(result.randomEntropy).toEqual(Float32Array.of(1));
  expect(result.uncorrelatedEntropy).toEqual(Float32Array.of(0.9182958340544896));
  expect(result.realEntropy).toEqual(Float32Array.of(1.188721875540867));
  expect(result.homeLocations).toEqual(Uint32Array.of(9));
  expect(result.workLocations).toEqual(Uint32Array.of(7));
  expect(result.excludedLocationCount).toEqual(Uint32Array.of(1));
  expect(result.candidateOverflow).toEqual(Uint32Array.of(0));

  for (const missing of [
    {...staypoints, locationIds: undefined},
    {...staypoints, locationIds: new Uint32Array(4).fill(0xffffffff)},
    {...staypoints, locationIterationLimitReached: Uint32Array.of(1)}
  ]) {
    const excluded = computeMobilityIndicatorsOnCPU({timestamps, staypoints: missing}, options);
    expect(excluded.randomEntropy).toEqual(Float32Array.of(NaN));
    expect(excluded.uncorrelatedEntropy).toEqual(Float32Array.of(NaN));
    expect(excluded.realEntropy).toEqual(Float32Array.of(NaN));
    expect(excluded.homeLocations).toEqual(Uint32Array.of(0xffffffff));
    expect(excluded.workLocations).toEqual(Uint32Array.of(0xffffffff));
    expect(excluded.excludedLocationCount).toEqual(Uint32Array.of(4));
    // All four centroids still contribute: mean x = 3, mean squared distance = 5.
    expect(excluded.radiusOfGyration).toEqual(Float32Array.of(Math.sqrt(5)));
    expect(excluded.jumpLengths).toEqual(Float32Array.of(0, 2, 2, 2));
    expect(excluded.waitingTimes).toEqual(Float32Array.of(0, 1, 1, 1));
  }
});

it('counts only active excluded keys across tracks and on upstream-loss return', () => {
  const timestamps = Float32Array.of(0, 1, 2);
  const staypoints = {
    tracks: Uint32Array.of(0, 0, 2, 0xffffffff),
    centroids: new Float32Array(8),
    startRows: Uint32Array.of(0, 1, 2, 0xffffffff),
    durations: Float32Array.of(1, 1, 1, 0),
    locationIds: Uint32Array.of(0, 0xffffffff, 4, 0xffffffff),
    count: Uint32Array.of(3)
  };
  // One active sentinel and one sentinel past the count. Track 1 is empty and key 0 is valid.
  const complete = computeMobilityIndicatorsOnCPU({timestamps, staypoints}, {trackCount: 3});
  expect(complete.excludedLocationCount).toEqual(Uint32Array.of(1));
  expect(complete.homeLocations).toEqual(Uint32Array.of(0, 0xffffffff, 4));

  const incomplete = {...staypoints, overflow: Uint32Array.of(1)};
  const suppressed = computeMobilityIndicatorsOnCPU(
    {timestamps, staypoints: incomplete},
    {trackCount: 3}
  );
  expect(suppressed.excludedLocationCount).toEqual(Uint32Array.of(1));
  expect(suppressed.randomEntropy).toEqual(Float32Array.of(NaN, NaN, NaN));
  const unsettled = computeMobilityIndicatorsOnCPU(
    {timestamps, staypoints: {...incomplete, locationIterationLimitReached: Uint32Array.of(1)}},
    {trackCount: 3}
  );
  expect(unsettled.excludedLocationCount).toEqual(Uint32Array.of(3));
});

it('suppresses tracking quality when supplied tripleg status reports loss', () => {
  const fixture = getMobilityFixture();
  const input = {...fixture, triplegs: {...fixture.triplegs!, overflow: Uint32Array.of(1)}};
  const result = computeMobilityIndicatorsOnCPU(input, {trackCount: 2});
  expect(Array.from(result.trackingQuality)).toEqual([NaN, NaN]);
});

it('B3 suppresses two ranked non-sentinel location IDs in pure and operation execution', async () => {
  const staypoints = {
    tracks: Uint32Array.of(0, 0, 0),
    centroids: new Float32Array(6),
    startRows: Uint32Array.of(0, 1, 2),
    durations: Float32Array.of(1, 4, 1),
    locationIds: Uint32Array.of(7, 9, 7),
    count: Uint32Array.of(3),
    locationIterationLimitReached: Uint32Array.of(0)
  };
  const props = getMobilityProps();
  const operation = new CPUMobilityIndicatorsOperation({
    ...props,
    trackCount: 1,
    rowCount: 3,
    staypointCapacity: 3,
    triplegs: undefined,
    maximumEntropySequenceRows: 0,
    staypoints: {
      ...props.staypoints,
      status: {count: 'stayCount', locationIterationLimitReached: 'locationLoss'}
    },
    output: {
      homeLocations: 'home',
      workLocations: 'work',
      radiusOfGyration: 'radius',
      status: {candidateOverflow: 'loss'}
    }
  });
  const vector = (
    format: 'uint32' | 'float32' | 'float32x2',
    values: Float32Array | Uint32Array
  ): GPUCPUVector => {
    const length = values.length / (format === 'float32x2' ? 2 : 1);
    return {format, length, chunks: [{values, length}], pack: () => values};
  };
  for (const loss of [0, 1]) {
    staypoints.locationIterationLimitReached[0] = loss;
    const result = computeMobilityIndicatorsOnCPU(
      {timestamps: Float32Array.of(0, 1, 2), staypoints},
      {trackCount: 1}
    );
    // Duration totals: location 9 has 4, location 7 has 2; home=9, work=7.
    expect(result.homeLocations).toEqual(Uint32Array.of(loss ? 0xffffffff : 9));
    expect(result.workLocations).toEqual(Uint32Array.of(loss ? 0xffffffff : 7));
    const executed = await operation.execute({
      inputs: {
        timestamps: vector('float32', Float32Array.of(0, 1, 2)),
        stayTracks: vector('uint32', staypoints.tracks),
        centroids: vector('float32x2', staypoints.centroids),
        stayStartRows: vector('uint32', staypoints.startRows),
        durations: vector('float32', staypoints.durations),
        locationIds: vector('uint32', staypoints.locationIds),
        stayCount: vector('uint32', staypoints.count),
        locationIterationLimitReached: vector('uint32', staypoints.locationIterationLimitReached)
      }
    });
    expect(executed['home']).toEqual(Uint32Array.of(loss ? 0xffffffff : 9));
    expect(executed['work']).toEqual(Uint32Array.of(loss ? 0xffffffff : 7));
    expect(executed['radius']).toEqual(Float32Array.of(0));
    expect(executed['loss']).toEqual(Uint32Array.of(loss));
  }
});

it('earlier entropy-cap note preserves the uncapped CPU default and ignores an unused cap', () => {
  const count = 1025;
  const input = {
    timestamps: new Float32Array(count),
    staypoints: {
      tracks: new Uint32Array(count),
      centroids: new Float32Array(count * 2),
      startRows: Uint32Array.from({length: count}, (_, row) => row),
      durations: new Float32Array(count),
      locationIds: Uint32Array.from({length: count}, (_, row) => row),
      count: Uint32Array.of(count)
    }
  };
  // Distinct keys have lambda=1 except the last lambda=2: 1025*log2(1025)/1026.
  const complete = computeMobilityIndicatorsOnCPU(input, {trackCount: 1});
  expect(complete.realEntropy).toEqual(Float32Array.of(9.991660118103027));
  expect(complete.candidateOverflow).toEqual(Uint32Array.of(0));
  const capped = computeMobilityIndicatorsOnCPU(input, {
    trackCount: 1,
    maximumEntropySequenceRows: 1024
  });
  expect(capped.realEntropy).toEqual(Float32Array.of(NaN));
  expect(capped.candidateOverflow).toEqual(Uint32Array.of(1));
  expect(() =>
    computeMobilityIndicatorsOnCPU(input, {
      trackCount: 1,
      includeRealEntropy: false,
      maximumEntropySequenceRows: 0
    })
  ).not.toThrow();
  expect(
    () => new CPUMobilityIndicatorsOperation({...getMobilityProps(), maximumEntropySequenceRows: 0})
  ).toThrow(/maximumEntropySequenceRows/);
});
import {
  computeMobilityIndicatorsOnCPU,
  CPUMobilityIndicatorsOperation
} from '../../../src/gpu-spatial-analysis/trajectory-mobility/cpu-mobility-indicators-operation';
import {GPUMobilityIndicatorsOperation} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-mobility-indicators-operation';
import scikitMobility from '../../oracles/fixtures/movement-scikit-mobility.json';
import {getMobilityFixture, getMobilityProps} from './mobility-indicators-operation-fixture';
import {computeTrajectoryStaypointsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-staypoints-operation';

it('reports location iteration loss separately from missing keys', () => {
  const input = {
    timestamps: Float32Array.of(0, 1),
    staypoints: {
      tracks: Uint32Array.of(0, 0),
      centroids: Float32Array.of(0, 0, 4, 0),
      startRows: Uint32Array.of(0, 1),
      durations: Float32Array.of(1, 1),
      locationIds: Uint32Array.of(0xffffffff, 0xffffffff),
      count: Uint32Array.of(2),
      locationIterationLimitReached: Uint32Array.of(0)
    }
  };
  const complete = computeMobilityIndicatorsOnCPU(input, {trackCount: 1});
  expect(Array.from(complete.randomEntropy)).toEqual([NaN]);
  expect(Array.from(complete.uncorrelatedEntropy)).toEqual([NaN]);
  expect(complete.candidateOverflow[0]).toBe(0);
  input.staypoints.locationIterationLimitReached[0] = 1;
  const incomplete = computeMobilityIndicatorsOnCPU(input, {trackCount: 1});
  for (const port of ['randomEntropy', 'uncorrelatedEntropy', 'realEntropy'] as const)
    expect(Array.from(incomplete[port])).toEqual([NaN]);
  expect(Array.from(incomplete.homeLocations)).toEqual([0xffffffff]);
  expect(Array.from(incomplete.workLocations)).toEqual([0xffffffff]);
  // Independent of location labels: points at 0 and 4 have RMS radius 2 and jump 4.
  expect(Array.from(incomplete.radiusOfGyration)).toEqual([2]);
  expect(Array.from(incomplete.jumpLengths)).toEqual([0, 4]);
  expect(incomplete.candidateOverflow[0]).toBe(1);
  for (const Operation of [GPUMobilityIndicatorsOperation, CPUMobilityIndicatorsOperation]) {
    const props = getMobilityProps();
    const operation = new Operation({
      ...props,
      staypoints: {
        ...props.staypoints,
        status: {...props.staypoints.status, locationIterationLimitReached: 'locationLoss'}
      }
    });
    expect(operation.inputs['locationIterationLimitReached']).toBe('locationLoss');
  }
});

it('suppresses truncated staypoint indicators and recovers both complete compositions', () => {
  for (const coordinates of [
    [0, 10, 20],
    [0, 0, 10, 10, 20, 20]
  ]) {
    const total = coordinates.length === 3 ? 2 : 3;
    const timestamps = Float32Array.from(coordinates, (_, row) => row);
    for (const capacity of [total - 1, total]) {
      const stays = computeTrajectoryStaypointsOnCPU(
        {
          positions: Float32Array.from(coordinates.flatMap(x => [x, 0])),
          timestamps,
          trackOffsets: Uint32Array.of(0, coordinates.length),
          parameters: Float32Array.of(1, 1, 0, Number(total === 3))
        },
        {capacity}
      );
      expect(stays.requiredCount[0]).toBe(total);
      expect(stays.overflow[0]).toBe(Number(capacity < total));
      const input = {
        timestamps,
        staypoints: {
          tracks: stays.ids,
          centroids: stays.centroids,
          startRows: stays.startRows,
          durations: stays.durations,
          locationIds: Uint32Array.from({length: capacity}, (_, row) => row),
          count: stays.count,
          overflow: stays.overflow,
          candidateOverflow: stays.candidateOverflow
        }
      };
      const result = computeMobilityIndicatorsOnCPU(input, {trackCount: 1});
      if (capacity < total) {
        expect(Array.from(result.radiusOfGyration)).toEqual([NaN]);
        expect(Array.from(result.randomEntropy)).toEqual([NaN]);
        expect(Array.from(result.homeLocations)).toEqual([0xffffffff]);
        expect(result.candidateOverflow[0]).toBe(1);
      } else {
        expect(result.radiusOfGyration[0]).toBeCloseTo(total === 2 ? 5 : Math.sqrt(200 / 3), 5);
        expect(result.randomEntropy[0]).toBeCloseTo(Math.log2(total), 6);
        expect(result.candidateOverflow[0]).toBe(0);
      }
      input.staypoints.overflow[0] = 0;
      input.staypoints.candidateOverflow[0] = 1;
      expect(
        Array.from(computeMobilityIndicatorsOnCPU(input, {trackCount: 1}).randomEntropy)
      ).toEqual([NaN]);
    }
  }
});

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
      gyrationCenter: 'coordinate-mean',
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
    // Independent Python haversine around the arithmetic mean of stored f32 coordinates;
    // .typecheck/r19/scratch/cells-trajectories-recipes-w5-stops-mobility-and-recursion/parity-radius.py.
    // Exact f32 literals distinguish arithmetic parity from the spherical default.
    expect(result.radiusOfGyration[track]).toBe(
      [19.66763687133789, 21.186420440673828, 18.397933959960938][track]
    );
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
  expect(missing.randomEntropy[0]).toBe(0); // The only usable key is 3.
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
  // @ts-expect-error Deliberately empty output exercises the JavaScript guard.
  expect(() => new CPUMobilityIndicatorsOperation({...props, output: {}})).toThrow(
    /^mobility-test requires at least one output/
  );
  expect(
    () => new CPUMobilityIndicatorsOperation({...props, maximumEntropySequenceRows: 1e9})
  ).not.toThrow();
});

it('writes NaN real entropy and the candidate flag for a track over the sequence cap (TRJ-103)', () => {
  // Expected 1.2967875003814697: scikit-mobility 1.3.1 _true_entropy transcribed in
  // /usr/bin/python3 on [0, 1, 0, 1, 2, 0, 1, 2, 3], 9 * log2(9) / 22, rounded to float32.
  const sequence = [0, 1, 0, 1, 2, 0, 1, 2, 3];
  const rows = sequence.length;
  const input = {
    timestamps: Float32Array.from(sequence, (_, row) => row),
    staypoints: {
      tracks: new Uint32Array(rows),
      centroids: new Float32Array(rows * 2),
      startRows: Uint32Array.from(sequence, (_, row) => row),
      durations: new Float32Array(rows).fill(1),
      locationIds: Uint32Array.from(sequence),
      count: Uint32Array.of(rows)
    }
  };
  const complete = computeMobilityIndicatorsOnCPU(input, {
    trackCount: 1,
    maximumEntropySequenceRows: rows
  });
  expect(Number.isFinite(complete.realEntropy[0])).toBe(true);
  expect(complete.realEntropy[0]).toBe(1.2967875003814697);
  expect(complete.candidateOverflow[0]).toBe(0);
  const capped = computeMobilityIndicatorsOnCPU(input, {
    trackCount: 1,
    maximumEntropySequenceRows: rows - 1
  });
  expect(Number.isNaN(capped.realEntropy[0])).toBe(true);
  expect(capped.candidateOverflow[0]).toBe(1);
  // The other indicators are still complete.
  expect(capped.uncorrelatedEntropy[0]).toBe(complete.uncorrelatedEntropy[0]);
});
