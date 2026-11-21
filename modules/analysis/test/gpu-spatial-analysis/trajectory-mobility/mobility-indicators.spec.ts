// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {
  GPUMobilityIndicators,
  GPUModalSplit,
  GPU_MOBILITY_DEFAULT_ENTROPY_SEQUENCE_ROWS
} from '../../../src/gpu-spatial-analysis/trajectory-mobility/index';
import {GPU_MOVEMENT_SENTINEL} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/index';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import scikitMobility from '../../oracles/fixtures/movement-scikit-mobility.json';
import trackintel from '../../oracles/fixtures/movement-trackintel.json';
import {GraphHarness, importTrajectory} from '../trajectory-staypoints/movement-spec-utils';

const SPHERICAL_KILOMETERS = {
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  sphereRadius: 6371
} as const;

function userSpans(offsets: number[]): {begin: number; end: number}[] {
  return offsets.slice(0, -1).map((begin, index) => ({begin, end: offsets[index + 1]}));
}

describe('GPUMobilityIndicators', () => {
  it('matches scikit-mobility individual measures', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {scenario} = scikitMobility;
    const userOffsets = scenario.userOffsets;
    const trackCount = userOffsets.length - 1;
    const rowCount = scenario.latLon.length;
    const harness = new GraphHarness(device, 'mobility-indicators');
    const trackRows = new Uint32Array(rowCount);
    userSpans(userOffsets).forEach(({begin, end}, track) => trackRows.fill(track, begin, end));
    const tracks = harness.input('stay-tracks', trackRows, 'uint32', rowCount);
    const centroids = harness.input(
      'centroids',
      Float32Array.from(scenario.latLon.flatMap(([latitude, longitude]) => [longitude, latitude])),
      'float32x2',
      rowCount
    );
    const startRows = harness.input(
      'start-rows',
      Uint32Array.from({length: rowCount}, (_, row) => row),
      'uint32',
      rowCount
    );
    const locationIds = harness.input(
      'location-ids',
      Uint32Array.from(scenario.locationIds),
      'uint32',
      rowCount
    );
    const durations = harness.input(
      'durations',
      new Float32Array(rowCount).fill(60),
      'float32',
      rowCount
    );
    const timestamps = importTrajectory(harness, {
      positions: new Float32Array(rowCount * 2),
      times: scenario.timesMs.map(value => BigInt(value)),
      trackOffsets: userOffsets
    }).timestamps;
    const countInput = harness.input('stay-count', Uint32Array.from([rowCount]), 'uint32', 1);
    const radius = harness.output('radius', 'float32', trackCount);
    const jumps = harness.output('jumps', 'float32', rowCount);
    const waits = harness.output('waits', 'float32', rowCount);
    const randomEntropy = harness.output('random-entropy', 'float32', trackCount);
    const uncorrelatedEntropy = harness.output('uncorrelated-entropy', 'float32', trackCount);
    const realEntropy = harness.output('real-entropy', 'float32', trackCount);
    harness.graph.add(
      new GPUMobilityIndicators({
        spatialContext: SPHERICAL_KILOMETERS,
        gyrationCenter: 'coordinate-mean',
        trackCount,
        timestamps,
        staypoints: {
          tracks,
          centroids,
          startRows,
          durations,
          locationIds,
          status: {count: countInput}
        },
        output: {
          radiusOfGyration: radius.view,
          jumpLengths: jumps.view,
          waitingTimes: waits.view,
          randomEntropy: randomEntropy.view,
          uncorrelatedEntropy: uncorrelatedEntropy.view,
          realEntropy: realEntropy.view
        }
      })
    );
    await harness.run(async () => {
      const radiusValues = await readFloat32(radius.buffer, trackCount);
      const randomValues = await readFloat32(randomEntropy.buffer, trackCount);
      const uncorrelatedValues = await readFloat32(uncorrelatedEntropy.buffer, trackCount);
      const realValues = await readFloat32(realEntropy.buffer, trackCount);
      for (let track = 0; track < trackCount; track++) {
        const key = String(track);
        expect(radiusValues[track]).toBeCloseTo(
          (scikitMobility.radiusOfGyrationKm as Record<string, number>)[key],
          2
        );
        expect(randomValues[track]).toBeCloseTo(
          (scikitMobility.randomEntropy as Record<string, number>)[key],
          4
        );
        expect(uncorrelatedValues[track]).toBeCloseTo(
          (scikitMobility.uncorrelatedEntropy as Record<string, number>)[key],
          4
        );
        expect(realValues[track]).toBeCloseTo(
          (scikitMobility.realEntropy as Record<string, number>)[key],
          4
        );
      }
      const jumpValues = await readFloat32(jumps.buffer, rowCount);
      const waitValues = await readFloat32(waits.buffer, rowCount);
      for (const [track, {begin, end}] of userSpans(userOffsets).entries()) {
        const key = String(track);
        const expectedJumps = (scikitMobility.jumpLengthsKm as Record<string, number[]>)[key];
        const expectedWaits = (scikitMobility.waitingTimesSeconds as Record<string, number[]>)[key];
        expect(jumpValues[begin]).toBe(0);
        for (let row = begin + 1; row < end; row++) {
          expect(jumpValues[row]).toBeCloseTo(expectedJumps[row - begin - 1], 2);
          expect(Math.abs(waitValues[row] - expectedWaits[row - begin - 1] * 1000)).toBeLessThan(4);
        }
      }
    });
  });

  it('reports overflow of the real entropy sequence cap', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new GraphHarness(device, 'entropy-overflow');
    const rowCount = 16;
    const tracks = harness.input('tracks', new Uint32Array(rowCount), 'uint32', rowCount);
    const timestamps = harness.input('times', new Float32Array(rowCount), 'float32', rowCount);
    const centroids = harness.input(
      'centroids',
      new Float32Array(rowCount * 2),
      'float32x2',
      rowCount
    );
    const startRows = harness.input(
      'starts',
      Uint32Array.from({length: rowCount}, (_, row) => row),
      'uint32',
      rowCount
    );
    const durations = harness.input('durations', new Float32Array(rowCount), 'float32', rowCount);
    const locationIds = harness.input('locations', new Uint32Array(rowCount), 'uint32', rowCount);
    const count = harness.input('count', Uint32Array.from([rowCount]), 'uint32', 1);
    const entropy = harness.output('entropy', 'float32', 1);
    const overflow = harness.output('overflow', 'uint32', 1);
    harness.graph.add(
      new GPUMobilityIndicators({
        trackCount: 1,
        timestamps,
        maximumEntropySequenceRows: 8,
        staypoints: {tracks, centroids, startRows, durations, locationIds, status: {count}},
        output: {realEntropy: entropy.view, status: {candidateOverflow: overflow.view}}
      })
    );
    const flag = await harness.run(async () => (await readUint32(overflow.buffer, 1))[0]);
    expect(flag).toBe(1);
    expect(GPU_MOBILITY_DEFAULT_ENTROPY_SEQUENCE_ROWS).toBeGreaterThan(8);
  });

  it('writes NaN real entropy over the sequence cap without a status view (TRJ-103)', async () => {
    // Expected 1.2967875003814697: scikit-mobility 1.3.1 _true_entropy transcribed in
    // /usr/bin/python3 on [0, 1, 0, 1, 2, 0, 1, 2, 3], 9 * log2(9) / 22, rounded to float32.
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const sequence = [0, 1, 0, 1, 2, 0, 1, 2, 3];
    const rowCount = sequence.length;
    const run = async (maximumEntropySequenceRows: number) => {
      const harness = new GraphHarness(device, `entropy-nan-${maximumEntropySequenceRows}`);
      const entropy = harness.output('entropy', 'float32', 1);
      harness.graph.add(
        new GPUMobilityIndicators({
          trackCount: 1,
          timestamps: harness.input(
            'times',
            Float32Array.from(sequence, (_, row) => row),
            'float32',
            rowCount
          ),
          maximumEntropySequenceRows,
          staypoints: {
            tracks: harness.input('tracks', new Uint32Array(rowCount), 'uint32', rowCount),
            centroids: harness.input(
              'centroids',
              new Float32Array(rowCount * 2),
              'float32x2',
              rowCount
            ),
            startRows: harness.input(
              'starts',
              Uint32Array.from(sequence, (_, row) => row),
              'uint32',
              rowCount
            ),
            durations: harness.input(
              'durations',
              new Float32Array(rowCount).fill(1),
              'float32',
              rowCount
            ),
            locationIds: harness.input('locations', Uint32Array.from(sequence), 'uint32', rowCount),
            status: {count: harness.input('count', Uint32Array.of(rowCount), 'uint32', 1)}
          },
          output: {realEntropy: entropy.view}
        })
      );
      return harness.run(async () => (await readFloat32(entropy.buffer, 1))[0]);
    };
    const complete = await run(rowCount);
    expect(Number.isFinite(complete)).toBe(true);
    expect(complete).toBeCloseTo(1.2967875003814697, 5);
    expect(Number.isNaN(await run(rowCount - 1))).toBe(true);
  });

  it('matches trackintel FREQ home and work, tracking quality and modal split', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const testCase = trackintel.cases.find(candidate => candidate.name === 'include-last')!;
    const {staypoints, triplegs, locationIds} = testCase;
    const trackCount = trackintel.scenario.trackOffsets.length - 1;
    const stayCount = staypoints.length;
    const harness = new GraphHarness(device, 'mobility-trackintel');
    const {timestamps, positions, trackOffsets, rowCount} = importTrajectory(harness, {
      positions: trackintel.scenario.positions,
      times: trackintel.scenario.timesMs.map(value => BigInt(value)),
      trackOffsets: trackintel.scenario.trackOffsets
    });
    const staypointColumns = {
      tracks: harness.input(
        'stay-tracks',
        Uint32Array.from(staypoints.map(row => row.track)),
        'uint32',
        stayCount
      ),
      centroids: harness.input(
        'stay-centroids',
        Float32Array.from(staypoints.flatMap(row => row.centroid)),
        'float32x2',
        stayCount
      ),
      startRows: harness.input(
        'stay-starts',
        Uint32Array.from(staypoints.map(row => row.startRow)),
        'uint32',
        stayCount
      ),
      durations: harness.input(
        'stay-durations',
        Float32Array.from(staypoints.map(row => row.durationMs)),
        'float32',
        stayCount
      ),
      locationIds: harness.input(
        'stay-locations',
        Uint32Array.from(locationIds),
        'uint32',
        stayCount
      ),
      status: {count: harness.input('stay-count', Uint32Array.from([stayCount]), 'uint32', 1)}
    };
    const stayFinishRows = harness.input(
      'stay-finish-rows',
      Uint32Array.from(staypoints.map(row => row.finishRow)),
      'uint32',
      stayCount
    );
    const triplegCount = triplegs.length;
    const triplegTracks = harness.input(
      'tripleg-tracks',
      Uint32Array.from(triplegs.map(row => row.track)),
      'uint32',
      triplegCount
    );
    const triplegStarts = harness.input(
      'tripleg-starts',
      Uint32Array.from(triplegs.map(row => row.startRow)),
      'uint32',
      triplegCount
    );
    const triplegEnds = harness.input(
      'tripleg-ends',
      Uint32Array.from(triplegs.map(row => row.endRow)),
      'uint32',
      triplegCount
    );
    const triplegCountView = harness.input(
      'tripleg-count',
      Uint32Array.from([triplegCount]),
      'uint32',
      1
    );
    const home = harness.output('home', 'uint32', trackCount);
    const work = harness.output('work', 'uint32', trackCount);
    const quality = harness.output('quality', 'float32', trackCount);
    harness.graph.add(
      new GPUMobilityIndicators({
        trackCount,
        timestamps,
        staypoints: staypointColumns,
        triplegs: {
          tracks: triplegTracks,
          startRows: triplegStarts,
          endRows: triplegEnds,
          status: {count: triplegCountView},
          staypointFinishRows: stayFinishRows
        },
        output: {homeLocations: home.view, workLocations: work.view, trackingQuality: quality.view}
      })
    );
    // trackintel's scenario: triplegs alternate slow (0) and motorized (1) in table order.
    const modalSplit = harness.output('modal-split-duration', 'float32', trackCount * 2);
    const countSplit = harness.output('modal-split-count', 'float32', trackCount * 2);
    const modes = harness.input(
      'modes',
      Uint32Array.from({length: triplegCount}, (_, index) => index % 2),
      'uint32',
      triplegCount
    );
    harness.graph.add(
      new GPUModalSplit({
        positions,
        timestamps,
        trackOffsets,
        triplegs: {
          tracks: triplegTracks,
          startRows: triplegStarts,
          endRows: triplegEnds,
          status: {count: triplegCountView}
        },
        modes,
        modeCount: 2,
        metric: 'duration',
        output: {modalSplit: modalSplit.view}
      })
    );
    harness.graph.add(
      new GPUModalSplit({
        id: 'modal-split-counts',
        positions,
        timestamps,
        trackOffsets,
        triplegs: {
          tracks: triplegTracks,
          startRows: triplegStarts,
          endRows: triplegEnds,
          status: {count: triplegCountView}
        },
        modes,
        modeCount: 2,
        metric: 'count',
        output: {modalSplit: countSplit.view}
      })
    );
    await harness.run(async () => {
      expect(rowCount).toBe(trackintel.scenario.timesMs.length);
      const homeLocations = await readUint32(home.buffer, trackCount);
      const workLocations = await readUint32(work.buffer, trackCount);
      const qualityValues = await readFloat32(quality.buffer, trackCount);
      for (let track = 0; track < trackCount; track++) {
        const key = String(track);
        const expectedHome = (testCase.homeLocations as Record<string, number>)[key];
        const expectedWork = (testCase.workLocations as Record<string, number>)[key];
        expect(homeLocations[track]).toBe(expectedHome < 0 ? GPU_MOVEMENT_SENTINEL : expectedHome);
        expect(workLocations[track]).toBe(expectedWork < 0 ? GPU_MOVEMENT_SENTINEL : expectedWork);
        expect(qualityValues[track]).toBeCloseTo(
          (testCase.trackingQuality as Record<string, number>)[key],
          4
        );
      }
      const durationSplit = await readFloat32(modalSplit.buffer, trackCount * 2);
      const countValues = await readFloat32(countSplit.buffer, trackCount * 2);
      for (let track = 0; track < trackCount; track++) {
        const key = String(track);
        const durationExpected = (
          testCase.modalSplit.duration as Record<string, Record<string, number>>
        )[key];
        const countExpected = (testCase.modalSplit.count as Record<string, Record<string, number>>)[
          key
        ];
        expect(durationSplit[track * 2]).toBeCloseTo(
          (durationExpected['slow_mobility'] ?? 0) * 1000,
          -1
        );
        expect(durationSplit[track * 2 + 1]).toBeCloseTo(
          (durationExpected['motorized_mobility'] ?? 0) * 1000,
          -1
        );
        expect(countValues[track * 2]).toBe(countExpected['slow_mobility'] ?? 0);
        expect(countValues[track * 2 + 1]).toBe(countExpected['motorized_mobility'] ?? 0);
      }
    });
  });
});

describe('GPUModalSplit speed classes', () => {
  // Hand-built planar triplegs of two rows each, one track. Speeds are distance over duration:
  // 2, 10, 4 (the first break), 25 (the last break), 60 (above every break), then a tripleg of
  // zero duration and one with a NaN position. Review finding trajectories-lines-02.
  const TRIPLEGS = [
    {length: 20, start: 0, finish: 10},
    {length: 100, start: 20, finish: 30},
    {length: 40, start: 40, finish: 50},
    {length: 250, start: 60, finish: 70},
    {length: 600, start: 80, finish: 90},
    {length: 50, start: 100, finish: 100},
    {length: Number.NaN, start: 110, finish: 120}
  ];

  async function runSpeedClasses(
    name: string,
    modeSpeedUpperBoundsFloat32: number[],
    metric: 'count' | 'distance'
  ): Promise<{modes: number[]; split: number[]}> {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new GraphHarness(device, name);
    const triplegCount = TRIPLEGS.length;
    const {positions, timestamps, trackOffsets} = importTrajectory(harness, {
      positions: TRIPLEGS.flatMap(tripleg => [
        [Number.isNaN(tripleg.length) ? Number.NaN : 0, 0],
        [Number.isNaN(tripleg.length) ? 50 : tripleg.length, 0]
      ]),
      times: Float32Array.from(TRIPLEGS.flatMap(tripleg => [tripleg.start, tripleg.finish])),
      trackOffsets: [0, 2 * triplegCount]
    });
    const predictedModes = harness.output('predicted-modes', 'uint32', triplegCount);
    const modalSplit = harness.output('modal-split', 'float32', modeSpeedUpperBoundsFloat32.length);
    harness.graph.add(
      new GPUModalSplit({
        positions,
        timestamps,
        trackOffsets,
        triplegs: {
          tracks: harness.input(
            'tripleg-tracks',
            new Uint32Array(triplegCount),
            'uint32',
            triplegCount
          ),
          startRows: harness.input(
            'tripleg-starts',
            Uint32Array.from({length: triplegCount}, (_, index) => 2 * index),
            'uint32',
            triplegCount
          ),
          endRows: harness.input(
            'tripleg-ends',
            Uint32Array.from({length: triplegCount}, (_, index) => 2 * index + 2),
            'uint32',
            triplegCount
          ),
          status: {count: harness.input('tripleg-count', Uint32Array.of(triplegCount), 'uint32', 1)}
        },
        modeSpeedUpperBoundsFloat32,
        modeCount: modeSpeedUpperBoundsFloat32.length,
        metric,
        output: {modalSplit: modalSplit.view, predictedModes: predictedModes.view}
      })
    );
    return harness.run(async () => ({
      modes: Array.from(await readUint32(predictedModes.buffer, triplegCount)),
      split: Array.from(await readFloat32(modalSplit.buffer, modeSpeedUpperBoundsFloat32.length))
    }));
  }

  it('puts a speed at or above a finite last break in the last class and leaves unusable triplegs out', async () => {
    const {modes, split} = await runSpeedClasses('modal-split-finite-break', [4, 25], 'distance');
    expect(modes).toEqual([0, 1, 1, 1, 1, GPU_MOVEMENT_SENTINEL, GPU_MOVEMENT_SENTINEL]);
    // Class 0: the 20-unit tripleg. Class 1: 100 + 40 + 250 + 600. The other two are in no class.
    expect(split).toEqual([20, 990]);
  });

  it('classifies left-closed with an infinite last break', async () => {
    const {modes, split} = await runSpeedClasses(
      'modal-split-infinite-break',
      [4, 25, Number.POSITIVE_INFINITY],
      'count'
    );
    expect(modes).toEqual([0, 1, 1, 2, 2, GPU_MOVEMENT_SENTINEL, GPU_MOVEMENT_SENTINEL]);
    expect(split).toEqual([1, 2, 2]);
  });
});
