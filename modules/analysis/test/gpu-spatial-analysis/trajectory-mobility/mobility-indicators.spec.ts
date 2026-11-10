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
import {requireTestDevice} from '../../utils/require-test-device';
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
