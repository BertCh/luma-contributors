// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {
  GPU_MOVEMENT_SENTINEL,
  GPUTrajectoryStaypoints,
  getGPUTrajectoryStaypointsParameterValues,
  type GPUTrajectoryStaypointsSettings
} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/index';
import {GPUTrajectoryMetrics} from '../../../src/gpu-spatial-analysis/trajectory-analysis/gpu-trajectory-metrics';
import {getGPUTrajectoryMetricsParameterValues} from '../../../src/gpu-spatial-analysis/trajectory-analysis/trajectory-metrics-parameters';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixture from '../../oracles/fixtures/movement-trackintel.json';
import {GraphHarness, importTrajectory, type TrajectoryColumns} from './movement-spec-utils';
import {
  detectStaypointsReference,
  euclideanDistance,
  haversineDistance,
  type ReferenceStaypoint
} from './staypoint-reference';

const GREAT_CIRCLE = {
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  sphereRadius: 6371000
} as const;

/** Runs GPUTrajectoryStaypoints with full outputs and reads them back. */
async function detectOnGPU(
  columns: TrajectoryColumns,
  settings: GPUTrajectoryStaypointsSettings,
  options: {
    capacity?: number;
    maximumSearchRows?: number;
    spatialContext?: typeof GREAT_CIRCLE;
  } = {}
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new GraphHarness(device, 'staypoints-test');
  const {positions, timestamps, trackOffsets, rowCount} = importTrajectory(harness, columns);
  const capacity = options.capacity ?? 64;
  const parameters = harness.input(
    'parameters',
    getGPUTrajectoryStaypointsParameterValues(settings),
    'float32',
    4
  );
  const ids = harness.output('ids', 'uint32', capacity);
  const count = harness.output('count', 'uint32', 1);
  const overflow = harness.output('overflow', 'uint32', 1);
  const requiredCount = harness.output('required', 'uint32', 1);
  const candidateOverflow = harness.output('candidate', 'uint32', 1);
  const startRows = harness.output('starts', 'uint32', capacity);
  const endRows = harness.output('ends', 'uint32', capacity);
  const finishRows = harness.output('finishes', 'uint32', capacity);
  const centroids = harness.output('centroids', 'float32x2', capacity);
  const durations = harness.output('durations', 'float32', capacity);
  const staypointIds = harness.output('row-ids', 'uint32', rowCount);
  harness.graph.add(
    new GPUTrajectoryStaypoints({
      positions,
      timestamps,
      trackOffsets,
      parameters,
      spatialContext: options.spatialContext,
      maximumSearchRows: options.maximumSearchRows,
      output: {
        ids: ids.view,
        status: {
          count: count.view,
          overflow: overflow.view,
          requiredCount: requiredCount.view,
          candidateOverflow: candidateOverflow.view
        },
        startRows: startRows.view,
        endRows: endRows.view,
        finishRows: finishRows.view,
        centroids: centroids.view,
        durations: durations.view,
        staypointIds: staypointIds.view
      }
    })
  );
  return harness.run(async () => {
    const stayCount = (await readUint32(count.buffer, 1))[0];
    const tracks = await readUint32(ids.buffer, stayCount);
    const starts = await readUint32(startRows.buffer, stayCount);
    const ends = await readUint32(endRows.buffer, stayCount);
    const finishes = await readUint32(finishRows.buffer, stayCount);
    const centroidValues = await readFloat32(centroids.buffer, stayCount * 2);
    const durationValues = await readFloat32(durations.buffer, stayCount);
    return {
      count: stayCount,
      overflow: (await readUint32(overflow.buffer, 1))[0],
      requiredCount: (await readUint32(requiredCount.buffer, 1))[0],
      candidateOverflow: (await readUint32(candidateOverflow.buffer, 1))[0],
      staypoints: starts.map((startRow, index) => ({
        track: tracks[index],
        startRow,
        endRow: ends[index],
        finishRow: finishes[index],
        centroid: [centroidValues[index * 2], centroidValues[index * 2 + 1]] as [number, number],
        duration: durationValues[index]
      })),
      rowStaypointIds: await readUint32(staypointIds.buffer, rowCount)
    };
  });
}

function expectStaypointsMatch(
  actual: ReferenceStaypoint[],
  expected: ReferenceStaypoint[],
  centroidTolerance: number
): void {
  expect(actual.length).toBe(expected.length);
  actual.forEach((staypoint, index) => {
    const reference = expected[index];
    expect(staypoint.track).toBe(reference.track);
    expect(staypoint.startRow).toBe(reference.startRow);
    expect(staypoint.endRow).toBe(reference.endRow);
    expect(staypoint.finishRow).toBe(reference.finishRow);
    expect(Math.abs(staypoint.centroid[0] - reference.centroid[0])).toBeLessThan(centroidTolerance);
    expect(Math.abs(staypoint.centroid[1] - reference.centroid[1])).toBeLessThan(centroidTolerance);
    expect(staypoint.duration).toBeCloseTo(reference.duration, 0);
  });
}

describe('GPUTrajectoryStaypoints', () => {
  it('matches the trackintel fixture on every case, words and float32 times', async () => {
    const {scenario, cases} = fixture;
    const positions = scenario.positions;
    const relativeOrigin = BigInt(scenario.timesMs[0]);
    for (const testCase of cases) {
      const settings = {
        spatialContext: GREAT_CIRCLE,
        distanceThreshold: testCase.distanceThreshold,
        timeThreshold: testCase.timeThresholdMinutes * 60_000,
        maximumGap: testCase.gapThresholdMinutes * 60_000,
        includeLast: testCase.includeLast
      };
      const wordResult = await detectOnGPU(
        {
          positions,
          times: scenario.timesMs.map(value => BigInt(value)),
          trackOffsets: scenario.trackOffsets
        },
        settings,
        {spatialContext: GREAT_CIRCLE}
      );
      const expected: ReferenceStaypoint[] = testCase.staypoints.map(staypoint => ({
        track: staypoint.track,
        startRow: staypoint.startRow,
        endRow: staypoint.endRow,
        finishRow: staypoint.finishRow,
        centroid: staypoint.centroid as [number, number],
        duration: staypoint.durationMs
      }));
      expect(wordResult.candidateOverflow).toBe(0);
      expect(wordResult.overflow).toBe(0);
      expect(wordResult.requiredCount).toBe(expected.length);
      // float32 degree positions round the centroid by about one float32 ulp of 47 degrees.
      expectStaypointsMatch(wordResult.staypoints, expected, 5e-6);
      const rowIds = testCase.rowStaypointIds.map(id => (id < 0 ? GPU_MOVEMENT_SENTINEL : id));
      expect(wordResult.rowStaypointIds).toEqual(rowIds);

      const floatResult = await detectOnGPU(
        {
          positions,
          times: Float32Array.from(scenario.timesMs, value => value - Number(relativeOrigin)),
          trackOffsets: scenario.trackOffsets
        },
        settings,
        {spatialContext: GREAT_CIRCLE}
      );
      expectStaypointsMatch(floatResult.staypoints, expected, 5e-6);
    }
  }, 120_000);

  it('matches a CPU sliding-window reference on random tracks', async () => {
    let seed = 12345;
    const random = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };
    const rows: number[][] = [];
    const times: number[] = [];
    const trackOffsets = [0];
    for (let track = 0; track < 4; track++) {
      let x = random() * 5000;
      let y = random() * 5000;
      let time = 0;
      for (let block = 0; block < 12; block++) {
        const moving = random() < 0.5;
        for (let step = 0; step < 15 + Math.floor(random() * 30); step++) {
          x += (random() - 0.5) * (moving ? 90 : 25);
          y += (random() - 0.5) * (moving ? 90 : 25);
          time += 10 + Math.floor(random() * 30) + (random() < 0.02 ? 900 : 0);
          rows.push([Math.round(x * 16) / 16, Math.round(y * 16) / 16]);
          times.push(time);
        }
      }
      trackOffsets.push(rows.length);
    }
    const referenceSettings = {
      distanceThreshold: 60,
      timeThreshold: 120,
      maximumGap: 600,
      includeLast: true,
      metric: euclideanDistance
    };
    const expected = detectStaypointsReference(rows, times, trackOffsets, referenceSettings);
    expect(expected.length).toBeGreaterThan(5);
    const result = await detectOnGPU(
      {positions: rows, times: Float32Array.from(times), trackOffsets},
      {distanceThreshold: 60, timeThreshold: 120, maximumGap: 600, includeLast: true},
      {capacity: 256}
    );
    expectStaypointsMatch(result.staypoints, expected, 1e-1);
  });

  it('keeps one stop as one stop under parked GPS jitter, where per-step speed splits it', async () => {
    let seed = 42;
    const random = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };
    // 1 Hz parked for 30 minutes with 3-5 m of multipath jitter, then a drive away.
    const rows: number[][] = [];
    const times: number[] = [];
    for (let second = 0; second < 1800; second++) {
      const angle = random() * 2 * Math.PI;
      const radius = 1 + random() * 4;
      rows.push([Math.cos(angle) * radius, Math.sin(angle) * radius]);
      times.push(second);
    }
    for (let second = 1800; second < 1860; second++) {
      rows.push([(second - 1799) * 40, 0]);
      times.push(second);
    }
    const reference = detectStaypointsReference(rows, times, [0, rows.length], {
      distanceThreshold: 100,
      timeThreshold: 300,
      metric: euclideanDistance
    });
    expect(reference).toHaveLength(1);
    expect(reference[0].startRow).toBe(0);

    const result = await detectOnGPU(
      {positions: rows, times: Float32Array.from(times), trackOffsets: [0, rows.length]},
      {distanceThreshold: 100, timeThreshold: 300}
    );
    expect(result.count).toBe(1);
    expect(result.staypoints[0].startRow).toBe(0);
    expect(result.staypoints[0].endRow).toBe(reference[0].endRow);
    expect(result.candidateOverflow).toBe(0);

    // The per-step speed rule the review criticised: the same data splits into many stops.
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new GraphHarness(device, 'jitter-metrics');
    const {positions, timestamps, trackOffsets} = importTrajectory(harness, {
      positions: rows,
      times: Float32Array.from(times),
      trackOffsets: [0, rows.length]
    });
    const stopCount = harness.output('stop-counts', 'uint32', 1);
    harness.graph.add(
      new GPUTrajectoryMetrics({
        positions,
        timestamps,
        trackOffsets,
        parameters: harness.input(
          'metric-parameters',
          getGPUTrajectoryMetricsParameterValues({stopSpeedThreshold: 2, stopMinimumDuration: 3}),
          'float32',
          4
        ),
        output: {trackStopCounts: stopCount.view}
      })
    );
    const perStepStops = await harness.run(async () => (await readUint32(stopCount.buffer, 1))[0]);
    expect(perStepStops).toBeGreaterThan(1);
  });

  it('flags a stay longer than maximumSearchRows and reports it by candidateOverflow', async () => {
    const rows: number[][] = [];
    const times: number[] = [];
    for (let second = 0; second < 900; second++) {
      rows.push([0, 0]);
      times.push(second);
    }
    rows.push([1000, 0]);
    times.push(900);
    const result = await detectOnGPU(
      {positions: rows, times: Float32Array.from(times), trackOffsets: [0, rows.length]},
      {distanceThreshold: 10, timeThreshold: 60},
      {maximumSearchRows: 256}
    );
    expect(result.candidateOverflow).toBe(1);
    expect(result.overflow).toBe(0);
    expect(result.count).toBe(4);
    expect(result.requiredCount).toBe(4);
    expect(result.staypoints.map(staypoint => staypoint.startRow)).toEqual([0, 257, 514, 771]);
  });

  it('honours the shared gap rule and capacity overflow', async () => {
    // A 100 s gap at the limit does not split; strictly longer does.
    const rows: number[][] = [];
    const times: number[] = [];
    for (let second = 0; second < 300; second++) {
      rows.push([0, 0]);
      times.push(second);
    }
    for (let second = 300; second < 600; second++) {
      rows.push([0, 0]);
      times.push(second + 100); // step 299 -> 300 (time 299 -> 400) is 101 long
    }
    rows.push([500, 0]);
    times.push(720); // no gap at the break itself
    const atLimit = await detectOnGPU(
      {positions: rows, times: Float32Array.from(times), trackOffsets: [0, rows.length]},
      {distanceThreshold: 10, timeThreshold: 60, maximumGap: 101},
      {capacity: 1}
    );
    expect(atLimit.count).toBe(1);
    expect(atLimit.overflow).toBe(0);
    // A step exactly at the limit is not a gap: the window spans it from row 0.
    expect(atLimit.staypoints[0].startRow).toBe(0);
    expect(atLimit.staypoints[0].endRow).toBe(600);
    const belowLimit = await detectOnGPU(
      {positions: rows, times: Float32Array.from(times), trackOffsets: [0, rows.length]},
      {distanceThreshold: 10, timeThreshold: 60, maximumGap: 100},
      {capacity: 1}
    );
    expect(belowLimit.requiredCount).toBe(1); // the gap cancels the first window; the second stays.
    const reference = detectStaypointsReference(rows, times, [0, rows.length], {
      distanceThreshold: 10,
      timeThreshold: 60,
      maximumGap: 100,
      metric: euclideanDistance
    });
    expect(reference).toHaveLength(belowLimit.requiredCount);
    expect(belowLimit.staypoints[0].startRow).toBe(reference[0].startRow);
  });

  it('computes haversine distances for great-circle contexts', () => {
    expect(haversineDistance([0, 0], [1, 0], 6371000)).toBeCloseTo(111194.9, 0);
  });
});
