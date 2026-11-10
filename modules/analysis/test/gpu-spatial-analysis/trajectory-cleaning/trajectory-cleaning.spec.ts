// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {GPULineSimplification} from '../../../src/gpu-spatial-analysis/line-simplification/gpu-line-simplification';
import {getGPULineSimplificationParameterValues} from '../../../src/gpu-spatial-analysis/line-simplification/line-simplification-parameters';
import {
  GPUTrajectoryHampel,
  GPUTrajectoryOutlierFlags,
  GPUTrajectorySplit,
  GPUTrajectoryThinning,
  GPUTrajectoryTrackRelativeTimes,
  GPU_TRAJECTORY_OUTLIER_FLAGS,
  getGPUTrajectoryHampelParameterValues,
  getGPUTrajectoryOutlierFlagsParameterValues,
  getGPUTrajectorySplitParameterValues,
  getGPUTrajectoryThinningParameterValues
} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/index';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixture from '../../oracles/fixtures/movement-cleaning.json';
import {GraphHarness, importTrajectory} from '../trajectory-staypoints/movement-spec-utils';

const EPOCH_MS = 1_700_000_000_000n;

function makeColumns() {
  const rows = fixture.trajectory.rows;
  return {
    rows,
    columns: {
      positions: rows.map(row => [row[0], row[1]]),
      times: rows.map(row => EPOCH_MS + BigInt(row[2]) * 1000n),
      trackOffsets: [0, rows.length]
    }
  };
}

describe('GPUTrajectorySplit', () => {
  it('matches the ObservationGapSplitter', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new GraphHarness(device, 'split-gap');
    const {rows, columns} = makeColumns();
    const {positions, timestamps, trackOffsets} = importTrajectory(harness, columns);
    const capacity = rows.length;
    const offsets = harness.output('offsets', 'uint32', capacity + 1);
    const tracks = harness.output('tracks', 'uint32', capacity);
    const rowIds = harness.output('rows', 'uint32', capacity);
    const count = harness.output('count', 'uint32', 1);
    const overflow = harness.output('overflow', 'uint32', 1);
    harness.graph.add(
      new GPUTrajectorySplit({
        mode: 'gap',
        positions,
        timestamps,
        trackOffsets,
        parameters: harness.input(
          'gap-parameters',
          getGPUTrajectorySplitParameterValues({maximumGap: fixture.trajectory.gapSeconds * 1000}),
          'float32',
          4
        ),
        output: {
          segmentOffsets: offsets.view,
          segmentTracks: tracks.view,
          rowIds: rowIds.view,
          status: {count: count.view, overflow: overflow.view}
        }
      })
    );
    const segments = await harness.run(async () => {
      const total = (await readUint32(count.buffer, 1))[0];
      const offsetValues = await readUint32(offsets.buffer, total + 1);
      const ids = await readUint32(rowIds.buffer, offsetValues[total]);
      return Array.from({length: total}, (_, index) =>
        ids.slice(offsetValues[index], offsetValues[index + 1])
      );
    });
    expect(segments).toEqual(fixture.gapSegments);
  });

  it('matches the ValueChangeSplitter including the repeated boundary row', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new GraphHarness(device, 'split-value');
    const {rows, columns} = makeColumns();
    const {positions, timestamps, trackOffsets} = importTrajectory(harness, columns);
    const capacity = rows.length * 2;
    const offsets = harness.output('offsets', 'uint32', capacity + 1);
    const tracks = harness.output('tracks', 'uint32', capacity);
    const rowIds = harness.output('rows', 'uint32', capacity);
    const count = harness.output('count', 'uint32', 1);
    const overflow = harness.output('overflow', 'uint32', 1);
    harness.graph.add(
      new GPUTrajectorySplit({
        mode: 'value-change',
        positions,
        timestamps,
        trackOffsets,
        values: harness.input(
          'values',
          Uint32Array.from(rows.map(row => row[3])),
          'uint32',
          rows.length
        ),
        output: {
          segmentOffsets: offsets.view,
          segmentTracks: tracks.view,
          rowIds: rowIds.view,
          status: {count: count.view, overflow: overflow.view}
        }
      })
    );
    const segments = await harness.run(async () => {
      const total = (await readUint32(count.buffer, 1))[0];
      const offsetValues = await readUint32(offsets.buffer, total + 1);
      const ids = await readUint32(rowIds.buffer, offsetValues[total]);
      return Array.from({length: total}, (_, index) =>
        ids.slice(offsetValues[index], offsetValues[index + 1])
      );
    });
    expect(segments).toEqual(fixture.valueSegments);
  });

  it('splits at staypoints and drops one-row runs, with a reference from the definition', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new GraphHarness(device, 'split-stop');
    const rowCount = 20;
    const staypointIds = new Uint32Array(rowCount).fill(0xffffffff);
    // Rows 3..8 and 12..14 are stays; row 9 is a one-row run, dropped; rows 15.. keep 5.
    for (let row = 3; row < 9; row++) staypointIds[row] = 0;
    for (let row = 12; row < 15; row++) staypointIds[row] = 1;
    const {positions, timestamps, trackOffsets} = importTrajectory(harness, {
      positions: Float32Array.from({length: rowCount * 2}, (_, index) => index),
      times: Float32Array.from({length: rowCount}, (_, index) => index),
      trackOffsets: [0, rowCount]
    });
    const offsets = harness.output('offsets', 'uint32', 8);
    const tracks = harness.output('tracks', 'uint32', 7);
    const rowIds = harness.output('rows', 'uint32', rowCount);
    const count = harness.output('count', 'uint32', 1);
    const overflow = harness.output('overflow', 'uint32', 1);
    harness.graph.add(
      new GPUTrajectorySplit({
        mode: 'stop',
        positions,
        timestamps,
        trackOffsets,
        staypointIds: harness.input('stays', staypointIds, 'uint32', rowCount),
        output: {
          segmentOffsets: offsets.view,
          segmentTracks: tracks.view,
          rowIds: rowIds.view,
          status: {count: count.view, overflow: overflow.view}
        }
      })
    );
    const segments = await harness.run(async () => {
      const total = (await readUint32(count.buffer, 1))[0];
      const offsetValues = await readUint32(offsets.buffer, total + 1);
      const ids = await readUint32(rowIds.buffer, offsetValues[total]);
      return Array.from({length: total}, (_, index) =>
        ids.slice(offsetValues[index], offsetValues[index + 1])
      );
    });
    expect(segments).toEqual([
      [0, 1, 2],
      [9, 10, 11],
      [15, 16, 17, 18, 19]
    ]);
  });
});

describe('GPUTrajectoryThinning', () => {
  for (const mode of ['minimum-distance', 'minimum-time'] as const) {
    it(`matches MovingPandas ${mode}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = new GraphHarness(device, `thinning-${mode}`);
      const {rows, columns} = makeColumns();
      const {positions, timestamps, trackOffsets} = importTrajectory(harness, columns);
      const expected = mode === 'minimum-distance' ? fixture.minimumDistance : fixture.minimumTime;
      const tolerance =
        mode === 'minimum-distance'
          ? fixture.minimumDistance.tolerance
          : fixture.minimumTime.toleranceSeconds * 1000;
      const rowIds = harness.output('row-ids', 'uint32', rows.length);
      const outOffsets = harness.output('out-offsets', 'uint32', 2);
      const count = harness.output('count', 'uint32', 1);
      const overflow = harness.output('overflow', 'uint32', 1);
      const candidateOverflow = harness.output('candidate', 'uint32', 1);
      harness.graph.add(
        new GPUTrajectoryThinning({
          mode,
          positions,
          timestamps,
          trackOffsets,
          parameters: harness.input(
            'thin-parameters',
            getGPUTrajectoryThinningParameterValues({tolerance}),
            'float32',
            4
          ),
          output: {
            rowIds: rowIds.view,
            trackOffsets: outOffsets.view,
            status: {
              count: count.view,
              overflow: overflow.view,
              candidateOverflow: candidateOverflow.view
            }
          }
        })
      );
      const result = await harness.run(async () => ({
        count: (await readUint32(count.buffer, 1))[0],
        rows: await readUint32(rowIds.buffer, expected.keptRows.length),
        offsets: await readUint32(outOffsets.buffer, 2),
        candidateOverflow: (await readUint32(candidateOverflow.buffer, 1))[0]
      }));
      expect(result.rows).toEqual(expected.keptRows);
      expect(result.count).toBe(expected.keptRows.length);
      expect(result.offsets).toEqual([0, expected.keptRows.length]);
      expect(result.candidateOverflow).toBe(0);
    });
  }
});

describe('time-aware simplification over word timestamps', () => {
  it('matches MovingPandas TopDownTimeRatio through per-track relative times', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new GraphHarness(device, 'tdtr-words');
    const {rows, columns} = makeColumns();
    const {positions, timestamps, trackOffsets, rowCount} = importTrajectory(harness, columns);
    const relative = harness.output('relative', 'float32', rowCount);
    harness.graph.add(
      new GPUTrajectoryTrackRelativeTimes({
        timestamps: timestamps as never,
        trackOffsets,
        output: relative.view
      })
    );
    const ids = harness.output('ids', 'uint32', rowCount);
    const count = harness.output('count', 'uint32', 1);
    const overflow = harness.output('overflow', 'uint32', 1);
    const importance = harness.output('importance', 'float32', rowCount);
    harness.graph.add(
      new GPULineSimplification({
        positions,
        lineOffsets: trackOffsets,
        timestamps: relative.view,
        metric: 'time-ratio',
        importance: importance.view,
        parameters: harness.input(
          'tdtr-parameters',
          getGPULineSimplificationParameterValues({tolerance: fixture.timeRatio.tolerance}),
          'float32',
          4
        ),
        output: {ids: ids.view, status: {count: count.view, overflow: overflow.view}}
      })
    );
    const kept = await harness.run(async () => {
      const total = (await readUint32(count.buffer, 1))[0];
      return readUint32(ids.buffer, total);
    });
    expect(kept).toEqual(fixture.timeRatio.keptRows);
    expect(rows.length).toBe(rowCount);
  });
});

describe('GPUTrajectoryOutlierFlags', () => {
  it('flags only the displaced fix, matching MovingPandas on isolated spikes', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new GraphHarness(device, 'outliers');
    const rows = fixture.spiky.rows;
    const {positions, timestamps, trackOffsets} = importTrajectory(harness, {
      positions: rows.map(row => [row[0], row[1]]),
      times: Float32Array.from(rows, row => row[2]),
      trackOffsets: [0, rows.length]
    });
    const flags = harness.output('flags', 'uint32', rows.length);
    const counts = harness.output('counts', 'uint32', 1);
    harness.graph.add(
      new GPUTrajectoryOutlierFlags({
        positions,
        timestamps,
        trackOffsets,
        parameters: harness.input(
          'outlier-parameters',
          getGPUTrajectoryOutlierFlagsParameterValues({maximumSpeed: fixture.spiky.vMax}),
          'float32',
          4
        ),
        output: {flags: flags.view, trackFlaggedCounts: counts.view}
      })
    );
    const result = await harness.run(async () => ({
      flags: await readUint32(flags.buffer, rows.length),
      count: (await readUint32(counts.buffer, 1))[0]
    }));
    const flagged = result.flags
      .map((value, index) => ((value & GPU_TRAJECTORY_OUTLIER_FLAGS.speed) !== 0 ? index : -1))
      .filter(index => index >= 0);
    expect(flagged).toEqual(fixture.spiky.removedRows);
    expect(result.count).toBe(fixture.spiky.removedRows.length);
    // The step test of addTrackQualityRecipe would flag rows 9, 10, 25, 26.
    expect(flagged).not.toContain(10);
    expect(flagged).not.toContain(26);
  });

  it('does not catch two consecutive displaced fixes, and flags acceleration spikes', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new GraphHarness(device, 'outliers-burst');
    // Steady 10 units per step; rows 5 and 6 jump together, so the step between them is slow.
    const rowCount = 12;
    const rows: number[][] = [];
    for (let index = 0; index < rowCount; index++) {
      rows.push([index === 5 || index === 6 ? index * 10 + 1000 : index * 10, 0]);
    }
    // Compute the expected flags by the point-centred definition on a separate track with no burst
    // for the acceleration rule: an acceleration spike at row 12 of a smooth ramp.
    const rampRows: number[][] = [];
    let x = 0;
    const speeds = [10, 10, 10, 10, 200, 10, 10, 10, 10, 10];
    rampRows.push([0, 0]);
    for (const speed of speeds) {
      x += speed;
      rampRows.push([x, 0]);
    }
    const expectedRampFlags = new Uint32Array(rampRows.length);
    const rampSpeeds = (row: number) => Math.abs(rampRows[row][0] - rampRows[row - 1][0]);
    const accelerationIn = (row: number) =>
      row >= 2 ? Math.abs(rampSpeeds(row) - rampSpeeds(row - 1)) > 50 : false;
    for (let row = 1; row + 1 < rampRows.length; row++) {
      if (accelerationIn(row) && accelerationIn(row + 1)) {
        expectedRampFlags[row] = GPU_TRAJECTORY_OUTLIER_FLAGS.acceleration;
      }
    }
    const allRows = [...rows, ...rampRows];
    const offsets = [0, rowCount, rowCount + rampRows.length];
    const {positions, timestamps, trackOffsets} = importTrajectory(harness, {
      positions: allRows,
      times: Float32Array.from({length: allRows.length}, (_, index) => index),
      trackOffsets: offsets
    });
    const flags = harness.output('flags', 'uint32', allRows.length);
    harness.graph.add(
      new GPUTrajectoryOutlierFlags({
        positions,
        timestamps,
        trackOffsets,
        parameters: harness.input(
          'outlier-burst-parameters',
          getGPUTrajectoryOutlierFlagsParameterValues({maximumSpeed: 100, maximumAcceleration: 50}),
          'float32',
          4
        ),
        output: {flags: flags.view}
      })
    );
    const values = await harness.run(async () => readUint32(flags.buffer, allRows.length));
    // Single-pass limitation documented on the class.
    expect(values[5] & GPU_TRAJECTORY_OUTLIER_FLAGS.speed).toBe(0);
    expect(values[6] & GPU_TRAJECTORY_OUTLIER_FLAGS.speed).toBe(0);
    const rampFlags = values
      .slice(rowCount)
      .map(value => value & GPU_TRAJECTORY_OUTLIER_FLAGS.acceleration);
    expect(rampFlags).toEqual(Array.from(expectedRampFlags));
    expect(rampFlags.some(value => value !== 0)).toBe(true);
  });
});

describe('GPUTrajectoryHampel', () => {
  it('matches the hampel package used by PTRAIL', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new GraphHarness(device, 'hampel');
    const {series, windowSize, nSigma, filtered, outlierIndices} = fixture.hampel;
    const rowCount = series.length;
    const values = harness.input('values', Float32Array.from(series), 'float32', rowCount);
    const trackOffsets = harness.input('offsets', Uint32Array.from([0, rowCount]), 'uint32', 2);
    const output = harness.output('filtered', 'float32', rowCount);
    const flags = harness.output('flags', 'uint32', rowCount);
    const count = harness.output('count', 'uint32', 1);
    harness.graph.add(
      new GPUTrajectoryHampel({
        values,
        trackOffsets,
        windowSize,
        parameters: harness.input(
          'hampel-parameters',
          getGPUTrajectoryHampelParameterValues({numberOfSigmas: nSigma}),
          'float32',
          4
        ),
        output: {filtered: output.view, flags: flags.view, outlierCount: count.view}
      })
    );
    const result = await harness.run(async () => ({
      filtered: await readFloat32(output.buffer, rowCount),
      flags: await readUint32(flags.buffer, rowCount),
      count: (await readUint32(count.buffer, 1))[0]
    }));
    expect(result.count).toBe(outlierIndices.length);
    expect(
      result.flags.map((value, index) => (value ? index : -1)).filter(index => index >= 0)
    ).toEqual(outlierIndices);
    filtered.forEach((value, index) => expect(result.filtered[index]).toBeCloseTo(value, 4));
  });
});
