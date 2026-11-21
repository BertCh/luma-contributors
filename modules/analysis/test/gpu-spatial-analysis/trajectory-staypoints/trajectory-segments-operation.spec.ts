// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUTrajectorySegmentsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-segments-operation';
import {GPUTrajectorySegmentsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-segments-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {getSegmentsFixture, getSegmentsProps} from './trajectory-segments-operation-fixture';

it('refuses omitted candidate destinations before creating either operation graph', () => {
  for (const Operation of [GPUTrajectorySegmentsOperation, CPUTrajectorySegmentsOperation]) {
    for (const capacity of [1, 2, 6]) {
      const props = getSegmentsProps(capacity, 5);
      delete props.output.trips!.status.candidateOverflow;
      expect(() => new Operation(props)).toThrow(/output.trips.status.candidateOverflow/);
      const upstream = getSegmentsProps(capacity, 5);
      upstream.staypoints!.status.candidateOverflow = 'stayCandidateOverflow';
      delete upstream.output.triplegs.status.candidateOverflow;
      expect(() => new Operation(upstream)).toThrow(/output.triplegs.status.candidateOverflow/);
    }
  }
});

it('propagates lost triplegs in a two-moving-track graph without final trip overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUTrajectorySegmentsOperation, CPUTrajectorySegmentsOperation]) {
    const props = {...getSegmentsProps(1, 2), rowCount: 4};
    const operation = new Operation(props);
    delete props.output.trips!.status.candidateOverflow;
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0)},
        timestamps: {format: 'float32', values: Float32Array.of(0, 1, 0, 1)},
        trackOffsets: {format: 'uint32', values: Uint32Array.of(0, 2, 4)},
        staypointIds: {format: 'uint32', values: new Uint32Array(4).fill(0xffffffff)},
        parameters: {format: 'float32', values: Float32Array.of(0, 1, 0, 0)},
        stayTracks: {format: 'uint32', values: Uint32Array.of(0)},
        stayStartRows: {format: 'uint32', values: Uint32Array.of(0)},
        stayFinishRows: {format: 'uint32', values: Uint32Array.of(0)},
        stayDurations: {format: 'float32', values: Float32Array.of(0)},
        stayCount: {format: 'uint32', values: Uint32Array.of(0)}
      },
      operations: [operation],
      outputs: ['tripCount', 'tripOverflow', 'tripCandidateOverflow', 'triplegOverflow']
    });
    expect(Array.from(result['tripCount'])).toEqual([1]);
    expect(Array.from(result['tripOverflow'])).toEqual([0]);
    expect(Array.from(result['tripCandidateOverflow'])).toEqual([1]);
    expect(Array.from(result['triplegOverflow'])).toEqual([1]);
  }
});

for (const scene of [
  'ordinary',
  'overflow',
  'stationary',
  'tripleg-loss',
  'upstream-loss'
] as const) {
  it(`segments compare every integer port exactly: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = getSegmentsFixture();
    const props =
      scene === 'overflow'
        ? getSegmentsProps(3, 1)
        : scene === 'tripleg-loss'
          ? getSegmentsProps(3, 5)
          : getSegmentsProps();
    if (scene === 'upstream-loss')
      props.staypoints!.status.candidateOverflow = 'stayCandidateOverflow';
    const inputs = {
      positions: {
        format: 'float32x2' as const,
        values: scene === 'stationary' ? new Float32Array(44) : fixture.positions
      },
      timestamps: {format: 'float32' as const, values: fixture.timestamps},
      trackOffsets: {format: 'uint32' as const, values: fixture.trackOffsets},
      staypointIds: {format: 'uint32' as const, values: fixture.staypointIds},
      parameters: {format: 'float32' as const, values: fixture.parameters},
      stayTracks: {format: 'uint32' as const, values: fixture.staypoints.tracks},
      stayStartRows: {format: 'uint32' as const, values: fixture.staypoints.startRows},
      stayFinishRows: {format: 'uint32' as const, values: fixture.staypoints.finishRows},
      stayDurations: {format: 'float32' as const, values: fixture.staypoints.durations},
      stayCount: {format: 'uint32' as const, values: fixture.staypoints.count},
      ...(scene === 'upstream-loss'
        ? {stayCandidateOverflow: {format: 'uint32' as const, values: Uint32Array.of(1)}}
        : {})
    };
    const gpuOperation = new GPUTrajectorySegmentsOperation(props);
    const outputs = Object.values(gpuOperation.outputs).map(output => output.value);
    const gpu = await runOperationGraph(device, {inputs, operations: [gpuOperation], outputs});
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUTrajectorySegmentsOperation(props)],
      outputs
    });
    // No floating outputs: exact comparisons, including sentinels and capacity rows.
    for (const output of outputs) expect(Array.from(gpu[output])).toEqual(Array.from(cpu[output]));
    expect(cpu['triplegRequiredCount'][0]).toBe(scene === 'stationary' ? 0 : 5);
    expect(cpu['tripRequiredCount'][0]).toBe(
      scene === 'stationary' ? 0 : scene === 'overflow' || scene === 'tripleg-loss' ? 2 : 4
    );
    expect(cpu['tripCandidateOverflow'][0]).toBe(
      Number(scene === 'overflow' || scene === 'tripleg-loss' || scene === 'upstream-loss')
    );
    expect(cpu['tripOverflow'][0]).toBe(Number(scene === 'overflow'));
    expect(cpu['triplegCandidateOverflow'][0]).toBe(Number(scene === 'upstream-loss'));
  });
}

it('parallel trip offsets preserve empty tracks, ordering and retained prefixes across workgroups', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const trackCount = 513;
  const moving = new Set([1, 257, 512]);
  const stationary = new Set([0, 256, 511]);
  const offsets = [0];
  const positions: number[] = [];
  const timestamps: number[] = [];
  const starts: number[] = [];
  for (let track = 0; track < trackCount; track++) {
    if (moving.has(track) || stationary.has(track)) {
      if (moving.has(track)) starts.push(timestamps.length);
      positions.push(0, 0, moving.has(track) ? 1 : 0, 0);
      timestamps.push(0, 1);
    }
    offsets.push(timestamps.length);
  }
  // Hand timeline: the only nonstationary tracks are 1,257,512; each makes one trip.
  // Other tracks have no rows or a stationary two-row run, hence zero trips.
  expect(starts).toEqual([2, 6, 10]);
  for (const [triplegCapacity, tripCapacity, count, required, finalLoss, candidateLoss] of [
    [6, 5, 3, 3, 0, 0],
    [6, 2, 2, 3, 1, 0],
    [2, 5, 2, 2, 0, 1]
  ]) {
    const operation = new GPUTrajectorySegmentsOperation({
      ...getSegmentsProps(triplegCapacity, tripCapacity),
      rowCount: timestamps.length
    });
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: Float32Array.from(positions)},
        timestamps: {format: 'float32', values: Float32Array.from(timestamps)},
        trackOffsets: {format: 'uint32', values: Uint32Array.from(offsets)},
        staypointIds: {
          format: 'uint32',
          values: new Uint32Array(timestamps.length).fill(0xffffffff)
        },
        parameters: {format: 'float32', values: Float32Array.of(0, 1, 0, 0)},
        stayTracks: {format: 'uint32', values: Uint32Array.of(0)},
        stayStartRows: {format: 'uint32', values: Uint32Array.of(0)},
        stayFinishRows: {format: 'uint32', values: Uint32Array.of(0)},
        stayDurations: {format: 'float32', values: Float32Array.of(0)},
        stayCount: {format: 'uint32', values: Uint32Array.of(0)}
      },
      operations: [operation],
      outputs: [
        'tripIds',
        'tripStartRows',
        'tripFinishRows',
        'tripCount',
        'tripRequiredCount',
        'tripOverflow',
        'tripCandidateOverflow',
        'triplegOverflow',
        'triplegRequiredCount'
      ]
    });
    expect(Array.from(result['tripCount'])).toEqual([count]);
    expect(Array.from(result['tripRequiredCount'])).toEqual([required]);
    expect(Array.from(result['tripOverflow'])).toEqual([finalLoss]);
    expect(Array.from(result['tripCandidateOverflow'])).toEqual([candidateLoss]);
    expect(Array.from(result['triplegRequiredCount'])).toEqual([3]);
    expect(Array.from(result['triplegOverflow'])).toEqual([triplegCapacity < 3 ? 1 : 0]);
    expect(Array.from(result['tripIds']).slice(0, count)).toEqual([1, 257, 512].slice(0, count));
    expect(Array.from(result['tripStartRows']).slice(0, count)).toEqual([2, 6, 10].slice(0, count));
    expect(Array.from(result['tripFinishRows']).slice(0, count)).toEqual(
      [3, 7, 11].slice(0, count)
    );
    expect(Array.from(result['tripIds']).slice(count)).toEqual(
      new Array(tripCapacity - count).fill(0xffffffff)
    );
  }
});
