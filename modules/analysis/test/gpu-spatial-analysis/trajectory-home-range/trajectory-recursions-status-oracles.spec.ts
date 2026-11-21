// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {CPUTrajectoryRecursionsOperation} from '../../../src/gpu-spatial-analysis/trajectory-recursion/cpu-trajectory-recursions-operation';
import {GPUTrajectoryRecursionsOperation} from '../../../src/gpu-spatial-analysis/trajectory-recursion/gpu-trajectory-recursions-operation';
import type {TrajectoryRecursionsOperationProps} from '../../../src/gpu-spatial-analysis/trajectory-recursion/gpu-trajectory-recursions-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/temporal-context';

for (const Operation of [GPUTrajectoryRecursionsOperation, CPUTrajectoryRecursionsOperation]) {
  it(`${Operation.name} retains constructed window and dimensions after caller mutation`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const spatialContext: GPUSpatialContext = {
      coordinateSpace: 'planar',
      metric: 'native',
      units: 'meters'
    };
    const temporalContext: GPUTemporalContext = {
      encoding: 'float32-relative',
      unit: 'second',
      epoch: 0n,
      axis: {kind: 'irregular'}
    };
    const props: TrajectoryRecursionsOperationProps = {
      positions: 'positions',
      timestamps: 'timestamps',
      trackOffsets: 'offsets',
      radii: 'radii',
      rowCount: 4,
      radiusCount: 1,
      timestampFormat: 'float32',
      maximumTrackSegments: 1,
      spatialContext,
      temporalContext,
      output: {residenceTimes: 'residence', status: {candidateOverflow: 'overflow'}}
    };
    const operation = new Operation(props);
    props.maximumTrackSegments = 3;
    props.rowCount = 3;
    props.radiusCount = 2;
    props.timestampFormat = 'uint32x2';
    Object.assign(spatialContext, {coordinateSpace: 'longitude-latitude', metric: 'great-circle'});
    Object.assign(temporalContext.axis, {kind: 'regular', step: 1});
    Object.assign(temporalContext, {encoding: 'int64-words', unit: 'nanosecond', epoch: 9n});
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: new Float32Array(8)},
        timestamps: {format: 'float32', values: Float32Array.from([0, 1, 2, 3])},
        offsets: {format: 'uint32', values: Uint32Array.from([0, 4])},
        radii: {format: 'float32', values: Float32Array.from([1])}
      },
      operations: [operation],
      outputs: ['overflow', 'residence']
    });
    // The constructed one-segment window keeps exactly one unit of residence,
    // and excludes two of the three unit-duration segments at every focal row.
    expect(Array.from(result['overflow'])).toEqual([1]);
    expect(Array.from(result['residence'])).toEqual([1, 1, 1, 1]);
  });
}

for (const Operation of [GPUTrajectoryRecursionsOperation, CPUTrajectoryRecursionsOperation]) {
  it(`${Operation.name} reads getter settings once and publishes inherited outputs`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const reads = {
      maximumTrackSegments: 0,
      rowCount: 0,
      radiusCount: 0,
      timestampFormat: 0,
      residence: 0,
      status: 0,
      candidate: 0
    };
    class Status {
      get candidateOverflow() {
        reads.candidate++;
        return 'overflow';
      }
    }
    const status = new Status();
    class Output {
      get residenceTimes() {
        reads.residence++;
        return 'residence';
      }
      get status() {
        reads.status++;
        return status;
      }
    }
    const props: TrajectoryRecursionsOperationProps = {
      positions: 'positions',
      timestamps: 'timestamps',
      trackOffsets: 'offsets',
      radii: 'radii',
      rowCount: 4,
      radiusCount: 1,
      timestampFormat: 'float32',
      maximumTrackSegments: 1,
      output: new Output()
    };
    for (const field of [
      'maximumTrackSegments',
      'rowCount',
      'radiusCount',
      'timestampFormat'
    ] as const) {
      const first = props[field];
      Object.defineProperty(props, field, {
        get() {
          reads[field]++;
          return reads[field] === 1
            ? first
            : field === 'timestampFormat'
              ? 'uint32x2'
              : field === 'radiusCount'
                ? 2
                : 3;
        }
      });
    }
    const operation = new Operation(props);
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: new Float32Array(8)},
        timestamps: {format: 'float32', values: Float32Array.from([0, 1, 2, 3])},
        offsets: {format: 'uint32', values: Uint32Array.from([0, 4])},
        radii: {format: 'float32', values: Float32Array.from([1])}
      },
      operations: [operation],
      outputs: ['overflow', 'residence']
    });
    // Four stationary fixes have three unit segments. The first getter values set
    // cap one, so each residence is exactly one and two omitted segments force loss.
    expect(Array.from(result['overflow'])).toEqual([1]);
    expect(Array.from(result['residence'])).toEqual([1, 1, 1, 1]);
    expect(reads).toEqual({
      maximumTrackSegments: 1,
      rowCount: 1,
      radiusCount: 1,
      timestampFormat: 1,
      residence: 1,
      status: 1,
      candidate: 1
    });
  });
}

// Four stationary fixes have THREE unit-duration segments. Cap 1 necessarily
// omits two segments per focal fix, forcing candidateOverflow=1. Cap 3 covers
// exactly the track and cap 4 exceeds it, both giving 0. Residence equals the
// window duration; all passages are censored at its end. These integer values
// are exactly representable in float32, so every assertion uses zero tolerance.
for (const scenario of [
  {budget: 1, overflow: [1], residence: [1, 1, 1, 1], passage: [1, 1, 1, 0]},
  {budget: 3, overflow: [0], residence: [3, 3, 3, 3], passage: [3, 2, 1, 0]},
  {budget: 4, overflow: [0], residence: [3, 3, 3, 3], passage: [3, 2, 1, 0]}
]) {
  for (const Operation of [GPUTrajectoryRecursionsOperation, CPUTrajectoryRecursionsOperation]) {
    it(`${Operation.name} writes independently counted window status ${scenario.budget}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: new Float32Array(8)},
          timestamps: {format: 'float32', values: new Float32Array([0, 1, 2, 3])},
          offsets: {format: 'uint32', values: new Uint32Array([0, 4])},
          radii: {format: 'float32', values: new Float32Array([1])}
        },
        operations: [
          new Operation({
            positions: 'positions',
            timestamps: 'timestamps',
            trackOffsets: 'offsets',
            radii: 'radii',
            rowCount: 4,
            radiusCount: 1,
            maximumTrackSegments: scenario.budget,
            output: {
              residenceTimes: 'residence',
              firstPassageTimes: 'passage',
              firstPassageCensored: 'censored',
              status: {candidateOverflow: 'overflow'}
            }
          })
        ],
        outputs: ['overflow', 'residence', 'passage', 'censored']
      });
      expect(Array.from(result['overflow'])).toEqual(scenario.overflow);
      expect(Array.from(result['residence'])).toEqual(scenario.residence);
      expect(Array.from(result['passage'])).toEqual(scenario.passage);
      expect(Array.from(result['censored'])).toEqual([1, 1, 1, 1]);
    });
  }
}

for (const Operation of [GPUTrajectoryRecursionsOperation, CPUTrajectoryRecursionsOperation]) {
  it(`${Operation.name} counts crossings in centred two-segment windows`, async () => {
    // Each window contains two segments, each spending exactly one unit in
    // the radius-1 disc. Windows [0,2), [0,2), [1,3), [1,3) give the counts
    // [1,0,0,1] after removing the focal visit. All expected values are exact.
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: new Float32Array([0, 0, 2, 0, 0, 0, 2, 0])},
        timestamps: {format: 'float32', values: new Float32Array([0, 2, 4, 6])},
        offsets: {format: 'uint32', values: new Uint32Array([0, 4])},
        radii: {format: 'float32', values: new Float32Array([1])}
      },
      operations: [
        new Operation({
          positions: 'positions',
          timestamps: 'timestamps',
          trackOffsets: 'offsets',
          radii: 'radii',
          rowCount: 4,
          radiusCount: 1,
          maximumTrackSegments: 2,
          output: {
            revisits: 'revisits',
            residenceTimes: 'residence',
            firstPassageTimes: 'passage',
            firstPassageCensored: 'censored',
            status: {candidateOverflow: 'overflow'}
          }
        })
      ],
      outputs: ['revisits', 'overflow', 'residence', 'passage', 'censored']
    });
    expect(Array.from(result['overflow'])).toEqual([1]);
    expect(Array.from(result['revisits'])).toEqual([1, 0, 0, 1]);
    expect(Array.from(result['residence'])).toEqual([2, 2, 2, 2]);
    expect(Array.from(result['passage'])).toEqual([1, 1, 1, 0]);
    expect(Array.from(result['censored'])).toEqual([0, 0, 0, 1]);
  });
}
