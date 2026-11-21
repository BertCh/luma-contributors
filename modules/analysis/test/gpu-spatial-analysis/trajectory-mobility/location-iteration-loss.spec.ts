// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUMobilityIndicators} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-mobility-indicators';
import {createOptimizationHarness} from '../../gpu-network/network-optimization/optimization-test-harness';
import {GPUMobilityIndicatorsOperation} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-mobility-indicators-operation';
import {CPUMobilityIndicatorsOperation} from '../../../src/gpu-spatial-analysis/trajectory-mobility/cpu-mobility-indicators-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

it.each([
  {name: 'mixed keys', loss: 0, ids: [7, 0xffffffff, 9, 7], excluded: 1},
  {
    name: 'only sentinel keys',
    loss: 0,
    ids: [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff],
    excluded: 4
  },
  {name: 'omitted keys', loss: 0, ids: undefined, excluded: 4},
  {name: 'unsettled keys', loss: 1, ids: [7, 8, 9, 7], excluded: 4}
])('excludes $name and preserves unrelated mobility measures', async ({loss, ids, excluded}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    randomEntropy: harness.output('float32', 1),
    uncorrelatedEntropy: harness.output('float32', 1),
    realEntropy: harness.output('float32', 1),
    homeLocations: harness.output('uint32', 1),
    workLocations: harness.output('uint32', 1),
    radiusOfGyration: harness.output('float32', 1),
    jumpLengths: harness.output('float32', 4),
    waitingTimes: harness.output('float32', 4),
    trackingQuality: harness.output('float32', 1),
    status: {
      candidateOverflow: harness.output('uint32', 1),
      excludedLocationCount: harness.output('uint32', 1)
    }
  };
  const read = await harness.run(
    new GPUMobilityIndicators({
      trackCount: 1,
      maximumEntropySequenceRows: 3,
      timestamps: harness.input('float32', Float32Array.of(0, 1, 2, 3, 4)),
      staypoints: {
        tracks: harness.input('uint32', Uint32Array.of(0, 0, 0, 0)),
        centroids: harness.input('float32x2', Float32Array.of(0, 0, 2, 0, 4, 0, 6, 0)),
        startRows: harness.input('uint32', Uint32Array.of(0, 1, 2, 3)),
        durations: harness.input('float32', Float32Array.of(1, 100, 4, 1)),
        locationIds: ids ? harness.input('uint32', Uint32Array.from(ids)) : undefined,
        status: {
          count: harness.input('uint32', Uint32Array.of(4)),
          locationIterationLimitReached: harness.input('uint32', Uint32Array.of(loss))
        }
      },
      // Four contiguous one-second stay intervals and no active triplegs give tracking quality 1.
      triplegs: {
        tracks: harness.input('uint32', Uint32Array.of(0)),
        startRows: harness.input('uint32', Uint32Array.of(0)),
        endRows: harness.input('uint32', Uint32Array.of(1)),
        status: {count: harness.input('uint32', Uint32Array.of(0))},
        staypointFinishRows: harness.input('uint32', Uint32Array.of(1, 2, 3, 4))
      },
      output
    })
  );
  if (excluded === 1) {
    // Retained sequence [7, 9, 7]: two keys with shares 2/3 and 1/3,
    // novelty lengths [1, 1, 2], and duration totals 2 and 4.
    expect(read(output.randomEntropy)).toEqual([1]);
    const uncorrelatedEntropy = read(output.uncorrelatedEntropy);
    const realEntropy = read(output.realEntropy);
    expect(uncorrelatedEntropy).toHaveLength(1);
    expect(realEntropy).toHaveLength(1);
    expect(uncorrelatedEntropy[0]).toBeCloseTo(0.9182958340544896, 6);
    expect(realEntropy[0]).toBeCloseTo(1.188721875540867, 6);
    expect(read(output.homeLocations)).toEqual([9]);
    expect(read(output.workLocations)).toEqual([7]);
  } else {
    expect(read(output.randomEntropy)).toEqual([NaN]);
    expect(read(output.uncorrelatedEntropy)).toEqual([NaN]);
    expect(read(output.realEntropy)).toEqual([NaN]);
    expect(read(output.homeLocations)).toEqual([0xffffffff]);
    expect(read(output.workLocations)).toEqual([0xffffffff]);
  }
  const radiusOfGyration = read(output.radiusOfGyration);
  expect(radiusOfGyration).toHaveLength(1);
  expect(radiusOfGyration[0]).toBeCloseTo(Math.sqrt(5), 6);
  expect(read(output.jumpLengths)).toEqual([0, 2, 2, 2]);
  expect(read(output.waitingTimes)).toEqual([0, 1, 1, 1]);
  expect(read(output.trackingQuality)).toEqual([1]);
  expect(read(output.status.excludedLocationCount)).toEqual([excluded]);
  expect(read(output.status.candidateOverflow)).toEqual([loss]);
});

it.each([
  GPUMobilityIndicatorsOperation,
  CPUMobilityIndicatorsOperation
])('B3 $name executes ranked home/work suppression through the operation', async Operation => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const loss of [0, 1]) {
    const output = await runOperationGraph(device, {
      inputs: {
        times: {format: 'float32', values: Float32Array.of(0, 1, 2)},
        tracks: {format: 'uint32', values: Uint32Array.of(0, 0, 0)},
        centroids: {format: 'float32x2', values: Float32Array.of(0, 0, 0, 0, 0, 0)},
        starts: {format: 'uint32', values: Uint32Array.of(0, 1, 2)},
        durations: {format: 'float32', values: Float32Array.of(1, 4, 1)},
        locations: {format: 'uint32', values: Uint32Array.of(7, 9, 7)},
        count: {format: 'uint32', values: Uint32Array.of(3)},
        locationLoss: {format: 'uint32', values: Uint32Array.of(loss)}
      },
      operations: [
        new Operation({
          trackCount: 1,
          rowCount: 3,
          staypointCapacity: 3,
          timestamps: 'times',
          staypoints: {
            tracks: 'tracks',
            centroids: 'centroids',
            startRows: 'starts',
            durations: 'durations',
            locationIds: 'locations',
            status: {count: 'count', locationIterationLimitReached: 'locationLoss'}
          },
          output: {
            homeLocations: 'home',
            workLocations: 'work',
            radiusOfGyration: 'radius',
            status: {candidateOverflow: 'loss', excludedLocationCount: 'excluded'}
          }
        })
      ],
      outputs: ['home', 'work', 'radius', 'loss', 'excluded']
    });
    // Location 9 accumulates duration 4; location 7 accumulates 2.
    expect(Array.from(output['home'])).toEqual([loss ? 0xffffffff : 9]);
    expect(Array.from(output['work'])).toEqual([loss ? 0xffffffff : 7]);
    expect(Array.from(output['radius'])).toEqual([0]);
    expect(Array.from(output['loss'])).toEqual([loss]);
    expect(Array.from(output['excluded'])).toEqual([loss ? 3 : 0]);
  }
});
