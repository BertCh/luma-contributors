// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUTrajectoryLocationsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-locations-operation';
import {
  GPUTrajectoryLocationsOperation,
  type TrajectoryLocationsOperationProps
} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-locations-operation';
import {getGPUTrajectoryLocationsParameterValues} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-locations';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

it('locations compare seam-equivalent and pole coordinates on the sphere', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: TrajectoryLocationsOperationProps = {
    centroids: 'centroids',
    tracks: 'tracks',
    count: 'count',
    parameters: 'parameters',
    capacity: 5,
    spatialContext: {
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      sphereRadius: 1
    },
    output: {
      locationIds: 'locationIds',
      centers: 'centers',
      status: {
        count: 'locationCount',
        converged: 'converged',
        iterationLimitReached: 'iterationLimitReached'
      }
    }
  };
  const inputs = {
    centroids: {
      format: 'float32x2' as const,
      values: new Float32Array([-180, 0, 180, 0, -179, 0, 0, 90, 180, 90])
    },
    tracks: {format: 'uint32' as const, values: new Uint32Array([0, 0, 0, 1, 1])},
    count: {format: 'uint32' as const, values: Uint32Array.of(5)},
    parameters: {
      format: 'float32' as const,
      values: getGPUTrajectoryLocationsParameterValues({epsilon: 0.1})
    }
  };
  const outputs = ['locationIds', 'centers', 'locationCount', 'converged', 'iterationLimitReached'];
  const gpu = await runOperationGraph(device, {
    inputs,
    operations: [new GPUTrajectoryLocationsOperation(props)],
    outputs
  });
  const cpu = await runOperationGraph(device, {
    inputs,
    operations: [new CPUTrajectoryLocationsOperation(props)],
    outputs
  });
  for (const output of outputs.filter(port => port !== 'centers'))
    expect(Array.from(gpu[output])).toEqual(Array.from(cpu[output]));
  // Centers: relative tolerance 1e-5, exact numeric zero for zero coordinates/capacity rows.
  for (let component = 0; component < cpu['centers'].length; component++) {
    const expected = cpu['centers'][component];
    if (expected === 0) expect(gpu['centers'][component] === 0).toBe(true);
    else
      expect(
        Math.abs(gpu['centers'][component] - expected) / Math.abs(expected)
      ).toBeLessThanOrEqual(1e-5);
  }
  expect(Array.from(cpu['locationIds'])).toEqual([0, 0, 0, 1, 1]);
  // Seam duplicates vote once: (-180 + -179)/2. Pole votes have longitude zero.
  for (const result of [cpu, gpu]) {
    expect(Array.from(result['centers']).slice(0, 4)).toEqual([-179.5, 0, 0, 90]);
    expect(Array.from(result['locationCount'])).toEqual([2]);
  }
});

for (const empty of [false, true]) {
  it(`locations compare every output (${empty ? 'zero active count' : 'two tracks and transitive chain'})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: TrajectoryLocationsOperationProps = {
      centroids: 'centroids',
      tracks: 'tracks',
      count: 'count',
      parameters: 'parameters',
      capacity: 10,
      output: {
        locationIds: 'locationIds',
        centers: 'centers',
        status: {
          count: 'locationCount',
          converged: 'converged',
          iterationLimitReached: 'iterationLimitReached'
        }
      }
    };
    const inputs = {
      centroids: {
        format: 'float32x2' as const,
        values: new Float32Array([0, 0, 2, 0, 4, 0, 0, 0, 10, 0, 0, 0, 2, 0, 5, 0, 5, 0, 99, 99])
      },
      tracks: {format: 'uint32' as const, values: new Uint32Array([0, 0, 0, 0, 0, 1, 1, 1, 1, 99])},
      count: {format: 'uint32' as const, values: Uint32Array.of(empty ? 0 : 9)},
      parameters: {
        format: 'float32' as const,
        values: getGPUTrajectoryLocationsParameterValues({epsilon: 2})
      }
    };
    const outputs = [
      'locationIds',
      'centers',
      'locationCount',
      'converged',
      'iterationLimitReached'
    ];
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUTrajectoryLocationsOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUTrajectoryLocationsOperation(props)],
      outputs
    });
    for (const output of outputs.filter(port => port !== 'centers'))
      expect(Array.from(gpu[output])).toEqual(Array.from(cpu[output]));
    // Relative center tolerance 2e-6; zero coordinates are exact in this scene. Unwritten GPU
    // center capacity rows start at zero in a fresh graph. Never compare the sign of zero.
    for (let component = 0; component < cpu['centers'].length; component++) {
      const expected = cpu['centers'][component];
      if (expected === 0) expect(gpu['centers'][component] === 0).toBe(true);
      else
        expect(
          Math.abs(gpu['centers'][component] - expected) / Math.abs(expected)
        ).toBeLessThanOrEqual(2e-6);
    }
    expect(cpu['locationCount'][0]).toBe(empty ? 0 : 4);
  });
}
