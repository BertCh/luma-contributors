// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUMobilityIndicatorsOperation} from '../../../src/gpu-spatial-analysis/trajectory-mobility/cpu-mobility-indicators-operation';
import {GPUMobilityIndicatorsOperation} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-mobility-indicators-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {getMobilityFixture, getMobilityProps} from './mobility-indicators-operation-fixture';

for (const [spatialContext, expectedError] of [
  [{coordinateSpace: 'planar', metric: 'native'}, undefined],
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
  it(`mobility matches distance-context acceptance for ${spatialContext.metric} ${'frame' in spatialContext ? spatialContext.frame.kind : ''}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props = {
      ...getMobilityProps(),
      trackCount: 1,
      rowCount: 2,
      staypointCapacity: 2,
      triplegs: undefined,
      spatialContext,
      output: {radiusOfGyration: 'radiusOfGyration', jumpLengths: 'jumpLengths'}
    };
    const inputs = {
      timestamps: {format: 'float32' as const, values: new Float32Array([0, 1])},
      stayTracks: {format: 'uint32' as const, values: new Uint32Array([0, 0])},
      centroids: {format: 'float32x2' as const, values: new Float32Array([0, 0, 3, 4])},
      stayStartRows: {format: 'uint32' as const, values: new Uint32Array([0, 1])},
      durations: {format: 'float32' as const, values: new Float32Array([1, 1])},
      locationIds: {format: 'uint32' as const, values: new Uint32Array([0, 1])},
      stayCount: {format: 'uint32' as const, values: Uint32Array.of(2)}
    };
    const outputs = ['radiusOfGyration', 'jumpLengths'];
    if (expectedError) {
      expect(() => new CPUMobilityIndicatorsOperation(props)).toThrow(expectedError);
      await expect(
        runOperationGraph(device, {
          inputs,
          operations: [new GPUMobilityIndicatorsOperation(props)],
          outputs
        })
      ).rejects.toThrow(expectedError);
    } else {
      const gpu = await runOperationGraph(device, {
        inputs,
        operations: [new GPUMobilityIndicatorsOperation(props)],
        outputs
      });
      const cpu = await runOperationGraph(device, {
        inputs,
        operations: [new CPUMobilityIndicatorsOperation(props)],
        outputs
      });
      // The 3-4-5 segment has an exactly representable midpoint radius of 2.5 and jump of 5.
      for (const result of [gpu, cpu]) {
        expect(result['radiusOfGyration'].length).toBe(1);
        expect(result['radiusOfGyration'][0] === 2.5).toBe(true);
        expect(result['jumpLengths'].length).toBe(2);
        expect(result['jumpLengths'][0] === 0).toBe(true);
        expect(result['jumpLengths'][1] === 5).toBe(true);
      }
    }
  });
}

for (const spherical of [false, true]) {
  for (const empty of [false, true]) {
    it(`mobility compares all outputs (spherical=${spherical}, ${empty ? 'no active stays or triplegs' : 'two tracks, entropy and duration tie'})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = getMobilityFixture();
      const props = {
        ...getMobilityProps(),
        ...(spherical
          ? {
              spatialContext: {
                coordinateSpace: 'longitude-latitude',
                metric: 'great-circle',
                sphereRadius: 1
              } as const
            }
          : {})
      };
      const inputs = {
        timestamps: {format: 'float32' as const, values: fixture.timestamps},
        stayTracks: {format: 'uint32' as const, values: fixture.staypoints.tracks},
        centroids: {format: 'float32x2' as const, values: fixture.staypoints.centroids},
        stayStartRows: {format: 'uint32' as const, values: fixture.staypoints.startRows},
        durations: {format: 'float32' as const, values: fixture.staypoints.durations},
        locationIds: {format: 'uint32' as const, values: fixture.staypoints.locationIds},
        stayCount: {
          format: 'uint32' as const,
          values: empty ? Uint32Array.of(0) : fixture.staypoints.count
        },
        triplegTracks: {format: 'uint32' as const, values: fixture.triplegs.tracks},
        triplegStartRows: {format: 'uint32' as const, values: fixture.triplegs.startRows},
        triplegEndRows: {format: 'uint32' as const, values: fixture.triplegs.endRows},
        triplegCount: {
          format: 'uint32' as const,
          values: empty ? Uint32Array.of(0) : fixture.triplegs.count
        },
        stayFinishRows: {format: 'uint32' as const, values: fixture.triplegs.staypointFinishRows}
      };
      const gpuOperation = new GPUMobilityIndicatorsOperation(props);
      const outputs = Object.values(gpuOperation.outputs).map(output => output.value);
      const gpu = await runOperationGraph(device, {inputs, operations: [gpuOperation], outputs});
      const cpu = await runOperationGraph(device, {
        inputs,
        operations: [new CPUMobilityIndicatorsOperation(props)],
        outputs
      });
      for (const output of outputs) {
        if (['homeLocations', 'workLocations', 'candidateOverflow'].includes(output))
          expect(Array.from(gpu[output])).toEqual(Array.from(cpu[output]));
        else {
          // Waiting times are exact differences of these small integer timestamps. Other nonzero
          // float results use relative tolerance 1e-5. Compare zeros numerically, without their sign.
          for (let row = 0; row < cpu[output].length; row++) {
            const expected = cpu[output][row];
            if (output === 'waitingTimes' || expected === 0)
              expect(gpu[output][row] === expected).toBe(true);
            else
              expect(
                Math.abs(gpu[output][row] - expected) / Math.abs(expected)
              ).toBeLessThanOrEqual(1e-5);
          }
        }
      }
    });
  }
}
