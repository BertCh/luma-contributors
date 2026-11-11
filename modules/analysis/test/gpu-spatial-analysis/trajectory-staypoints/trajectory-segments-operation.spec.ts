// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUTrajectorySegmentsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-segments-operation';
import {GPUTrajectorySegmentsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-segments-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {getSegmentsFixture, getSegmentsProps} from './trajectory-segments-operation-fixture';

for (const scene of ['ordinary', 'overflow', 'stationary'] as const) {
  it(`segments compare every integer port exactly: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = getSegmentsFixture();
    const props = scene === 'overflow' ? getSegmentsProps(3, 1) : getSegmentsProps();
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
      stayCount: {format: 'uint32' as const, values: fixture.staypoints.count}
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
      scene === 'stationary' ? 0 : scene === 'overflow' ? 2 : 4
    );
  });
}
