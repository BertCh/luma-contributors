// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUModalSplitOperation} from '../../../src/gpu-spatial-analysis/trajectory-mobility/cpu-modal-split-operation';
import {GPUModalSplitOperation} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-modal-split-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {getModalFixture, getModalProps} from './modal-split-operation-fixture';

for (const metric of ['count', 'distance'] as const) {
  for (const start of [0, 1, 2]) {
    it(`modal supplied-mode ${metric} preserves an empty range at row ${start}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const props = {
        ...getModalProps(metric, false),
        rowCount: 2,
        trackCount: 1,
        triplegCapacity: 1,
        modeCount: 1
      };
      const inputs = {
        positions: {format: 'float32x2' as const, values: new Float32Array([0, 0, 1, 0])},
        timestamps: {format: 'float32' as const, values: new Float32Array([0, 1])},
        trackOffsets: {format: 'uint32' as const, values: new Uint32Array([0, 2])},
        triplegTracks: {format: 'uint32' as const, values: Uint32Array.of(0)},
        triplegStartRows: {format: 'uint32' as const, values: Uint32Array.of(start)},
        triplegEndRows: {format: 'uint32' as const, values: Uint32Array.of(start)},
        triplegCount: {format: 'uint32' as const, values: Uint32Array.of(1)},
        modes: {format: 'uint32' as const, values: Uint32Array.of(0)}
      };
      const outputs = ['modalSplit'];
      const gpu = await runOperationGraph(device, {
        inputs,
        operations: [new GPUModalSplitOperation(props)],
        outputs
      });
      const cpu = await runOperationGraph(device, {
        inputs,
        operations: [new CPUModalSplitOperation(props)],
        outputs
      });
      // Kernel-implied result, pending execution on the integrator's GPU: literal count one,
      // or distance zero from the empty path. Exact numeric comparison ignores zero sign.
      const expected = metric === 'count' ? 1 : 0;
      expect(cpu['modalSplit'].length).toBe(1);
      expect(gpu['modalSplit'].length).toBe(1);
      expect(cpu['modalSplit'][0] === expected).toBe(true);
      expect(gpu['modalSplit'][0] === expected).toBe(true);
    });
  }
}

for (const metric of ['count', 'distance', 'duration'] as const) {
  for (const prediction of [false, true]) {
    for (const empty of [false, true]) {
      it(`modal ${metric}, prediction=${prediction}, empty=${empty} compares every output`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const fixture = getModalFixture();
        const props = getModalProps(metric, prediction);
        const inputs = {
          positions: {format: 'float32x2' as const, values: fixture.positions},
          timestamps: {format: 'float32' as const, values: fixture.timestamps},
          trackOffsets: {format: 'uint32' as const, values: fixture.trackOffsets},
          triplegTracks: {format: 'uint32' as const, values: fixture.triplegs.tracks},
          triplegStartRows: {format: 'uint32' as const, values: fixture.triplegs.startRows},
          triplegEndRows: {format: 'uint32' as const, values: fixture.triplegs.endRows},
          triplegCount: {
            format: 'uint32' as const,
            values: empty ? Uint32Array.of(0) : fixture.triplegs.count
          },
          ...(!prediction ? {modes: {format: 'uint32' as const, values: fixture.modes}} : {})
        };
        const outputs = prediction ? ['modalSplit', 'predictedModes'] : ['modalSplit'];
        const gpu = await runOperationGraph(device, {
          inputs,
          operations: [new GPUModalSplitOperation(props)],
          outputs
        });
        const cpu = await runOperationGraph(device, {
          inputs,
          operations: [new CPUModalSplitOperation(props)],
          outputs
        });
        if (prediction)
          expect(Array.from(gpu['predictedModes'])).toEqual(Array.from(cpu['predictedModes']));
        // Counts and these integer-time duration sums are exact. Distances use 2e-6 relative
        // tolerance, with exact zero bins. Compare zeros numerically, without their sign.
        for (let row = 0; row < cpu['modalSplit'].length; row++) {
          const expected = cpu['modalSplit'][row];
          if (metric === 'count' || metric === 'duration' || expected === 0)
            expect(gpu['modalSplit'][row] === expected).toBe(true);
          else
            expect(
              Math.abs(gpu['modalSplit'][row] - expected) / Math.abs(expected)
            ).toBeLessThanOrEqual(2e-6);
        }
      });
    }
  }
}
