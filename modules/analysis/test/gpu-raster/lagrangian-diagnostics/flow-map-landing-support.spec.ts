// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPULagrangianFlowMap,
  getGPULagrangianFlowMapParameterValues
} from '../../../src/raster-analysis/lagrangian-diagnostics/gpu-lagrangian-flow-map';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GraphHarness} from '../../gpu-spatial-analysis/trajectory-staypoints/movement-spec-utils';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {expectLostPosition} from '../flow-texture/flow-output-assertions';

for (const cellWidth of [1, -1]) {
  for (const missingLanding of [true, false]) {
    it(`RT-124 FlowMap validates final landing support: spacing ${cellWidth}, missing ${missingLanding}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = new GraphHarness(device, 'flow-map-landing');
      const originX = cellWidth === 1 ? 0 : 4;
      const extent = [originX, 1, cellWidth, -1] as const;
      const velocity = 3 * cellWidth;
      const field = Float32Array.of(
        velocity,
        0,
        velocity,
        0,
        velocity,
        0,
        missingLanding ? NaN : velocity,
        missingLanding ? NaN : 0
      );
      const positions = harness.output('positions', 'float32x2', 1);
      const validity = harness.output('validity', 'uint32', 1);
      harness.graph.add(
        new GPULagrangianFlowMap({
          seeds: harness.input(
            'seeds',
            Float32Array.of(originX + 0.5 * cellWidth, 0.5),
            'float32x2',
            1
          ),
          velocitiesStart: harness.input('start', field, 'float32x2', 4),
          velocitiesEnd: harness.input('end', field, 'float32x2', 4),
          fieldWidth: 4,
          fieldHeight: 1,
          stepCount: 1,
          parameters: harness.input(
            'parameters',
            getGPULagrangianFlowMapParameterValues({
              fieldPlacement: {origin: [extent[0], extent[1]], cellSize: [extent[2], extent[3]]},
              startTime: 0,
              timeStep: 1,
              sliceStartTime: 0,
              sliceEndTime: 1
            }),
            'float32',
            8
          ),
          output: {positions: positions.view, validity: validity.view}
        })
      );
      await harness.run(async () => {
        const values = await readFloat32(positions.buffer, 2);
        expect(await readUint32(validity.buffer, 1)).toEqual([missingLanding ? 0 : 1]);
        // Closed form for identical constant slices: local x .5 -> 2 -> 3.5.
        // The landing remains inside the rectangle even when its velocity cell is missing.
        if (missingLanding) {
          expectLostPosition(values);
        } else {
          expect(values.every(Number.isFinite)).toBe(true);
          expect(values).toEqual([originX + 3.5 * cellWidth, 0.5]);
        }
      });
    });
  }
}

for (const missingEndLanding of [true, false]) {
  it(`B3 FlowMap requires end-slice support at a backward landing: missing ${missingEndLanding}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new GraphHarness(device, 'flow-map-end-hole');
    const start = Float32Array.of(0, 2, 0, 2, 0, 2, 0, 2, 0, 2);
    const end = start.slice();
    if (missingEndLanding) end.set([NaN, NaN], 8);
    const positions = harness.output('positions', 'float32x2', 1);
    const validity = harness.output('validity', 'uint32', 1);
    harness.graph.add(
      new GPULagrangianFlowMap({
        seeds: harness.input('seeds', Float32Array.of(0.5, 3.5), 'float32x2', 1),
        velocitiesStart: harness.input('start', start, 'float32x2', 5),
        velocitiesEnd: harness.input('end', end, 'float32x2', 5),
        fieldWidth: 1,
        fieldHeight: 5,
        stepCount: 1,
        parameters: harness.input(
          'parameters',
          getGPULagrangianFlowMapParameterValues({
            fieldPlacement: {origin: [0, 5], cellSize: [1, -1]},
            startTime: 1.5,
            timeStep: -1.5,
            sliceStartTime: 0,
            sliceEndTime: 1.5
          }),
          'float32',
          8
        ),
        output: {positions: positions.view, validity: validity.view}
      })
    );
    await harness.run(async () => {
      const values = await readFloat32(positions.buffer, 2);
      expect(await readUint32(validity.buffer, 1)).toEqual([missingEndLanding ? 0 : 1]);
      // Closed form .5,3.5 -> midpoint .5,2 -> landing .5,.5 (row 4).
      // Even at the start-slice time, the sampler requires both slices' support.
      if (missingEndLanding) expectLostPosition(values);
      else expect(values).toEqual([0.5, 0.5]);
    });
  });
}
