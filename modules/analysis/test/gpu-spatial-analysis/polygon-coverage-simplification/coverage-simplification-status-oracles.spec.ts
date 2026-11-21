// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUCoverageSimplification} from '../../../src/gpu-spatial-analysis/polygon-coverage-simplification/gpu-coverage-simplification';
import {getGPULineSimplificationParameterValues} from '../../../src/gpu-spatial-analysis/line-simplification/index';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

for (const capacity of [2, 3]) {
  it(`coverage triangle has a bounded complete solve and three required vertices at capacity ${capacity}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    try {
      const converged = rig.output('uint32', 1);
      const required = rig.output('uint32', 1);
      const overflow = rig.output('uint32', 1);
      rig.run(
        new GPUCoverageSimplification({
          positions: rig.input(Float32Array.of(0, 0, 2, 0, 0, 2), 'float32x2', 3),
          ringOffsets: rig.input(Uint32Array.of(0, 3), 'uint32', 2),
          polygonOffsets: rig.input(Uint32Array.of(0, 1), 'uint32', 2),
          featureOffsets: rig.input(Uint32Array.of(0, 1), 'uint32', 2),
          parameters: rig.input(
            getGPULineSimplificationParameterValues({tolerance: 0}),
            'float32',
            4
          ),
          maximumIterations: 16,
          output: {
            positions: rig.output('float32x2', capacity).view,
            ringOffsets: rig.output('uint32', 2).view,
            status: {
              converged: converged.view,
              requiredCount: required.view,
              overflow: overflow.view
            }
          }
        })
      );
      // The closed three-vertex arc has two interior rows after repeating the first endpoint.
      // At most two splitting decisions and their terminal intervals finish well within 16
      // rounds. All three noncollinear corners have positive importance, so tolerance 0 keeps 3.
      expect(await readUint32(converged.buffer, 1)).toEqual([1]);
      expect(await readUint32(required.buffer, 1)).toEqual([3]);
      expect(await readUint32(overflow.buffer, 1)).toEqual([capacity === 2 ? 1 : 0]);
    } finally {
      rig.destroy();
    }
  });
}

for (const maximumIterations of [1, 16]) {
  it(`coverage 72-corner ring distinguishes unfinished and complete solves at budget ${maximumIterations}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const positions = Float32Array.from(
      Array.from({length: 72}, (_, row) => {
        const angle = (2 * Math.PI * row) / 72;
        return [Math.cos(angle), Math.sin(angle)];
      }).flat()
    );
    const rig = new WeightsRig(device);
    try {
      const converged = rig.output('uint32', 1);
      const props = {
        positions: rig.input(positions, 'float32x2', 72),
        ringOffsets: rig.input(Uint32Array.of(0, 72), 'uint32', 2),
        polygonOffsets: rig.input(Uint32Array.of(0, 1), 'uint32', 2),
        featureOffsets: rig.input(Uint32Array.of(0, 1), 'uint32', 2),
        parameters: rig.input(
          getGPULineSimplificationParameterValues({tolerance: 0}),
          'float32',
          4
        ),
        maximumIterations,
        topologyIterationCount: 0,
        output: {
          positions: rig.output('float32x2', 72).view,
          ringOffsets: rig.output('uint32', 2).view,
          status: {converged: converged.view, overflow: rig.output('uint32', 1).view}
        }
      };
      const contributor = new GPUCoverageSimplification(props);
      // The validated one-round case must stay incomplete even if the caller later asks
      // for 16; the complete 16-round case must likewise retain its captured budget.
      props.maximumIterations = maximumIterations === 1 ? 16 : 1;
      rig.run(contributor);
      // The repeated-endpoint arc first splits at the opposite row 36. Each side still
      // has 35 interior rows, exceeding finishSpanLimit 32, so one round cannot finish.
      // Subsequent bisection leaves fewer than 32 interior rows per span; 16 rounds suffice.
      // The companion node test independently pins the same literals on the CPU line solver.
      expect(await readUint32(converged.buffer, 1)).toEqual([maximumIterations === 1 ? 0 : 1]);
    } finally {
      rig.destroy();
    }
  });
}
