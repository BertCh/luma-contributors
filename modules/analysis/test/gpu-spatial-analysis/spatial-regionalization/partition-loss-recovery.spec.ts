// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUAZPRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

// Depends on the staging adjacency-loss suppression fix; the same compiled graph must reset it.
for (const name of ['overflow', 'candidateOverflow'] as const) {
  it(`clears AZP adjacency ${name} loss on resubmission of the same graph`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    const loss = rig.input(Uint32Array.of(1), 'uint32', 1);
    const labels = rig.output('uint32', 4),
      objective = rig.output('float32', 1),
      status = rig.output('uint32', 4);
    rig.graph.add(
      new GPUAZPRegions({
        weights: {
          offsets: rig.input(Uint32Array.of(0, 1, 2, 3, 4), 'uint32', 5),
          neighbors: rig.input(Uint32Array.of(1, 0, 3, 2), 'uint32', 4),
          weights: rig.input(Float32Array.of(1, 1, 1, 1), 'float32', 4),
          status: {[name]: loss}
        },
        values: rig.input(Float32Array.of(0, 2, 10, 14), 'float32', 4),
        initialLabels: rig.input(Uint32Array.of(0, 0, 1, 1), 'uint32', 4),
        regionCapacity: 2,
        parameters: rig.input(Uint32Array.of(619), 'uint32', 1),
        standardize: false,
        maximumIterations: 4,
        output: {labels: labels.view, objective: objective.view, status: status.view}
      })
    );
    const compiled = await rig.graph.compileAsync();
    try {
      for (const flag of [1, 0, 1, 0]) {
        rig.bufferOf(loss).write(Uint32Array.of(flag));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(labels.buffer, 4)).toEqual(
          flag ? new Array(4).fill(0xffffffff) : [0, 0, 1, 1]
        );
        const [score] = await readFloat32(objective.buffer, 1);
        if (flag) expect(Number.isNaN(score)).toBe(true);
        else {
          expect(Number.isFinite(score)).toBe(true);
          expect(score).toBe(10);
        } // 2²/2 + 4²/2.
      }
    } finally {
      compiled.destroy();
      rig.destroy();
      device.destroy();
    }
  });
}
