// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUMaxPRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {createPartitionLossPorts} from './partition-loss-ports';
import {expectRecoveredPairLabels} from './partition-recovery-assertions';

// Depends on staging adjacency-loss suppression. Two disconnected pairs of unit extensive
// weight and threshold 2 force exactly those pairs: SSD = (2 - 0)^2/2 + (14 - 10)^2/2 = 10.
for (const name of ['overflow', 'candidateOverflow'] as const) {
  it(`clears Max-p adjacency ${name} loss twice on the same compiled graph`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    const losses = createPartitionLossPorts(rig);
    const labels = rig.output('uint32', 4);
    const objective = rig.output('float32', 1);
    const status = rig.output('uint32', 6);
    rig.graph.add(
      new GPUMaxPRegions({
        weights: {
          offsets: rig.input(Uint32Array.of(0, 1, 2, 3, 4), 'uint32', 5),
          neighbors: rig.input(Uint32Array.of(1, 0, 3, 2), 'uint32', 4),
          weights: rig.input(Float32Array.of(1, 1, 1, 1), 'float32', 4),
          status: losses
        },
        values: rig.input(Float32Array.of(0, 2, 10, 14), 'float32', 4),
        extensiveValues: rig.input(Float32Array.of(1, 1, 1, 1), 'float32', 4),
        threshold: rig.input(Float32Array.of(2), 'float32', 1),
        parameters: rig.input(Uint32Array.of(619), 'uint32', 1),
        standardize: false,
        startCount: 4,
        maximumGrowthSteps: 16,
        maximumIterations: 0,
        output: {labels: labels.view, objective: objective.view, status: status.view}
      })
    );
    const compiled = await rig.graph.compileAsync();
    try {
      for (const flag of [1, 0, 1, 0]) {
        // Bind both producer ports throughout: nullish-coalescing one port must not hide the other.
        rig.bufferOf(losses.overflow).write(Uint32Array.of(name === 'overflow' ? flag : 0));
        rig
          .bufferOf(losses.candidateOverflow)
          .write(Uint32Array.of(name === 'candidateOverflow' ? flag : 0));
        submitGraph(device, compiled, undefined);
        const actualLabels = await readUint32(labels.buffer, 4);
        const [score] = await readFloat32(objective.buffer, 1);
        if (flag) {
          expect(actualLabels).toEqual([0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]);
          expect(Number.isNaN(score)).toBe(true);
          expect(await readUint32(status.buffer, 6)).toEqual([0, 0, 0, 0, 0, 0]);
        } else {
          expectRecoveredPairLabels(actualLabels);
          expect(Number.isFinite(score)).toBe(true);
          expect(score).toBe(10);
          expect(await readUint32(status.buffer, 6)).toEqual([2, 1, 0, 0, 0, 0]);
        }
      }
    } finally {
      compiled.destroy();
      rig.destroy();
      device.destroy();
    }
  });
}
