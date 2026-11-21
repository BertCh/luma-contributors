// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPURegionPartitionEvaluation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-region-partition-evaluation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {createPartitionLossPorts} from './partition-loss-ports';

for (const name of ['overflow', 'candidateOverflow'] as const) {
  it(`clears partition evaluation adjacency ${name} loss twice on the same compiled graph`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    const losses = createPartitionLossPorts(rig);
    const summary = rig.output('float32', 9);
    const sizes = rig.output('uint32', 2);
    const within = rig.output('float32', 2);
    const means = rig.output('float32', 2);
    const invalid = rig.output('uint32', 1);
    rig.graph.add(
      new GPURegionPartitionEvaluation({
        weights: {
          offsets: rig.input(Uint32Array.of(0, 1, 3, 5, 6), 'uint32', 5),
          neighbors: rig.input(Uint32Array.of(1, 0, 2, 1, 3, 2), 'uint32', 6),
          weights: rig.input(Float32Array.of(1, 1, 1, 1, 1, 1), 'float32', 6),
          status: losses
        },
        values: rig.input(Float32Array.of(0, 2, 10, 14), 'float32', 4),
        labels: rig.input(Uint32Array.of(0, 0, 1, 1), 'uint32', 4),
        labelCapacity: 2,
        output: {
          summary: summary.view,
          regionSizes: sizes.view,
          regionWithinSumOfSquares: within.view,
          regionMeans: means.view,
          invalidRowCount: invalid.view
        }
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
        const actual = await readFloat32(summary.buffer, 9);
        // Means 1,12; SSDs 2,8; overall mean 6.5 gives total 131 and between 121.
        // Pseudo-F = (121/1)/(10/2) = 24.2. Only crossLinkFraction depends on adjacency.
        const expected = [2, 10, 121, 131, 2, 2, 1 / 3, 0, 24.2];
        for (let word = 0; word < expected.length; word++) {
          if (word === 6 && flag) expect(Number.isNaN(actual[word])).toBe(true);
          else {
            expect(Number.isFinite(actual[word])).toBe(true);
            expect(actual[word]).toBeCloseTo(expected[word], 5);
          }
        }
        expect(await readUint32(sizes.buffer, 2)).toEqual([2, 2]);
        expect(await readFloat32(within.buffer, 2)).toEqual([2, 8]);
        expect(await readFloat32(means.buffer, 2)).toEqual([1, 12]);
        expect(await readUint32(invalid.buffer, 1)).toEqual([0]);
      }
    } finally {
      compiled.destroy();
      rig.destroy();
      device.destroy();
    }
  });
}
