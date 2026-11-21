// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUCoverageDissolve,
  GPUCoverageValidity
} from '../../../src/gpu-spatial-analysis/polygon-coverage-topology';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

it('CPL-14 preserves an incumbent dissolve and rewrites its label provenance on reuse', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const inputComplete = rig.input(Uint32Array.of(0), 'uint32', 1);
    const complete = rig.input(Uint32Array.of(99, 0x12345678), 'uint32', 2);
    const count = rig.output('uint32', 1),
      overflow = rig.output('uint32', 1);
    rig.graph.add(
      new GPUCoverageDissolve({
        positions: rig.input(Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1), 'float32x2', 4),
        ringOffsets: rig.input(Uint32Array.of(0, 4), 'uint32', 2),
        polygonOffsets: rig.input(Uint32Array.of(0, 1), 'uint32', 2),
        featureOffsets: rig.input(Uint32Array.of(0, 1), 'uint32', 2),
        labels: rig.input(Uint32Array.of(3), 'uint32', 1),
        labelsComplete: inputComplete,
        output: {
          positions: rig.output('float32x2', 5).view,
          ringOffsets: rig.output('uint32', 2).view,
          status: {count: count.view, overflow: overflow.view, labelsComplete: complete}
        }
      })
    );
    const compiled = rig.graph.compile();
    try {
      for (const value of [0, 1, 0]) {
        rig.bufferOf(inputComplete).write(Uint32Array.of(value));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(rig.bufferOf(complete), 2)).toEqual([value, 0x12345678]);
        expect(await readUint32(count.buffer, 1)).toEqual([1]);
        expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

it('WRT-08 coverage validity preserves the second seeded status word', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const total = rig.input(Uint32Array.of(99, 0x12345678), 'uint32', 2);
    rig.run(
      new GPUCoverageValidity({
        positions: rig.input(Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1), 'float32x2', 4),
        ringOffsets: rig.input(Uint32Array.of(0, 4), 'uint32', 2),
        polygonOffsets: rig.input(Uint32Array.of(0, 1), 'uint32', 2),
        featureOffsets: rig.input(Uint32Array.of(0, 1), 'uint32', 2),
        parameters: rig.input(new Float32Array(4), 'float32', 4),
        output: {segmentFlags: rig.output('uint32', 4).view, status: {invalidCount: total}}
      })
    );
    expect(await readUint32(rig.bufferOf(total), 2)).toEqual([0, 0x12345678]);
  } finally {
    rig.destroy();
  }
});

it('B6 dissolve completeness includes snap uncertainty without an optional uncertainty output', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const complete = rig.output('uint32', 1),
      overflow = rig.output('uint32', 1);
    rig.run(
      new GPUCoverageDissolve({
        positions: rig.input(
          Float32Array.of(0, 0, 2, 0, 0, 2, 300000, 300000, 300002, 300000, 300000, 300002),
          'float32x2',
          6
        ),
        ringOffsets: rig.input(Uint32Array.of(0, 3, 6), 'uint32', 3),
        polygonOffsets: rig.input(Uint32Array.of(0, 1, 2), 'uint32', 3),
        featureOffsets: rig.input(Uint32Array.of(0, 1, 2), 'uint32', 3),
        labels: rig.input(Uint32Array.of(1, 2), 'uint32', 2),
        snapGridSize: 0.0001,
        output: {
          positions: rig.output('float32x2', 12).view,
          ringOffsets: rig.output('uint32', 4).view,
          status: {
            count: rig.output('uint32', 1).view,
            overflow: overflow.view,
            complete: complete.view
          }
        }
      })
    );
    expect(await readUint32(complete.buffer, 1)).toEqual([0]);
    expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
  } finally {
    rig.destroy();
  }
});
