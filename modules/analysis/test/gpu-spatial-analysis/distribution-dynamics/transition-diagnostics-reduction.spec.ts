// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUTransitionDiagnostics} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-transition-diagnostics';
import {GraphRig} from '../areal-interpolation/areal-interpolation-harness';

it('B5 identical mixed-magnitude regimes have exactly zero Pearson statistic', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device),
    conditionCount = 65537;
  try {
    const counts = Uint32Array.from({length: conditionCount * 4}, (_, index) => {
      const cell = index % 4;
      if (cell > 1) return 0;
      return (cell === 0 ? 1 : 3) * (index < 4 ? 2 ** 24 : 1);
    });
    const homogeneity = rig.output('float32', 5);
    rig.run(
      new GPUTransitionDiagnostics({
        counts: rig.input(counts, 'uint32'),
        classCount: 2,
        conditionCount,
        output: {homogeneity: homogeneity.view}
      })
    );
    // All nonempty rows have probabilities [.25,.75], regardless of their mass.
    // Pooled counts [16842752,50528256], Q=LR=0, df=(65537-1)*(2-1).
    const actual = await homogeneity.readFloat32();
    expect(actual).toHaveLength(5);
    expect(actual.every(Number.isFinite)).toBe(true);
    expect(actual).toEqual([0, 1, 0, 1, 65536]);
  } finally {
    rig.destroy();
  }
});

it('W3-02 parallel condition summaries retain 513 conditions and empty behavior', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device);
  const conditionCount = 513;
  try {
    const counts = rig.output('uint32', conditionCount * 4),
      homogeneity = rig.output('float32', 5),
      stateSparsity = rig.output('uint32', 2);
    counts.buffer.write(new Uint32Array(conditionCount * 4).fill(2));
    rig.run(
      new GPUTransitionDiagnostics({
        counts: counts.view,
        classCount: 2,
        conditionCount,
        output: {homogeneity: homogeneity.view, stateSparsity: stateSparsity.view}
      })
    );
    // Every chain has identical probabilities [1/2,1/2]. Q=LR=0; df=2*(513-1)=1024.
    const summary = await homogeneity.readFloat32();
    expect(summary.every(Number.isFinite)).toBe(true);
    expect(summary).toEqual([0, 1, 0, 1, 1024]);
    // 2*513 rows have 4 transitions (< default minimum 5).
    expect(await stateSparsity.readUint32()).toEqual([1026, 0]);
    counts.buffer.write(new Uint32Array(conditionCount * 4));
    rig.resubmit();
    const empty = await homogeneity.readFloat32();
    expect(empty[0]).toBe(0);
    expect(empty[1]).toBeNaN();
    expect(empty[2]).toBe(0);
    expect(empty[3]).toBeNaN();
    expect(empty[4]).toBe(0);
    expect(await stateSparsity.readUint32()).toEqual([0, 1026]);
  } finally {
    rig.destroy();
  }
});
