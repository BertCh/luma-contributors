// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUDBSCAN} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-spatial-clustering';
import {GPUOPTICS} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

for (const epsilon of [1e-23, 2 ** -63, 2 ** 63, 2e19]) {
  it(`enforces the planar squared-distance range on raw parameters at epsilon ${epsilon}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    const positions = rig.input(Float32Array.of(0, 0, 2 * epsilon, 0), 'float32x2', 2);
    const parameters = rig.input(
      Float32Array.of(0, 0, 2 * epsilon, 0, epsilon, 2, 0, 0),
      'float32',
      8
    );
    const labels = rig.output('uint32', 2);
    rig.graph.add(
      new GPUDBSCAN({
        positions,
        parameters,
        gridSize: [1, 1],
        output: {labels: labels.view}
      })
    );
    const compiled = await rig.graph.compileAsync();
    try {
      submitGraph(device, compiled, undefined);
      // Two points separated by 2*epsilon never form a two-point epsilon cluster.
      expect(await readUint32(labels.buffer, 2)).toEqual([0xffffffff, 0xffffffff]);
      rig.bufferOf(positions).write(new Float32Array(4));
      submitGraph(device, compiled, undefined);
      const supported = epsilon >= 2 ** -63 && epsilon <= 2 ** 63;
      expect(await readUint32(labels.buffer, 2)).toEqual(
        supported ? [0, 0] : [0xffffffff, 0xffffffff]
      );
    } finally {
      compiled.destroy();
      rig.destroy();
      device.destroy();
    }
  });
}

it('keeps a noncoincident pair connected under a saturated OPTICS radius', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const labels = rig.output('uint32', 2);
  rig.graph.add(
    new GPUOPTICS({
      positions: rig.input(Float32Array.of(0, 0, 2 ** 70, 2 ** 70), 'float32x2', 2),
      minimumSamples: 1,
      maximumDistance: 3.4028234663852886e38,
      extractionDistance: 2 ** 72,
      neighborIndex: 'grid',
      neighborCapacity: 2,
      output: {
        ordering: rig.output('uint32', 2).view,
        coreDistances: rig.output('float32', 2).view,
        reachability: rig.output('float32', 2).view,
        predecessors: rig.output('uint32', 2).view,
        labels: labels.view
      }
    })
  );
  const compiled = await rig.graph.compileAsync();
  try {
    submitGraph(device, compiled, undefined);
    // Distance sqrt(2)*2^70 is below both the search and extraction radii.
    expect(await readUint32(labels.buffer, 2)).toEqual([0, 0]);
  } finally {
    compiled.destroy();
    rig.destroy();
    device.destroy();
  }
});

for (const neighborIndex of ['brute-force', 'grid'] as const) {
  it(`keeps two-sample OPTICS core distances finite at a large radius with ${neighborIndex}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    const separation = 2 ** 69;
    const coreDistances = rig.output('float32', 2);
    const labels = rig.output('uint32', 2);
    try {
      rig.graph.add(
        new GPUOPTICS({
          positions: rig.input(Float32Array.of(0, 0, separation, 0), 'float32x2', 2),
          minimumSamples: 2,
          maximumDistance: 2 ** 72,
          extractionDistance: separation,
          neighborIndex,
          output: {
            ordering: rig.output('uint32', 2).view,
            coreDistances: coreDistances.view,
            reachability: rig.output('float32', 2).view,
            predecessors: rig.output('uint32', 2).view,
            labels: labels.view
          }
        })
      );
      const compiled = await rig.graph.compileAsync();
      try {
        submitGraph(device, compiled, undefined);
        // Each row needs the other row. Their exact axial distance is 2^69.
        expect(await readFloat32(coreDistances.buffer, 2)).toEqual([separation, separation]);
        expect(await readUint32(labels.buffer, 2)).toEqual([0, 0]);
      } finally {
        compiled.destroy();
      }
    } finally {
      rig.destroy();
      device.destroy();
    }
  });
}

for (const temporal of [false, true]) {
  it(`executes finite saturated OPTICS ${temporal ? 'time' : 'distance'} radius on a tiny grid (S3)`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    const positions = rig.input(new Float32Array(4), 'float32x2', 2);
    const labels = rig.output('uint32', 2),
      reason = rig.output('uint32', 1);
    rig.graph.add(
      new GPUOPTICS({
        positions,
        minimumSamples: 1,
        maximumDistance: temporal ? 1 : 3.4028234663852886e38,
        extractionDistance: 1,
        neighborIndex: 'grid',
        neighborCapacity: 2,
        ...(temporal
          ? {
              times: rig.input(Float32Array.of(0, 0), 'float32', 2),
              maximumTimeDistance: 3.4028234663852886e38,
              temporalContext: {
                encoding: 'float32-relative' as const,
                epoch: 0n,
                unit: 'second' as const,
                axis: {kind: 'irregular' as const}
              }
            }
          : {}),
        output: {
          ordering: rig.output('uint32', 2).view,
          coreDistances: rig.output('float32', 2).view,
          reachability: rig.output('float32', 2).view,
          predecessors: rig.output('uint32', 2).view,
          labels: labels.view,
          ports: {status: {reason: reason.view}}
        }
      })
    );
    const compiled = await rig.graph.compileAsync();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(labels.buffer, 2)).toEqual([0, 0]);
      expect(await readUint32(reason.buffer, 1)).toEqual([0]);
    } finally {
      compiled.destroy();
      rig.destroy();
      device.destroy();
    }
  });
}
