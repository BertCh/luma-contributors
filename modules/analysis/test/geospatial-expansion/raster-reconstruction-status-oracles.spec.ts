// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPURasterReconstruction} from '../../src/raster-analysis/gpu-raster-reconstruction';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {createParityGraph} from '../oracles/parity-wp2-geodesic-harness';
import {readFloat32, readUint32, submitGraph} from '../utils/gpu-contributor-test-utils';

it('reconstruction verifies a one-round frontier and rewrites convergence on a stable replay', async () => {
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  const rig = createParityGraph(device);
  const marker = rig.input('marker', [1, 0, 0, 0], 'float32');
  const values = rig.output('values', 4, 'float32');
  const validity = rig.output('validity', 4, 'uint32');
  const converged = rig.output('converged', 1, 'uint32');
  const iterationLimitReached = rig.output('limit', 1, 'uint32');
  const iterationCount = rig.output('iterations', 1, 'uint32');
  rig.graph.add(
    new GPURasterReconstruction({
      grid: {
        width: 4,
        height: 1,
        affine: [1, 0, 0, 0, 1, 0],
        pixelInterpretation: 'area'
      },
      marker: marker.view,
      mask: rig.input('mask', [1, 1, 1, 1], 'float32').view,
      maximumIterations: 1,
      output: {
        values: values.view,
        validity: validity.view,
        status: {
          converged: converged.view,
          iterationLimitReached: iterationLimitReached.view,
          iterationCount: iterationCount.view
        }
      }
    })
  );
  const compiled = rig.graph.compile();
  try {
    for (const frame of [
      {marker: [1, 0, 0, 0], values: [1, 1, 0, 0], status: [0, 1, 2]},
      {marker: [1, 1, 1, 1], values: [1, 1, 1, 1], status: [1, 0, 1]},
      {marker: [1, 0, 0, 0], values: [1, 1, 0, 0], status: [0, 1, 2]}
    ]) {
      // By hand: round one raises cell 1; cell 2 would rise in round two, so work remains.
      // Verification counts as a pass but cannot advance the published values. The uniform
      // marker is already a fixed point and needs exactly one unchanged pass.
      marker.buffer.write(Float32Array.from(frame.marker));
      for (const port of [converged, iterationLimitReached, iterationCount]) {
        port.buffer.write(Uint32Array.of(0xdeadbeef));
      }
      validity.buffer.write(new Uint32Array(4).fill(0xdeadbeef));
      submitGraph(device, compiled, undefined);
      expect(await readFloat32(values.buffer, 4)).toEqual(frame.values);
      expect(await readUint32(validity.buffer, 4)).toEqual([1, 1, 1, 1]);
      expect([
        (await readUint32(converged.buffer, 1))[0],
        (await readUint32(iterationLimitReached.buffer, 1))[0],
        (await readUint32(iterationCount.buffer, 1))[0]
      ]).toEqual(frame.status);
    }
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});
