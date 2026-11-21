// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkInhomogeneousK} from '../../src/gpu-network/gpu-network-inhomogeneous-k';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {createStatusOracleHarness} from './network-analysis/network-status-oracles-harness';

it('R5 radius-order bit is per row and clears after repairing the radii', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createStatusOracleHarness(device);
  const radii = harness.input('float32', Float32Array.of(4, 3));
  const status = harness.output('uint32', 2);
  // All data describe a valid length-4 edge. Only the second radius violates
  // increasing order: its complete diagnostic word is bit 16, with bits
  // 1/2/4/8/32 absent. The first radius must remain exactly zero.
  await harness.run(
    new GPUNetworkInhomogeneousK({
      eventNodeIds: harness.input('uint32', Uint32Array.of(0, 1)),
      intensity: harness.input('float32', Float32Array.of(1, 2)),
      nodeDistances: harness.input('float32', Float32Array.of(0, 4, 4, 0)),
      nodeCount: 2,
      edgeNodes: harness.input('uint32x2', Uint32Array.of(0, 1)),
      edgeLengths: harness.input('float32', Float32Array.of(4)),
      radii,
      correction: 'none',
      output: {
        k: harness.output('float32', 2),
        numerator: harness.output('float32', 2),
        normalization: harness.output('float32', 3),
        status
      }
    }),
    [
      read => {
        expect(read(status)).toEqual([0, 16]);
        harness.rewrite(radii, Float32Array.of(3, 4));
      },
      read => expect(read(status)).toEqual([0, 0])
    ]
  );
});
