// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkAnalyticsColumns} from '../../../src/gpu-network/network-analysis/gpu-network-analytics-columns';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createStatusOracleHarness} from './network-status-oracles-harness';

it('derives weak-component convergence from an unresolved chain edge and isolated replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createStatusOracleHarness(device);
  const neighbors = harness.input('uint32', Uint32Array.of(1, 0, 2, 1, 3, 2));
  const labels = harness.output('uint32', 4);
  const converged = harness.output('uint32', 1);
  // Undirected 0-1-2-3 starts at labels [0,1,2,3]. At least node 1 must
  // change to 0 in the first sweep, so that sweep cannot certify unchanged
  // labels regardless of invocation order. Removing all edges leaves labels stable.
  await harness.run(
    new GPUNetworkAnalyticsColumns({
      topology: {offsets: harness.input('uint32', Uint32Array.of(0, 1, 3, 5, 6)), neighbors},
      components: {output: labels, maximumIterations: 1, status: {converged}}
    }),
    [
      read => {
        expect(read(converged)).toEqual([0]);
        harness.rewrite(neighbors, new Uint32Array(6).fill(99));
      },
      read => {
        expect(read(converged)).toEqual([1]);
        expect(read(labels)).toEqual([0, 1, 2, 3]);
      }
    ]
  );
});
