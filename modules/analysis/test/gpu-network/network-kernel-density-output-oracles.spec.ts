// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkKernelDensity} from '../../src/gpu-network/gpu-network-kernel-density';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {createOracleContext} from './adjacency-matrix/network-output-oracle-test-utils';

it('network KDE invalidCount is four after global mass invalidation and zero after repair', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const withCount of [true, false]) {
    const context = createOracleContext(device);
    const {graph, input, output, read, write} = context;
    const distances = input('distances', 'float32', [
      0,
      Infinity,
      -1,
      Infinity,
      1,
      Infinity,
      Infinity,
      0
    ]);
    const intensity = output('intensity', 'float32', 4);
    const kernelMass = output('mass', 'float32', 2);
    const validity = output('validity', 'uint32', 4);
    const invalidCount = withCount ? output('invalid-count', 'uint32') : undefined;
    graph.add(
      new GPUNetworkKernelDensity({
        distances,
        eventCount: 2,
        quadratureLengths: input('lengths', 'float32', [1, 2, 1, 3]),
        bandwidth: 2,
        output: {
          intensity,
          kernelMass,
          validity,
          ...(invalidCount ? {status: {invalidCount}} : {})
        }
      })
    );
    try {
      // A negative distance invalidates event 0's mass, hence all FOUR queries, even the
      // disconnected query. This expectation comes from the documented global mass requirement.
      context.run();
      expect(await read(validity)).toEqual([0, 0, 0, 0]);
      if (invalidCount) expect(await read(invalidCount)).toEqual([4]);
      expect(await read(intensity)).toEqual([NaN, NaN, NaN, NaN]);
      // Repair the input on the same compiled graph. Triangular masses at bandwidth 2 are
      // (1+2+1/2)/2=7/4 and 3/2; corrected intensities are [2/7,2/7,1/7,1/3].
      write(distances, [0, Infinity, 0, Infinity, 1, Infinity, Infinity, 0]);
      context.run();
      expect(await read(validity)).toEqual([1, 1, 1, 1]);
      if (invalidCount) expect(await read(invalidCount)).toEqual([0]);
      expect(await read(kernelMass)).toEqual([1.75, 1.5]);
      const values = await read(intensity);
      expect(values).toHaveLength(4);
      for (const [index, expected] of [2 / 7, 2 / 7, 1 / 7, 1 / 3].entries()) {
        expect(Number.isFinite(values[index])).toBe(true);
        expect(values[index]).toBeCloseTo(expected, 6); // float32 division, < 2 ulps.
      }
    } finally {
      context.destroy();
    }
  }
});
