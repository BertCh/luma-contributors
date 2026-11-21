// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkInhomogeneousK} from '../../src/gpu-network/gpu-network-inhomogeneous-k';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {createOracleContext} from './adjacency-matrix/network-output-oracle-test-utils';

it('inhomogeneous network K publishes all three independently derived normalizers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A single length-4 edge, one event at each end, intensities 1 and 2.
  // Sum inverse intensities=1+1/2=1.5; denominators are L=4, S=1.5, S²/L=0.5625.
  // Two ordered pairs contribute 1/(1*2) each at radius 4; numerator=1.
  for (const [normalization, denominator, expectedK] of [
    ['network-length', 4, 0.25],
    ['inverse-intensity', 1.5, 2 / 3],
    ['inverse-intensity-squared', 0.5625, 16 / 9]
  ] as const) {
    const context = createOracleContext(device);
    const {graph, input, output, read} = context;
    const result = {
      k: output('k', 'float32', 2),
      numerator: output('numerator', 'float32', 2),
      normalization: output('normalization', 'float32', 3),
      status: output('status', 'uint32', 2)
    };
    graph.add(
      new GPUNetworkInhomogeneousK({
        eventNodeIds: input('events', 'uint32', [0, 1]),
        intensity: input('intensity', 'float32', [1, 2]),
        nodeCount: 2,
        nodeDistances: input('distances', 'float32', [0, 4, 4, 0]),
        edgeNodes: input('edges', 'uint32x2', [0, 1]),
        edgeLengths: input('lengths', 'float32', [4]),
        radii: input('radii', 'float32', [3, 4]),
        correction: 'none',
        normalization,
        output: result
      })
    );
    try {
      context.run();
      expect(await read(result.normalization)).toEqual([4, 1.5, denominator]);
      expect(await read(result.numerator)).toEqual([0, 1]);
      expect(await read(result.status)).toEqual([0, 0]);
      const values = await read(result.k);
      expect(values[0]).toBe(0);
      expect(Number.isFinite(values[1])).toBe(true);
      // One float32 quotient, comfortably within two ulps.
      expect(values[1]).toBeCloseTo(expectedK, 6);
    } finally {
      context.destroy();
    }
  }
});
