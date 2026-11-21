// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createIsochronesDecodeNode,
  createIsochronesSplatNode
} from '../../../src/gpu-network/network-isochrones/network-isochrones-passes';
import {getGPUNetworkIsochronesParameterValues} from '../../../src/gpu-network/network-isochrones/index';
import type {IsochroneOracleHarness} from './isochrones-output-oracle-utils';

/** Three samples of [-2^30, 2^30] are its endpoints and midpoint. Only the midpoint
 * reaches [-1, 0, 1] at radius 1; t=1/2 gives cost (0 + 2/2) = 1 in each pixel.
 * Exercise the splat directly to retain the independent reference's 3x1 grid.
 */
export function createLongRoadSamplingFixture(
  harness: Pick<IsochroneOracleHarness, 'graph' | 'input' | 'output'>
) {
  const values = harness.output('values', 'float32', 3);
  const encoded = harness.input('uint32', [0xffffffff, 0xffffffff, 0xffffffff]);
  const nodes = [
    createIsochronesSplatNode(harness.graph, {
      id: 'long-road-splat',
      nodeCount: 2,
      width: 3,
      height: 1,
      mode: 'min',
      maximumBufferPixels: 6,
      maximumSamplesPerEdge: 3,
      offsets: harness.input('uint32', [0, 1, 1]),
      neighbors: harness.input('uint32', [1]),
      weights: harness.input('float32', [2]),
      costs: harness.input('float32', [0, Infinity]),
      nodePositions: harness.input('float32x2', [-(2 ** 30), 0, 2 ** 30, 0]),
      parameters: harness.input(
        'float32',
        Array.from(
          getGPUNetworkIsochronesParameterValues({
            breakCount: 1,
            extent: [-1.5, -0.5, 1.5, 0.5],
            bufferRadius: 1,
            walkCostPerUnit: 0
          })
        )
      ),
      encoded
    }),
    createIsochronesDecodeNode(harness.graph, {
      id: 'long-road-decode',
      mode: 'min',
      unreachedCost: 1e30,
      encoded,
      values,
      pixelCount: 3
    })
  ];
  return {values, nodes};
}
