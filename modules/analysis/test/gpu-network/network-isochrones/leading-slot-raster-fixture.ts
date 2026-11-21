// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUNetworkIsochrones,
  getGPUNetworkIsochronesParameterValues
} from '../../../src/gpu-network/network-isochrones/index';
import type {createOptimizationHarness} from '../network-optimization/optimization-test-harness';

/** Shared fixture so host lowering and device readback exercise the same output request. */
export function createLeadingSlotRasterFixture(
  harness: Pick<ReturnType<typeof createOptimizationHarness>, 'input' | 'output'>,
  sourcePaths = false
) {
  const values = harness.output('float32', 4);
  const topology = {
    offsets: harness.input('uint32', Uint32Array.of(1, 1, 2)),
    neighbors: harness.input('uint32', Uint32Array.of(1, 1)),
    weights: harness.input('float32', Float32Array.of(1, 1)),
    nodePositions: harness.input('float32x2', Float32Array.of(0.5, 0.5, 1.5, 0.5))
  };
  const costs = harness.input('float32', Float32Array.of(0, Infinity));
  const sourcePathOffsets = sourcePaths
    ? harness.input('uint32', Uint32Array.of(0, 1, 1))
    : undefined;
  const paths = sourcePathOffsets
    ? {
        topology: {
          offsets: sourcePathOffsets,
          neighbors: harness.input('uint32', Uint32Array.of(1)),
          weights: harness.input('float32', Float32Array.of(0)),
          nodePositions: harness.input('float32x2', Float32Array.of(0.5, 0.5, 0.5, 0.5))
        },
        costs: harness.input('float32', Float32Array.of(7, Infinity))
      }
    : undefined;
  const contributor = new GPUNetworkIsochrones({
    topology,
    costs,
    costBreaks: harness.input('float32', Float32Array.of(1)),
    parameters: harness.input(
      'float32',
      getGPUNetworkIsochronesParameterValues({
        breakCount: 1,
        extent: [0, 0, 2, 2],
        bufferRadius: 1,
        walkCostPerUnit: 0
      })
    ),
    raster: {
      width: 2,
      height: 2,
      maximumSamplesPerEdge: 2,
      ...(paths ? {sourcePaths: paths} : {}),
      output: {values, bandClasses: harness.output('uint32', 4)}
    }
  });
  return {contributor, values, sourcePathOffsets};
}
