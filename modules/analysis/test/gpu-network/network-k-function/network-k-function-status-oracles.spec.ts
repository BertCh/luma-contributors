// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUNetworkKFunction,
  getGPUNetworkKFunctionParameterValues
} from '../../../src/gpu-network/network-k-function/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createStatusOracleHarness} from '../network-analysis/network-status-oracles-harness';

for (const candidateCapacity of [1, 3, 4]) {
  it(`derives snapping overflow from four point-edge candidates (${candidateCapacity})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createStatusOracleHarness(device);
    const points = harness.input('float32x2', Float32Array.of(0, 0, 1, 0));
    const overflow = harness.output('uint32', 1);
    const converged = harness.output('uint32', 1);
    // Two endpoint events and both directed copies of the same unit road:
    // all FOUR point-edge boxes overlap the radius-2 query boxes. Capacity 1
    // loses candidates; capacity 4 holds all. The one-edge shortest searches
    // need no more than one relaxing hop plus an unchanged certification round.
    await harness.run(
      new GPUNetworkKFunction({
        positions: points,
        topology: {
          offsets: harness.input('uint32', Uint32Array.of(0, 1, 2)),
          neighbors: harness.input('uint32', Uint32Array.of(1, 0)),
          weights: harness.input('float32', Float32Array.of(1, 1)),
          nodePositions: harness.input('float32x2', Float32Array.of(0, 0, 1, 0))
        },
        candidateCapacity,
        maximumSnapDistance: harness.input('float32', Float32Array.of(2)),
        maximumDistance: harness.input('float32', Float32Array.of(2)),
        networkLength: harness.input('float32', Float32Array.of(1)),
        parameters: harness.input(
          'uint32',
          getGPUNetworkKFunctionParameterValues({seed: 1, activeSimulations: 0})
        ),
        bandCount: 2,
        maximumIterations: 4,
        rowsPerBlock: 2,
        laneCount: 1,
        output: {kValues: harness.output('float32', 2), status: {overflow, converged}}
      }),
      [
        read => {
          expect(read(overflow)).toEqual([candidateCapacity < 4 ? 1 : 0]);
          expect(read(converged)).toEqual([1]);
          // Remote events have zero candidate boxes; reuse must clear any prior overflow.
          harness.rewrite(points, Float32Array.of(10, 10, 20, 20));
        },
        read => {
          expect(read(overflow)).toEqual([0]);
          expect(read(converged)).toEqual([1]);
        }
      ]
    );
  });
}

for (const maximumIterations of [1, 8])
  it(`R4 network K publishes both independently derived stopping words (${maximumIterations})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createStatusOracleHarness(device);
    const converged = harness.output('uint32', 1);
    const iterationLimitReached = harness.output('uint32', 1);
    // Opposite endpoints of a six-node unit chain seed only their incident
    // endpoints. One local hop cannot reach the other end; eight rounds settle
    // the five-edge path and its empty frontier. These are stopping expectations,
    // independently of K's numeric value and snapping's capacity status.
    await harness.run(
      new GPUNetworkKFunction({
        topology: {
          offsets: harness.input('uint32', Uint32Array.of(0, 1, 3, 5, 7, 9, 10)),
          neighbors: harness.input('uint32', Uint32Array.of(1, 0, 2, 1, 3, 2, 4, 3, 5, 4)),
          weights: harness.input('float32', new Float32Array(10).fill(1)),
          nodePositions: harness.input(
            'float32x2',
            Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0)
          )
        },
        positions: harness.input('float32x2', Float32Array.of(0, 0, 5, 0)),
        maximumDistance: harness.input('float32', Float32Array.of(6)),
        networkLength: harness.input('float32', Float32Array.of(5)),
        parameters: harness.input(
          'uint32',
          getGPUNetworkKFunctionParameterValues({seed: 1, activeSimulations: 0})
        ),
        bandCount: 2,
        maximumIterations,
        localIterations: 1,
        rowsPerBlock: 2,
        laneCount: 1,
        output: {kValues: harness.output('float32', 2), status: {converged, iterationLimitReached}}
      }),
      [
        read =>
          expect([...read(converged), ...read(iterationLimitReached)]).toEqual(
            maximumIterations === 1 ? [0, 1] : [1, 0]
          )
      ]
    );
  });
