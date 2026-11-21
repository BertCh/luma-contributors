// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {
  GPUNetworkIsochrones,
  getGPUNetworkIsochronesParameterValues
} from '../../../src/gpu-network/network-isochrones/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createIsochroneOracleHarness,
  createOracleRingOutput,
  expectOracleValues
} from './isochrones-output-oracle-utils';

// Queued for the forwarded GPUIsobandRings edge-loss handoff; requested status must describe it.
it('writes poisoned raster ring completeness and polygon count and recovers (B5)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createIsochroneOracleHarness(device);
  const ringOutput = createOracleRingOutput(harness, 'all', 2, 5);
  const breaks = harness.input('float32', [1]);
  const contributor = new GPUNetworkIsochrones({
    topology: {
      offsets: harness.input('uint32', [0, 1]),
      neighbors: harness.input('uint32', [0]),
      weights: harness.input('float32', [0]),
      nodePositions: harness.input('float32x2', [1, 1])
    },
    costs: harness.input('float32', [0]),
    costBreaks: breaks,
    parameters: harness.input(
      'float32',
      Array.from(
        getGPUNetworkIsochronesParameterValues({
          breakCount: 1,
          extent: [0, 0, 2, 2],
          bufferRadius: 2
        })
      )
    ),
    raster: {
      width: 2,
      height: 2,
      output: {bandClasses: harness.output('classes', 'uint32', 4)},
      rings: {
        edgeCapacity: 2,
        output: {
          ...ringOutput,
          status: {
            ...ringOutput.status,
            edgeOverflow: harness.output('ringEdgeOverflow', 'uint32', 1),
            candidateOverflow: harness.output('ringCandidateOverflow', 'uint32', 1),
            complete: harness.output('ringComplete', 'uint32', 1),
            polygonCount: harness.output('polygonCount', 'uint32', 1)
          }
        }
      }
    }
  });
  const result = await harness.run(contributor, first => {
    // The constant square needs four edges; two retained edges cannot form a closed polygon.
    expectOracleValues(first['ringEdgeOverflow'], [1]);
    expectOracleValues(first['ringOverflow'], [0]);
    expectOracleValues(first['ringCandidateOverflow'], [1]);
    expectOracleValues(first['ringComplete'], [0]);
    expectOracleValues(first['polygonCount'], [0]);
    harness.write(breaks, [-1]);
  });
  expectOracleValues(result['ringEdgeOverflow'], [0]);
  expectOracleValues(result['ringOverflow'], [0]);
  expectOracleValues(result['ringCandidateOverflow'], [0]);
  expectOracleValues(result['ringComplete'], [1]);
  expectOracleValues(result['polygonCount'], [0]);
});
