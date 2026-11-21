// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUPointPatternEnvelopeOperation} from '../../../src/gpu-spatial-analysis/point-process/gpu-point-pattern-envelope-operation';
import {GPUTrajectoryStaypointsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-staypoints-operation';
import {GPUTrajectoryLocationsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-locations-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

it('B2 maximum-deviation retains the signed-reference residual through comparison', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runOperationGraph(device, {
    inputs: {
      observed: {format: 'float32', values: Float32Array.of(-1)},
      reference: {format: 'float32', values: Float32Array.of(2 ** -26)},
      simulations: {format: 'float32', values: new Float32Array(7).fill(1)}
    },
    operations: [
      new GPUPointPatternEnvelopeOperation({
        observed: 'observed',
        reference: 'reference',
        simulations: 'simulations',
        curveRowCount: 1,
        statistic: 'maximum-deviation',
        nullModel: 'fixed external curves',
        fittedParameters: 'fixed-external',
        output: {lower: 'lower', upper: 'upper', statistics: 'statistics', status: 'status'}
      })
    ],
    outputs: ['lower', 'upper', 'statistics', 'status']
  });
  // Exact deviations are 1+2^-26 and 1-2^-26. Zero of seven exceed: p=(0+1)/(7+1).
  expect([...result['statistics']].every(Number.isFinite)).toBe(true);
  expect([...result['statistics']]).toEqual([1, 0.125, 0.125, 7]);
  expect([...result['status']]).toEqual([0]);
  expect([...result['lower']]).toEqual([1]);
  expect([...result['upper']]).toEqual([1]);
});

it('B6 staypoint votes identify equivalent longitudes one float32 step from the seam', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runOperationGraph(device, {
    inputs: {
      positions: {
        format: 'float32x2',
        values: Float32Array.of(179.99998474121094, 0, -180.00001525878906, 0, 177, 0)
      },
      timestamps: {format: 'float32', values: Float32Array.of(0, 1, 2)},
      offsets: {format: 'uint32', values: Uint32Array.of(0, 3)},
      parameters: {format: 'float32', values: Float32Array.of(500000, 1, 0, 1)}
    },
    operations: [
      new GPUTrajectoryStaypointsOperation({
        positions: 'positions',
        timestamps: 'timestamps',
        trackOffsets: 'offsets',
        parameters: 'parameters',
        rowCount: 3,
        capacity: 1,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
        output: {
          ids: 'tracks',
          centroids: 'centroids',
          startRows: 'starts',
          endRows: 'ends',
          staypointIds: 'ids',
          status: {count: 'count', overflow: 'overflow', candidateOverflow: 'candidateOverflow'}
        }
      })
    ],
    outputs: ['centroids', 'starts', 'ends', 'ids', 'count', 'overflow']
  });
  // The seam spellings differ by exactly 360 degrees and contribute one vote.
  // Mean of two distinct longitudes: (179.99998474121094+177)/2=178.49999237060547.
  expect(result['centroids']).toHaveLength(2);
  expect([...result['centroids']].every(Number.isFinite)).toBe(true);
  expect(result['centroids'][0]).toBeCloseTo(178.49999237060547, 4);
  expect(result['centroids'][1]).toBe(0);
  expect([...result['count']]).toEqual([1]);
  expect([...result['overflow']]).toEqual([0]);
  expect([...result['starts']]).toEqual([0]);
  expect([...result['ends']]).toEqual([3]);
  expect([...result['ids']]).toEqual([0, 0, 0]);
});

it('B6 location votes identify equivalent longitudes one float32 step from the seam', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runOperationGraph(device, {
    inputs: {
      positions: {
        format: 'float32x2',
        values: Float32Array.of(179.99998474121094, 0, -180.00001525878906, 0, 177, 0)
      },
      tracks: {format: 'uint32', values: Uint32Array.of(0, 0, 0)},
      active: {format: 'uint32', values: Uint32Array.of(3)},
      parameters: {format: 'float32', values: Float32Array.of(400000, 0, 0, 0)}
    },
    operations: [
      new GPUTrajectoryLocationsOperation({
        centroids: 'positions',
        tracks: 'tracks',
        count: 'active',
        parameters: 'parameters',
        capacity: 3,
        maximumIterations: 2,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
        output: {
          locationIds: 'ids',
          centers: 'centers',
          status: {count: 'count', iterationLimitReached: 'limit'}
        }
      })
    ],
    outputs: ['centers', 'ids', 'count']
  });
  // Same two distinct physical longitude votes as the staypoint fixture above.
  expect(result['centers']).toHaveLength(6);
  expect(Number.isFinite(result['centers'][0])).toBe(true);
  expect(result['centers'][0]).toBeCloseTo(178.49999237060547, 4);
  expect(result['centers'][1]).toBe(0);
  expect([...result['count']]).toEqual([1]);
  expect([...result['ids']]).toEqual([0, 0, 0]);
});
