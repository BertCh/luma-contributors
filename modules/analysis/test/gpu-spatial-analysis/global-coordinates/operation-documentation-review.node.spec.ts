// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {computeTrackSimilarityOnCPU} from '../../../src/gpu-spatial-analysis/track-similarity/cpu-track-similarity-operation';
import {computeTrajectoryThinningOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-thinning-operation';
import {computeSphericalArcOverlayOnCPU} from '../../../src/gpu-spatial-analysis/global-coordinates/cpu-spherical-arc-overlay-operation';
import {getArcScalingOperand} from './global-coordinate-scaling-fixtures';

const movement = readFileSync('docs/api/gpu-spatial-analysis/movement-coverage.md', 'utf8');
it('documents the similarity active suffix, geographic route and explicit CPU cap', () => {
  expect(movement).toContain('inactive suffix');
  expect(movement).toContain('longitude/latitude');
  expect(movement).toContain('maximumFrechetVertexCount');
});
it('lists the required timeOrigin for word-time restriction', () => {
  const restriction = movement
    .split('\n')
    .find(line => line.startsWith('| `GPUTrajectoryRestrictionOperation`'))!;
  expect(restriction).toContain('timeOrigin');
  expect(restriction).toContain('uint32x2');
});
it('qualifies the shared arc output capacity ceiling as GPU-only', () => {
  const source = readFileSync(
    'modules/analysis/src/gpu-spatial-analysis/global-coordinates/gpu-spherical-arc-overlay-operation.ts',
    'utf8'
  );
  expect(source).toContain('GPU-only');
});
it('pins the shared inactive/invalid status and the optional CPU Frechet cap on geographic input', () => {
  const input = {
    linesA: {positions: Float32Array.of(0, 0, 1, 0, 2, 0), lineOffsets: Uint32Array.of(0, 3)},
    pairA: Uint32Array.of(0, 9, 0),
    pairB: Uint32Array.of(0, 0, 0),
    activePairCount: Uint32Array.of(2)
  };
  const options = {
    measures: ['frechet'] as const,
    spatialContext: {
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      sphereRadius: 1
    } as const
  };
  const capped = computeTrackSimilarityOnCPU(input, {...options, maximumFrechetVertexCount: 1});
  expect(Array.from(capped.pairStatus)).toEqual([2, 4, 4]);
  const uncapped = computeTrackSimilarityOnCPU(input, options);
  expect(Array.from(uncapped.pairStatus)).toEqual([0, 4, 4]);
  // Identical curves have Frechet distance zero by choosing equal points at every coupling step.
  expect(Number.isFinite(uncapped.frechet[0])).toBe(true);
  expect(uncapped.frechet[0]).toBe(0);
});
it('pins explicit CPU thinning search loss and the uncapped default', () => {
  const input = {
    positions: new Float32Array(8),
    timestamps: Float32Array.of(0, 1, 2, 3),
    trackOffsets: Uint32Array.of(0, 4),
    parameters: Float32Array.of(10, 0, 0, 0)
  };
  const capped = computeTrajectoryThinningOnCPU(input, {
    mode: 'minimum-time',
    capacity: 4,
    maximumSearchRows: 1
  });
  // No later row reaches ten seconds. The explicit cap advances to row 2, splitting the chain.
  expect(Array.from(capped.sourceRows)).toEqual([0, 2, 3, 0xffffffff]);
  expect(Array.from(capped.candidateOverflow)).toEqual([1]);
  const uncapped = computeTrajectoryThinningOnCPU(input, {mode: 'minimum-time', capacity: 4});
  expect(Array.from(uncapped.sourceRows)).toEqual([0, 3, 0xffffffff, 0xffffffff]);
  expect(Array.from(uncapped.candidateOverflow)).toEqual([0]);
});
it('allows a CPU arc output capacity above the GPU-only ceiling', () => {
  const operand = (right: boolean) => ({
    positions: Float32Array.from(getArcScalingOperand(8, right).flat()),
    ringOffsets: Uint32Array.of(0, 4),
    referenceInside: true
  });
  const result = computeSphericalArcOverlayOnCPU(
    {left: operand(false), right: operand(true)},
    {
      referencePoint: [0, 0, 1],
      operation: 'intersection',
      capacity: 1536,
      ringCapacity: 64
    }
  );
  expect(Array.from(result.status)).toEqual([0]);
  expect(result.positions).toHaveLength(1536 * 3);
  expect(Array.from(result.counts)).toEqual([4, 1, 4, 1]);
});
