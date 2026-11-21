// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUCoverageDissolve} from '../../../src/gpu-spatial-analysis/polygon-coverage-topology/gpu-coverage-dissolve';
import {requireTestDevice} from '../../utils/require-test-device';
import {createOutputOracleFixture} from '../outline-geometry/output-oracle-fixture';
import {
  createDissolveCapacityOracle,
  createDissolveTouchOracle
} from './coverage-dissolve-output-oracles';

it('publishes two written rings and the exact three-ring retry requirement', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createDissolveCapacityOracle(device);
  try {
    const result = await fixture.runGPU();
    expect(result['count']).toEqual([2]);
    expect(result['requiredCount']).toEqual([3]);
    expect(result['overflow']).toEqual([1]);
    expect(result['offsets']).toEqual([0, 5, 10]);
  } finally {
    fixture.destroy();
  }
});

it('publishes labeled shells and a hole with each optional assembly output alone and absent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Disjoint squares: first is a 4x4 shell minus a 2x2 hole (label 37, area 12),
  // second a 2x2 shell (label 91, area 4). No edge cancels. Input order fixes ring leaders:
  // shell 0, hole 1, shell 2. Assembly closes rings; polygon output omits closing duplicates.
  const coordinates = [0, 0, 4, 0, 4, 4, 0, 4, 1, 1, 1, 3, 3, 3, 3, 1, 10, 0, 12, 0, 12, 2, 10, 2];
  for (const optional of [
    undefined,
    'polygons',
    'polygonsWithoutIds',
    'segmentRings',
    'segmentVertices',
    'segmentFlags',
    'requiredCount',
    'touchingSegmentCount'
  ] as const) {
    const fixture = createOutputOracleFixture(device);
    const {column, output, graph} = fixture;
    const polygons =
      optional === 'polygons' || optional === 'polygonsWithoutIds'
        ? {
            kind: 'polygons' as const,
            positions: output('polygonPositions', 'float32x2', 15),
            featureOffsets: output('featureOffsets', 'uint32', 4),
            polygonOffsets: output('polygonOffsets', 'uint32', 4),
            ringOffsets: output('polygonRingOffsets', 'uint32', 4),
            ...(optional === 'polygons' ? {sourceIds: output('sourceIds', 'uint32', 3)} : {})
          }
        : undefined;
    const status = {
      count: output('count', 'uint32', 1),
      overflow: output('overflow', 'uint32', 1),
      ...(optional === 'requiredCount' || optional === 'touchingSegmentCount'
        ? {[optional]: output(optional, 'uint32', 1)}
        : {})
    };
    graph.add(
      new GPUCoverageDissolve({
        positions: column('input', 'float32x2', new Float32Array(coordinates)),
        ringOffsets: column('rings', 'uint32', new Uint32Array([0, 4, 8, 12])),
        polygonOffsets: column('polygons', 'uint32', new Uint32Array([0, 2, 3])),
        featureOffsets: column('features', 'uint32', new Uint32Array([0, 1, 2])),
        labels: column('labels', 'uint32', new Uint32Array([37, 91])),
        output: {
          positions: output('positions', 'float32x2', 15),
          ringOffsets: output('offsets', 'uint32', 4),
          status,
          ...(polygons ? {polygons} : {}),
          ...(optional === 'segmentRings' ||
          optional === 'segmentVertices' ||
          optional === 'segmentFlags'
            ? {[optional]: output(optional, 'uint32', 12)}
            : {})
        }
      })
    );
    try {
      const result = await fixture.runGPU();
      expect(result['count']).toEqual([3]);
      expect(result['overflow']).toEqual([0]);
      expect(result['offsets']).toEqual([0, 5, 10, 15]);
      expect(result['positions']).toEqual([
        ...coordinates.slice(0, 8),
        0,
        0,
        ...coordinates.slice(8, 16),
        1,
        1,
        ...coordinates.slice(16),
        10,
        0
      ]);
      if (polygons) {
        expect(result['featureOffsets']).toEqual([0, 1, 2, 2]);
        expect(result['polygonOffsets']).toEqual([0, 2, 3, 3]);
        expect(result['polygonRingOffsets']).toEqual([0, 4, 8, 12]);
        expect(result['polygonPositions'].slice(0, 24)).toEqual(coordinates);
        if (optional === 'polygons') expect(result['sourceIds']).toEqual([37, 91, 0xffffffff]);
      }
      if (optional === 'segmentRings')
        expect(result['segmentRings']).toEqual([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]);
      if (optional === 'segmentVertices')
        expect(result['segmentVertices']).toEqual([0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]);
      if (optional === 'segmentFlags') expect(result['segmentFlags']).toEqual(Array(12).fill(0));
      if (optional === 'requiredCount') expect(result['requiredCount']).toEqual([3]);
      if (optional === 'touchingSegmentCount') expect(result['touchingSegmentCount']).toEqual([0]);
    } finally {
      fixture.destroy();
    }
  }
});

it('counts the two incoming segments with each diagnostic alone and both absent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const optional of ['both', 'segmentFlags', 'touchingSegmentCount', undefined] as const) {
    const fixture = createDissolveTouchOracle(device, optional);
    try {
      const result = await fixture.runGPU();
      if (optional === 'both' || optional === 'touchingSegmentCount')
        expect(result['touchingSegmentCount']).toEqual([2]);
      if (optional === 'both' || optional === 'segmentFlags')
        expect(result['segmentFlags']).toEqual([0, 1, 0, 0, 0, 0, 0, 1]);
      expect(result['count']).toEqual([2]);
      expect(result['overflow']).toEqual([0]);
      expect(result['offsets']).toEqual([0, 5, 10]);
    } finally {
      fixture.destroy();
    }
  }
});
