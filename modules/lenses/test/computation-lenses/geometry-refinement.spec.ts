// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {getLensScenario} from '../../src/computation-lenses/registry';
import {GEOMETRY_REFINEMENT_SCENARIOS} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/geometry-refinement.lens';
import {requireTestDevice} from '@luma-contributors/graph/testing';

it.each(
  GEOMETRY_REFINEMENT_SCENARIOS
)('$element captures an executed graph and meaningful output', async ({element, slug}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scenario = getLensScenario(`gpu-spatial-analysis.${slug}`);
  expect(scenario).toBeDefined();
  const result = await (await scenario!.load()).run(device);
  expect(result.graphs.length).toBeGreaterThan(0);
  const snapshots = result.captures?.snapshots ?? [];
  expect(snapshots.some(snapshot => snapshot.role === 'input')).toBe(true);
  expect(snapshots.some(snapshot => snapshot.role === 'output')).toBe(true);
  const values = (name: string) =>
    Array.from(snapshots.find(snapshot => snapshot.name === name)?.values ?? [], Number);
  switch (element) {
    case 'GPUNormalizeGeometry':
      expect(values('status')).toEqual([0, 0, 0]);
      expect(values('normalized-positions')).toEqual([0, 0, 4, 0, 2, 2]);
      expect(values('normalized-children')).toEqual([0, 1]);
      break;
    case 'GPUCanonicalGeometryEquality':
      expect(values('equality')).toEqual([1]);
      break;
    case 'GPUVectorStraightSkeleton':
      expect(values('status')[3]).toBe(0);
      expect(values('status')[0]).toBeGreaterThan(8);
      expect(values('status')[2]).toBe(8);
      break;
    case 'GPUOffsetCurveComponents':
      expect(values('status').slice(4)).toEqual([0, 0, 0, 0]);
      expect(values('status')[0]).toBe(1);
      expect(values('components')[2]).toBe(0);
      break;
    case 'GPUNormalizePolygons':
      expect(values('normalized-positions')).toEqual([0, 0, 0, 2, 4, 2, 4, 0]);
      break;
    case 'GPUCanonicalPolygonEquality':
      expect(values('equality')).toEqual([1]);
      break;
    case 'GPUSubdividePolygons':
      expect(values('status')).toEqual([1, 1, 0, 0]);
      break;
    case 'GPUPolygonPrecisionReduction':
      expect(values('result-overflow')).toEqual([0]);
      expect(values('closed-ring-count')[0]).toBeGreaterThan(0);
      expect(values('lattice')[2]).toBe(1);
      break;
    case 'GPULinePrecisionReduction':
      expect(values('status')).toEqual([2]);
      expect(values('rounded-offsets')).toEqual([0, 2]);
      break;
    case 'GPUSegmentSnap':
      expect(values('counts')).toEqual([4, 4, 0, 0]);
      expect(values('snapped-positions').slice(0, 8)).toEqual([
        0,
        0,
        2,
        0,
        4,
        Math.fround(0.1),
        6,
        0
      ]);
      break;
    case 'GPULineLoopRemoval':
      expect(values('status')).toEqual([3, 3, 0, 0]);
      expect(values('erased-positions').slice(0, 6)).toEqual([0, 0, 2, 2, 4, 0]);
      break;
    case 'GPURatioConcaveHull':
      expect(values('converged')).toEqual([1]);
      expect(values('edge-validity').reduce((sum, value) => sum + value, 0)).toBe(5);
      break;
    case 'GPUMaximumInscribedCircle':
      expect(values('circles')[2]).toBeCloseTo(1, 4);
      expect([0, 1]).toContain(values('status')[0]);
      break;
    case 'GPUConvexStraightSkeleton':
      expect(values('status')).toEqual([5, 5, 4, 0]);
      break;
  }
});
