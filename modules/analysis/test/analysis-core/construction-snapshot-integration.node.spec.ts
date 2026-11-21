// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {snapshotConstructionFields} from '../../src/analysis-core/kernels/construction-snapshot';
import {computePointsInPolygonsChoroplethOnCPU} from '@luma-contributors/analysis/gpu-spatial-analysis';

it('R2a captures own, inherited and non-enumerable declared fields exactly once', () => {
  for (const placement of ['own', 'inherited', 'non-enumerable'] as const) {
    let reads = 0;
    const descriptor = {
      get: () => (++reads === 1 ? 'sum' : 'count'),
      enumerable: placement !== 'non-enumerable'
    };
    const props =
      placement === 'inherited'
        ? Object.create(Object.defineProperty({}, 'statistic', descriptor))
        : Object.defineProperty({}, 'statistic', descriptor);
    expect(snapshotConstructionFields(props, ['statistic']).statistic).toBe('sum');
    expect(reads).toBe(1);
  }
});

it('R2a retains proxy get-only values and distinguishes absent from present undefined', () => {
  const reads: PropertyKey[] = [];
  const props = new Proxy<{statistic?: string; absent?: string; present?: string}>(
    {present: undefined},
    {
      get(target, field, receiver) {
        reads.push(field);
        return field === 'statistic' ? 'sum' : Reflect.get(target, field, receiver);
      }
    }
  );
  const snapshot = snapshotConstructionFields(props, ['statistic', 'absent', 'present']);
  expect(snapshot.statistic).toBe('sum');
  expect(Object.hasOwn(snapshot, 'absent')).toBe(false);
  expect(Object.hasOwn(snapshot, 'present')).toBe(true);
  expect(reads).toEqual(['statistic', 'absent', 'present']);
});

it.each(['zonal', 'group'] as const)('R2a public %s choropleth retains proxy sum', backend => {
  const options = new Proxy(
    {backend, candidateCapacity: 8},
    {
      get(target, field, receiver) {
        return field === 'statistic' ? 'sum' : Reflect.get(target, field, receiver);
      }
    }
  );
  const result = computePointsInPolygonsChoroplethOnCPU(
    {
      points: {kind: 'points', positions: Float32Array.of(0.5, 0.5)},
      polygons: {
        kind: 'polygons',
        positions: Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1),
        featureOffsets: Uint32Array.of(0, 1),
        polygonOffsets: Uint32Array.of(0, 1),
        ringOffsets: Uint32Array.of(0, 4)
      },
      values: Float32Array.of(7)
    },
    options
  );
  // One interior match contributes its value 7, rather than its count 1.
  expect(Array.from(result.featureValues)).toEqual([7]);
});
