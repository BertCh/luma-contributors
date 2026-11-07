// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {getGPUProjectionSpatialContext} from '../../src/geospatial-analysis/projection-spatial-context';
import {
  getGPUSpatialContextCompatibility,
  resolveGPUSpatialContext
} from '../../src/analysis-core/contracts/spatial-context';

describe('GPUProjection output spatial context', () => {
  // Units are declared when resolvable without a CRS provider; EPSG:2263 (US feet) is not.
  it.each([
    ['EPSG:32633', 'meters'],
    ['EPSG:3857', 'meters'],
    ['EPSG:2263', undefined]
  ])('describes target %s as a crs frame', (crs, units) => {
    const context = getGPUProjectionSpatialContext(crs);
    expect(context).toMatchObject({
      coordinateSpace: 'planar',
      metric: 'native',
      frame: {kind: 'crs', id: crs, crs}
    });
    expect(context?.units).toBe(units);
    expect(
      resolveGPUSpatialContext('op', context, {coordinateSpaces: ['planar'], measure: 'distance'})
    ).toBe(context);
    expect(
      resolveGPUSpatialContext('op', context, {coordinateSpaces: ['planar'], measure: 'area'})
    ).toBe(context);
  });

  it('keeps rows of different target CRSs apart', () => {
    const utm = getGPUProjectionSpatialContext('EPSG:32633')!;
    const mercator = getGPUProjectionSpatialContext('EPSG:3857')!;
    expect(getGPUSpatialContextCompatibility(utm, mercator).compatible).toBe(false);
    expect(getGPUSpatialContextCompatibility(utm, utm).compatible).toBe(true);
  });

  it.each([undefined, ''])('has no context without a target (%s)', targetCrs => {
    expect(getGPUProjectionSpatialContext(targetCrs)).toBeUndefined();
  });
});
