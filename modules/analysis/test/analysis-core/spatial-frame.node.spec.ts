// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {projectionEngine} from '@math.gl/geospatial/projection';
import {
  GPU_LOCAL_FRAME_KINDS,
  createGPULocalSpatialFrame,
  getGPUSpatialFrameContextUnits,
  getGPUSpatialFrameKind
} from '../../src/analysis-core/contracts/spatial-frame';
import {createGPUSpatialFrameFromCRS} from '../../src/analysis-core/contracts/spatial-frame-crs';
import {createGPULocalFrame} from '../../src/analysis-core/contracts/local-frame';
import {getGPUSpatialContextCompatibility} from '../../src/analysis-core/contracts/spatial-context';
import {getGPUProjectionSpatialContext} from '../../src/geospatial-analysis/projection-spatial-context';

describe('GPUSpatialFrame facade', () => {
  it.each([
    ['EPSG:4326', 'degrees', undefined],
    ['EPSG:3857', 'meters', 'meters'],
    ['EPSG:32633', 'meters', 'meters'],
    ['EPSG:32733', 'meters', 'meters'],
    ['+proj=utm +zone=33 +units=us-ft +datum=NAD83', 'us-survey-feet', undefined],
    ['+proj=lcc +lat_1=40 +lat_2=60 +units=m', 'meters', 'meters'],
    ['+proj=utm +zone=33 +units=km', 'kilometers', 'kilometers']
  ])('resolves the unit of %s', (crs, unitName, contextUnits) => {
    const frame = createGPUSpatialFrameFromCRS(crs);
    expect(frame.kind).toBe('crs');
    expect(frame.unit?.name).toBe(unitName);
    expect(getGPUSpatialFrameContextUnits(frame)).toBe(contextUnits);
  });

  it('leaves the unit of unknown registry codes unresolved', () => {
    const frame = createGPUSpatialFrameFromCRS('EPSG:2263');
    expect(frame).toEqual({id: 'EPSG:2263', kind: 'crs', crs: 'EPSG:2263'});
  });

  it('normalizes equivalent identifier spellings to one id', () => {
    const ids = ['EPSG:3857', 'epsg:3857', 'urn:ogc:def:crs:EPSG::3857'].map(
      crs => createGPUSpatialFrameFromCRS(crs).id
    );
    expect(ids).toEqual(['EPSG:3857', 'EPSG:3857', 'EPSG:3857']);
    const a = getGPUProjectionSpatialContext(
      'epsg:3857',
      createGPUSpatialFrameFromCRS('epsg:3857')
    )!;
    const b = getGPUProjectionSpatialContext('EPSG:3857')!;
    expect(getGPUSpatialContextCompatibility(a, b).compatible).toBe(true);
    expect(a.frame?.crs).toBe('epsg:3857');
  });

  it('fills metadata from resolved math.gl descriptors', async () => {
    const metadata = await projectionEngine.getCRS('+proj=utm +zone=33 +units=us-ft +datum=NAD83');
    const frame = createGPUSpatialFrameFromCRS('+proj=utm +zone=33 +units=us-ft +datum=NAD83', {
      metadata,
      precision: 'float64-split'
    });
    expect(frame.signature).toBe(metadata.signature);
    expect(frame.unit?.metersPerUnit).toBeCloseTo(0.3048006, 6);
    expect(frame.axisOrder).toBe('enu');
    expect(frame.precision).toBe('float64-split');
    expect(getGPUSpatialFrameContextUnits(frame)).toBeUndefined();

    const mercator = createGPUSpatialFrameFromCRS('EPSG:3857', {
      metadata: await projectionEngine.getCRS('EPSG:3857')
    });
    expect(mercator.id).toBe('EPSG:3857');
    expect(mercator.signature).toBe('EPSG:3857');
    expect(getGPUSpatialFrameContextUnits(mercator)).toBe('meters');
  });

  it('declares meters on projection contexts when the unit is known', () => {
    expect(getGPUProjectionSpatialContext('EPSG:32633')?.units).toBe('meters');
    expect(getGPUProjectionSpatialContext('EPSG:3857')?.units).toBe('meters');
    expect(getGPUProjectionSpatialContext('EPSG:2263')?.units).toBeUndefined();
  });

  it('maps local frame kinds in one place', () => {
    expect(getGPUSpatialFrameKind('azimuthal-equidistant')).toBe('local-azimuthal-equidistant');
    expect(Object.keys(GPU_LOCAL_FRAME_KINDS)).toEqual([
      'azimuthal-equidistant',
      'lambert-equal-area'
    ]);
  });

  it('describes local frames and keeps legacy ids', () => {
    const frame = createGPULocalSpatialFrame({kind: 'lambert-equal-area', origin: [10, 20]});
    expect(frame).toMatchObject({
      id: 'local-lambert-equal-area(10,20)',
      kind: 'local-lambert-equal-area',
      origin: [10, 20],
      anchor: 'projection-center',
      precision: 'float32-origin-relative',
      axisOrder: 'enu'
    });
    expect(frame.unit?.metersPerUnit).toBe(1);

    const local = createGPULocalFrame({origin: [10, 20], sphereRadius: 6000000});
    expect(local.spatialContext.frame?.id).toBe('local-azimuthal-equidistant(10,20,R=6000000)');
    expect(local.spatialContext.units).toBe('meters');
    expect(createGPULocalFrame({origin: [10, 20], id: 'mine'}).spatialContext.frame?.id).toBe(
      'mine'
    );
  });
});
