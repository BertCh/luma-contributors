// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import * as arrow from 'apache-arrow';
import {prepareKeplerGeometrySourceFromArrow} from '../../../../examples/deck/kepler-compute/src/geometry-preparation/geometry-gpu-vectors';

/** 3-wide interleaved coordinates whose child field keeps the legacy name 'value' (no dimension). */
function makeLegacyPointVector(): arrow.Vector {
  const type = new arrow.FixedSizeList(3, new arrow.Field('value', new arrow.Float64(), false));
  const values = arrow.makeData({
    type: new arrow.Float64(),
    data: new Float64Array([1, 2, 3, 4, 5, 6])
  });
  return new arrow.Vector([arrow.makeData({type, length: 2, child: values})]);
}

/** Polygon with legacy-named XYZ ring coordinates. */
function makeLegacyPolygonVector(): arrow.Vector {
  const coordinates = [0, 0, 1, 4, 0, 2, 4, 4, 3, 0, 4, 4, 0, 0, 1];
  const coordinateType = new arrow.FixedSizeList(
    3,
    new arrow.Field('value', new arrow.Float64(), false)
  );
  const ringType = new arrow.List(new arrow.Field('rings', coordinateType, false));
  const polygonType = new arrow.List(new arrow.Field('polygons', ringType, false));
  const values = arrow.makeData({type: new arrow.Float64(), data: new Float64Array(coordinates)});
  const points = arrow.makeData({type: coordinateType, length: 5, child: values});
  const rings = arrow.makeData({
    type: ringType,
    length: 1,
    valueOffsets: new Int32Array([0, 5]),
    child: points
  });
  return new arrow.Vector([
    arrow.makeData({
      type: polygonType,
      length: 1,
      valueOffsets: new Int32Array([0, 1]),
      child: rings
    })
  ]);
}

describe('legacy unnamed XYZ coordinates', () => {
  it('prepares points on the default and generic routes', () => {
    const fast = prepareKeplerGeometrySourceFromArrow(makeLegacyPointVector(), {
      encoding: 'geoarrow.point'
    });
    const generic = prepareKeplerGeometrySourceFromArrow(makeLegacyPointVector(), {
      encoding: 'geoarrow.point',
      nativeFastPath: false
    });
    expect(fast.metadata.rowCount).toBe(2);
    expect(generic.metadata.rowCount).toBe(2);
  });

  it('honours an explicit dimension on the generic route', () => {
    const source = prepareKeplerGeometrySourceFromArrow(makeLegacyPointVector(), {
      encoding: 'geoarrow.point',
      dimension: 'xyz',
      nativeFastPath: false
    });
    expect(source.metadata.rowCount).toBe(2);
  });

  it('prepares polygons on the generic route and with globe preparation', () => {
    const generic = prepareKeplerGeometrySourceFromArrow(makeLegacyPolygonVector(), {
      encoding: 'geoarrow.polygon',
      nativeFastPath: false
    });
    const globe = prepareKeplerGeometrySourceFromArrow(makeLegacyPolygonVector(), {
      encoding: 'geoarrow.polygon',
      globe: {}
    });
    expect(generic.metadata.rowCount).toBe(1);
    expect(globe.metadata.rowCount).toBe(1);
  });
});
