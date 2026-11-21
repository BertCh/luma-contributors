// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getCatalogBinding, runCatalogOperation} from '../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {getThinSeamTriangleRing, SEAM_S2_COVER_PARAMETERS} from './s2-catalog-cover-fixtures';

it('recheck-status-spatial-b-R1 CPU catalog thin seam exterior excludes antipodal faces', () => {
  const polygons: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      {geometry: {type: 'Polygon', coordinates: [getThinSeamTriangleRing()]}, properties: {}}
    ]
  };
  const result = runCatalogOperation(
    getCatalogBinding('s2-polygon-covering')!,
    SEAM_S2_COVER_PARAMETERS,
    {polygons}
  );
  // The short equatorial edge straddles the positive x=y seam; the third vertex
  // stays north of that seam. Neither antipodal face 3 nor 4 touches this triangle.
  expect(result.tables[0].columns['cell_id']).toEqual(['1000000000000000', '3000000000000000']);
  expect(result.summary['cellCount']).toBe(2);
  expect(result.summary['truncatedCount']).toBe(0);
  expect(result.diagnostics).toEqual([]);
});

it.each([
  'closed',
  'open'
] as const)('spherical catalog excludes the short-edge antipode (%s)', boundary => {
  const halfAngleDegrees = (0.75e-6 * 180) / Math.PI;
  const points: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [180, 0].map(longitude => ({
      geometry: {type: 'Point', coordinates: [longitude, 0]},
      properties: {}
    }))
  };
  const polygons: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      {
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [-halfAngleDegrees, 0],
              [halfAngleDegrees, 0],
              [0, 90],
              [-halfAngleDegrees, 0]
            ]
          ]
        },
        properties: {}
      }
    ]
  };
  // The finite equatorial edge has midpoint +x. Its antipode is at least pi/2
  // from the triangle, far beyond the binding's 1e-6-radian boundary tolerance.
  const result = runCatalogOperation(
    getCatalogBinding('spherical-point-in-polygon')!,
    {boundary},
    {points, polygons}
  );
  expect(result.summary['containedCount']).toBe(boundary === 'closed' ? 1 : 0);
  expect(result.tables[0].columns['point_feature_index']).toEqual(boundary === 'closed' ? [1] : []);
  expect(result.tables[0].columns['on_boundary']).toEqual(boundary === 'closed' ? [true] : []);
  expect(result.diagnostics).toEqual([]);
});

it('rejects a closed zero-area equatorial ring for containment', () => {
  // All vertices lie on the equator: their central-projection area is exactly zero.
  const result = runCatalogOperation(
    getCatalogBinding('spherical-point-in-polygon')!,
    {boundary: 'closed'},
    {
      points: {
        type: 'FeatureCollection',
        features: [{geometry: {type: 'Point', coordinates: [2, 0]}, properties: {}}]
      },
      polygons: {
        type: 'FeatureCollection',
        features: [
          {
            geometry: {
              type: 'Polygon',
              coordinates: [
                [
                  [0, 0],
                  [2, 0],
                  [4, 0],
                  [0, 0]
                ]
              ]
            },
            properties: {}
          }
        ]
      }
    }
  );
  expect(result.summary['containedCount']).toBe(0);
  expect(
    result.diagnostics.some(diagnostic => diagnostic.code === 'invalid-spherical-polygon')
  ).toBe(true);
});
