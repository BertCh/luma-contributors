// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogGeometry} from '../../src/analysis/catalog-geojson';

/** Shared by Node preparation/triangulation checks and genuine WebGPU publication checks. */
export const DASYMETRIC_GEOMETRY_CASES = {
  // GPUPolygonTriangulation explicitly completes collapsed shells with zero triangles.
  collapsedShell: {
    type: 'Polygon',
    coordinates: [
      [
        [0, 0],
        [1, 0],
        [2, 0],
        [0, 0]
      ]
    ]
  },
  // The hole's leftward bridge ray cannot hit a shell entirely to its right.
  unbridgeableHole: {
    type: 'Polygon',
    coordinates: [
      [
        [0.5, 0],
        [2.5, 0],
        [2.5, 1],
        [0.5, 1],
        [0.5, 0]
      ],
      [
        [-1, 0.25],
        [-1, 0.75],
        [-0.5, 0.75],
        [-0.5, 0.25],
        [-1, 0.25]
      ]
    ]
  }
} satisfies Record<string, CatalogGeometry>;
