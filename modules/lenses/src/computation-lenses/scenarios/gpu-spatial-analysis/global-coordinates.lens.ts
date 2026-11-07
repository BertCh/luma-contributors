// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerS2LensScenarios} from './define-s2-lens-scenarios';

registerS2LensScenarios(
  {
    dataset: {
      id: 'global-coordinate-illustrations/1',
      source:
        'Repository-authored analytic unit-sphere examples; synthetic, not observed geographic data.',
      licence: 'MIT'
    },
    loadBuilder: async () =>
      (await import('./global-coordinates-examples')).buildGlobalCoordinateScenario
  },
  [
    {
      id: 'gpu-spatial-analysis.s2-cell-hierarchy',
      element: 'GPUS2CellHierarchy',
      title: 'Exact children of the six S2 root faces'
    },
    {
      id: 'gpu-spatial-analysis.s2-cell-measures',
      element: 'GPUS2CellMeasures',
      title: 'Unit-sphere area and perimeter of the six S2 root faces'
    },
    {
      id: 'gpu-spatial-analysis.s2-cap-covering',
      element: 'GPUS2CapCovering',
      title: 'Exterior root-face covering of a small north-pole cap'
    },
    {
      id: 'gpu-spatial-analysis.spherical-convex-contains',
      element: 'GPUSphericalConvexContains',
      title: 'Interior, exterior and closed-boundary queries on an octant'
    },
    {
      id: 'gpu-spatial-analysis.spherical-convex-intersection',
      element: 'GPUSphericalConvexIntersection',
      title: 'Identical octants with intersection area and overlap fractions'
    },
    {
      id: 'gpu-spatial-analysis.s2-polygon-covering',
      element: 'GPUS2PolygonCovering',
      title: 'Adaptive covering of an octant'
    },
    {
      id: 'gpu-spatial-analysis.spherical-polygon-contains',
      element: 'GPUSphericalPolygonContains',
      title: 'Nonconvex winding containment with explicit boundaries'
    },
    {
      id: 'gpu-spatial-analysis.spherical-polygon-overlay',
      element: 'GPUSphericalPolygonOverlay',
      title: 'Chart polygon intersection with captured topology'
    },
    {
      id: 'gpu-spatial-analysis.spherical-arc-overlay',
      element: 'GPUSphericalArcOverlay',
      title: 'Intrinsic spherical intersection with anchored membership'
    }
  ]
);
