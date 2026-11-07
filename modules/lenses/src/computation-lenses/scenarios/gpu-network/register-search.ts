// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Registers the gpu-network search-family lens scenarios (one entry per scenario).

import {registerLensScenarios} from '../../define-lens-scenarios';

const dataset = {
  id: 'n-chicago-loop-roads',
  licence: 'ODbL 1.0',
  source: 'OpenStreetMap via BBBike Chicago extract'
};

registerLensScenarios(
  {
    entry: 'gpu-network',
    audience: 'analyst',
    defaultRung: 'result',
    dataset
  },
  [
    {
      id: 'gpu-network.network-service-areas',
      element: 'GPUNetworkServiceAreas',
      title: 'Chicago Loop service areas of four facilities',
      rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
      load: () => import('./network-service-areas.lens')
    },
    {
      id: 'gpu-network.network-path-extraction',
      element: 'GPUNetworkPathExtraction',
      title: 'Chicago Loop shortest path to the West Loop library',
      rungs: ['result', 'datum', 'kernel', 'graph'],
      load: () => import('./network-path-extraction.lens')
    },
    {
      id: 'gpu-network.network-neighborhood',
      element: 'GPUNetworkNeighborhood',
      title: 'Chicago Loop six-hop neighborhood',
      rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
      load: () => import('./network-neighborhood.lens')
    },
    {
      id: 'gpu-network.network-isochrones',
      element: 'GPUNetworkIsochrones',
      title: 'Chicago Loop travel-time bands',
      rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
      load: () => import('./network-isochrones.lens')
    },
    {
      id: 'gpu-network.network-snapping',
      element: 'GPUNetworkSnapping',
      title: 'Chicago Loop facilities snapped to the nearest road edge',
      rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
      load: () => import('./network-snapping.lens')
    },
    {
      id: 'gpu-network.network-cost-matrix',
      element: 'GPUNetworkCostMatrix',
      title: 'Chicago Loop travel times from four facilities',
      rungs: ['result', 'datum', 'kernel', 'graph'],
      load: () => import('./network-cost-matrix.lens')
    },
    {
      id: 'gpu-network.network-accessibility',
      element: 'GPUNetworkAccessibility',
      title: 'Chicago Loop gravity access to four facilities',
      rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
      load: () => import('./network-accessibility.lens')
    },
    {
      id: 'gpu-network.drive-time-catchment-recipe',
      element: 'addDriveTimeCatchmentRecipe',
      title: 'Chicago Loop drive-time catchment of four facilities',
      rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
      load: () => import('./drive-time-catchment-recipe.lens')
    }
  ]
);
