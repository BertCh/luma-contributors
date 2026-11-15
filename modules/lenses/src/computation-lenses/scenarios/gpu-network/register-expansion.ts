// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenarios} from '../../define-lens-scenarios';

registerLensScenarios(
  {
    entry: 'gpu-network',
    audience: 'analyst',
    rungs: ['result', 'datum', 'kernel', 'graph'],
    defaultRung: 'result'
  },
  [
    {
      id: 'gpu-network.network-structure',
      element: 'GPUNetworkStructure',
      title: 'Synthetic directed diamond: network-structure',
      dataset: {
        id: 'synthetic-four-node-directed-diamond-v1',
        source: 'Deterministic example constructed in expansion.lens.ts',
        licence: 'MIT'
      },
      load: async () => {
        const module = await import('./expansion.lens');
        return {
          run: device => module.runExpansion(device, 'network-structure'),
          describeStructure: device => module.describeExpansion(device, 'network-structure')
        };
      }
    },
    {
      id: 'gpu-network.k-shortest-paths',
      element: 'GPUKShortestPaths',
      title: 'Synthetic directed diamond: two certified Yen routes',
      dataset: {
        id: 'synthetic-four-node-directed-diamond-v1',
        source: 'Deterministic example constructed in expansion.lens.ts',
        licence: 'MIT'
      },
      load: async () => {
        const module = await import('./expansion.lens');
        return {
          run: device => module.runExpansion(device, 'k-shortest-paths'),
          describeStructure: device => module.describeExpansion(device, 'k-shortest-paths')
        };
      }
    },
    {
      id: 'gpu-network.alternative-routes',
      element: 'GPUAlternativeRoutes',
      title: 'Synthetic directed diamond: alternative-routes',
      dataset: {
        id: 'synthetic-four-node-directed-diamond-v1',
        source: 'Deterministic example constructed in expansion.lens.ts',
        licence: 'MIT'
      },
      load: async () => {
        const module = await import('./expansion.lens');
        return {
          run: device => module.runExpansion(device, 'alternative-routes'),
          describeStructure: device => module.describeExpansion(device, 'alternative-routes')
        };
      }
    },
    {
      id: 'gpu-network.network-disruption-recipe',
      element: 'addNetworkDisruptionRecipe',
      title: 'Synthetic directed diamond: network-disruption-recipe',
      dataset: {
        id: 'synthetic-four-node-directed-diamond-v1',
        source: 'Deterministic example constructed in expansion.lens.ts',
        licence: 'MIT'
      },
      load: async () => {
        const module = await import('./expansion.lens');
        return {
          run: device => module.runExpansion(device, 'network-disruption-recipe'),
          describeStructure: device => module.describeExpansion(device, 'network-disruption-recipe')
        };
      }
    },
    {
      id: 'gpu-network.facility-scenario-recipe',
      element: 'addFacilityScenarioRecipe',
      title: 'Synthetic directed diamond: facility-scenario-recipe',
      dataset: {
        id: 'synthetic-four-node-directed-diamond-v1',
        source: 'Deterministic example constructed in expansion.lens.ts',
        licence: 'MIT'
      },
      load: async () => {
        const module = await import('./expansion.lens');
        return {
          run: device => module.runExpansion(device, 'facility-scenario-recipe'),
          describeStructure: device => module.describeExpansion(device, 'facility-scenario-recipe')
        };
      }
    },
    {
      id: 'gpu-network.track-quality-recipe',
      element: 'addTrackQualityRecipe',
      title: 'Raw trajectory quality flags and metrics',
      dataset: {
        id: 'synthetic-four-observation-trajectory-v1',
        source: 'Deterministic projected trajectory in track-quality-recipe.lens.ts',
        licence: 'MIT'
      },
      load: () => import('./track-quality-recipe.lens')
    }
  ]
);
