// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerS2LensScenarios} from './define-s2-lens-scenarios';

registerS2LensScenarios(
  {
    dataset: {
      id: 'urban-ecology-illustrative/1',
      source:
        'Repository-authored synthetic geometry in arbitrary planar units and explicit categorical/population fixtures.',
      licence: 'MIT (repository-authored illustrative inputs)'
    },
    loadBuilder: async () => (await import('./urban-ecology-examples')).buildUrbanEcologyScenario
  },
  [
    {
      id: 'gpu-spatial-analysis.street-profiles-recipe',
      element: 'addStreetProfilesRecipe',
      title: 'Side-specific street distances and height/width ratios'
    },
    {
      id: 'gpu-spatial-analysis.building-adjacency-recipe',
      element: 'addBuildingAdjacencyRecipe',
      title: 'Shared-wall length and axial orientation'
    },
    {
      id: 'gpu-spatial-analysis.isovists-recipe',
      element: 'addIsovistsRecipe',
      title: 'Eight-ray isovists inside a square enclosure'
    },
    {
      id: 'gpu-spatial-analysis.visibility-graph-recipe',
      element: 'addVisibilityGraphRecipe',
      title: 'Three-point visibility with bounded edge output'
    },
    {
      id: 'gpu-spatial-analysis.building-tessellation-recipe',
      element: 'addBuildingTessellationRecipe',
      title: 'Two boundary-sampled buildings inside a convex enclosure'
    },
    {
      id: 'gpu-spatial-analysis.landscape-metrics-recipe',
      element: 'addLandscapeMetricsRecipe',
      title: 'Two-class patch areas, edges and aggregation'
    },
    {
      id: 'gpu-spatial-analysis.patch-proximity-recipe',
      element: 'addPatchProximityRecipe',
      title: 'Two patches separated by background'
    },
    {
      id: 'gpu-spatial-analysis.resistance-connectivity-recipe',
      element: 'addResistanceConnectivityRecipe',
      title: 'Three-node conductance network'
    },
    {
      id: 'gpu-spatial-analysis.segregation-decomposition-recipe',
      element: 'addSegregationDecompositionRecipe',
      title: 'Four supplied aligned population counterfactuals'
    }
  ]
);
registerS2LensScenarios(
  {
    dataset: {
      id: 'urban-ecology-illustrative/2',
      source: 'Repository-authored synthetic conductance networks and aligned population counts.',
      licence: 'MIT (repository-authored illustrative inputs)'
    },
    loadBuilder: async () => (await import('./urban-ecology-examples')).buildUrbanEcologyScenario
  },
  [
    {
      id: 'gpu-spatial-analysis.segregation-counterfactual-recipe',
      element: 'addSegregationCounterfactualRecipe',
      title: 'Empirical-quantile population counterfactuals and Shapley components'
    },
    {
      id: 'gpu-spatial-analysis.population-support-alignment-recipe',
      element: 'addPopulationSupportAlignmentRecipe',
      title: 'Explicit population support identifiers reordered with completeness evidence'
    },
    {
      id: 'gpu-spatial-analysis.sparse-resistance-connectivity-recipe',
      element: 'addSparseResistanceConnectivityRecipe',
      title: 'Corner-to-corner current through a 3 x 3 conductance lattice'
    },
    {
      id: 'gpu-spatial-analysis.raster-resistance-connectivity-recipe',
      element: 'addRasterResistanceConnectivityRecipe',
      title: 'Current through the gap of a no-data wall on a resistance raster'
    }
  ]
);
