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
      choices: {maximumDistance: 10, sampleSpacing: 1},
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
      choices: {rayCount: 8, maximumDistance: 10},
      title: 'Eight-ray isovists inside a square enclosure'
    },
    {
      id: 'gpu-spatial-analysis.visibility-graph-recipe',
      element: 'addVisibilityGraphRecipe',
      choices: {maximumDistance: 1.1, edgeCapacity: 1},
      title: 'Three-point visibility with bounded edge output'
    },
    {
      id: 'gpu-spatial-analysis.building-tessellation-recipe',
      element: 'addBuildingTessellationRecipe',
      choices: {sampleSpacing: 1, vertexTolerance: 1 / 1024},
      title: 'Two boundary-sampled buildings inside a convex enclosure'
    },
    {
      id: 'gpu-spatial-analysis.landscape-metrics-recipe',
      element: 'addLandscapeMetricsRecipe',
      choices: {connectivity: 4},
      title: 'Two-class patch areas, edges and aggregation'
    },
    {
      id: 'gpu-spatial-analysis.patch-proximity-recipe',
      element: 'addPatchProximityRecipe',
      choices: {edgeDepth: 0.4, searchRadius: 2},
      title: 'Two patches separated by background'
    },
    {
      id: 'gpu-spatial-analysis.resistance-connectivity-recipe',
      element: 'addResistanceConnectivityRecipe',
      choices: {tolerance: 1e-5, maximumIterations: 256},
      title: 'Three-node conductance network'
    },
    {
      id: 'gpu-spatial-analysis.segregation-decomposition-recipe',
      element: 'addSegregationDecompositionRecipe',
      choices: {counterfactualMethod: 'explicit-supplied-counterfactuals'},
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
      id: 'gpu-spatial-analysis.population-support-alignment',
      element: 'GPUPopulationSupportAlignment',
      title: 'Support-ID alignment with missing, duplicate and excluded rows'
    },
    {
      id: 'gpu-spatial-analysis.sparse-resistance-connectivity-recipe',
      element: 'addSparseResistanceConnectivityRecipe',
      choices: {tolerance: 1e-5, maximumIterations: 64, connectivityIterations: 128},
      title: 'Corner-to-corner current through a 3 x 3 conductance lattice'
    },
    {
      id: 'gpu-spatial-analysis.raster-resistance-connectivity-recipe',
      element: 'addRasterResistanceConnectivityRecipe',
      choices: {neighborhood: 8, cellSizeMode: 'uniform', tolerance: 1e-5, maximumIterations: 64},
      title: 'Current through the gap of a no-data wall on a resistance raster'
    }
  ]
);
