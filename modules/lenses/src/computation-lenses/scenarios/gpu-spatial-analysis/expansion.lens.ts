// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerS2LensScenarios} from './define-s2-lens-scenarios';

registerS2LensScenarios(
  {
    dataset: {
      id: 'geospatial-expansion-illustrative/1',
      source:
        'Deterministic synthetic examples in arbitrary planar units; H3 examples use a valid resolution-nine cell.',
      licence: 'MIT (repository-authored illustrative inputs)'
    },
    loadBuilder: async () => (await import('./expansion-examples')).buildExpansionScenario
  },
  [
    {
      id: 'gpu-spatial-analysis.local-spatial-heteroscedasticity',
      element: 'GPULocalSpatialHeteroscedasticity',
      title: 'GPULocalSpatialHeteroscedasticity: explicit illustrative inputs'
    },
    {
      id: 'gpu-spatial-analysis.multivariate-local-geary',
      element: 'GPUMultivariateLocalGeary',
      title: 'GPUMultivariateLocalGeary: explicit illustrative inputs'
    },
    {
      id: 'gpu-spatial-analysis.generalized-geographically-weighted-regression',
      element: 'GPUGeneralizedGeographicallyWeightedRegression',
      title: 'GPUGeneralizedGeographicallyWeightedRegression: explicit illustrative inputs'
    },
    {
      id: 'gpu-spatial-analysis.multiscale-geographically-weighted-regression',
      element: 'GPUMultiscaleGeographicallyWeightedRegression',
      choices: {
        'bandwidthSelection.criterion': 'disabled',
        'bandwidthSelection.searchMethod': 'disabled'
      },
      title: 'GPUMultiscaleGeographicallyWeightedRegression: explicit illustrative inputs'
    },
    {
      id: 'gpu-spatial-analysis.spatial-panel-regression',
      element: 'GPUSpatialPanelRegression',
      title: 'GPUSpatialPanelRegression: explicit illustrative inputs'
    },
    {
      id: 'gpu-spatial-analysis.spatial-regime-regression',
      element: 'GPUSpatialRegimeRegression',
      title: 'GPUSpatialRegimeRegression: explicit illustrative inputs'
    },
    {
      id: 'gpu-spatial-analysis.concave-hull',
      element: 'GPUConcaveHull',
      title: 'GPUConcaveHull: explicit illustrative inputs'
    },
    {
      id: 'gpu-spatial-analysis.group-concave-hull',
      element: 'GPUGroupConcaveHull',
      title: 'GPUGroupConcaveHull: explicit illustrative inputs'
    },
    {
      id: 'gpu-spatial-analysis.constrained-delaunay-triangulation',
      element: 'GPUConstrainedDelaunayTriangulation',
      title: 'GPUConstrainedDelaunayTriangulation: explicit illustrative inputs'
    },
    {
      id: 'gpu-spatial-analysis.h3-cell-identities',
      element: 'GPUH3CellIdentities',
      title: 'GPUH3CellIdentities: explicit illustrative inputs'
    },
    {
      id: 'gpu-spatial-analysis.bounded-facility-allocation',
      element: 'GPUBoundedFacilityAllocation',
      title: 'GPUBoundedFacilityAllocation: explicit illustrative inputs'
    },
    {
      id: 'gpu-spatial-analysis.zonal-distribution-recipe',
      element: 'addZonalDistributionRecipe',
      title: 'addZonalDistributionRecipe: explicit illustrative inputs'
    },
    {
      id: 'gpu-spatial-analysis.raster-zonal-distribution-recipe',
      element: 'addRasterZonalDistributionRecipe',
      title: 'addRasterZonalDistributionRecipe: explicit illustrative inputs'
    },
    {
      id: 'gpu-spatial-analysis.exposure-by-zone-recipe',
      element: 'addExposureByZoneRecipe',
      title: 'addExposureByZoneRecipe: explicit illustrative inputs'
    },
    {
      id: 'gpu-spatial-analysis.regional-raster-change-recipe',
      element: 'addRegionalRasterChangeRecipe',
      title: 'addRegionalRasterChangeRecipe: explicit illustrative inputs'
    },
    {
      id: 'gpu-spatial-analysis.surface-preparation-recipe',
      element: 'addSurfacePreparationRecipe',
      title: 'addSurfacePreparationRecipe: explicit illustrative inputs'
    },
    {
      id: 'gpu-spatial-analysis.interpolation-diagnostics-recipe',
      element: 'addInterpolationDiagnosticsRecipe',
      title: 'addInterpolationDiagnosticsRecipe: explicit illustrative inputs'
    },
    {
      id: 'gpu-spatial-analysis.drainage-network-recipe',
      element: 'addDrainageNetworkRecipe',
      title: 'addDrainageNetworkRecipe: explicit illustrative inputs'
    },
    {
      id: 'gpu-spatial-analysis.local-variability-recipe',
      element: 'addLocalVariabilityRecipe',
      title: 'addLocalVariabilityRecipe: explicit illustrative inputs'
    },
    {
      id: 'gpu-spatial-analysis.cell-flow-topology-recipe',
      element: 'addCellFlowTopologyRecipe',
      title: 'addCellFlowTopologyRecipe: explicit illustrative inputs'
    },
    {
      id: 'gpu-spatial-analysis.losh-residual-permutation-test',
      element: 'GPULOSHResidualPermutationTest',
      title: 'GPULOSHResidualPermutationTest: explicit illustrative inputs'
    }
  ]
);
