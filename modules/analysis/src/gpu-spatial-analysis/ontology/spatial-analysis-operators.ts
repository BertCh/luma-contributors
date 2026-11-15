// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_SPATIAL_ANALYSIS_CAPABILITIES} from './spatial-analysis-catalog';
import {KEPLER_ANALYSIS_OPERATOR_OVERRIDES} from './kepler-analysis-catalog';
import {COVERAGE_EXPANSION_OPERATOR_OVERRIDES} from './coverage-expansion-catalog';
import {OPERATOR_ANNOTATIONS} from './all-entry-points/operator-annotations';
import {SPATIOTEMPORAL_OPERATOR_NAMES} from './spatiotemporal-review-catalog';
import {NESTED_VALIDATION_OPERATOR_OVERRIDE} from '../spatial-validation/spatial-validation-descriptors';
import type {
  GPUSpatialAnalysisAudit,
  GPUSpatialAnalysisCapability,
  GPUSpatialAnalysisDataKind,
  GPUSpatialAnalysisOperator,
  GPUSpatialAnalysisOperatorQuery
} from './spatial-analysis-ontology';

const DOCUMENTATION = 'docs/api/gpu-spatial-analysis/operators.md';
const TEST_ROOT = 'modules/analysis/test/gpu-spatial-analysis';

type OperatorOverride = Partial<
  Pick<
    GPUSpatialAnalysisOperator,
    | 'package'
    | 'tags'
    | 'inputs'
    | 'outputs'
    | 'coordinateModels'
    | 'dimensionality'
    | 'topology'
    | 'cardinality'
    | 'semantics'
    | 'deterministic'
    | 'status'
    | 'readback'
  >
> & {contract?: string; evidence?: GPUSpatialAnalysisOperator['evidence']['status']};

type CompleteOperatorOverride = OperatorOverride & {tests?: string; documentation?: string};

const BOUNDED_STATUS = ['count', 'requiredCount', 'overflow'] as const;

/**
 * Material operator-level differences that cannot be inferred from a capability family.
 *
 * Everything else is deliberately inherited from its owning family. Keeping this table beside
 * the catalogue makes additions reviewable without importing any contributor constructor.
 */
const OPERATOR_OVERRIDES: Readonly<Record<string, CompleteOperatorOverride>> = {
  GPUHierarchicalDensityClustering: {
    inputs: ['points', 'attributes', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['planar'],
    dimensionality: '2d+time',
    cardinality: 'source-sized',
    semantics: 'exact',
    deterministic: true,
    status: ['count'],
    documentation: 'docs/api/gpu-spatial-analysis/space-time-clustering.md',
    tests: `${TEST_ROOT}/spatial-clustering/space-time-hdbscan.spec.ts`,
    contract: 'GPUHierarchicalDensityClusteringProps'
  },
  ...KEPLER_ANALYSIS_OPERATOR_OVERRIDES,
  ...COVERAGE_EXPANSION_OPERATOR_OVERRIDES,
  GPUGeodesicPairs: {
    coordinateModels: ['spherical', 'ellipsoidal'],
    dimensionality: '2d',
    topology: 'fixed',
    semantics: 'approximate',
    deterministic: true,
    status: ['converged', 'approximation', 'iterationLimitReached'],
    evidence: 'partial',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/geometry-measures/cr01-geodesic-envelopes.spec.ts',
    documentation: 'modules/analysis/docs/geodesic-precision.md',
    contract: 'GPUGeodesicPairsProps'
  },
  GPUGeodesicDestination: {
    coordinateModels: ['spherical', 'ellipsoidal'],
    dimensionality: '2d',
    topology: 'fixed',
    semantics: 'approximate',
    deterministic: true,
    status: ['converged', 'iterationLimitReached', 'approximation'],
    evidence: 'partial',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/geometry-measures/cr01-geodesic-envelopes.spec.ts',
    documentation: 'modules/analysis/docs/geodesic-precision.md',
    contract: 'GPUGeodesicDestinationProps'
  },
  GPUNestedInterpolationValidation:
    NESTED_VALIDATION_OPERATOR_OVERRIDE.GPUNestedInterpolationValidation,
  addDasymetricRedistributionRecipe: {
    inputs: ['polygons', 'surface', 'attributes'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['planar', 'grid'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'mixed',
    semantics: 'approximate',
    deterministic: true,
    status: ['invalidCount', 'overflow'],
    evidence: 'partial',
    tests: 'modules/analysis/test/gpu-spatial-analysis/recipes/dasymetric-redistribution.spec.ts',
    documentation: 'docs/api/gpu-spatial-analysis/dasymetric-redistribution.md',
    contract: 'GPUDasymetricRedistributionRecipeProps'
  },
  GPULocalSpatialHeteroscedasticity: {
    inputs: ['attributes', 'spatial-weights'],
    outputs: ['attributes'],
    coordinateModels: ['topological'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'approximate',
    deterministic: true,
    status: ['invalidCount'],
    evidence: 'partial',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/spatial-regression/geospatial-expansion.spec.ts',
    documentation: 'docs/api/gpu-spatial-analysis/statistical-expansion.md',
    contract: 'GPULocalSpatialHeteroscedasticityProps'
  },
  GPULOSHResidualPermutationTest: {
    inputs: ['attributes', 'spatial-weights'],
    outputs: ['attributes'],
    coordinateModels: ['topological'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'approximate',
    deterministic: true,
    status: ['overflow'],
    evidence: 'partial',
    tests: 'modules/analysis/test/geospatial-expansion/losh-backlog-inference.spec.ts',
    documentation: 'docs/api/gpu-spatial-analysis/losh-residual-permutation.md',
    contract: 'GPULOSHResidualPermutationTestProps'
  },
  GPUMultivariateLocalGeary: {
    inputs: ['attributes', 'spatial-weights'],
    outputs: ['attributes'],
    coordinateModels: ['topological'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'approximate',
    deterministic: true,
    status: ['invalidCount'],
    evidence: 'partial',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/spatial-regression/geospatial-expansion.spec.ts',
    documentation: 'docs/api/gpu-spatial-analysis/statistical-expansion.md',
    contract: 'GPUMultivariateLocalGearyProps'
  },
  GPUGeneralizedGeographicallyWeightedRegression: {
    inputs: ['points', 'attributes', 'spatial-weights', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['planar', 'topological'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'model-based',
    deterministic: true,
    status: ['converged', 'iterationLimitReached'],
    evidence: 'partial',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/spatial-regression/generalized-gwr-selection.spec.ts',
    documentation: 'docs/api/gpu-spatial-analysis/adaptive-generalized-gwr.md',
    contract: 'GPUGeneralizedGeographicallyWeightedRegressionProps'
  },
  GPUMultiscaleGeographicallyWeightedRegression: {
    inputs: ['points', 'attributes', 'spatial-weights', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['planar', 'topological'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'model-based',
    deterministic: true,
    status: ['converged'],
    evidence: 'partial',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/spatial-regression/geospatial-expansion.spec.ts',
    documentation: 'docs/api/gpu-spatial-analysis/statistical-expansion.md',
    contract: 'GPUMultiscaleGeographicallyWeightedRegressionProps'
  },
  GPUSpatialPanelRegression: {
    inputs: ['points', 'attributes', 'spatial-weights', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['planar', 'topological'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'model-based',
    deterministic: true,
    status: ['converged'],
    evidence: 'partial',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/spatial-regression/geospatial-expansion.spec.ts',
    documentation: 'docs/api/gpu-spatial-analysis/statistical-expansion.md',
    contract: 'GPUSpatialPanelRegressionProps'
  },
  GPUSpatialRegimeRegression: {
    inputs: ['points', 'attributes', 'spatial-weights', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['planar', 'topological'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'model-based',
    deterministic: true,
    status: ['converged'],
    evidence: 'partial',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/spatial-regression/geospatial-expansion.spec.ts',
    documentation: 'docs/api/gpu-spatial-analysis/statistical-expansion.md',
    contract: 'GPUSpatialRegimeRegressionProps'
  },
  GPUConcaveHull: {
    inputs: ['points', 'polygons', 'attributes'],
    outputs: ['lines', 'render-geometry', 'attributes'],
    coordinateModels: ['planar'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'capacity-bounded',
    semantics: 'approximate',
    deterministic: true,
    status: ['converged'],
    evidence: 'partial',
    tests: 'modules/analysis/test/geospatial-expansion/raster-terrain-expansion.spec.ts',
    documentation: 'docs/api/gpu-raster/expansion.md',
    contract: 'GPUConcaveHullProps'
  },
  GPUGroupConcaveHull: {
    inputs: ['points', 'polygons', 'attributes'],
    outputs: ['lines', 'render-geometry', 'attributes'],
    coordinateModels: ['planar'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'capacity-bounded',
    semantics: 'approximate',
    deterministic: true,
    status: ['converged'],
    evidence: 'partial',
    tests: 'modules/analysis/test/geospatial-expansion/raster-terrain-expansion.spec.ts',
    documentation: 'docs/api/gpu-raster/expansion.md',
    contract: 'GPUGroupConcaveHullProps'
  },
  GPUConstrainedDelaunayTriangulation: {
    inputs: ['points', 'polygons', 'attributes'],
    outputs: ['lines', 'render-geometry', 'attributes'],
    coordinateModels: ['planar'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'capacity-bounded',
    semantics: 'approximate',
    deterministic: true,
    status: ['converged'],
    evidence: 'partial',
    tests: 'modules/analysis/test/geospatial-expansion/raster-terrain-expansion.spec.ts',
    documentation: 'docs/api/gpu-raster/expansion.md',
    contract: 'GPUConstrainedDelaunayTriangulationProps'
  },
  GPUH3CellIdentities: {
    inputs: ['cells'],
    outputs: ['attributes'],
    coordinateModels: ['grid', 'topological'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'exact',
    deterministic: true,
    status: [],
    evidence: 'partial',
    tests: 'modules/analysis/test/geospatial-expansion/network-expansion.spec.ts',
    documentation: 'docs/api/gpu-network/expansion.md',
    contract: 'GPUH3CellIdentitiesProps'
  },
  GPUBoundedFacilityAllocation: {
    inputs: ['attributes', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['none'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'approximate',
    deterministic: true,
    status: ['invalidCount', 'overflow'],
    evidence: 'partial',
    tests: 'modules/analysis/test/geospatial-expansion/network-expansion.spec.ts',
    documentation: 'docs/api/gpu-network/expansion.md',
    contract: 'GPUBoundedFacilityAllocationProps'
  },
  addZonalDistributionRecipe: {
    inputs: [
      'points',
      'polygons',
      'trajectories',
      'cells',
      'surface',
      'spatial-weights',
      'attributes',
      'parameters'
    ],
    outputs: ['attributes', 'surface', 'network', 'summary'],
    coordinateModels: ['mixed'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'mixed',
    semantics: 'approximate',
    deterministic: true,
    status: ['invalidCount', 'overflow'],
    evidence: 'partial',
    tests: 'modules/analysis/test/gpu-spatial-analysis/recipes/expansion-recipes.spec.ts',
    documentation: 'docs/api/gpu-spatial-analysis/expansion-recipes.md',
    contract: 'GPUZonalDistributionRecipeProps'
  },
  addRasterZonalDistributionRecipe: {
    inputs: [
      'points',
      'polygons',
      'trajectories',
      'cells',
      'surface',
      'spatial-weights',
      'attributes',
      'parameters'
    ],
    outputs: ['attributes', 'surface', 'network', 'summary'],
    coordinateModels: ['mixed'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'mixed',
    semantics: 'approximate',
    deterministic: true,
    status: ['invalidCount', 'overflow'],
    evidence: 'partial',
    tests: 'modules/analysis/test/gpu-spatial-analysis/recipes/expansion-recipes.spec.ts',
    documentation: 'docs/api/gpu-spatial-analysis/expansion-recipes.md',
    contract: 'GPURasterZonalDistributionRecipeProps'
  },
  addInterpolationDiagnosticsRecipe: {
    inputs: [
      'points',
      'polygons',
      'trajectories',
      'cells',
      'surface',
      'spatial-weights',
      'attributes',
      'parameters'
    ],
    outputs: ['attributes', 'surface', 'network', 'summary'],
    coordinateModels: ['mixed'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'mixed',
    semantics: 'approximate',
    deterministic: true,
    status: ['invalidCount', 'overflow'],
    evidence: 'partial',
    tests: 'modules/analysis/test/gpu-spatial-analysis/recipes/expansion-recipes.spec.ts',
    documentation: 'docs/api/gpu-spatial-analysis/expansion-recipes.md',
    contract: 'GPUInterpolationDiagnosticsRecipeProps'
  },
  addExposureByZoneRecipe: {
    inputs: [
      'points',
      'polygons',
      'trajectories',
      'cells',
      'surface',
      'spatial-weights',
      'attributes',
      'parameters'
    ],
    outputs: ['attributes', 'surface', 'network', 'summary'],
    coordinateModels: ['mixed'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'mixed',
    semantics: 'approximate',
    deterministic: true,
    status: ['invalidCount', 'overflow'],
    evidence: 'partial',
    tests: 'modules/analysis/test/gpu-spatial-analysis/recipes/expansion-recipes.spec.ts',
    documentation: 'docs/api/gpu-spatial-analysis/expansion-recipes.md',
    contract: 'GPUExposureByZoneRecipeProps'
  },
  addRegionalRasterChangeRecipe: {
    inputs: [
      'points',
      'polygons',
      'trajectories',
      'cells',
      'surface',
      'spatial-weights',
      'attributes',
      'parameters'
    ],
    outputs: ['attributes', 'surface', 'network', 'summary'],
    coordinateModels: ['mixed'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'mixed',
    semantics: 'approximate',
    deterministic: true,
    status: ['invalidCount', 'overflow'],
    evidence: 'partial',
    tests: 'modules/analysis/test/gpu-spatial-analysis/recipes/expansion-recipes.spec.ts',
    documentation: 'docs/api/gpu-spatial-analysis/expansion-recipes.md',
    contract: 'GPURegionalRasterChangeRecipeProps'
  },
  addDrainageNetworkRecipe: {
    inputs: [
      'points',
      'polygons',
      'trajectories',
      'cells',
      'surface',
      'spatial-weights',
      'attributes',
      'parameters'
    ],
    outputs: ['attributes', 'surface', 'network', 'summary'],
    coordinateModels: ['mixed'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'mixed',
    semantics: 'approximate',
    deterministic: true,
    status: ['invalidCount', 'overflow'],
    evidence: 'partial',
    tests: 'modules/analysis/test/gpu-spatial-analysis/recipes/expansion-recipes.spec.ts',
    documentation: 'docs/api/gpu-spatial-analysis/expansion-recipes.md',
    contract: 'GPUDrainageNetworkRecipeProps'
  },
  addLocalVariabilityRecipe: {
    inputs: [
      'points',
      'polygons',
      'trajectories',
      'cells',
      'surface',
      'spatial-weights',
      'attributes',
      'parameters'
    ],
    outputs: ['attributes', 'surface', 'network', 'summary'],
    coordinateModels: ['mixed'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'mixed',
    semantics: 'approximate',
    deterministic: true,
    status: ['invalidCount', 'overflow'],
    evidence: 'partial',
    tests: 'modules/analysis/test/gpu-spatial-analysis/recipes/expansion-recipes.spec.ts',
    documentation: 'docs/api/gpu-spatial-analysis/expansion-recipes.md',
    contract: 'GPULocalVariabilityRecipeProps'
  },
  addCellFlowTopologyRecipe: {
    inputs: [
      'points',
      'polygons',
      'trajectories',
      'cells',
      'surface',
      'spatial-weights',
      'attributes',
      'parameters'
    ],
    outputs: ['attributes', 'surface', 'network', 'summary'],
    coordinateModels: ['mixed'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'mixed',
    semantics: 'approximate',
    deterministic: true,
    status: ['invalidCount', 'overflow'],
    evidence: 'partial',
    tests: 'modules/analysis/test/gpu-spatial-analysis/recipes/expansion-recipes.spec.ts',
    documentation: 'docs/api/gpu-network/expansion.md',
    contract: 'GPUCellFlowTopologyRecipeProps'
  },
  addSurfacePreparationRecipe: {
    inputs: [
      'points',
      'polygons',
      'trajectories',
      'cells',
      'surface',
      'spatial-weights',
      'attributes',
      'parameters'
    ],
    outputs: ['attributes', 'surface', 'network', 'summary'],
    coordinateModels: ['mixed'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'mixed',
    semantics: 'approximate',
    deterministic: true,
    status: ['invalidCount', 'overflow'],
    evidence: 'partial',
    tests: 'modules/analysis/test/gpu-spatial-analysis/recipes/expansion-recipes.spec.ts',
    documentation: 'docs/api/gpu-spatial-analysis/expansion-recipes.md',
    contract: 'GPUSurfacePreparationRecipeProps'
  },
  GPULonLatPolygonPreparation: {
    inputs: ['polygons', 'parameters'],
    outputs: ['polygons', 'attributes'],
    coordinateModels: ['spherical'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'capacity-bounded',
    semantics: 'approximate',
    deterministic: true,
    status: [...BOUNDED_STATUS, 'invalidCount', 'approximation'],
    contract: 'GPULonLatPolygonPreparationOutput',
    evidence: 'partial',
    tests: `${TEST_ROOT}/lon-lat-polygons/gpu-lon-lat-polygon-preparation.spec.ts`
  },
  GPUStudyWindowPreparation: {
    inputs: ['polygons'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['planar'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'exact',
    deterministic: true,
    status: [],
    contract: 'GPUPolygonGeometryFields',
    evidence: 'partial',
    tests: `${TEST_ROOT}/study-window/gpu-study-window-preparation.spec.ts`
  },
  GPUTerrainPeakProminence: {
    package: '@luma-contributors/analysis/gpu-terrain',
    inputs: ['surface'],
    outputs: ['surface', 'attributes'],
    coordinateModels: ['grid'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'exact',
    deterministic: true,
    status: [],
    contract: 'GPUTerrainPeakProminenceOutput',
    evidence: 'partial',
    tests: 'modules/analysis/test/gpu-terrain/terrain-features/terrain-peak-prominence.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md'
  },
  GPUParameterBuffer: {
    package: '@luma-contributors/analysis/analysis-core',
    inputs: [],
    outputs: ['parameters'],
    coordinateModels: ['none'],
    dimensionality: 'mixed',
    topology: 'dynamic-parameters',
    cardinality: 'fixed-sized',
    semantics: 'exact',
    status: [],
    contract: 'GPUSpatialParameterSchema',
    evidence: 'proven',
    tests: 'modules/analysis/test/gpu-spatial-analysis/contracts/contracts.node.spec.ts'
  },
  GPUSpatialPredicateJoin: {
    inputs: ['points', 'lines', 'polygons', 'parameters'],
    outputs: ['pairs', 'spatial-weights'],
    coordinateModels: ['planar', 'spherical'],
    dimensionality: '2d',
    semantics: 'robust-exact',
    status: [...BOUNDED_STATUS, 'candidateOverflow', 'uncertainCount'],
    contract: 'GPUCompactPairPort',
    evidence: 'proven',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join.spec.ts'
  },
  GPUPairGather: {
    inputs: ['pairs', 'attributes'],
    outputs: ['pairs', 'attributes'],
    coordinateModels: ['none'],
    dimensionality: 'topology',
    status: BOUNDED_STATUS,
    contract: 'GPUCompactPairPort',
    evidence: 'proven',
    tests: 'modules/analysis/test/gpu-spatial-analysis/pair-gather/gpu-pair-gather.spec.ts'
  },
  GPULineSplit: {
    inputs: ['lines'],
    outputs: ['lines'],
    coordinateModels: ['planar'],
    dimensionality: '2d',
    semantics: 'robust-exact',
    status: [...BOUNDED_STATUS, 'candidateOverflow', 'uncertainCount'],
    contract: 'GPUGeneratedGeometryPort',
    evidence: 'proven',
    tests: 'modules/analysis/test/gpu-spatial-analysis/line-split/gpu-line-split.spec.ts'
  },
  GPUSegmentRingAssembly: {
    inputs: ['lines'],
    outputs: ['polygons', 'summary'],
    coordinateModels: ['planar', 'topological'],
    dimensionality: 'topology',
    status: [...BOUNDED_STATUS, 'invalidCount'],
    contract: 'GPUPolygonGeometryPort & GPUGeneratedGeometryPort',
    evidence: 'proven',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/ring-assembly/gpu-segment-ring-assembly.spec.ts'
  },
  GPULineNoding: {
    inputs: ['lines'],
    outputs: ['lines', 'summary'],
    coordinateModels: ['planar'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'capacity-bounded',
    semantics: 'robust-exact',
    deterministic: true,
    status: [...BOUNDED_STATUS, 'candidateOverflow', 'uncertainCount'],
    contract: 'GPUNodedSegmentPort',
    evidence: 'proven',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/geometry-topology/gpu-geometry-topology.spec.ts'
  },
  GPUPolygonize: {
    inputs: ['lines'],
    outputs: ['polygons', 'summary'],
    coordinateModels: ['planar', 'topological'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'capacity-bounded',
    semantics: 'robust-exact',
    deterministic: true,
    status: [...BOUNDED_STATUS, 'candidateOverflow', 'uncertainCount'],
    contract: 'GPUPolygonizeOutput',
    evidence: 'proven',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/geometry-topology/gpu-geometry-topology.spec.ts'
  },
  GPUMakeValid: {
    inputs: ['polygons'],
    outputs: ['polygons', 'summary'],
    coordinateModels: ['planar', 'topological'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'capacity-bounded',
    semantics: 'robust-exact',
    deterministic: true,
    status: [...BOUNDED_STATUS, 'candidateOverflow', 'uncertainCount'],
    contract: 'GPUPolygonizeOutput',
    evidence: 'proven',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/geometry-topology/gpu-geometry-topology.spec.ts'
  },
  GPUDelaunayTessellation: {
    inputs: ['points'],
    outputs: ['render-geometry', 'summary'],
    coordinateModels: ['planar'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'capacity-bounded',
    semantics: 'approximate',
    deterministic: true,
    status: [...BOUNDED_STATUS, 'invalidCount'],
    contract: 'GPUDelaunayTessellationOutput & GPUBoundedResultStatusPort',
    evidence: 'proven',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation.spec.ts'
  },
  GPUPolygonOverlay: {
    inputs: ['polygons'],
    outputs: ['polygons'],
    coordinateModels: ['planar', 'topological'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'capacity-bounded',
    semantics: 'approximate',
    deterministic: true,
    // Rule 8: no feature `requiredCount` (unknown once rings drop); rings carry the requirement.
    status: [
      'count',
      'overflow',
      'candidateOverflow',
      'uncertainCount',
      'ringCount',
      'requiredRingCount',
      'nodedSegmentRequiredCount'
    ],
    contract: 'GPUPolygonOverlayOutput',
    evidence: 'proven',
    tests: 'modules/analysis/test/gpu-spatial-analysis/polygon-overlay/polygon-overlay.spec.ts'
  },
  GPUBufferSurface: {
    inputs: ['lines', 'polygons', 'parameters'],
    outputs: ['polygons'],
    coordinateModels: ['planar', 'topological'],
    dimensionality: '2d',
    topology: 'dynamic-parameters',
    cardinality: 'capacity-bounded',
    semantics: 'approximate',
    deterministic: true,
    // Rule 8: no feature `requiredCount` (unknown once rings drop); rings carry the requirement.
    status: [
      'count',
      'overflow',
      'candidateOverflow',
      'uncertainCount',
      'ringCount',
      'requiredRingCount',
      'nodedSegmentRequiredCount'
    ],
    contract: 'GPUPolygonOverlayOutput',
    evidence: 'proven',
    tests: 'modules/analysis/test/gpu-spatial-analysis/polygon-buffer/polygon-buffer.spec.ts'
  },
  GPUVoronoiDiagram: {
    inputs: ['points', 'render-geometry', 'parameters'],
    outputs: ['lines', 'summary'],
    coordinateModels: ['planar'],
    dimensionality: '2d',
    topology: 'dynamic-parameters',
    cardinality: 'capacity-bounded',
    semantics: 'approximate',
    deterministic: true,
    status: [...BOUNDED_STATUS, 'invalidCount'],
    contract: 'GPUVoronoiDiagramOutput & GPUBoundedResultStatusPort',
    evidence: 'proven',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation.spec.ts'
  },
  GPURegionStatisticsReadback: {
    inputs: ['summary'],
    outputs: ['summary'],
    coordinateModels: ['none'],
    dimensionality: 'mixed',
    readback: 'explicit-adapter',
    contract: 'GPURegionStatisticsReadback',
    evidence: 'proven',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/region-statistics/gpu-region-statistics.spec.ts'
  },
  addPointsInPolygonsChoroplethRecipe: {
    inputs: ['points', 'polygons', 'attributes'],
    outputs: ['attributes', 'render-geometry'],
    coordinateModels: ['planar'],
    dimensionality: '2d',
    tags: ['spatial-joins', 'zonal-and-region-statistics'],
    status: ['overflow', 'uncertainCount'],
    contract: 'GPURecipeResult',
    evidence: 'proven',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/recipes/points-in-polygons-choropleth-recipe.spec.ts'
  },
  addFleetDwellRecipe: {
    inputs: ['trajectories', 'polygons'],
    outputs: ['summary'],
    coordinateModels: ['planar'],
    tags: ['movement-analysis', 'zonal-and-region-statistics'],
    dimensionality: '2d+time',
    status: ['count', 'overflow'],
    contract: 'GPURecipeResult',
    evidence: 'proven',
    tests: 'modules/analysis/test/gpu-spatial-analysis/recipes/fleet-dwell-recipe.spec.ts'
  },
  GPULocalGeary: {
    inputs: ['attributes', 'spatial-weights', 'parameters'],
    outputs: ['attributes'],
    coordinateModels: ['topological'],
    dimensionality: 'topology',
    topology: 'dynamic-parameters',
    cardinality: 'source-sized',
    semantics: 'exact',
    deterministic: true,
    contract: 'GPUInferenceResultPort',
    evidence: 'proven',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/spatial-autocorrelation/additional-spatial-statistics.spec.ts'
  },
  GPUSpatialPearson: {
    inputs: ['attributes', 'spatial-weights'],
    outputs: ['summary'],
    coordinateModels: ['topological'],
    dimensionality: 'topology',
    topology: 'fixed',
    cardinality: 'fixed-sized',
    semantics: 'exact',
    deterministic: true,
    contract: 'GPUInferenceResultPort',
    evidence: 'proven',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/spatial-autocorrelation/additional-spatial-statistics.spec.ts'
  },
  GPUGammaStatistic: {
    inputs: ['attributes', 'spatial-weights'],
    outputs: ['summary'],
    coordinateModels: ['topological'],
    dimensionality: 'topology',
    topology: 'fixed',
    cardinality: 'fixed-sized',
    semantics: 'exact',
    deterministic: true,
    contract: 'GPUInferenceResultPort',
    evidence: 'proven',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/spatial-autocorrelation/additional-spatial-statistics.spec.ts'
  },
  GPURoseStatistic: {
    inputs: ['lines', 'attributes'],
    outputs: ['summary', 'attributes'],
    coordinateModels: ['planar'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'fixed-sized',
    semantics: 'exact',
    deterministic: true,
    contract: 'GPUCircularStatisticsContract',
    evidence: 'proven',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/geographic-distribution/gpu-rose-statistic.spec.ts'
  },
  GPUAZPRegions: {
    inputs: ['attributes', 'spatial-weights', 'regions'],
    outputs: ['regions', 'summary'],
    coordinateModels: ['topological'],
    dimensionality: 'topology',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'model-based',
    deterministic: true,
    status: ['converged', 'iterationLimitReached', 'invalidCount'],
    contract: 'GPUAZPRegionsOutput',
    evidence: 'proven',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions.spec.ts'
  },
  GPUMaxPRegions: {
    inputs: ['attributes', 'spatial-weights', 'parameters'],
    outputs: ['regions', 'summary'],
    coordinateModels: ['topological'],
    dimensionality: 'topology',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'model-based',
    deterministic: true,
    status: ['count', 'invalidCount'],
    contract: 'GPUMaxPRegionsOutput',
    evidence: 'proven',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions.spec.ts'
  },
  GPUWardRegions: {
    inputs: ['attributes', 'spatial-weights'],
    outputs: ['regions', 'summary'],
    coordinateModels: ['topological'],
    dimensionality: 'topology',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'model-based',
    deterministic: true,
    status: ['converged', 'iterationLimitReached', 'invalidCount'],
    contract: 'GPUWardRegionsOutput',
    evidence: 'proven',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/spatial-regionalization/gpu-ward-regions.spec.ts'
  },
  GPULocationAllocation: {
    inputs: ['network', 'attributes', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['topological'],
    dimensionality: 'topology',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'model-based',
    deterministic: true,
    status: ['count', 'invalidCount'],
    contract: 'GPULocationAllocationOutput',
    evidence: 'proven',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/location-allocation/gpu-location-allocation.spec.ts'
  }
};

function makeOperator(
  capability: GPUSpatialAnalysisCapability,
  exportName: string
): GPUSpatialAnalysisOperator {
  const override = OPERATOR_OVERRIDES[exportName] ?? {};
  const annotation = OPERATOR_ANNOTATIONS[exportName] ?? {};
  const capabilityEvidence = capability.evidence;
  if (!capabilityEvidence) {
    throw new Error(`available capability ${capability.id} needs execution evidence`);
  }
  return {
    exportName,
    primaryCapability: capability.id,
    tags: override.tags ?? [],
    package: override.package ?? '@luma-contributors/analysis/gpu-spatial-analysis',
    inputs: override.inputs ?? [],
    outputs: override.outputs ?? [],
    dimensionality: SPATIOTEMPORAL_OPERATOR_NAMES.has(exportName)
      ? '2d+time'
      : (override.dimensionality ?? 'unspecified'),
    coordinateModels: override.coordinateModels ?? ['unspecified'],
    topology: override.topology ?? 'unspecified',
    cardinality: override.cardinality ?? 'unspecified',
    semantics: override.semantics ?? 'unspecified',
    deterministic: override.deterministic ?? 'unspecified',
    status: override.status ?? [],
    readback: override.readback ?? 'none',
    role: annotation.role ?? 'operator',
    tier: annotation.tier ?? 'scale',
    ...(annotation.limits ? {limits: annotation.limits} : {}),
    ...(annotation.limitation ? {limitation: annotation.limitation} : {}),
    evidence: {
      status: override.evidence ?? 'partial',
      tests: override.tests ?? TEST_ROOT,
      documentation: override.documentation ?? DOCUMENTATION,
      contract: override.contract ?? 'unaudited',
      representativeSourceRows: capabilityEvidence.representativeSourceRows,
      representativeOutputRows: capabilityEvidence.representativeOutputRows,
      transientMemory: capabilityEvidence.transientMemory,
      dispatch: capabilityEvidence.dispatch,
      correctnessOracle: capabilityEvidence.correctnessOracle
    }
  };
}

/** Operator-level discovery catalogue. It contains names and metadata, never constructors. */
export const GPU_SPATIAL_ANALYSIS_OPERATORS: readonly GPUSpatialAnalysisOperator[] =
  GPU_SPATIAL_ANALYSIS_CAPABILITIES.flatMap(capability =>
    capability.operators.map(exportName => makeOperator(capability, exportName))
  );

const OPERATOR_BY_NAME = new Map(
  GPU_SPATIAL_ANALYSIS_OPERATORS.map(operator => [operator.exportName, operator])
);

/** Returns one public operator descriptor. */
export function getGPUSpatialAnalysisOperator(
  exportName: string
): GPUSpatialAnalysisOperator | undefined {
  return OPERATOR_BY_NAME.get(exportName);
}

/** Finds public operators by task, data kind, semantics, evidence or free text. */
export function queryGPUSpatialAnalysisOperators(
  query: GPUSpatialAnalysisOperatorQuery = {}
): readonly GPUSpatialAnalysisOperator[] {
  const search = query.search?.trim().toLowerCase();
  return GPU_SPATIAL_ANALYSIS_OPERATORS.filter(operator => {
    if (query.primaryCapability && operator.primaryCapability !== query.primaryCapability)
      return false;
    if (query.tag && !operator.tags.includes(query.tag)) return false;
    if (query.input && !operator.inputs.includes(query.input)) return false;
    if (query.output && !operator.outputs.includes(query.output)) return false;
    if (query.coordinateModel && !operator.coordinateModels.includes(query.coordinateModel))
      return false;
    if (query.status && !operator.status.includes(query.status)) return false;
    if (query.semantics && operator.semantics !== query.semantics) return false;
    if (query.evidence && operator.evidence.status !== query.evidence) return false;
    if (!search) return true;
    return [
      operator.exportName,
      operator.primaryCapability,
      ...operator.tags,
      ...operator.inputs,
      ...operator.outputs,
      operator.evidence.contract
    ]
      .join(' ')
      .toLowerCase()
      .includes(search);
  });
}

const operatorsByCapability: Record<string, number> = {};
for (const operator of GPU_SPATIAL_ANALYSIS_OPERATORS) {
  operatorsByCapability[operator.primaryCapability] =
    (operatorsByCapability[operator.primaryCapability] ?? 0) + 1;
}

/** Generated evidence baseline for tests, docs and maintainer tooling. */
export const GPU_SPATIAL_ANALYSIS_AUDIT: GPUSpatialAnalysisAudit = {
  capabilityFamilyCount: GPU_SPATIAL_ANALYSIS_CAPABILITIES.length,
  availableCapabilityFamilyCount: GPU_SPATIAL_ANALYSIS_CAPABILITIES.filter(
    capability => capability.status === 'available'
  ).length,
  opportunityCount: GPU_SPATIAL_ANALYSIS_CAPABILITIES.filter(
    capability => capability.status === 'opportunity'
  ).length,
  operatorCount: GPU_SPATIAL_ANALYSIS_OPERATORS.length,
  provenOperatorCount: GPU_SPATIAL_ANALYSIS_OPERATORS.filter(
    operator => operator.evidence.status === 'proven'
  ).length,
  partialOperatorCount: GPU_SPATIAL_ANALYSIS_OPERATORS.filter(
    operator => operator.evidence.status === 'partial'
  ).length,
  helperOperatorCount: GPU_SPATIAL_ANALYSIS_OPERATORS.filter(operator => operator.role === 'helper')
    .length,
  referenceOperatorCount: GPU_SPATIAL_ANALYSIS_OPERATORS.filter(
    operator => operator.tier === 'reference'
  ).length,
  fullySpecifiedCapabilityFamilyCount: GPU_SPATIAL_ANALYSIS_CAPABILITIES.filter(
    capability => capability.status === 'available' && capability.evidence
  ).length,
  operatorsByCapability
};

/** Formats the current baseline without introducing a generated file into package builds. */
export function formatGPUSpatialAnalysisAuditMarkdown(): string {
  const rows = GPU_SPATIAL_ANALYSIS_CAPABILITIES.map(capability => {
    const operators = queryGPUSpatialAnalysisOperators({primaryCapability: capability.id});
    const proven = operators.filter(operator => operator.evidence.status === 'proven').length;
    return `| ${capability.id} | ${capability.status} | ${operators.length} | ${proven} |`;
  });
  return [
    '| Capability | Catalogue status | Operators | Proven |',
    '| --- | --- | ---: | ---: |',
    ...rows
  ].join('\n');
}

/** Narrows arbitrary strings to known logical data kinds for catalogue generators. */
export function isGPUSpatialAnalysisDataKind(value: string): value is GPUSpatialAnalysisDataKind {
  return GPU_SPATIAL_ANALYSIS_CAPABILITIES.some(
    capability =>
      capability.inputs.includes(value as GPUSpatialAnalysisDataKind) ||
      capability.outputs.includes(value as GPUSpatialAnalysisDataKind)
  );
}
