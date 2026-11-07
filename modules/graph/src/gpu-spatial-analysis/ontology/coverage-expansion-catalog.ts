// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUSpatialAnalysisCapability,
  GPUSpatialAnalysisOperator
} from './spatial-analysis-ontology';

type CoverageFamily = {
  capability: GPUSpatialAnalysisCapability;
  tests: string;
  documentation: string;
};

const DOCUMENTATION_ROOT = 'docs/api-reference/experimental/gpu-spatial-analysis';
const TEST_ROOT = 'modules/graph/test/gpu-spatial-analysis';

/** Bounded implementations, with numerical evidence kept distinct from full backlog completion. */
const COVERAGE_FAMILIES: readonly CoverageFamily[] = [
  {
    capability: {
      id: 'coverage-movement',
      title: 'Trajectory estimation and similarity search',
      status: 'available',
      domain: 'movement',
      stage: 'analyze',
      summary:
        'Kalman state estimation, geometric restrictions and bounded sequence similarity search.',
      inputs: ['points', 'attributes', 'parameters'],
      outputs: ['attributes', 'summary'],
      coordinateModels: ['planar'],
      outputBound: 'source-sized',
      operators: ['GPUTrackSimilaritySearch', 'GPUTrajectoryKalman', 'GPUTrajectoryRestriction'],
      limitation: 'Explicit sampling, process models and bounded sequence lengths apply.',
      evidence: {
        representativeSourceRows: 4,
        transientMemory: 'solver-matrix',
        dispatch: 'multi-pass',
        correctnessOracle: `${TEST_ROOT}/trajectory-estimation/gpu-trajectory-estimation.spec.ts`
      }
    },
    tests: `${TEST_ROOT}/trajectory-estimation/gpu-trajectory-estimation.spec.ts`,
    documentation: 'docs/api-reference/experimental/gpu-network/network-movement-coverage.md'
  },
  {
    capability: {
      id: 'coverage-geometry-refinement',
      title: 'Geometry normalization and refinement',
      status: 'available',
      domain: 'geometry',
      stage: 'construct',
      summary:
        'Canonical mixed geometry trees, arbitrary precision grids, rectilinear subdivision, offset components and event-based vector skeletons.',
      inputs: ['polygons', 'lines', 'points', 'parameters'],
      outputs: ['polygons', 'lines', 'attributes'],
      coordinateModels: ['planar'],
      outputBound: 'capacity-bounded',
      operators: [
        'GPUCanonicalGeometryEquality',
        'GPUNormalizeGeometry',
        'GPUOffsetCurveComponents',
        'GPUVectorStraightSkeleton',
        'GPUCanonicalPolygonEquality',
        'GPUConvexStraightSkeleton',
        'GPULineLoopRemoval',
        'GPUNormalizePolygons',
        'GPUMaximumInscribedCircle',
        'GPUSegmentSnap',
        'GPURatioConcaveHull',
        'GPUPolygonPrecisionReduction',
        'GPULinePrecisionReduction',
        'GPUSubdividePolygons'
      ],
      limitation:
        'Postorder collection depth is bounded at 64. Geometry event ordering uses float32 tolerances; incomplete topology and output capacities are explicit.',
      evidence: {
        representativeSourceRows: 1536,
        transientMemory: 'linear-source-and-output',
        dispatch: 'bounded-iterative',
        correctnessOracle: `${TEST_ROOT}/geometry-refinement/geometry-advanced.spec.ts`
      }
    },
    tests: `${TEST_ROOT}/geometry-refinement/geometry-refinement.spec.ts`,
    documentation: `docs/api-reference/experimental/gpu-geometry-refinement.md`
  },
  {
    capability: {
      id: 'coverage-global-coordinates',
      title: 'Spherical convex geometry and S2 cells',
      status: 'available',
      domain: 'discrete-global-grids',
      stage: 'relate',
      summary:
        'Exact S2 hierarchy, stable fine-cell measures, adaptive polygon and cap covers, spherical containment and Boolean topology.',
      inputs: ['cells', 'polygons', 'points', 'parameters'],
      outputs: ['cells', 'polygons', 'attributes'],
      coordinateModels: ['spherical', 'grid'],
      outputBound: 'capacity-bounded',
      operators: [
        'GPUS2CellHierarchy',
        'GPUS2CellMeasures',
        'GPUS2CapCovering',
        'GPUSphericalConvexIntersection',
        'GPUSphericalConvexContains',
        'GPUSphericalPolygonContains',
        'GPUSphericalPolygonOverlay',
        'GPUSphericalArcOverlay',
        'GPUS2PolygonCovering'
      ],
      limitation:
        'Hierarchy and measures levels 0–30; adaptive polygon covers through 15; cap covers through 8. Intrinsic global overlay accepts at most 128 input vertices and 1024 output vertices, with explicit uncertain topology status; chart overlay follows its snap lattice contract.',
      evidence: {
        representativeSourceRows: 166,
        transientMemory: 'linear-output',
        dispatch: 'source-parallel',
        correctnessOracle: `${TEST_ROOT}/global-coordinates/global-coordinates.spec.ts`
      }
    },
    tests: `${TEST_ROOT}/global-coordinates/global-coordinates.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/global-coordinates.md`
  },
  {
    capability: {
      id: 'coverage-local-prediction',
      title: 'Local prediction and conditional simulation',
      status: 'available',
      domain: 'density-and-interpolation',
      stage: 'infer',
      summary:
        'Bounded local kriging, radial basis prediction, stolen-area natural neighbors and joint conditional Gaussian simulation.',
      inputs: ['points', 'attributes', 'parameters'],
      outputs: ['attributes', 'summary'],
      coordinateModels: ['planar'],
      outputBound: 'source-sized',
      operators: [
        'GPULocalKriging',
        'GPULocalRBF',
        'GPUNaturalNeighbor',
        'GPUConditionalGaussianSimulation'
      ],
      limitation:
        'Neighbor, cavity and simulation sizes are explicitly bounded. Rank failure, hull exclusion and capacity exhaustion remain observable.',
      evidence: {
        representativeSourceRows: 4,
        transientMemory: 'solver-matrix',
        dispatch: 'multi-pass',
        correctnessOracle: `${TEST_ROOT}/spatial-interpolation/interpolation-coverage.spec.ts`
      }
    },
    tests: `${TEST_ROOT}/spatial-interpolation/interpolation-coverage.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/interpolation-expansion.md`
  },
  {
    capability: {
      id: 'coverage-statistical-contracts',
      title: 'Robust inference and spatial validation',
      status: 'available',
      domain: 'spatial-models',
      stage: 'infer',
      summary:
        'Robust instrumental-variable inference, restricted regimes, panel error likelihood, SUR, probit, local diagnostics, permutation inference, rank dynamics, fuzzy weights, held-out regression and spatial interaction.',
      inputs: ['points', 'attributes', 'spatial-weights', 'parameters'],
      outputs: ['attributes', 'summary', 'spatial-weights'],
      coordinateModels: ['planar', 'topological'],
      outputBound: 'mixed',
      operators: [
        'GPUInstrumentalVariablesCovariance',
        'GPURestrictedRegimeRegression',
        'GPUSpatialPanelErrorMaximumLikelihood',
        'GPUSpatialPanelLagMaximumLikelihood',
        'GPUSeeminglyUnrelatedRegression',
        'GPUProbitRegression',
        'GPULocalCollinearity',
        'GPUMultivariateLocalGearyPermutationTest',
        'GPURankMarkov',
        'GPUFuzzyContiguity',
        'GPUIslandAttachment',
        'GPUSpatialCrossValidation',
        'GPUSpatialCrossValidatedOLS',
        'GPUSpatialInteraction'
      ],
      limitation:
        'Estimator-specific bounded numerical conventions apply. Held-out OLS fits use training masks; other preprocessing and model selection must honor the same partition.',
      evidence: {
        representativeSourceRows: 4,
        transientMemory: 'solver-matrix',
        dispatch: 'multi-pass',
        correctnessOracle: `${TEST_ROOT}/spatial-regression/coverage-statistics.spec.ts`
      }
    },
    tests: `${TEST_ROOT}/spatial-regression/coverage-statistics.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/coverage-statistics.md`
  },
  {
    capability: {
      id: 'coverage-point-processes',
      title: 'Point-process inference and sampling',
      status: 'available',
      domain: 'spatial-statistics',
      stage: 'infer',
      summary:
        'Inhomogeneous polygon-window summaries, supplied-curve Monte Carlo inference, bounded OPTICS and finite/continuous-frame probability sampling.',
      inputs: ['points', 'attributes', 'polygons', 'parameters'],
      outputs: ['attributes', 'summary'],
      coordinateModels: ['planar'],
      outputBound: 'capacity-bounded',
      operators: [
        'GPUInhomogeneousPointPattern',
        'GPUPointPatternEnvelope',
        'GPUSpatialBalancedSample',
        'GPUContinuousSpatialSample',
        'GPUOPTICS'
      ],
      limitation:
        'Null models, fitted-parameter treatment, sample inclusion design and OPTICS extraction modes are explicit; external-package parity is not inferred.',
      evidence: {
        representativeSourceRows: 4,
        transientMemory: 'quadratic-source',
        dispatch: 'bounded-iterative',
        correctnessOracle: `${TEST_ROOT}/point-process/point-process.spec.ts`
      }
    },
    tests: `${TEST_ROOT}/point-process/point-process.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/point-process.md`
  },
  {
    capability: {
      id: 'coverage-urban-morphology',
      title: 'Building and street analysis',
      status: 'available',
      domain: 'regional-analysis',
      stage: 'analyze',
      summary:
        'Boundary-sampled building cells, street cross-sections, shared walls, sampled isovists and visibility graphs.',
      inputs: ['polygons', 'lines', 'points', 'attributes'],
      outputs: ['polygons', 'network', 'attributes'],
      coordinateModels: ['planar'],
      outputBound: 'capacity-bounded',
      operators: [
        'addBuildingTessellationRecipe',
        'addStreetProfilesRecipe',
        'addBuildingAdjacencyRecipe',
        'addIsovistsRecipe',
        'addVisibilityGraphRecipe'
      ],
      limitation:
        'Convex and simple nonconvex enclosures with holes use bounded sampling and snapped overlay. Straight-corridor facade visibility, vertex-event isovists and normalized depth metrics have explicit boundary and capacity contracts; continuous visibility and whole-tool parity are not claimed.',
      evidence: {
        representativeSourceRows: 4,
        transientMemory: 'linear-source-and-output',
        dispatch: 'orchestrated',
        correctnessOracle: `${TEST_ROOT}/urban-ecology/urban-ecology.spec.ts`
      }
    },
    tests: `${TEST_ROOT}/urban-ecology/urban-ecology.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/urban-morphology.md`
  },
  {
    capability: {
      id: 'coverage-landscape-ecology',
      title: 'Landscape and resistance connectivity',
      status: 'available',
      domain: 'regional-analysis',
      stage: 'analyze',
      summary:
        'Class and landscape summaries, patch core/proximity, grounded resistance/current maps and common-support segregation decomposition.',
      inputs: ['surface', 'regions', 'network', 'attributes'],
      outputs: ['attributes', 'summary'],
      coordinateModels: ['grid', 'topological'],
      outputBound: 'source-sized',
      operators: [
        'addLandscapeMetricsRecipe',
        'addPatchProximityRecipe',
        'addResistanceConnectivityRecipe',
        'addSparseResistanceConnectivityRecipe',
        'addRasterResistanceConnectivityRecipe',
        'addSegregationDecompositionRecipe',
        'addSegregationCounterfactualRecipe',
        'addPopulationSupportAlignmentRecipe'
      ],
      limitation:
        'Global tile IDs and resident halo-free support enable patch reconciliation. Sparse Jacobi-PCG exposes convergence and disconnected support; raster conductance, quantile counterfactuals, ID alignment and paired randomization have explicit contracts. Scaling evidence covers 65,538 vertices and 65,536 cells, not all conditioning regimes.',
      evidence: {
        representativeSourceRows: 65538,
        transientMemory: 'solver-matrix',
        dispatch: 'bounded-iterative',
        correctnessOracle: `${TEST_ROOT}/urban-ecology/urban-ecology-advanced.spec.ts`
      }
    },
    tests: `${TEST_ROOT}/urban-ecology/urban-ecology-advanced.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/landscape-ecology.md`
  }
];

export const COVERAGE_EXPANSION_CAPABILITIES = COVERAGE_FAMILIES.map(({capability}) => capability);

type CoverageOperatorOverride = Pick<
  GPUSpatialAnalysisOperator,
  'inputs' | 'outputs' | 'coordinateModels'
> &
  Partial<
    Pick<
      GPUSpatialAnalysisOperator,
      | 'dimensionality'
      | 'topology'
      | 'cardinality'
      | 'semantics'
      | 'deterministic'
      | 'status'
      | 'readback'
    >
  > & {
    tests: string;
    documentation: string;
    contract: string;
    evidence: 'partial';
  };

/** Precise global-coordinate contracts; per-row status codes are documented, not mislabeled as aggregate invalid counts. */
const GLOBAL_COORDINATE_OPERATOR_OVERRIDES: Readonly<
  Record<string, Partial<CoverageOperatorOverride>>
> = {
  GPUS2CellHierarchy: {
    inputs: ['cells', 'parameters'],
    outputs: ['cells', 'attributes'],
    coordinateModels: ['grid', 'topological'],
    dimensionality: 'topology',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'exact',
    deterministic: true,
    readback: 'none',
    status: []
  },
  GPUS2CellMeasures: {
    inputs: ['cells'],
    outputs: ['attributes'],
    coordinateModels: ['spherical', 'grid'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'approximate',
    deterministic: true,
    readback: 'none',
    status: []
  },
  GPUS2CapCovering: {
    inputs: ['points', 'parameters'],
    outputs: ['cells', 'attributes', 'summary'],
    coordinateModels: ['spherical', 'grid'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'capacity-bounded',
    semantics: 'approximate',
    deterministic: true,
    readback: 'none',
    status: ['count', 'requiredCount', 'overflow']
  },
  GPUSphericalConvexIntersection: {
    inputs: ['polygons', 'parameters'],
    outputs: ['polygons', 'attributes', 'summary'],
    coordinateModels: ['spherical'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'capacity-bounded',
    semantics: 'approximate',
    deterministic: true,
    readback: 'none',
    status: ['count', 'requiredCount', 'overflow']
  },
  GPUSphericalConvexContains: {
    inputs: ['polygons', 'points', 'parameters'],
    outputs: ['attributes'],
    coordinateModels: ['spherical'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'approximate',
    deterministic: true,
    readback: 'none',
    status: []
  },
  GPUSphericalPolygonContains: {
    inputs: ['polygons', 'points', 'parameters'],
    outputs: ['attributes'],
    coordinateModels: ['spherical'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'approximate',
    deterministic: true,
    readback: 'none',
    status: []
  },
  GPUSphericalPolygonOverlay: {
    inputs: ['polygons', 'parameters'],
    outputs: ['polygons', 'attributes', 'summary'],
    coordinateModels: ['spherical'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'capacity-bounded',
    semantics: 'approximate',
    deterministic: true,
    readback: 'none',
    status: ['count', 'requiredCount', 'overflow', 'candidateOverflow']
  },
  GPUSphericalArcOverlay: {
    inputs: ['polygons', 'parameters'],
    outputs: ['polygons', 'attributes', 'summary'],
    coordinateModels: ['spherical'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'capacity-bounded',
    semantics: 'approximate',
    deterministic: true,
    readback: 'none',
    status: ['count', 'requiredCount', 'overflow']
  },
  GPUS2PolygonCovering: {
    inputs: ['polygons', 'parameters'],
    outputs: ['cells', 'attributes', 'summary'],
    coordinateModels: ['spherical'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'capacity-bounded',
    semantics: 'approximate',
    deterministic: true,
    readback: 'none',
    status: ['count', 'requiredCount', 'overflow']
  }
};

/** Geometry status rows carry documented codes; they are not generic invalid-count views. */
const GEOMETRY_OPERATOR_OVERRIDES: Readonly<Record<string, Partial<CoverageOperatorOverride>>> = {
  GPUNormalizeGeometry: {
    inputs: ['points', 'lines', 'polygons', 'attributes'],
    outputs: ['points', 'lines', 'polygons', 'attributes'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'approximate',
    deterministic: true,
    readback: 'none',
    status: [],
    tests: `${TEST_ROOT}/geometry-refinement/geometry-advanced.spec.ts`
  },
  GPUCanonicalGeometryEquality: {
    inputs: ['points', 'lines', 'polygons', 'parameters'],
    outputs: ['attributes'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'approximate',
    deterministic: true,
    readback: 'none',
    status: [],
    tests: `${TEST_ROOT}/geometry-refinement/geometry-advanced.spec.ts`
  },
  GPUOffsetCurveComponents: {
    inputs: ['lines', 'parameters'],
    outputs: ['lines', 'attributes'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'capacity-bounded',
    semantics: 'approximate',
    deterministic: true,
    readback: 'none',
    status: ['count', 'requiredCount', 'overflow'],
    tests: `${TEST_ROOT}/geometry-refinement/geometry-advanced.spec.ts`
  },
  GPUVectorStraightSkeleton: {
    inputs: ['polygons', 'parameters'],
    outputs: ['lines', 'polygons', 'attributes'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'capacity-bounded',
    semantics: 'approximate',
    deterministic: true,
    readback: 'none',
    status: [],
    tests: `${TEST_ROOT}/geometry-refinement/geometry-advanced.spec.ts`
  }
};

/** Continuous sampling has per-wave status records and continuous design intensity, not point inclusion probabilities. */
const CONTINUOUS_SAMPLING_OPERATOR_OVERRIDES: Readonly<
  Record<string, Partial<CoverageOperatorOverride>>
> = {
  GPUContinuousSpatialSample: {
    inputs: ['lines', 'polygons', 'points', 'parameters'],
    outputs: ['points', 'attributes', 'summary'],
    coordinateModels: ['planar'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'capacity-bounded',
    semantics: 'approximate',
    deterministic: true,
    readback: 'none',
    status: [],
    tests: `${TEST_ROOT}/point-process/continuous-spatial-sample.node.spec.ts`
  }
};

/** Descriptor data imports no GPU constructors and does not promote incomplete oracle coverage. */
export const COVERAGE_EXPANSION_OPERATOR_OVERRIDES: Readonly<
  Record<string, CoverageOperatorOverride>
> = Object.fromEntries(
  COVERAGE_FAMILIES.flatMap(({capability, tests, documentation}) =>
    capability.operators.map(name => [
      name,
      {
        inputs: capability.inputs,
        outputs: capability.outputs,
        coordinateModels: capability.coordinateModels,
        tests,
        documentation,
        contract: `${name.startsWith('add') ? `GPU${name.slice(3, -6)}` : name}Props`,
        evidence: 'partial',
        ...GLOBAL_COORDINATE_OPERATOR_OVERRIDES[name],
        ...GEOMETRY_OPERATOR_OVERRIDES[name],
        ...CONTINUOUS_SAMPLING_OPERATOR_OVERRIDES[name],
        ...(name === 'GPUSpatialPanelLagMaximumLikelihood'
          ? {
              tests: `${TEST_ROOT}/spatial-regression/statistics-advanced.spec.ts`,
              inputs: ['attributes', 'spatial-weights', 'parameters'],
              outputs: ['attributes', 'summary']
            }
          : {})
      }
    ])
  )
);
