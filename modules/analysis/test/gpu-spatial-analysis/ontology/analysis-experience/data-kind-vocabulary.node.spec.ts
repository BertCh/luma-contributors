// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {describe, expect, it} from 'vitest';
import {getGPUAnalysisOperatorCatalog} from '../../../../src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog';
import {
  GPU_ANALYSIS_CATALOG_INPUT_GEOMETRIES,
  GPU_ANALYSIS_CATALOG_OUTPUT_GEOMETRIES,
  GPU_ANALYSIS_DATA_KIND_RECORDS,
  GPU_ANALYSIS_WEBSITE_DATA_TYPES,
  GPU_ANALYSIS_WEBSITE_TYPE_KINDS,
  type GPUAnalysisCatalogOutputGeometry,
  type GPUAnalysisDataKindRecord,
  type GPUAnalysisWebsiteDataType,
  getGPUAnalysisCatalogGeometryForDataKind,
  getGPUAnalysisDataKindForCatalogGeometry,
  getGPUAnalysisDataKindRecord,
  getGPUAnalysisDataKindsForWebsiteDataType,
  getGPUAnalysisWebsiteDataTypes,
  listGPUAnalysisLoadableDataKinds
} from '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/data-kind-vocabulary';
import {
  GPU_ANALYSIS_HANDLE_OPERATORS,
  GPU_ANALYSIS_MAXIMUM_PATH_CANDIDATES,
  GPU_ANALYSIS_MAXIMUM_PATH_COUNT,
  GPU_ANALYSIS_MAXIMUM_PATH_COUNT_LIMIT,
  GPU_ANALYSIS_MAXIMUM_PATH_STEPS,
  GPU_ANALYSIS_OPERATOR_KIND_SUPPLEMENTS,
  findGPUAnalysisOperatorPaths,
  getGPUAnalysisNextOperators,
  queryGPUAnalysisOperatorsByKind,
  resolveGPUAnalysisOperatorKinds
} from '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-operator-kind-supplements';
import {getGPUAnalysisOperatorQuestions} from '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-operator-questions';
import {
  GPU_SPATIAL_ANALYSIS_DATA_KINDS,
  type GPUSpatialAnalysisDataKind
} from '../../../../src/gpu-spatial-analysis/ontology/spatial-analysis-ontology';

const SOURCE_ROOT = fileURLToPath(new URL('../../../../src/', import.meta.url));

describe('data kind records', () => {
  it('has one record per ontology kind, in ontology order, with prose', () => {
    expect(GPU_ANALYSIS_DATA_KIND_RECORDS.map(record => record.id)).toEqual([
      ...GPU_SPATIAL_ANALYSIS_DATA_KINDS
    ]);
    for (const record of GPU_ANALYSIS_DATA_KIND_RECORDS) {
      expect(record.label.length, record.id).toBeGreaterThan(2);
      expect(record.description.endsWith('.'), record.id).toBe(true);
    }
    expect(getGPUAnalysisDataKindRecord('surface').label).toBe('Surface (raster or terrain)');
    expect(() => getGPUAnalysisDataKindRecord('nonsense' as GPUSpatialAnalysisDataKind)).toThrow(
      /getGPUAnalysisDataKindRecord kind "nonsense"/
    );
  });

  it('marks exactly the kinds a user can load as a dataset', () => {
    expect([...listGPUAnalysisLoadableDataKinds()].sort()).toEqual(
      [
        'attributes',
        'cells',
        'lines',
        'network',
        'points',
        'polygons',
        'spatial-weights',
        'surface',
        'time-series',
        'trajectories'
      ].sort()
    );
    for (const produced of [
      'pairs',
      'regions',
      'summary',
      'parameters',
      'render-geometry'
    ] as const) {
      expect(getGPUAnalysisDataKindRecord(produced).loadable, produced).toBe(false);
    }
  });
});

describe('vocabulary translation', () => {
  it('gives every ontology kind a website answer, empty only for internal kinds', () => {
    for (const kind of GPU_SPATIAL_ANALYSIS_DATA_KINDS) {
      const types = getGPUAnalysisWebsiteDataTypes(kind);
      for (const type of types) expect(GPU_ANALYSIS_WEBSITE_DATA_TYPES).toContain(type);
      const isInternal = kind === 'parameters' || kind === 'render-geometry';
      expect(types.length === 0, kind).toBe(isInternal);
    }
  });

  it('maps raster and terrain to surface and surface to both', () => {
    expect(getGPUAnalysisDataKindsForWebsiteDataType('raster')).toEqual(['surface']);
    expect(getGPUAnalysisDataKindsForWebsiteDataType('terrain')).toEqual(['surface']);
    expect(getGPUAnalysisWebsiteDataTypes('surface')).toEqual(['raster', 'terrain']);
  });

  it('maps table to attributes, summary and pairs (and regions), attributes first', () => {
    const kinds = getGPUAnalysisDataKindsForWebsiteDataType('table');
    expect(kinds[0]).toBe('attributes');
    expect(kinds).toEqual(expect.arrayContaining(['attributes', 'summary', 'pairs']));
  });

  it('round trips every website type except point-cloud, which is lossy', () => {
    for (const type of GPU_ANALYSIS_WEBSITE_DATA_TYPES) {
      const kinds = getGPUAnalysisDataKindsForWebsiteDataType(type);
      expect(kinds.length, type).toBeGreaterThan(0);
      const comesBack = kinds.some(kind => getGPUAnalysisWebsiteDataTypes(kind).includes(type));
      // point-cloud becomes points, and points come back as points only.
      expect(comesBack, type).toBe(type !== 'point-cloud');
    }
    expect(getGPUAnalysisDataKindsForWebsiteDataType('point-cloud')).toEqual(['points']);
    expect(getGPUAnalysisWebsiteDataTypes('points')).toEqual(['points']);
  });

  it('round trips each ontology kind that has a website type, back to the same kind', () => {
    for (const kind of GPU_SPATIAL_ANALYSIS_DATA_KINDS) {
      for (const type of getGPUAnalysisWebsiteDataTypes(kind)) {
        expect(getGPUAnalysisDataKindsForWebsiteDataType(type), `${kind} via ${type}`).toContain(
          kind
        );
      }
    }
  });

  it('maps catalog geometries to kinds, with none meaning a plain table', () => {
    expect(
      GPU_ANALYSIS_CATALOG_OUTPUT_GEOMETRIES.map(getGPUAnalysisDataKindForCatalogGeometry)
    ).toEqual(['points', 'lines', 'polygons', 'attributes']);
  });

  it('round trips catalog geometry through the ontology kind', () => {
    for (const geometry of GPU_ANALYSIS_CATALOG_INPUT_GEOMETRIES) {
      const kind = getGPUAnalysisDataKindForCatalogGeometry(geometry);
      expect(getGPUAnalysisCatalogGeometryForDataKind(kind, 'input')).toBe(geometry);
      expect(getGPUAnalysisCatalogGeometryForDataKind(kind, 'output')).toBe(geometry);
    }
    // none goes to attributes and attributes comes back as none as an output only.
    expect(getGPUAnalysisCatalogGeometryForDataKind('attributes', 'output')).toBe('none');
    expect(getGPUAnalysisCatalogGeometryForDataKind('attributes', 'input')).toBeUndefined();
  });

  it('says undefined for kinds the catalog cannot carry', () => {
    for (const kind of [
      'surface',
      'network',
      'trajectories',
      'cells',
      'spatial-weights',
      'time-series',
      'parameters',
      'render-geometry'
    ] as const) {
      expect(
        getGPUAnalysisCatalogGeometryForDataKind(kind, 'input'),
        `${kind} input`
      ).toBeUndefined();
      expect(
        getGPUAnalysisCatalogGeometryForDataKind(kind, 'output'),
        `${kind} output`
      ).toBeUndefined();
    }
    for (const kind of ['summary', 'pairs', 'regions'] as const) {
      expect(getGPUAnalysisCatalogGeometryForDataKind(kind, 'output')).toBe('none');
      expect(getGPUAnalysisCatalogGeometryForDataKind(kind, 'input')).toBeUndefined();
    }
  });
});

describe('operator kind supplements', () => {
  const catalogue = getGPUAnalysisOperatorCatalog();
  const catalogueByName = new Map(catalogue.map(operator => [operator.exportName, operator]));

  const BASELINE_OPERATORS = [
    'GPUQuantileLISA',
    'GPUDorlingCartogram',
    'addExcessRiskRecipe',
    'addSpatialRateRecipe',
    'GPUNestedInterpolationValidation',
    'GPUGeometryMeasures',
    'GPUShapeDescriptors',
    'GPUMinimumBounds',
    'GPUMinimumClearance',
    'GPULabelPoint',
    'GPUGeographicDistribution',
    'GPUGeodesicPairs',
    'GPUGeodesicDestination',
    'GPULocalFrameProjection',
    'GPUAffineTransform',
    'GPUGeometryOrientation',
    'GPUGeometryCleanup',
    'GPUVertexSnap',
    'GPUOffsetCurve',
    'GPUGridGenerator',
    'GPUShapeGenerator',
    'GPUDotDensity',
    'GPURandomPointsInPolygon',
    'GPURandomPointsOnLine',
    'GPUPolygonTriangulation',
    'GPUOutlineGeometry',
    'GPUGeometryPredicates',
    'GPUGeometryValidity',
    'GPUCoverageValidity',
    'GPUGreatCircleArcs',
    'GPULineChunk',
    'GPULineSegmentize',
    'GPULineSmooth',
    'GPULineSimplification',
    'GPULineMerge',
    'GPULineClipByPolygon',
    'GPUSharedPaths',
    'GPURectangleClip',
    'GPULinearReferencing',
    'GPULineLocate',
    'GPUSegmentIntersection',
    'GPUCoverageDissolve',
    'GPUCoverageSimplification',
    'GPUHilbertKeys',
    'GPUSpatialJoinCandidates',
    'GPUSpatialJoinPrepared',
    'GPUPointInPolygonJoin',
    'GPUNearestFeatureJoin',
    'GPUBufferSelection',
    'GPUOffsetExpansion',
    'GPUBoundsFilter',
    'GPUNeighborSearch',
    'GPUContiguityWeights',
    'GPULatticeWeights',
    'GPUNearestFeatureWeights',
    'GPUSpatialWeightsTransform',
    'GPUSpatialWeightsAlgebra',
    'GPUSpatialWeightsSummary',
    'GPUSpatialWeightsTranspose',
    'GPUSpatialWeightsValidation',
    'GPUChunkedSpatialWeightsTransform',
    'GPUChunkedSpatialWeightsAlgebra',
    'GPUChunkedSpatialWeightsTranspose',
    'GPUChunkedSpatialWeightsValidation',
    'GPUSpatialLag',
    'GPUMapColoring',
    'GPUPointToCell',
    'GPUCellGeometry',
    'GPUCellTopology',
    'GPUCellCompaction',
    'GPUCellGridPath',
    'GPUCellMeasures',
    'GPUCellCover',
    'GPUCellAggregation',
    'GPUCellRollup',
    'GPUCellPyramid',
    'GPUCellLevelSelection',
    'GPUCellSetOutline',
    'GPUCellTableCompare',
    'GPUPointDensity',
    'GPUKernelDensity',
    'GPULineDensity',
    'GPULineLengthPerPolygon',
    'GPUInverseDistanceWeighting',
    'GPUKriging',
    'GPUFocalStatistics',
    'GPUZonalStatistics',
    'GPURegionMask',
    'GPUPickRegionMask',
    'GPURegionStatistics',
    'GPUSpatialClustering',
    'GPUSpatialKMeans',
    'GPUSpatialWeightsMinimumSpanningTree',
    'GPUSkaterRegions',
    'GPURegionPartitionEvaluation',
    'GPUGroupGeometry',
    'GPUGroupConvexHull',
    'GPULocalOutlierFactor',
    'GPUSimilarLocations',
    'GPUGlobalSpatialStatistics',
    'GPUHotSpotAnalysis',
    'GPULocalMoran',
    'GPULocalPermutationTest',
    'GPUGlobalPermutationTest',
    'GPUNeighborhoodSummary',
    'GPUEmpiricalBayesRates',
    'GPUSpatialEmpiricalBayesRates',
    'GPUAssuncaoReisStandardization',
    'GPUMultipleTestingAdjustment',
    'GPUSpatialScanStatistic',
    'GPUKnoxTest',
    'GPUMantelTest',
    'GPUOrdinaryLeastSquares',
    'GPUSpatialRegressionDiagnostics',
    'GPUSpatialTwoStageLeastSquares',
    'GPUSpatialErrorGM',
    'GPUSpatialLagMaximumLikelihood',
    'GPUSpatialErrorMaximumLikelihood',
    'GPUSpatialWeightsTraces',
    'GPUSpatialLagImpacts',
    'GPUGeographicallyWeightedRegression',
    'GPUGeographicallyWeightedRegressionNonstationarityTest',
    'GPUClassAssignment',
    'GPUClassificationFit',
    'GPUTransitionMatrix',
    'GPUSpatialMarkov',
    'GPULISAMarkov',
    'GPUTransitionDiagnostics',
    'GPUEmergingHotSpots',
    'GPUCatchmentAccessibility',
    'GPUHuffTradeAreas',
    'GPUSegregation',
    'GPUArealInterpolation',
    'GPUPycnophylactic',
    'GPUTrajectoryMetrics',
    'GPUTrajectoryPlayhead',
    'GPUTrajectoryResample',
    'GPUZoneEvents',
    'GPUTrajectoryEncounters',
    'GPUTrajectoryClosestApproach',
    'GPUTrackSimilarity',
    'addClockEncounters',
    'addChangeOfSupportRecipe',
    'addClusterAndOutlineRecipe',
    'addFleetDwellZoneEventsRecipe',
    'addHotSpotAnalysisRecipe',
    'addPeriodComparisonRecipe',
    'addRateClusterMapRecipe',
    'addSpaceTimeHotSpotsRecipe',
    'addSpatialRegressionRecipe',
    'addStraightLineCatchmentsRecipe'
  ];
  // The operators whose catalogue records had no kinds when the supplement rows were written. Other
  // sessions add operators often, so an operator outside this list that resolves as unknown is
  // reported, not failed.
  it('resolves every baseline operator to catalogue, supplement or mixed kinds', () => {
    const unresolved = BASELINE_OPERATORS.filter(name => {
      const operator = catalogueByName.get(name);
      return !operator || resolveGPUAnalysisOperatorKinds(operator).source === 'unknown';
    });
    expect(
      unresolved,
      `baseline operators that are missing or unknown: ${unresolved.join(', ')}`
    ).toEqual([]);
    const baseline = new Set<string>(BASELINE_OPERATORS);
    const newlyUnknown = catalogue
      .filter(
        operator =>
          !baseline.has(operator.exportName) &&
          resolveGPUAnalysisOperatorKinds(operator).source === 'unknown'
      )
      .map(operator => operator.exportName);
    if (newlyUnknown.length > 0) {
      console.info(
        `operators added since the baseline with no kinds and no supplement row: ${newlyUnknown.join(', ')}`
      );
    }
    // Every baseline operator has a row; a row is the only way one of them can resolve.
    for (const name of BASELINE_OPERATORS) {
      expect(Object.hasOwn(GPU_ANALYSIS_OPERATOR_KIND_SUPPLEMENTS, name), name).toBe(true);
    }
  });

  it('has no row for an operator that is absent or that the catalogue describes on both sides', () => {
    const absent = Object.keys(GPU_ANALYSIS_OPERATOR_KIND_SUPPLEMENTS).filter(
      name => !catalogueByName.has(name)
    );
    expect(absent, `stale rows, operator not in the catalogue: ${absent.join(', ')}`).toEqual([]);
    const redundant = Object.keys(GPU_ANALYSIS_OPERATOR_KIND_SUPPLEMENTS).filter(name => {
      const operator = catalogueByName.get(name);
      return operator && operator.inputs.length > 0 && operator.outputs.length > 0;
    });
    expect(redundant, `stale rows, catalogue now has kinds: ${redundant.join(', ')}`).toEqual([]);
  });

  it('uses only valid kinds, each at most once, with outputs empty only for handle operators', () => {
    const valid = new Set<string>(GPU_SPATIAL_ANALYSIS_DATA_KINDS);
    for (const [name, kinds] of Object.entries(GPU_ANALYSIS_OPERATOR_KIND_SUPPLEMENTS)) {
      expect(kinds.inputs.length, `${name} inputs`).toBeGreaterThan(0);
      if (GPU_ANALYSIS_HANDLE_OPERATORS.includes(name)) {
        expect(kinds.outputs, `${name} is a handle operator`).toEqual([]);
      } else {
        expect(kinds.outputs.length, `${name} outputs`).toBeGreaterThan(0);
      }
      for (const list of [kinds.inputs, kinds.outputs]) {
        expect(new Set(list).size, `${name} repeats a kind`).toBe(list.length);
        for (const kind of list) expect(valid.has(kind), `${name}: ${kind}`).toBe(true);
      }
    }
    for (const name of GPU_ANALYSIS_HANDLE_OPERATORS) {
      expect(catalogueByName.has(name), `handle operator ${name} is in the catalogue`).toBe(true);
    }
  });

  it('supplements the baseline operators, all from the gpu-spatial-analysis entry point', () => {
    const supplemented = catalogue.filter(operator =>
      BASELINE_OPERATORS.includes(operator.exportName)
    );
    expect(supplemented.length).toBe(BASELINE_OPERATORS.length);
    expect(new Set(supplemented.map(operator => operator.package))).toEqual(
      new Set(['@luma-contributors/analysis/gpu-spatial-analysis'])
    );
  });

  it('prefers the catalogue, falls back to the supplement and reports unknown without throwing', () => {
    expect(
      resolveGPUAnalysisOperatorKinds({
        exportName: 'GPUKriging',
        inputs: ['network'],
        outputs: ['network']
      })
    ).toEqual({
      inputs: ['network'],
      outputs: ['network'],
      source: 'catalogue',
      inputsSource: 'catalogue',
      outputsSource: 'catalogue'
    });
    const supplemented = resolveGPUAnalysisOperatorKinds({
      exportName: 'GPUKriging',
      inputs: [],
      outputs: []
    });
    expect(supplemented.source).toBe('supplement');
    expect(supplemented.outputs).toContain('surface');
    for (const exportName of ['GPUNotYetCatalogued', '', 'constructor', 'toString', '__proto__']) {
      expect(resolveGPUAnalysisOperatorKinds({exportName, inputs: [], outputs: []})).toEqual({
        inputs: [],
        outputs: [],
        source: 'unknown',
        inputsSource: 'unknown',
        outputsSource: 'unknown'
      });
    }
  });

  it('resolves each side separately when the catalogue lists only one side', () => {
    const onlyOutputs = resolveGPUAnalysisOperatorKinds({
      exportName: 'GPUKriging',
      inputs: [],
      outputs: ['network']
    });
    expect(onlyOutputs.outputs).toEqual(['network']);
    expect(onlyOutputs.outputsSource).toBe('catalogue');
    expect(onlyOutputs.inputs).toEqual(GPU_ANALYSIS_OPERATOR_KIND_SUPPLEMENTS['GPUKriging'].inputs);
    expect(onlyOutputs.inputsSource).toBe('supplement');
    expect(onlyOutputs.source).toBe('mixed');
    const onlyInputs = resolveGPUAnalysisOperatorKinds({
      exportName: 'GPUKriging',
      inputs: ['network'],
      outputs: []
    });
    expect(onlyInputs.inputs).toEqual(['network']);
    expect(onlyInputs.outputs).toEqual(
      GPU_ANALYSIS_OPERATOR_KIND_SUPPLEMENTS['GPUKriging'].outputs
    );
    expect(onlyInputs.source).toBe('mixed');
    // One side known and no row for the other: that side's source stands alone, the other is unknown.
    const noRow = resolveGPUAnalysisOperatorKinds({
      exportName: 'GPUParameterBuffer',
      inputs: [],
      outputs: ['parameters']
    });
    expect(noRow).toEqual({
      inputs: [],
      outputs: ['parameters'],
      source: 'catalogue',
      inputsSource: 'unknown',
      outputsSource: 'catalogue'
    });
  });

  it('gives a handle operator empty outputs from the supplement, so it produces nothing', () => {
    const prepared = resolveGPUAnalysisOperatorKinds({
      exportName: 'GPUSpatialJoinPrepared',
      inputs: [],
      outputs: []
    });
    expect(prepared.outputs).toEqual([]);
    expect(prepared.outputsSource).toBe('supplement');
    expect(prepared.source).toBe('supplement');
    const producers = queryGPUAnalysisOperatorsByKind({output: 'pairs'}).map(
      operator => operator.exportName
    );
    expect(producers).not.toContain('GPUSpatialJoinPrepared');
    expect(queryGPUAnalysisOperatorsByKind({input: 'polygons'}).map(o => o.exportName)).toContain(
      'GPUSpatialJoinPrepared'
    );
    for (const {paths} of [findGPUAnalysisOperatorPaths('points', 'attributes', 3, 500)]) {
      expect(paths.some(path => path.includes('GPUSpatialJoinPrepared'))).toBe(false);
    }
    expect(
      resolveGPUAnalysisOperatorKinds({
        exportName: 'GPUCellLevelSelection',
        inputs: [],
        outputs: []
      }).inputs
    ).toEqual(['attributes', 'parameters']);
  });

  it('follows the catalogue convention for label kinds', () => {
    const outputs = (name: string) => GPU_ANALYSIS_OPERATOR_KIND_SUPPLEMENTS[name].outputs;
    // The catalogue gives OPTICS and HDBSCAN labels attributes; regions are for regionalization.
    expect(catalogueByName.get('GPUOPTICS')?.outputs).toContain('attributes');
    expect(catalogueByName.get('GPUOPTICS')?.outputs).not.toContain('regions');
    expect(outputs('GPUSpatialClustering')).toContain('attributes');
    expect(outputs('GPUSpatialClustering')).not.toContain('regions');
    expect(outputs('GPUSpatialKMeans')).not.toContain('regions');
    expect(outputs('GPUSkaterRegions')).toContain('regions');
    expect(GPU_ANALYSIS_OPERATOR_KIND_SUPPLEMENTS['GPUGroupConvexHull'].inputs).toEqual(
      expect.arrayContaining(['attributes', 'regions'])
    );
  });

  it('cannot be corrupted through the exported table or its rows', () => {
    expect(Object.isFrozen(GPU_ANALYSIS_OPERATOR_KIND_SUPPLEMENTS)).toBe(true);
    expect(Object.isFrozen(GPU_ANALYSIS_OPERATOR_KIND_SUPPLEMENTS['GPUKriging'])).toBe(true);
    expect(() =>
      (
        GPU_ANALYSIS_OPERATOR_KIND_SUPPLEMENTS['GPUKriging'].outputs as GPUSpatialAnalysisDataKind[]
      ).push('network')
    ).toThrow(TypeError);
    expect(() => {
      (GPU_ANALYSIS_OPERATOR_KIND_SUPPLEMENTS as Record<string, unknown>)['GPUExtra'] = {};
    }).toThrow(TypeError);
    expect(Object.isFrozen(GPU_ANALYSIS_HANDLE_OPERATORS)).toBe(true);
    expect(
      resolveGPUAnalysisOperatorKinds({exportName: 'GPUKriging', inputs: [], outputs: []}).outputs
    ).toContain('surface');
  });

  // Evidence for a spread of rows. `ports` are strings that must occur in the source file (relative
  // to modules/analysis/src); the kinds are an independent statement of what those ports imply.
  const EVIDENCE: readonly {
    name: string;
    file: string;
    ports: readonly string[];
    inputs: readonly GPUSpatialAnalysisDataKind[];
    outputs: readonly GPUSpatialAnalysisDataKind[];
  }[] = [
    {
      name: 'GPUQuantileLISA',
      file: 'gpu-spatial-analysis/spatial-autocorrelation/gpu-quantile-lisa.ts',
      ports: [
        'weights: GPUSpatialWeights',
        "values: GraphDataView<'float32'>",
        'membership: GraphDataView'
      ],
      inputs: ['attributes', 'spatial-weights'],
      outputs: ['attributes']
    },
    {
      name: 'GPUDorlingCartogram',
      file: 'gpu-spatial-analysis/dorling-cartogram/gpu-dorling-cartogram.ts',
      ports: ["positions: GraphDataView<'float32x2'>", 'radii: GraphDataView'],
      inputs: ['points', 'attributes'],
      outputs: ['points']
    },
    {
      name: 'GPUShapeDescriptors',
      file: 'gpu-spatial-analysis/shape-descriptors/gpu-shape-descriptors.ts',
      ports: ['polsbyPopper?: GraphDataView', 'ringOffsets'],
      inputs: ['polygons'],
      outputs: ['attributes']
    },
    {
      name: 'GPULabelPoint',
      file: 'gpu-spatial-analysis/label-point/gpu-label-point.ts',
      ports: ["points: GraphDataView<'float32x2'>", 'ringOffsets'],
      inputs: ['polygons'],
      outputs: ['points']
    },
    {
      name: 'GPUGeodesicPairs',
      file: 'gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs.ts',
      ports: [
        "origins: GraphDataView<'float32x2'>",
        "targets: GraphDataView<'float32x2'>",
        'distances?: GraphDataView'
      ],
      inputs: ['points'],
      outputs: ['attributes']
    },
    {
      name: 'GPUGridGenerator',
      file: 'gpu-spatial-analysis/grid-generators/gpu-grid-generator.ts',
      ports: [
        'gridType: GPUGridType',
        "parameters: GraphDataView<'float32'>",
        "centers?: GraphDataView<'float32x2'>"
      ],
      inputs: ['parameters'],
      outputs: ['polygons', 'points']
    },
    {
      name: 'GPUDotDensity',
      file: 'gpu-spatial-analysis/dot-density/gpu-dot-density.ts',
      ports: [
        'GPUPolygonGeometryFields',
        "values: GraphDataView<'float32'>",
        "parameters: GraphDataView<'uint32'>"
      ],
      inputs: ['polygons', 'attributes'],
      outputs: ['points']
    },
    {
      name: 'GPUPolygonTriangulation',
      file: 'analysis-core/polygon-triangulation/gpu-polygon-triangulation.ts',
      ports: ['GPUPolygonGeometryFields', "indices: GraphDataView<'uint32'>"],
      inputs: ['polygons'],
      outputs: ['render-geometry']
    },
    {
      name: 'GPUOutlineGeometry',
      file: 'gpu-spatial-analysis/outline-geometry/gpu-outline-geometry.ts',
      ports: ["geometryType: 'points'", "positions: GraphDataView<'float32x2'>"],
      inputs: ['points', 'lines'],
      outputs: ['render-geometry']
    },
    {
      name: 'GPUGeometryValidity',
      file: 'gpu-spatial-analysis/geometry-validity/gpu-geometry-validity.ts',
      ports: [
        'polygons?: GPUPackedGeometryPort<GPUPolygonGeometryPort>',
        'lines?: GPUPackedGeometryPort<GPULineGeometryPort>',
        "mask: GraphDataView<'uint32'>"
      ],
      inputs: ['polygons', 'lines', 'points'],
      outputs: ['attributes']
    },
    {
      name: 'GPUGreatCircleArcs',
      file: 'gpu-spatial-analysis/line-segmentize/gpu-great-circle-arcs.ts',
      ports: ["sources: GraphDataView<'float32x2'>", 'output: GPULineGeometryOutput'],
      inputs: ['points'],
      outputs: ['lines']
    },
    {
      name: 'GPULineSegmentize',
      file: 'gpu-spatial-analysis/line-segmentize/gpu-line-segmentize.ts',
      ports: ["lineOffsets: GraphDataView<'uint32'>", 'output: GPULineGeometryOutput'],
      inputs: ['lines', 'parameters'],
      outputs: ['lines']
    },
    {
      name: 'GPULineClipByPolygon',
      file: 'gpu-spatial-analysis/line-clip/gpu-line-clip-by-polygon.ts',
      ports: [
        'lines: GPUPackedGeometryPort<GPULineGeometryPort>',
        'polygons: GPUPackedGeometryPort<GPUPolygonGeometryPort>',
        'pieces: GPULineSplitPieces'
      ],
      inputs: ['lines', 'polygons'],
      outputs: ['lines']
    },
    {
      name: 'GPULinearReferencing',
      file: 'gpu-spatial-analysis/linear-referencing/gpu-linear-referencing.ts',
      ports: ["footPoints?: GraphDataView<'float32x2'>", 'measures?: GraphDataView'],
      inputs: ['points', 'lines'],
      outputs: ['attributes', 'points']
    },
    {
      name: 'GPUSpatialJoinCandidates',
      file: 'gpu-spatial-analysis/spatial-join/gpu-spatial-join-candidates.ts',
      ports: ['left: GPUSpatialJoinGeometry', 'output: GPUCompactPairPort'],
      inputs: ['points', 'polygons'],
      outputs: ['pairs']
    },
    {
      name: 'GPUPointInPolygonJoin',
      file: 'gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join.ts',
      ports: [
        'points: GPUPointGeometryPort',
        'polygons: GPUPackedGeometryPort<GPUPolygonGeometryPort>',
        'pointFeatureIds: GPUUint32Rows'
      ],
      inputs: ['points', 'polygons'],
      outputs: ['attributes']
    },
    {
      name: 'GPUNeighborSearch',
      file: 'gpu-spatial-analysis/neighbor-search/gpu-neighbor-search.ts',
      ports: ["positions: GraphDataView<'float32x2'>", 'GPUSpatialWeights} CSR'],
      inputs: ['points'],
      outputs: ['spatial-weights']
    },
    {
      name: 'GPUContiguityWeights',
      file: 'gpu-spatial-analysis/spatial-weights/gpu-contiguity-weights.ts',
      ports: ['polygons: GPUPolygonGeometryPort', 'output: GPUSpatialWeightsOutput'],
      inputs: ['polygons'],
      outputs: ['spatial-weights']
    },
    {
      name: 'GPUSpatialWeightsSummary',
      file: 'gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-summary.ts',
      ports: ['weights: GPUSpatialWeights', "statistics: GraphDataView<'float32'>"],
      inputs: ['spatial-weights'],
      outputs: ['summary']
    },
    {
      name: 'GPUSpatialLag',
      file: 'gpu-spatial-analysis/spatial-weights/gpu-spatial-lag.ts',
      ports: [
        'values: GPUFloat32OrUint32Values',
        'weights: GPUSpatialWeights',
        "output: GraphDataView<'float32'>"
      ],
      inputs: ['attributes', 'spatial-weights'],
      outputs: ['attributes']
    },
    {
      name: 'GPUPointToCell',
      file: 'gpu-spatial-analysis/cell-indexing/gpu-point-to-cell.ts',
      ports: ["positions: GraphDataView<'float32x2'>", "cells: GraphDataView<'uint32x2'>"],
      inputs: ['points'],
      outputs: ['cells']
    },
    {
      name: 'GPUCellCover',
      file: 'gpu-spatial-analysis/cell-cover/gpu-cell-cover.ts',
      ports: [
        'Polyfills polygon features',
        "cells: GraphDataView<'uint32x2'>",
        'featureIds: GraphDataView'
      ],
      inputs: ['polygons'],
      outputs: ['cells']
    },
    {
      name: 'GPUCellTableCompare',
      file: 'gpu-spatial-analysis/cell-table-compare/gpu-cell-table-compare.ts',
      ports: ['before: GPUCellTable', 'after: GPUCellTable', 'delta?: GraphDataView'],
      inputs: ['cells'],
      outputs: ['cells', 'attributes']
    },
    {
      name: 'GPUKernelDensity',
      file: 'gpu-spatial-analysis/point-density/gpu-kernel-density.ts',
      ports: [
        "positions: GraphDataView<'float32x2'>",
        'grid: GPURasterMetadata',
        "values: GraphDataView<'float32'>"
      ],
      inputs: ['points', 'attributes'],
      outputs: ['surface']
    },
    {
      name: 'GPUInverseDistanceWeighting',
      file: 'gpu-spatial-analysis/spatial-interpolation/gpu-inverse-distance-weighting.ts',
      ports: ["positions: GraphDataView<'float32x2'>", 'grid: GPURasterMetadata'],
      inputs: ['points', 'attributes'],
      outputs: ['surface']
    },
    {
      name: 'GPUFocalStatistics',
      file: 'gpu-spatial-analysis/spatial-interpolation/gpu-focal-statistics.ts',
      ports: ['grid: GPURasterMetadata', 'moving-window statistics over a float32 raster'],
      inputs: ['surface'],
      outputs: ['surface']
    },
    {
      name: 'GPUZonalStatistics',
      file: 'gpu-spatial-analysis/zonal-statistics/gpu-zonal-statistics.ts',
      ports: ['features: GPUZonalStatisticsFeatures', "sums?: GraphDataView<'float32'>"],
      inputs: ['points', 'polygons'],
      outputs: ['attributes']
    },
    {
      name: 'GPUPickRegionMask',
      file: 'gpu-spatial-analysis/region-statistics/gpu-pick-region-mask.ts',
      ports: ["picks: GraphDataView<'uint32'>", 'GPURegionMaskOutput'],
      inputs: ['attributes'],
      outputs: ['attributes']
    },
    {
      name: 'GPUSpatialClustering',
      file: 'gpu-spatial-analysis/spatial-clustering/gpu-spatial-clustering.ts',
      ports: ["positions: GraphDataView<'float32x2'>", "labels: GraphDataView<'uint32'>"],
      inputs: ['points'],
      outputs: ['attributes']
    },
    {
      name: 'GPUGroupConvexHull',
      file: 'gpu-spatial-analysis/group-geometry/gpu-group-convex-hull.ts',
      ports: [
        "positions: GraphDataView<'float32x2'>",
        "labels: GraphDataView<'uint32'>",
        'vertexPositions?: GraphDataView'
      ],
      inputs: ['points', 'regions'],
      outputs: ['polygons']
    },
    {
      name: 'GPULocalMoran',
      file: 'gpu-spatial-analysis/spatial-autocorrelation/gpu-local-moran.ts',
      ports: [
        'weights: GPUSpatialWeights',
        'values: GPUFloat32OrUint32Values',
        'zScores: GraphDataView'
      ],
      inputs: ['attributes', 'spatial-weights'],
      outputs: ['attributes']
    },
    {
      name: 'GPUGlobalPermutationTest',
      file: 'gpu-spatial-analysis/permutation-inference/gpu-global-permutation-test.ts',
      ports: ['weights: GPUSpatialWeights', "summary: GraphDataView<'float32'>"],
      inputs: ['attributes', 'spatial-weights'],
      outputs: ['summary']
    },
    {
      name: 'GPUMultipleTestingAdjustment',
      file: 'gpu-spatial-analysis/multiple-testing/gpu-multiple-testing-adjustment.ts',
      ports: ["pValues: GraphDataView<'float32'>", 'adjustedPValues: GraphDataView'],
      inputs: ['attributes'],
      outputs: ['attributes']
    },
    {
      name: 'GPUKnoxTest',
      file: 'gpu-spatial-analysis/space-time-tests/gpu-knox-test.ts',
      ports: [
        'pairs: GPUSpatialWeights',
        "times: GraphDataView<'float32'>",
        'summary: GraphDataView'
      ],
      inputs: ['spatial-weights', 'attributes'],
      outputs: ['summary']
    },
    {
      name: 'GPUOrdinaryLeastSquares',
      file: 'gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares.ts',
      ports: [
        "predictors: GraphDataView<'float32'>",
        "response: GraphDataView<'float32'>",
        'coefficients: GraphDataView'
      ],
      inputs: ['attributes'],
      outputs: ['summary']
    },
    {
      name: 'GPUSpatialErrorMaximumLikelihood',
      file: 'gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood.ts',
      ports: [
        'weights: GPUSpatialWeights',
        "table: GraphDataView<'float32'>",
        'residuals?: GraphDataView'
      ],
      inputs: ['attributes', 'spatial-weights'],
      outputs: ['summary']
    },
    {
      name: 'GPUTransitionMatrix',
      file: 'gpu-spatial-analysis/distribution-dynamics/gpu-transition-matrix.ts',
      ports: ["classes: GraphDataView<'uint32'>", 'periodCount: number'],
      inputs: ['time-series'],
      outputs: ['summary']
    },
    {
      name: 'GPUEmergingHotSpots',
      file: 'gpu-spatial-analysis/emerging-hot-spots/gpu-emerging-hot-spots.ts',
      ports: ['sliceCount: number', 'giZScores: GraphDataView', 'category: GraphDataView'],
      inputs: ['time-series'],
      outputs: ['attributes']
    },
    {
      name: 'GPUCatchmentAccessibility',
      file: 'gpu-spatial-analysis/catchment-accessibility/gpu-catchment-accessibility.ts',
      ports: [
        "supply: GraphDataView<'float32'>",
        'facilityWeights: GPUSpatialWeights',
        'accessibility: GraphDataView'
      ],
      inputs: ['attributes', 'spatial-weights'],
      outputs: ['attributes']
    },
    {
      name: 'GPUArealInterpolation',
      file: 'gpu-spatial-analysis/areal-interpolation/gpu-areal-interpolation.ts',
      ports: [
        'grid: GPURasterMetadata',
        "sourceZones: GraphDataView<'uint32'>",
        'weights: GPUSpatialWeights'
      ],
      inputs: ['surface', 'attributes'],
      outputs: ['spatial-weights']
    },
    {
      name: 'GPUTrajectoryMetrics',
      file: 'gpu-spatial-analysis/trajectory-analysis/gpu-trajectory-metrics.ts',
      ports: ['trackOffsets', 'trackLengths?: GraphDataView'],
      inputs: ['trajectories'],
      outputs: ['attributes']
    },
    {
      name: 'GPUZoneEvents',
      file: 'gpu-spatial-analysis/trajectory-zones/gpu-zone-events.ts',
      ports: [
        'tracks: GPUPackedGeometryPort<GPUTrajectoryGeometryPort>',
        'zones: GPUPackedGeometryPort<GPUPolygonGeometryPort>'
      ],
      inputs: ['trajectories', 'polygons'],
      outputs: ['attributes']
    },
    {
      name: 'GPUTrajectoryClosestApproach',
      file: 'gpu-spatial-analysis/trajectory-encounters/gpu-trajectory-closest-approach.ts',
      ports: ['pairs: GPUTrajectoryClosestApproachPairs', "distance: GraphDataView<'float32'>"],
      inputs: ['trajectories'],
      outputs: ['pairs']
    },
    {
      name: 'GPUTrackSimilarity',
      file: 'gpu-spatial-analysis/track-similarity/gpu-track-similarity.ts',
      ports: [
        'linesA: GPUPackedGeometryPort<GPULineGeometryPort>',
        "pairA: GraphDataView<'uint32'>",
        'hausdorff?: GraphDataView'
      ],
      inputs: ['lines', 'pairs'],
      outputs: ['attributes']
    },
    {
      name: 'addHotSpotAnalysisRecipe',
      file: 'gpu-spatial-analysis/recipes/hot-spot-analysis-recipe.ts',
      ports: ['points on a study area, or a raster', 'GPUHotSpotAnalysis'],
      inputs: ['points', 'surface'],
      outputs: ['cells']
    },
    {
      name: 'addStraightLineCatchmentsRecipe',
      file: 'gpu-spatial-analysis/recipes/straight-line-catchments-recipe.ts',
      ports: ['GPUDistanceField', 'GPURasterZonalStatistics'],
      inputs: ['points', 'surface'],
      outputs: ['surface']
    },
    {
      name: 'GPUCellLevelSelection',
      file: 'gpu-spatial-analysis/cell-aggregation/gpu-cell-level-selection.ts',
      ports: ["levelCounts: GraphDataView<'uint32'>", "activeLevel: GraphDataView<'uint32'>"],
      inputs: ['attributes'],
      outputs: ['render-geometry']
    },
    {
      name: 'GPUSpatialJoinPrepared',
      file: 'gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared.ts',
      ports: ['geometry: GPUSpatialJoinGeometry', 'storage?: GPUSpatialJoinPreparedStorage'],
      inputs: ['points', 'lines', 'polygons'],
      outputs: []
    }
  ];

  it('documents at least 30 rows with a source file and ports that exist in it', () => {
    expect(EVIDENCE.length).toBeGreaterThanOrEqual(30);
    // Spread over families: at least 15 distinct source directories.
    expect(
      new Set(EVIDENCE.map(row => row.file.split('/').slice(0, 2).join('/'))).size
    ).toBeGreaterThanOrEqual(15);
    for (const row of EVIDENCE) {
      const source = readFileSync(SOURCE_ROOT + row.file, 'utf8');
      expect(source, `${row.name} is defined in ${row.file}`).toMatch(
        new RegExp(`export (class|function|const) ${row.name}\\b`)
      );
      for (const port of row.ports) {
        expect(source.includes(port), `${row.name}: "${port}" not found in ${row.file}`).toBe(true);
      }
      const supplement = GPU_ANALYSIS_OPERATOR_KIND_SUPPLEMENTS[row.name];
      expect(supplement, `${row.name} has a supplement row`).toBeDefined();
      for (const kind of row.inputs) expect(supplement.inputs, `${row.name} input`).toContain(kind);
      for (const kind of row.outputs)
        expect(supplement.outputs, `${row.name} output`).toContain(kind);
    }
  });

  it('lets queries find supplemented operators that the catalogue alone would miss', () => {
    const viaKinds = queryGPUAnalysisOperatorsByKind({input: 'points'}).map(
      operator => operator.exportName
    );
    const viaCatalogue = catalogue
      .filter(operator => operator.inputs.includes('points'))
      .map(operator => operator.exportName);
    expect(viaKinds.length).toBeGreaterThan(viaCatalogue.length);
    expect(viaKinds).toContain('GPUKernelDensity');
    expect(viaCatalogue).not.toContain('GPUKernelDensity');
    const densityMakers = queryGPUAnalysisOperatorsByKind({input: 'points', output: 'surface'}).map(
      operator => operator.exportName
    );
    expect(densityMakers).toEqual(expect.arrayContaining(['GPUKernelDensity', 'GPUPointDensity']));
    expect(queryGPUAnalysisOperatorsByKind({}).length).toBeGreaterThan(400);
  });
});

describe('operator paths', () => {
  const kindOf = (name: string) =>
    resolveGPUAnalysisOperatorKinds(
      getGPUAnalysisOperatorCatalog().find(entry => entry.exportName === name)!
    );

  it('lists next operators sorted, including supplemented ones', () => {
    const next = getGPUAnalysisNextOperators('spatial-weights');
    expect(next).toContain('GPULocalMoran');
    expect(next).toContain('GPUSpatialWeightsTransform');
    expect([...next]).toEqual([...next].sort());
    expect(getGPUAnalysisNextOperators('parameters').length).toBeGreaterThan(0);
  });

  it('finds hull or buffer style operators from points to polygons in one step', () => {
    const {paths} = findGPUAnalysisOperatorPaths('points', 'polygons', 1);
    expect(paths.length).toBeGreaterThan(0);
    for (const path of paths) expect(path.length).toBe(1);
    expect(paths.some(([name]) => /Hull|Buffer/.test(name))).toBe(true);
    expect(paths).toContainEqual(['GPUGroupConvexHull']);
  });

  it('finds multi-step chains that use a supplemented operator', () => {
    // cells -> polygons (GPUCellGeometry, supplemented) -> surface (GPUPolygonRasterization).
    const {paths} = findGPUAnalysisOperatorPaths(
      'cells',
      'surface',
      2,
      GPU_ANALYSIS_MAXIMUM_PATH_COUNT_LIMIT
    );
    expect(paths).toContainEqual(['GPUCellGeometry', 'GPUPolygonRasterization']);
  });

  it('returns nothing for an unreachable pair and for the same kind', () => {
    const empty = {paths: [], truncated: false, totalFound: 0};
    // Nothing turns data into settings, and summaries are only consumed by summaries.
    expect(findGPUAnalysisOperatorPaths('points', 'parameters', 3)).toEqual(empty);
    expect(findGPUAnalysisOperatorPaths('summary', 'points', 3)).toEqual(empty);
    expect(findGPUAnalysisOperatorPaths('points', 'points', 3)).toEqual(empty);
  });

  it('caps steps and paths, and says what the cap left out', () => {
    expect(GPU_ANALYSIS_MAXIMUM_PATH_STEPS).toBe(3);
    const capped = findGPUAnalysisOperatorPaths('attributes', 'summary', 99);
    for (const path of capped.paths) {
      expect(path.length).toBeLessThanOrEqual(GPU_ANALYSIS_MAXIMUM_PATH_STEPS);
    }
    expect(capped).toEqual(findGPUAnalysisOperatorPaths('attributes', 'summary', 3));
    const wide = findGPUAnalysisOperatorPaths('points', 'attributes', 1);
    expect(wide.paths.length).toBe(GPU_ANALYSIS_MAXIMUM_PATH_COUNT);
    expect(wide.truncated).toBe(true);
    expect(wide.totalFound).toBeGreaterThan(GPU_ANALYSIS_MAXIMUM_PATH_COUNT);
    for (const path of wide.paths) expect(path.length).toBe(1);
    // Asking for everything returns everything and is not truncated.
    const all = findGPUAnalysisOperatorPaths(
      'points',
      'attributes',
      1,
      GPU_ANALYSIS_MAXIMUM_PATH_COUNT_LIMIT
    );
    if (all.totalFound <= GPU_ANALYSIS_MAXIMUM_PATH_COUNT_LIMIT) {
      expect(all.truncated).toBe(false);
      expect(all.paths.length).toBe(all.totalFound);
    }
    // totalFound never exceeds its bound, and a search that reaches it says so.
    const deep = findGPUAnalysisOperatorPaths('points', 'attributes', 3);
    expect(deep.totalFound).toBeLessThanOrEqual(GPU_ANALYSIS_MAXIMUM_PATH_CANDIDATES);
    expect(deep.truncated).toBe(true);
    const few = findGPUAnalysisOperatorPaths('points', 'polygons', 1, 3);
    expect(few.paths.length).toBe(3);
    expect(few.truncated).toBe(true);
  });

  it('keeps recipes and non-plumbing operators in a capped answer', () => {
    const {paths} = findGPUAnalysisOperatorPaths('points', 'attributes', 1);
    expect(paths.some(([name]) => name.startsWith('add'))).toBe(true);
    // Plumbing (prepare-data) operators are not ahead of analysis operators.
    const names = paths.map(([name]) => name);
    expect(names.indexOf('GPUSpatialJoinCandidates')).toBe(-1);
  });

  it('lists shorter chains first, never repeats an operator and is stable between calls', () => {
    const first = findGPUAnalysisOperatorPaths('points', 'summary', 3);
    expect(findGPUAnalysisOperatorPaths('points', 'summary', 3)).toEqual(first);
    const lengths = first.paths.map(path => path.length);
    expect(lengths).toEqual([...lengths].sort((left, right) => left - right));
    for (const path of first.paths) expect(new Set(path).size).toBe(path.length);
    expect(new Set(first.paths.map(path => path.join('>'))).size).toBe(first.paths.length);
  });

  it('chains consecutive operators whose kinds connect, without revisiting a kind', () => {
    for (const path of findGPUAnalysisOperatorPaths('points', 'polygons', 3).paths) {
      const resolved = path.map(kindOf);
      expect(resolved[0].inputs).toContain('points');
      expect(resolved[resolved.length - 1].outputs).toContain('polygons');
      for (let index = 1; index < resolved.length; index++) {
        const shared = resolved[index - 1].outputs.filter(kind =>
          resolved[index].inputs.includes(kind)
        );
        expect(shared.length, path.join(' > ')).toBeGreaterThan(0);
      }
    }
  });

  it('answers the unreachable worst case from the cache quickly', () => {
    findGPUAnalysisOperatorPaths('points', 'attributes', 3);
    const start = performance.now();
    for (const kind of GPU_SPATIAL_ANALYSIS_DATA_KINDS) {
      findGPUAnalysisOperatorPaths(kind, 'parameters', 3);
    }
    // Generous bound (15 searches); a rebuild of the operator list per call would exceed it.
    expect(performance.now() - start).toBeLessThan(500);
  });

  it('throws for arguments that are not kinds or are out of range', () => {
    for (const bad of [0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => findGPUAnalysisOperatorPaths('points', 'polygons', bad)).toThrow(
        /findGPUAnalysisOperatorPaths maximumSteps/
      );
    }
    for (const bad of [0, 1.5, GPU_ANALYSIS_MAXIMUM_PATH_COUNT_LIMIT + 1, Number.NaN]) {
      expect(() => findGPUAnalysisOperatorPaths('points', 'polygons', 1, bad)).toThrow(
        /findGPUAnalysisOperatorPaths maximumPaths/
      );
    }
    expect(() =>
      findGPUAnalysisOperatorPaths('point' as GPUSpatialAnalysisDataKind, 'polygons', 1)
    ).toThrow(
      /findGPUAnalysisOperatorPaths fromKind "point" is not a data kind; accepted: parameters/
    );
    expect(() =>
      findGPUAnalysisOperatorPaths('points', 'polygon' as GPUSpatialAnalysisDataKind, 1)
    ).toThrow(/toKind "polygon"/);
    expect(() => getGPUAnalysisNextOperators('constructor' as GPUSpatialAnalysisDataKind)).toThrow(
      /getGPUAnalysisNextOperators outputKind "constructor"/
    );
    expect(() =>
      queryGPUAnalysisOperatorsByKind({input: 'nope' as GPUSpatialAnalysisDataKind})
    ).toThrow(/queryGPUAnalysisOperatorsByKind input "nope"/);
    expect(() =>
      queryGPUAnalysisOperatorsByKind({output: 'point' as GPUSpatialAnalysisDataKind})
    ).toThrow(/queryGPUAnalysisOperatorsByKind output "point"/);
    expect(queryGPUAnalysisOperatorsByKind({input: undefined}).length).toBeGreaterThan(400);
  });
});

describe('vocabulary input checks and immutability', () => {
  it('throws for names that are not kinds, types or geometries, including inherited keys', () => {
    for (const bad of [
      'constructor',
      'toString',
      '__proto__',
      'hasOwnProperty',
      'point',
      '',
      'Points'
    ]) {
      const asKind = bad as GPUSpatialAnalysisDataKind;
      expect(() => getGPUAnalysisWebsiteDataTypes(asKind), bad).toThrow(
        /getGPUAnalysisWebsiteDataTypes kind/
      );
      expect(() => getGPUAnalysisDataKindRecord(asKind), bad).toThrow(
        /getGPUAnalysisDataKindRecord kind/
      );
      expect(() => getGPUAnalysisCatalogGeometryForDataKind(asKind), bad).toThrow(
        /getGPUAnalysisCatalogGeometryForDataKind kind/
      );
      expect(
        () => getGPUAnalysisDataKindsForWebsiteDataType(bad as GPUAnalysisWebsiteDataType),
        bad
      ).toThrow(/getGPUAnalysisDataKindsForWebsiteDataType type/);
      if (bad !== '' && bad !== 'point') {
        expect(
          () => getGPUAnalysisDataKindForCatalogGeometry(bad as GPUAnalysisCatalogOutputGeometry),
          bad
        ).toThrow(/getGPUAnalysisDataKindForCatalogGeometry geometry/);
      }
    }
    expect(() => getGPUAnalysisCatalogGeometryForDataKind('points', 'both' as 'input')).toThrow(
      /role "both" is not one of: input, output/
    );
    expect(() =>
      getGPUAnalysisDataKindForCatalogGeometry(
        undefined as unknown as GPUAnalysisCatalogOutputGeometry
      )
    ).toThrow(/is not one of: point, line, polygon, none/);
  });

  it('cannot be corrupted by a caller', () => {
    const table = getGPUAnalysisDataKindsForWebsiteDataType(
      'table'
    ) as GPUSpatialAnalysisDataKind[];
    expect(() => table.push('surface')).toThrow(TypeError);
    expect(getGPUAnalysisDataKindsForWebsiteDataType('table')).toEqual([
      'attributes',
      'summary',
      'pairs',
      'regions'
    ]);
    expect(() => (getGPUAnalysisWebsiteDataTypes('surface') as string[]).push('table')).toThrow(
      TypeError
    );
    expect(() => {
      (GPU_ANALYSIS_WEBSITE_TYPE_KINDS as Record<string, unknown>)['extra'] = [];
    }).toThrow(TypeError);
    expect(() => {
      (GPU_ANALYSIS_WEBSITE_DATA_TYPES as unknown as string[]).push('x');
    }).toThrow(TypeError);
    expect(() => {
      (GPU_ANALYSIS_CATALOG_OUTPUT_GEOMETRIES as unknown as string[]).pop();
    }).toThrow(TypeError);
    expect(() => {
      (GPU_ANALYSIS_DATA_KIND_RECORDS as GPUAnalysisDataKindRecord[]).pop();
    }).toThrow(TypeError);
    expect(() => {
      (GPU_ANALYSIS_DATA_KIND_RECORDS[0] as {label: string}).label = 'changed';
    }).toThrow(TypeError);
    const loadable = listGPUAnalysisLoadableDataKinds();
    loadable.pop();
    expect(listGPUAnalysisLoadableDataKinds().length).toBe(loadable.length + 1);
  });
});

describe('operator path ranking beyond the candidate limit', () => {
  it('returns the globally best 500 cells-to-network chains', () => {
    // Independent exhaustive enumeration from the review probe, without the production search cap.
    const operators = getGPUAnalysisOperatorCatalog()
      .filter(operator => operator.role !== 'helper')
      .map(operator => ({
        name: operator.exportName,
        kinds: resolveGPUAnalysisOperatorKinds(operator),
        plumbing: getGPUAnalysisOperatorQuestions(operator)[0] === 'prepare-data',
        recipe: /^add[A-Z]/.test(operator.exportName)
      }));
    const byName = new Map(operators.map(operator => [operator.name, operator]));
    const chains = new Map<string, string[]>();
    const enumerate = (
      kind: GPUSpatialAnalysisDataKind,
      chain: string[],
      visitedKinds: ReadonlySet<GPUSpatialAnalysisDataKind>
    ): void => {
      if (chain.length === 3) return;
      for (const operator of operators) {
        if (!operator.kinds.inputs.includes(kind) || chain.includes(operator.name)) continue;
        for (const output of operator.kinds.outputs) {
          if (visitedKinds.has(output)) continue;
          const next = [...chain, operator.name];
          if (output === 'network') chains.set(next.join('\u0000'), next);
          else enumerate(output, next, new Set([...visitedKinds, output]));
        }
      }
    };
    enumerate('cells', [], new Set(['cells']));
    const count = (chain: readonly string[], property: 'plumbing' | 'recipe'): number =>
      chain.reduce((sum, name) => sum + Number(byName.get(name)?.[property]), 0);
    const compareNames = (left: string, right: string): number =>
      left < right ? -1 : left > right ? 1 : 0;
    const expected = [...chains.values()].sort(
      (left, right) =>
        left.length - right.length ||
        count(left, 'plumbing') - count(right, 'plumbing') ||
        count(right, 'recipe') - count(left, 'recipe') ||
        compareNames(left.join('\u0000'), right.join('\u0000'))
    );
    expect(chains.size).toBeGreaterThan(GPU_ANALYSIS_MAXIMUM_PATH_CANDIDATES);
    const result = findGPUAnalysisOperatorPaths('cells', 'network', 3, 500);
    expect(result.paths).toEqual(expected.slice(0, 500));
    expect(result).toMatchObject({
      totalFound: GPU_ANALYSIS_MAXIMUM_PATH_CANDIDATES,
      truncated: true
    });
    expect(findGPUAnalysisOperatorPaths('cells', 'network', 3).paths).toEqual(
      expected.slice(0, 50)
    );
  }, 60000);
});
