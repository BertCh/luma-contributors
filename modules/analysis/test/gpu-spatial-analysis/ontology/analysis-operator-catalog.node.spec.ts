// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync} from 'node:fs';
import {resolve} from 'node:path';
import {expect, it} from 'vitest';
import * as analysisCore from '../../../src/analysis-core/index';
import * as computationRecord from '../../../src/computation-record/index';
import * as dataframeAnalysis from '../../../src/dataframe-analysis/index';
import * as geospatialAnalysis from '../../../src/geospatial-analysis/index';
import * as gpuNetwork from '../../../src/gpu-network/index';
import * as gpuSpatialAnalysis from '../../../src/gpu-spatial-analysis/index';
import * as gpuTerrain from '../../../src/gpu-terrain/index';
import * as pointcloudAnalysis from '../../../src/pointcloud-analysis/index';
import * as rasterAnalysis from '../../../src/raster-analysis/index';
import {
  REUSED_ANALYSIS_OPERATORS,
  GPU_SPATIAL_ANALYSIS_AUDIT,
  GPU_SPATIAL_ANALYSIS_CAPABILITIES,
  GPU_SPATIAL_ANALYSIS_OPERATORS,
  getGPUAnalysisCapabilityCatalog,
  getGPUAnalysisOperator,
  getGPUAnalysisOperatorCatalog,
  queryGPUAnalysisOperators,
  queryGPUSpatialAnalysisOperators
} from '../../../src/gpu-spatial-analysis/ontology/index';
import type {GPUAnalysisPackage} from '../../../src/gpu-spatial-analysis/ontology/index';
import {FITTING_KERNEL_OPERATORS} from '../../../src/gpu-spatial-analysis/ontology/fitting-kernel-catalog';
import {SPATIOTEMPORAL_REVIEW_OPERATORS} from '../../../src/gpu-spatial-analysis/ontology/spatiotemporal-review-catalog';
import {
  describeOutputs,
  searchGPUAnalysisTasks
} from '../../../src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog';

it('DST-11 ranks task aliases and keeps upstream tasks outside the analysis census', () => {
  expect(searchGPUAnalysisTasks('find similar pixels')[0]?.identity.exportName).toBe(
    'GPURasterEmbeddingNearestPixels'
  );
  expect(
    searchGPUAnalysisTasks('change point')
      .slice(0, 2)
      .map(result => result.identity.exportName)
      .sort()
  ).toEqual(['GPULocationChangePoints', 'GPUPettittTest']);
  expect(searchGPUAnalysisTasks('NDVI')[0]).toMatchObject({
    owner: 'upstream',
    identity: {package: '@luma.gl/graph/gpu-raster', exportName: 'GPURasterNDVI'}
  });
  expect(getGPUAnalysisOperator('GPURasterNDVI')).toBeUndefined();
  expect(
    searchGPUAnalysisTasks('Pettitt', {question: 'what-changed'})[0]?.identity.exportName
  ).toBe('GPUPettittTest');
  for (const search of ['Pettitt', 'similar pixels']) {
    expect(queryGPUAnalysisOperators({search}).length).toBeGreaterThan(0);
  }
});

it('UCO-17 describes reserved and readable output lengths explicitly', () => {
  const packageName = '@luma-contributors/analysis/gpu-spatial-analysis' as const;
  const ols = describeOutputs(
    {package: packageName, exportName: 'GPUOrdinaryLeastSquares'},
    'default',
    {predictorCount: 2, rowCount: 10}
  );
  expect(ols.declaration).toBe('declared');
  expect(ols.outputs.find(output => output.path === 'output.coefficients')).toMatchObject({
    format: 'float32',
    capacity: 16,
    activeLength: 3,
    minimumCapacity: 3
  });
  const line = describeOutputs({package: packageName, exportName: 'GPULineChunk'}, 'chunk', {
    lineCapacity: 5,
    vertexCapacity: 20
  });
  expect(line.outputs.find(output => output.path === 'output.lineOffsets')).toMatchObject({
    capacity: 6,
    activeLengthRule: 'output.status.count + 1',
    status: ['output.status.count', 'output.status.overflow']
  });
  expect(
    describeOutputs({package: packageName, exportName: 'GPUHotSpot'}, 'default', {}).declaration
  ).toBe('absent');
  expect(() =>
    describeOutputs({package: packageName, exportName: 'GPULineChunk'}, 'typo', {})
  ).toThrow(/chunk.*substring/);
});

it('FRM-13 links all graph pairs and array references to public qualified exports', () => {
  let pairs = 0;
  const arrays = new Set<string>();
  for (const operator of getGPUAnalysisOperatorCatalog()) {
    if (operator.forms?.cpu && operator.forms.gpu) pairs++;
    for (const graphForm of [operator.forms?.cpu, operator.forms?.gpu]) {
      if (graphForm)
        expect(
          ENTRY_POINTS[graphForm.package]?.[graphForm.exportName],
          graphForm.exportName
        ).toBeTypeOf('function');
    }
    for (const form of operator.forms?.arrays ?? []) {
      arrays.add(`${form.package}:${form.exportName}`);
      expect(ENTRY_POINTS[form.package]?.[form.exportName], form.exportName).toBeTypeOf('function');
    }
  }
  const expectedPairs = Object.values(ENTRY_POINTS).flatMap(entry =>
    Object.keys(entry).filter(name => /^CPU\w+Operation$/.test(name))
  );
  const expectedArrays = Object.entries(ENTRY_POINTS).flatMap(([packageName, entry]) =>
    Object.keys(entry)
      .filter(name => /^compute\w+OnCPU$/.test(name))
      .map(name => `${packageName}:${name}`)
  );
  expect(pairs).toBe(expectedPairs.length);
  expect([...arrays].sort()).toEqual(expectedArrays.sort());
});

it('SCL-16 distinguishes crossover, allocation and enforced route bounds', () => {
  for (const exportName of [
    'GPUSpatialJoinPrepared',
    'GPUIslandAttachment',
    'GPUSpatialWeightsTraces',
    'GPURankMarkov',
    'GPUSpatialInteraction',
    'GPUGeographicDistribution',
    'GPUSpatialPredicateJoin',
    'GPUNeighborSearch',
    'GPUCellPyramid',
    'GPUPointDensity',
    'GPUKernelDensity',
    'GPUClassBreaks',
    'GPUExpectationBasedSpaceTimeScan'
  ]) {
    const routes = getGPUAnalysisOperator(exportName)?.routes;
    expect(routes?.length, exportName).toBeGreaterThan(0);
    for (const route of routes ?? []) {
      expect(existsSync(route.source), route.source).toBe(true);
      expect(route.appliesWhen).toBeTruthy();
      expect(route.accuracyImplications).toBeTruthy();
      expect(
        route.maximum !== undefined ||
          route.minimum !== undefined ||
          route.conditionExpression !== undefined ||
          route.byteExpression !== undefined
      ).toBe(true);
      if (route.planner) {
        const planner: unknown = ENTRY_POINTS[route.planner.package]?.[route.planner.exportName];
        expect(
          (planner as {getExecutionPlan?: unknown})?.getExecutionPlan,
          `${route.planner.package}:${route.planner.exportName}`
        ).toBeTypeOf('function');
      }
    }
  }
  expect(getGPUAnalysisOperator('GPUGeographicDistribution')?.routes?.[0]).toMatchObject({
    kind: 'auto-crossover',
    maximum: 64
  });
  expect(getGPUAnalysisOperator('GPUPointDensity')?.routes?.[0]).toMatchObject({
    kind: 'allocation',
    scope: 'binding'
  });
});

it('FRM-13 qualifies prepared/chunked forms and conditional recipe stages', () => {
  for (const operator of getGPUAnalysisOperatorCatalog()) {
    for (const form of [
      ...(operator.forms?.variants ?? []),
      ...(operator.forms?.recipeStages ?? [])
    ]) {
      expect(ENTRY_POINTS[form.package]?.[form.exportName], form.exportName).toBeTypeOf('function');
    }
  }
  expect(
    getGPUAnalysisOperator('GPUSpatialWeightsValidation')?.forms?.variants?.[0]?.exportName
  ).toBe('GPUChunkedSpatialWeightsValidation');
  expect(
    getGPUAnalysisOperator('GPUContiguityWeights')?.forms?.arrays?.map(form => form.exportName)
  ).toContain('computePolygonContiguityOnCPU');
});

it('UCO-17 rejects missing or invalid sizes and freezes declarations', () => {
  const identity = {
    package: '@luma-contributors/analysis/gpu-spatial-analysis',
    exportName: 'GPUOrdinaryLeastSquares'
  } as const;
  for (const predictorCount of [0, 16, Number.NaN, 1.5]) {
    expect(() => describeOutputs(identity, 'default', {predictorCount, rowCount: 10})).toThrow(
      /predictorCount/
    );
  }
  expect(() => describeOutputs(identity, 'default', {predictorCount: 2})).toThrow(/rowCount/);
  const description = describeOutputs(identity, 'default', {predictorCount: 15, rowCount: 10});
  expect(description.outputs[0]?.activeLength).toBe(16);
  expect(Object.isFrozen(description.outputs[0])).toBe(true);
});

const ENTRY_POINTS: Record<string, Record<string, unknown>> = {
  '@luma-contributors/analysis/analysis-core': analysisCore,
  '@luma-contributors/analysis/computation-record': computationRecord,
  '@luma-contributors/analysis/gpu-spatial-analysis': gpuSpatialAnalysis,
  '@luma-contributors/analysis/gpu-network': gpuNetwork,
  '@luma-contributors/analysis/gpu-terrain': gpuTerrain,
  '@luma-contributors/analysis/raster-analysis': rasterAnalysis,
  '@luma-contributors/analysis/dataframe-analysis': dataframeAnalysis,
  '@luma-contributors/analysis/pointcloud-analysis': pointcloudAnalysis,
  '@luma-contributors/analysis/geospatial-analysis': geospatialAnalysis
};

/** Classes that declare getCommandNodes, and add*Recipe functions, exported by an entry point. */
function listExportedContributors(entry: Record<string, unknown>): string[] {
  return Object.entries(entry)
    .filter(([name, value]) => {
      if (typeof value !== 'function') return false;
      if (/^add\w+Recipe$/.test(name)) return true;
      const prototype = (value as {prototype?: Record<string, unknown>}).prototype;
      return typeof prototype?.['getCommandNodes'] === 'function';
    })
    .map(([name]) => name)
    .sort();
}

it('has a catalog entry for every exported contributor and recipe of every entry point', () => {
  const missing: Record<string, string[]> = {};
  for (const [packageName, entry] of Object.entries(ENTRY_POINTS)) {
    const names = listExportedContributors(entry).filter(
      exportName =>
        !getGPUAnalysisOperator({package: packageName as GPUAnalysisPackage, exportName})
    );
    if (names.length > 0) missing[packageName] = names;
  }
  // A new contributor or add*Recipe export needs an entry in the operator catalog.
  expect(missing).toEqual({});
});

it('resolves every catalog operator from the entry point it declares', () => {
  const operators = getGPUAnalysisOperatorCatalog();
  for (const operator of operators) {
    expect(
      ENTRY_POINTS[operator.package]?.[operator.exportName],
      `${operator.package}: ${operator.exportName}`
    ).toBeTypeOf('function');
  }
  expect(
    new Set(operators.map(({package: packageName, exportName}) => `${packageName}:${exportName}`))
      .size
  ).toBe(operators.length);
});

it('keeps the original exports unchanged and adds the other entry points beside them', () => {
  // Consumers (the website inventory generator, the kepler catalog bindings) enumerate these.
  expect(GPU_SPATIAL_ANALYSIS_OPERATORS).toHaveLength(261);
  expect(GPU_SPATIAL_ANALYSIS_CAPABILITIES).toHaveLength(49);
  expect(GPU_SPATIAL_ANALYSIS_AUDIT.operatorCount).toBe(261);
  const packages = new Set(GPU_SPATIAL_ANALYSIS_OPERATORS.map(({package: name}) => name));
  // GPUParameterBuffer is catalogued here but exported only from analysis-core (STD-09).
  expect([...packages].sort()).toEqual([
    '@luma-contributors/analysis/analysis-core',
    '@luma-contributors/analysis/gpu-spatial-analysis',
    '@luma-contributors/analysis/gpu-terrain'
  ]);

  const all = getGPUAnalysisOperatorCatalog();
  // 226 operators of the other entry points, then the operators of `gpu-spatial-analysis` added
  // after the original 261 (spatiotemporal review, fitting kernels, reused analysis operators).
  expect(all.length).toBe(
    261 +
      226 +
      SPATIOTEMPORAL_REVIEW_OPERATORS.length +
      FITTING_KERNEL_OPERATORS.length +
      REUSED_ANALYSIS_OPERATORS.length +
      2 + // Named reference and local tessellation routes added after the historical catalog.
      1 + // Qualified network addVisibilityGraphRecipe, appended separately.
      1 // Point-cloud coverage validity, registered alongside ground/surface operations.
  );
  expect(
    all
      .slice(0, 261)
      .map(({forms: _forms, outputDeclarations: _outputs, routes: _routes, ...operator}) => ({
        ...operator,
        status: operator.status === 'unknown' ? [] : operator.status
      }))
  ).toEqual(GPU_SPATIAL_ANALYSIS_OPERATORS);
  const capabilities = getGPUAnalysisCapabilityCatalog();
  expect(capabilities.slice(0, 49).map(({id}) => id)).toEqual(
    GPU_SPATIAL_ANALYSIS_CAPABILITIES.map(({id}) => id)
  );
  expect(new Set(capabilities.map(({id}) => id)).size).toBe(capabilities.length);
  for (const operator of all) {
    expect(capabilities.some(({id}) => id === operator.primaryCapability)).toBe(true);
  }
});

it('names the new capability families for what they do, never for a development wave', () => {
  const original = new Set(GPU_SPATIAL_ANALYSIS_CAPABILITIES.map(({id}) => id));
  for (const capability of getGPUAnalysisCapabilityCatalog()) {
    if (original.has(capability.id)) continue;
    expect(capability.id).toMatch(/^[a-z0-9]+(-[a-z0-9]+)*$/);
    expect(capability.id).not.toMatch(/^(expansion|coverage|wave|tranche|platform)-/);
    expect(capability.operators.length).toBeGreaterThan(0);
  }
});

it('records a tier for every operator and the limits of every reference operator', () => {
  for (const operator of getGPUAnalysisOperatorCatalog()) {
    expect(['scale', 'reference', 'unknown'], operator.exportName).toContain(operator.tier);
    if (operator.tier === 'reference') {
      expect(
        (operator.limits?.length ?? 0) > 0 || Boolean(operator.limitation),
        `${operator.exportName} is a reference operator and must record its limits`
      ).toBe(true);
    }
    for (const limit of operator.limits ?? []) {
      expect(limit.maximum, `${operator.exportName} ${limit.name}`).toBeGreaterThan(0);
      expect(limit.reason, `${operator.exportName} ${limit.name}`).toBeTruthy();
      if (limit.source) {
        expect(existsSync(resolve(limit.source)), `${operator.exportName}: ${limit.source}`).toBe(
          true
        );
      }
    }
  }
  const reference = queryGPUAnalysisOperators({tier: 'reference'}).map(
    ({exportName}) => exportName
  );
  expect(reference).toContain('GPUNetworkFacilityOptimization');
  expect(reference).not.toContain('GPUNetworkFlow');
  expect(reference).not.toContain('GPUNetworkReachability');
});

it('records the facility search limit as candidates, not demand rows', () => {
  const facility = getGPUAnalysisOperator('GPUNetworkFacilityOptimization');
  expect(facility?.limits?.find(({name}) => name === 'candidateCount')?.maximum).toBe(2 ** 20);
});

it('describes added operators with a summary, existing documentation and a spec', () => {
  const added = getGPUAnalysisOperatorCatalog().slice(261);
  for (const operator of added) {
    expect(operator.summary, operator.exportName).toBeTruthy();
    expect(existsSync(resolve(operator.evidence.tests)), operator.evidence.tests).toBe(true);
    expect(
      existsSync(resolve(operator.evidence.documentation)),
      operator.evidence.documentation
    ).toBe(true);
  }
});

it('marks temporal operators as 2d+time', () => {
  const temporal = queryGPUAnalysisOperators({dimensionality: '2d+time'}).map(
    ({exportName}) => exportName
  );
  for (const name of [
    'GPUNetworkTimeRouting',
    'GPUMapMatching',
    'GPUCalendarBuckets',
    'GPUTemporalReduction',
    'GPUTimeWindowFilter',
    'GPUSpaceTimeVariogram',
    'GPUChangeDetection',
    'GPURasterTemporal',
    'GPURasterSeasonalBreak',
    'GPUSolarIrradiance',
    // TC-08: operators of gpu-spatial-analysis that read times or tracks.
    'GPUKnoxTest',
    'GPUMantelTest',
    'GPUEmergingHotSpots',
    'GPUSpatialMarkov',
    'GPUSpatialPanelRegression',
    'GPUTrajectoryMetrics',
    'GPUTrackSimilarity',
    'addSpaceTimeHotSpotsRecipe',
    'addPeriodComparisonRecipe',
    ...SPATIOTEMPORAL_REVIEW_OPERATORS.filter(
      ({dimensionality}) => dimensionality === '2d+time'
    ).map(({exportName}) => exportName)
  ]) {
    expect(temporal, name).toContain(name);
  }
  // Every trajectory contributor is time-aware.
  const untagged = getGPUAnalysisOperatorCatalog()
    .filter(({exportName}) => /^GPUTraj|^GPUTrack|^GPUMobility|^GPUModalSplit/.test(exportName))
    .filter(({dimensionality}) => dimensionality !== '2d+time')
    .map(({exportName}) => exportName);
  expect(untagged).toEqual([]);
});

it('finds time-series consumers of the dataframe and raster entry points by data kind', () => {
  const consumers = queryGPUAnalysisOperators({input: 'time-series'}).map(
    ({exportName}) => exportName
  );
  for (const name of [
    'GPUCalendarBuckets',
    'GPUCubeLayoutTranspose',
    'GPUPeriodIndex',
    'GPUTimeWordsRelative',
    'GPUValueCube',
    'GPUCellPeriodKeys',
    'GPURasterTemporal',
    'GPUChangeDetection',
    'GPUSpaceTimeCrossCorrelation',
    'GPUSpaceTimeKriging'
  ]) {
    expect(consumers, name).toContain(name);
  }
});

it('lists the parameter buffer as a helper and leaves it out of operator queries', () => {
  expect(getGPUAnalysisOperator('GPUParameterBuffer')?.role).toBe('helper');
  expect(queryGPUAnalysisOperators().map(({exportName}) => exportName)).not.toContain(
    'GPUParameterBuffer'
  );
  expect(
    queryGPUAnalysisOperators({includeHelpers: true}).map(({exportName}) => exportName)
  ).toContain('GPUParameterBuffer');
  // Counts of the original exports are unchanged: the helper is still listed.
  expect(GPU_SPATIAL_ANALYSIS_AUDIT.helperOperatorCount).toBe(1);
});

it('queries operators by package, tier and capability', () => {
  const network = queryGPUAnalysisOperators({
    package: '@luma-contributors/analysis/gpu-network'
  });
  expect(network.length).toBeGreaterThanOrEqual(37);
  expect(network.map(({exportName}) => exportName)).toContain('GPUNetworkReachability');
  expect(
    queryGPUAnalysisOperators({primaryCapability: 'pointcloud-ground'}).map(
      ({exportName}) => exportName
    )
  ).toEqual([
    'GPUPointCloudGround',
    'GPUPointCloudSMRF',
    'GPUPointCloudSurfaces',
    'GPUPointCloudCoverageValidity',
    'GPUPointCloudHeight'
  ]);
});

it('classifies extracted preparation and exact summaries as operations, and their arithmetic/upload building blocks as helpers', () => {
  for (const operator of REUSED_ANALYSIS_OPERATORS) {
    expect(getGPUAnalysisOperator(operator.exportName)).toBe(operator);
    expect(operator.evidence.contract).not.toBe('unaudited');
  }
  for (const helper of [
    'makeDecimal128WordVectors',
    'makeDecimal128SelectionExpression',
    'makeInteger64RangeSelectionExpression',
    'mergeNumericNarrowingReports',
    'runWideNumericKernel',
    'splitBigIntToWords'
  ] as const) {
    expect(dataframeAnalysis[helper]).toBeTypeOf('function');
    expect(getGPUAnalysisOperator(helper)).toBeUndefined();
  }
});

it('RCP-107 discovers both named tessellation routes in the qualified urban family', () => {
  const packageName = '@luma-contributors/analysis/gpu-spatial-analysis' as const;
  const cases = [
    [
      'addBuildingTessellationReferenceRecipe',
      ['lines', 'polygons', 'attributes'],
      [
        'count',
        'requiredCount',
        'overflow',
        'candidateOverflow',
        'requiredRingCount',
        'ringCount',
        'uncertainCount',
        'nodedSegmentRequiredCount'
      ],
      'dissolved polygon rings',
      'result.outputs[i].status',
      64
    ],
    [
      'addLocalBuildingTessellationRecipe',
      ['lines', 'polygons', 'attributes', 'parameters'],
      ['count', 'requiredCount', 'overflow', 'candidateOverflow', 'invalidCount'],
      'convex polygon pieces',
      'result.status',
      1000000
    ]
  ] as const;
  const family = getGPUAnalysisCapabilityCatalog().find(
    record => record.id === 'coverage-urban-morphology'
  );
  for (const [exportName, inputs, status, representation, statusPath, maximum] of cases) {
    const operator = getGPUAnalysisOperator({package: packageName, exportName});
    expect(operator, exportName).toBeDefined();
    expect(operator?.primaryCapability).toBe('coverage-urban-morphology');
    expect(operator?.inputs).toEqual(inputs);
    expect(operator?.outputs).toEqual(['polygons', 'attributes', 'summary']);
    expect(operator?.status).toEqual(status);
    expect(operator?.summary).toContain(representation);
    expect(operator?.limitation).toContain(statusPath);
    expect(
      operator?.limits?.find(limit => limit.name === 'buildingCount (per call)')?.maximum
    ).toBe(maximum);
    expect(family?.operators).toContain(exportName);
    expect(family?.operatorIdentities).toContainEqual({package: packageName, exportName});
  }
});

it('GEO-129 records package-qualified CPU and GPU operation forms without extra contributors', () => {
  const packageName = '@luma-contributors/analysis/gpu-spatial-analysis' as const;
  for (const [exportName, cpu, gpu] of [
    [
      'GPUConstrainedDelaunayTriangulation',
      'CPUConstrainedDelaunayTriangulationOperation',
      'GPUConstrainedDelaunayTriangulationOperation'
    ],
    [
      'GPUDelaunayTessellation',
      'CPUDelaunayTessellationOperation',
      'GPUDelaunayTessellationOperation'
    ],
    ['GPUMinimumBounds', 'CPUMinimumBoundsOperation', 'GPUMinimumBoundsOperation'],
    [
      'GPUPolygonTriangulation',
      'CPUPolygonTriangulationOperation',
      'GPUPolygonTriangulationOperation'
    ]
  ]) {
    const operator = getGPUAnalysisOperator({package: packageName, exportName});
    expect(operator?.forms, exportName).toMatchObject({
      cpu: {package: packageName, exportName: cpu},
      gpu: {package: packageName, exportName: gpu}
    });
    for (const form of [operator?.forms?.cpu, operator?.forms?.gpu]) {
      expect(form).toBeDefined();
      if (!form) throw new Error('Missing graph form');
      expect(ENTRY_POINTS[form.package]?.[form.exportName], form.exportName).toBeTypeOf('function');
      expect(getGPUAnalysisOperator(form)).toBeUndefined();
    }
  }
  expect(GPU_SPATIAL_ANALYSIS_OPERATORS).toHaveLength(261);
});

it('review B4 queries the dispatcher by its route kinds without a network result', () => {
  const names = [
    'addBuildingTessellationRecipe',
    'addBuildingTessellationReferenceRecipe',
    'addLocalBuildingTessellationRecipe'
  ];
  // Both routes accept building attributes. Only local props accept a parameter view.
  for (const [input, expected] of [
    ['attributes', names],
    ['parameters', ['addBuildingTessellationRecipe', 'addLocalBuildingTessellationRecipe']]
  ] as const) {
    expect(
      queryGPUAnalysisOperators({input, output: 'summary'})
        .map(record => record.exportName)
        .filter(name => names.includes(name))
        .sort()
    ).toEqual([...expected].sort());
  }
  for (const query of [queryGPUAnalysisOperators, queryGPUSpatialAnalysisOperators]) {
    expect(
      query({input: 'parameters', output: 'summary'}).map(record => record.exportName)
    ).toContain('addBuildingTessellationRecipe');
    expect(
      query({input: 'points', output: 'network'}).map(record => record.exportName)
    ).not.toContain('addBuildingTessellationRecipe');
  }
  expect(getGPUAnalysisOperator('addBuildingTessellationRecipe')?.inputs).toEqual([
    'lines',
    'polygons',
    'attributes',
    'parameters'
  ]);
  expect(getGPUAnalysisOperator('addBuildingTessellationRecipe')?.outputs).toEqual([
    'polygons',
    'attributes',
    'summary'
  ]);
});
