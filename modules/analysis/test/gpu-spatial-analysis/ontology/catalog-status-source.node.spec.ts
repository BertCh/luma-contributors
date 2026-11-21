// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes,
  type WGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUIslandAttachment} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment';
import {createWeightsLossNodes} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights-loss';
import {readdirSync, readFileSync} from 'node:fs';
import {join} from 'node:path';
import {
  getGPUAnalysisOperator,
  getGPUAnalysisOperatorCatalog,
  queryGPUAnalysisOperators
} from '../../../src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog';
import {
  getGPUSpatialAnalysisOperator,
  queryGPUSpatialAnalysisOperators
} from '../../../src/gpu-spatial-analysis/ontology/spatial-analysis-operators';

import {AUDITED_EMPTY_OPERATOR_STATUSES} from '../../../src/gpu-spatial-analysis/ontology/all-entry-points/operator-status-audits';

const SOURCE_ROOT = 'modules/analysis/src';

/**
 * Records whose `status` is `[]` although the public output type declares a `status`,
 * `overflow` or `candidateOverflow` member, and whose status kinds have not been audited. Remove an
 * entry once its record lists them (the spec fails on stale entries). Keep this list empty.
 */
const UNAUDITED_STATUS_RECORDS: readonly string[] = [];

/**
 * Records whose output type has a `status` member that is not a one-row catalog status view, so
 * `status: []` is the correct claim: per-row or per-feature status codes, packed status words,
 * a solver `code`, or one-row counts under names that are not catalog kinds. Audited by hand
 * against each class's output type (catalog CAT-101).
 */
const CODED_STATUS_RECORDS: Readonly<Record<string, string>> = {
  GPUSegmentSnap: 'candidateOverflow is per-line; counts is a packed uint32x4 row per line',
  GPUAreaOfApplicability: 'one status code word (GPU_AREA_OF_APPLICABILITY_STATUS_*)',
  GPUConditionalGaussianSimulation: 'one status code per replicate',
  GPUConformalPredictionIntervals: 'status bits per query row',
  GPUContinuousSpatialSample: 'packed uint32x4 status row per wave',
  GPUConvexStraightSkeleton: 'packed uint32x4 status row per polygon',
  GPUGeoConformalPredictionIntervals: 'status bits per query row',
  GPUInhomogeneousPointPattern: 'one status code word',
  GPUInstrumentalVariablesCovariance: 'solver code only (Pick<GPUSolverStatusPort, code>)',
  GPULineLoopRemoval: 'packed uint32x4 status row [written, required, overflow, incomplete]',
  GPULinePrecisionReduction: 'status code per line',
  GPULocalCollinearity: 'status code per row',
  GPULocalKriging: 'status code per query row',
  GPULocalRBF: 'status code per query row',
  GPUMaximumInscribedCircle: 'status code per feature',
  GPUNaturalNeighbor: 'status code per query row',
  GPUNestedInterpolationValidation: 'status bits per fold',
  GPUNormalizeGeometry: 'status code per node',
  GPUOrdinaryLeastSquares: 'solver code only (Pick<GPUSolverStatusPort, code>)',
  GPUPointPatternEnvelope: 'one status code word',
  GPUPopulationSupportAlignment: 'packed uint32x4 alignment counts',
  GPUProbitRegression: 'model and diagnostic status code words',
  GPURestrictedRegimeRegression: 'one status code word',
  GPUS2CellHierarchy: 'status code per cell row',
  GPUS2CellMeasures: 'status code per cell row',
  GPUSeeminglyUnrelatedRegression: 'status code words (per equation and FGLS)',
  GPUSpaceTimeRelativeRisk: 'caseInvalidCount and controlInvalidCount, not catalog kinds',
  GPUSpatialBalancedSample: 'packed [selectedCount, status] per stratum',
  GPUSpatialCrossValidation: 'status bits per fold',
  GPUSpatialErrorGM: 'solver code only (Pick<GPUSolverStatusPort, code>)',
  GPUSpatialInteraction: 'one status code word',
  GPUSpatialPanelLagMaximumLikelihood: 'one status code word with bit flags',
  GPUSpatialRegressionDiagnostics: 'solver code only (Pick<GPUSolverStatusPort, code>)',
  GPUSpatialTwoStageLeastSquares: 'solver code only (Pick<GPUSolverStatusPort, code>)',
  GPUSphericalConvexContains: 'status code per query row',
  GPUSphericalPolygonContains: 'status code per query row',
  GPUSubdividePolygons: 'packed uint32x4 status row',
  GPUTrackSimilaritySearch: 'status bits per pair',
  GPUVecchiaFisherScoring: 'packed [failed, excluded, used] site counts',
  GPUVecchiaLogLikelihood: 'packed [failed, excluded, used] site counts',
  GPUVecchiaSpaceTimeLogLikelihood: 'packed [failed, excluded, used] site counts',
  GPUVectorStraightSkeleton: 'packed uint32x4 status row per polygon'
};

/**
 * Records claiming `status: []` whose output type this text scan cannot find: functions and
 * recipes (results are not a `<Name>Output` alias), handles, and classes whose props reach
 * `output` only through generic intersections. Each was checked by hand (catalog CAT-101); the
 * reason says why no catalog status kind applies. A record that gains status kinds leaves the set.
 */
const UNINSPECTABLE_STATUS_RECORDS: Readonly<Record<string, string>> = {
  'analysis-core:GPUParameterBuffer': 'parameter upload; no output status',
  'dataframe-analysis:aggregateDecimal128Column': 'host result after readback; no GPU status view',
  'dataframe-analysis:aggregateDecimal128OnGpu': 'host result after readback; no GPU status view',
  'dataframe-analysis:aggregateInteger64OnGpu': 'host result after readback; no GPU status view',
  'dataframe-analysis:aggregateInteger64Range': 'host result after readback; no GPU status view',
  'dataframe-analysis:diagnoseFloat32Narrowing':
    'CPU report; overflowCount is Float32 range overflow, not capacity',
  'dataframe-analysis:scanWideNumericColumnOnGpu': 'host result after readback; no GPU status view',
  'gpu-spatial-analysis:GPUGammaStatistic':
    'GPUEdgeAssociationOutput holds only `statistic` (props reach it through an intersection)',
  'gpu-spatial-analysis:GPURegionStatisticsReadback': 'readback adapter; host results',
  'gpu-spatial-analysis:GPUSpaceTimeKriging': 'status code per query row',
  'gpu-spatial-analysis:GPUSpatialPearson':
    'GPUEdgeAssociationOutput holds only `statistic` (props reach it through an intersection)',
  'gpu-spatial-analysis:GPUSpatialJoinPrepared':
    'prepared index handle; its BVH storage words are read by the joins that use it',
  'gpu-spatial-analysis:GPUTrajectoryKalman': 'status bits per observation',
  'gpu-spatial-analysis:GPUTrajectoryRestriction':
    'status bits per interval slot and a packed compactSummary',
  'gpu-spatial-analysis:addIsovistsRecipe': 'result has no status',
  'gpu-spatial-analysis:addLandscapeMetricsRecipe': 'result has no status',
  'gpu-spatial-analysis:addResistanceConnectivityRecipe': 'packed uint32x4 status row',
  'gpu-spatial-analysis:addSegregationDecompositionRecipe': 'result has no status',
  'gpu-spatial-analysis:addSparseResistanceConnectivityRecipe':
    'packed uint32x4 [converged, iterations, reachable, invalid] row',
  'gpu-spatial-analysis:addStreetProfilesRecipe': 'result has no status',
  'gpu-spatial-analysis:computePolygonContiguityOnCPU': 'CPU function; host diagnostics',
  'gpu-spatial-analysis:getHealpixCellBoundary': 'CPU function; no status',
  'gpu-spatial-analysis:preparePolygonContiguityTopology': 'CPU function; host diagnostics'
};

function listSourceFiles(directory: string): string[] {
  return readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return listSourceFiles(path);
    return entry.name.endsWith('.ts') ? [path] : [];
  });
}

/** Text of one declaration: an interface body, or a type alias up to its top-level semicolon. */
function readDeclaration(text: string, start: number): string {
  const isInterface = /^\s*(?:export\s+)?interface\b/.test(text.slice(start, start + 40));
  let depth = 0;
  for (let index = start; index < text.length; index++) {
    const character = text[index];
    if (character === '{' || character === '(' || character === '[') depth++;
    else if (character === '}' || character === ')' || character === ']') {
      depth--;
      if (depth === 0 && isInterface && character === '}') return text.slice(start, index + 1);
    } else if (character === ';' && depth === 0) return text.slice(start, index + 1);
  }
  return text.slice(start);
}

type SourceIndex = {
  declarations: Map<string, string[]>;
  classTexts: Map<string, string[]>;
};

function indexSource(): SourceIndex {
  const declarations = new Map<string, string[]>();
  const classTexts = new Map<string, string[]>();
  const append = (map: Map<string, string[]>, key: string, value: string) =>
    map.set(key, [...(map.get(key) ?? []), value]);
  for (const file of listSourceFiles(SOURCE_ROOT)) {
    const text = readFileSync(file, 'utf8');
    for (const match of text.matchAll(/^(?:export\s+)?(?:type|interface)\s+(\w+)/gm))
      append(declarations, match[1], readDeclaration(text, match.index));
    // Each class body runs to the next exported class, so a file with several classes does not
    // lend the first class's constructor props to the others.
    const classes = [...text.matchAll(/^export\s+(?:abstract\s+)?class\s+(\w+)/gm)];
    for (const [position, match] of classes.entries())
      append(classTexts, match[1], text.slice(match.index, classes[position + 1]?.index));
  }
  return {declarations, classTexts};
}

const STATUS_MEMBER =
  /(?:^|[{;,\s])(?:readonly\s+)?(status|count|requiredCount|matchedCount|overflow|candidateOverflow|excludedLocationCount|uncertainCount|invalidCount|converged|iterationLimitReached|iterationCount|zeroWeightCycle|approximation|vertexCount|requiredVertexCount|ringCount|requiredRingCount|nodedSegmentRequiredCount)\??\s*:/m;
const OUTPUT_MEMBER = /(?:^|[{;,\s])(?:readonly\s+)?outputs?\??\s*:/gm;
const REFERENCED_OUTPUT_TYPE = /\b[A-Z]\w*(?:Output|Outputs|Status|Ports?|Result|Table)\w*\b/g;

/**
 * Finds counterevidence to an existing empty audit. A bounded walk with no match cannot
 * establish a new audit: unresolved references and deeper declarations can hide status.
 */
function findStatusMember(
  index: SourceIndex,
  texts: readonly string[],
  seen: Set<string>,
  depth: number
): string | null {
  for (const text of texts) {
    const member = text.match(STATUS_MEMBER);
    if (member) return member[1];
  }
  if (depth > 3) return null;
  for (const text of texts) {
    for (const [name] of text.matchAll(REFERENCED_OUTPUT_TYPE)) {
      const referenced = index.declarations.get(name);
      if (seen.has(name) || !referenced) continue;
      seen.add(name);
      const found = findStatusMember(index, referenced, seen, depth + 1);
      if (found) return `${found} via ${name}`;
    }
  }
  return null;
}

/**
 * Reads the `<Name>Output` alias and the `output` member of the class props (`<Name>Props` or the
 * constructor's props type). Returns `undefined` when no output type text is found.
 */
function inspectOutputStatus(index: SourceIndex, exportName: string): string | null | undefined {
  const roots = [
    ...(index.declarations.get(`${exportName}Output`) ?? []),
    ...(index.declarations.get(`${exportName}Outputs`) ?? [])
  ];
  const props = [...(index.declarations.get(`${exportName}Props`) ?? [])];
  for (const text of index.classTexts.get(exportName) ?? []) {
    const constructorProps = text.match(/constructor\s*\(\s*props\s*:\s*([A-Z]\w*)/)?.[1];
    if (constructorProps) props.push(...(index.declarations.get(constructorProps) ?? []));
  }
  for (const text of props) {
    for (const match of text.matchAll(OUTPUT_MEMBER))
      roots.push(readDeclaration(text, match.index + match[0].length).slice(0, 2000));
  }
  return roots.length ? findStatusMember(index, roots, new Set(), 0) : undefined;
}

describe('catalog status claims against the source', () => {
  const LINE_GEOMETRY_STATUS = [
    'count',
    'requiredCount',
    'overflow',
    'vertexCount',
    'requiredVertexCount'
  ];
  // Independent expectation: the `status` member of `GPULineGeometryOutput` in
  // line-segmentize/line-segmentize-types.ts and each class's publish node (`approximation` is
  // accepted only by segmentize and great-circle arcs; offset curve has it in compact layout).
  const EXPECTED_STATUS: Record<string, readonly string[]> = {
    GPULineSegmentize: [...LINE_GEOMETRY_STATUS, 'approximation'],
    GPUGreatCircleArcs: [...LINE_GEOMETRY_STATUS, 'approximation'],
    GPULineSmooth: LINE_GEOMETRY_STATUS,
    GPULineChunk: LINE_GEOMETRY_STATUS,
    GPURectangleClip: LINE_GEOMETRY_STATUS,
    GPUOffsetCurve: LINE_GEOMETRY_STATUS
  };

  it('finds the six line operators with overflow ports in both overflow queries', () => {
    const all = new Set(queryGPUAnalysisOperators({status: 'overflow'}).map(r => r.exportName));
    const legacy = new Set(
      queryGPUSpatialAnalysisOperators({status: 'overflow'}).map(r => r.exportName)
    );
    for (const [exportName, status] of Object.entries(EXPECTED_STATUS)) {
      expect(getGPUAnalysisOperator(exportName)?.status, exportName).toEqual(status);
      expect(getGPUSpatialAnalysisOperator(exportName)?.status, exportName).toEqual(status);
      expect(all.has(exportName), exportName).toBe(true);
      expect(legacy.has(exportName), exportName).toBe(true);
    }
    expect(getGPUAnalysisOperator('GPUOffsetCurve')?.limitation).toContain(
      "only with `output.layout: 'compact'`"
    );
  });

  it('requires affirmative qualified audits for every known-empty status', () => {
    const emptyRecords = getGPUAnalysisOperatorCatalog().filter(
      record => record.status !== 'unknown' && record.status.length === 0
    );
    const identities = emptyRecords.map(record => `${record.package}:${record.exportName}`).sort();
    expect(identities).toEqual(Object.keys(AUDITED_EMPTY_OPERATOR_STATUSES).sort());
    for (const identity of identities) {
      expect(identity).toMatch(/^@luma-contributors\/analysis\/[^:]+:[^:]+$/);
      expect(AUDITED_EMPTY_OPERATOR_STATUSES[identity]?.trim(), identity).toBeTruthy();
    }
  });

  it('checks affirmative empty audits for newly visible status members', () => {
    const index = indexSource();
    const unexpected: string[] = [];
    const flagged = new Set<string>();
    const uninspectable: string[] = [];
    for (const record of getGPUAnalysisOperatorCatalog()) {
      if (record.status === 'unknown' || record.status.length > 0) continue;
      const shortPackage = record.package.split('/').pop()!;
      const found = inspectOutputStatus(index, record.exportName);
      if (found === undefined) uninspectable.push(`${shortPackage}:${record.exportName}`);
      else if (found) {
        flagged.add(record.exportName);
        if (
          !UNAUDITED_STATUS_RECORDS.includes(record.exportName) &&
          !Object.hasOwn(CODED_STATUS_RECORDS, record.exportName)
        )
          unexpected.push(`${shortPackage}:${record.exportName} (${found})`);
      }
    }
    expect(unexpected, 'records claiming status [] with a status member in source').toEqual([]);
    const stale = [...UNAUDITED_STATUS_RECORDS, ...Object.keys(CODED_STATUS_RECORDS)].filter(
      name => !flagged.has(name)
    );
    expect(stale, 'allowlisted records no longer claiming status []').toEqual([]);
    expect(uninspectable.sort()).toEqual(Object.keys(UNINSPECTABLE_STATUS_RECORDS).sort());
  });

  it('lists the audited status kinds of each catalog route', () => {
    // Independent expectation: the one-row members each class writes, read from its output type
    // (cell-cover/gpu-cell-cover.ts status; spatial-clustering/gpu-optics.ts ports.status;
    // spatial-weights/gpu-island-attachment.ts, which writes optional candidate loss and count;
    // space-time-hawkes/gpu-space-time-hawkes.ts status; trajectory-cleaning/gpu-trajectory-split.ts
    // status; recipes/rate-mapping-recipes.ts stage `excess-risk`).
    const expected: Record<string, readonly string[]> = {
      GPUCellCover: [
        'count',
        'requiredCount',
        'overflow',
        'candidateOverflow',
        'uncertainCount',
        'invalidCount'
      ],
      GPUOPTICS: [
        'candidateOverflow',
        'converged',
        'iterationLimitReached',
        'iterationCount',
        'invalidCount'
      ],
      GPUIslandAttachment: [
        'count',
        'requiredCount',
        'overflow',
        'candidateOverflow',
        'uncertainCount'
      ],
      GPUSpaceTimeHawkes: [
        'candidateOverflow',
        'invalidCount',
        'converged',
        'iterationLimitReached',
        'iterationCount',
        'approximation'
      ],
      GPUTrajectorySplit: ['count', 'requiredCount', 'overflow', 'candidateOverflow'],
      GPUMobilityIndicators: ['candidateOverflow', 'excludedLocationCount'],
      addExcessRiskRecipe: ['invalidCount']
    };
    const overflow = new Set(
      queryGPUAnalysisOperators({status: 'overflow'}).map(r => r.exportName)
    );
    for (const [exportName, status] of Object.entries(expected)) {
      expect(getGPUAnalysisOperator(exportName)?.status, exportName).toEqual(status);
      expect(overflow.has(exportName), exportName).toBe(status.includes('overflow'));
    }
    // The original catalog view agrees for the records it holds.
    for (const exportName of ['GPUCellCover', 'GPUOPTICS', 'GPUIslandAttachment']) {
      expect(getGPUSpatialAnalysisOperator(exportName)?.status, exportName).toEqual(
        expected[exportName]
      );
    }
  });

  it('detects a direct scalar and a status nested in GPUCellTable', () => {
    const index = indexSource();
    expect(inspectOutputStatus(index, 'GPUShapeGenerator')).toBe('vertexCount');
    expect(inspectOutputStatus(index, 'GPUCellAggregation')).toBe('status via GPUCellTable');
    expect(inspectOutputStatus(index, 'GPUCellPyramid')).toBe(
      'status via GPUCellTable via GPUCellPyramidOutput'
    );
  });

  it('detects a status member reached through a referenced output type', () => {
    // Discriminating control for the scan itself: GPULineSegmentize reaches `status` only through
    // `GPULineGeometryOutput`, and GPUGridGenerator's output has no status member.
    const index = indexSource();
    expect(inspectOutputStatus(index, 'GPULineSegmentize')).toBe(
      'status via GPULineGeometryOutput'
    );
    expect(inspectOutputStatus(index, 'GPUGridGenerator')).toBeNull();
  });
});

describe('wave three staged writer registrations', () => {
  const packageName = '@luma-contributors/analysis/gpu-spatial-analysis' as const;
  it('W3-02 discovers inherited island loss and optional emitted count in both catalogs', () => {
    // The staged writer reduces inherited final/candidate loss and writes offsets[rows] to count.
    const source = readFileSync(
      `${SOURCE_ROOT}/gpu-spatial-analysis/spatial-weights/gpu-island-attachment.ts`,
      'utf8'
    );
    expect(source).toContain('if (props.output.status.count)');
    expect(source).toContain('count[countOffset] = offsets[offsetsOffset +');
    expect(source).toContain('[...inherited.overflow, ...inherited.candidateOverflow]');
    expect(source).toContain('createWeightsLossNodes(');
    const lossWriter = readFileSync(
      `${SOURCE_ROOT}/gpu-spatial-analysis/spatial-weights/spatial-weights-loss.ts`,
      'utf8'
    );
    // Empty loss inputs still execute the first batch and initialize the destination to zero.
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      recordWGSLKernelNodes(graph);
      const destination = createTransientView(graph, 'empty-loss', 'uint32', 1);
      expect(createWeightsLossNodes(graph, 'empty-loss-writer', [], destination)).toHaveLength(1);
      const record = getWGSLKernelNodeRecord(graph, 'empty-loss-writer-0');
      expect(record?.bindings).toHaveLength(1);
      expect(record?.bindings[0].view).toBe(destination);
      // With no inherited flags, the hand-derived OR identity is zero, still written once.
      expect(record?.body).toContain('var flag = 0u;');
      expect(record?.body).toContain('loss[lossOffset] = flag;');
    } finally {
      device.destroy();
    }
    expect(lossWriter).toContain("${merge || first > 0 ? 'u32(loss[lossOffset] != 0u)' : '0u'}");
    expect(lossWriter).toContain('loss[lossOffset] = flag;');
    expect(getGPUAnalysisOperator('GPUIslandAttachment')?.summary).toContain(
      'optional count is the emitted link count'
    );
    for (const status of ['candidateOverflow', 'count'] as const) {
      expect(
        queryGPUAnalysisOperators({package: packageName, status}).map(record => record.exportName)
      ).toContain('GPUIslandAttachment');
      expect(queryGPUSpatialAnalysisOperators({status}).map(record => record.exportName)).toContain(
        'GPUIslandAttachment'
      );
    }
  });
});

it('W3-03 discovers optional requiredCount beside mandatory selected-level count', () => {
  // Source-backed expectation: both destinations receive the selected level's occupied count;
  // GPUCardinalityPort requires count and makes requiredCount optional.
  const source = readFileSync(
    `${SOURCE_ROOT}/gpu-spatial-analysis/cell-aggregation/gpu-cell-level-selection.ts`,
    'utf8'
  );
  const contract = readFileSync(
    `${SOURCE_ROOT}/analysis-core/kernels/gpu-contributor-types.ts`,
    'utf8'
  );
  expect(contract).toContain('count: GPUStatusScalar');
  expect(contract).toContain('requiredCount?: GPUStatusScalar');
  expect(source).toContain('let count = levelCounts[levelCountsOffset + level];');
  expect(source).toContain('countOut[countOutOffset] = count;');
  expect(source).toContain(
    "output.status.requiredCount ? 'requiredCountOut[requiredCountOutOffset] = count;'"
  );
  expect(
    getGPUAnalysisOperator({
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUCellLevelSelection'
    })?.status
  ).toEqual(['count', 'requiredCount']);
  expect(getGPUAnalysisOperator('GPUCellLevelSelection')?.summary).toContain(
    'already-clamped occupied row count'
  );
  expect(getGPUAnalysisOperator('GPUCellLevelSelection')?.summary).toContain(
    'cannot recover the unclamped requirement'
  );
  expect(
    queryGPUAnalysisOperators({status: 'requiredCount'}).map(record => record.exportName)
  ).toContain('GPUCellLevelSelection');
  expect(
    queryGPUSpatialAnalysisOperators({status: 'requiredCount'}).map(record => record.exportName)
  ).toContain('GPUCellLevelSelection');
});

it('W3-04 discovers trajectory iteration budget beside final convergence flags', () => {
  const source = readFileSync(
    `${SOURCE_ROOT}/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-locations.ts`,
    'utf8'
  );
  // This reports the configured hook/compress round budget, not the first stable round.
  expect(source).toContain('iterationCount[iterationCountOffset] = ${maximumIterations}u;');
  expect(source).toContain('converged[convergedOffset] = select(1u, 0u, didChange);');
  expect(source).toContain(
    'iterationLimitReached[iterationLimitReachedOffset] = select(0u, 1u, didChange);'
  );
  expect(
    getGPUAnalysisOperator({
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUTrajectoryLocations'
    })?.status
  ).toEqual(['count', 'converged', 'iterationLimitReached', 'iterationCount', 'complete']);
  expect(
    queryGPUAnalysisOperators({status: 'iterationCount'}).map(record => record.exportName)
  ).toContain('GPUTrajectoryLocations');
  // This post-original operator has no historical spatial projection.
  expect(getGPUSpatialAnalysisOperator('GPUTrajectoryLocations')).toBeUndefined();
});

it('STA-08 discovers remaining drainage work without erasing resolved fallback discovery', () => {
  const source = readFileSync(
    `${SOURCE_ROOT}/gpu-spatial-analysis/recipes/drainage-network-recipe.ts`,
    'utf8'
  );
  // These are two distinct scalar populations: remaining distance cells and unresolved stream cells.
  expect(source).toMatch(
    /stage:\s*'downstream-distance',\s*status:\s*\{\s*unresolvedCount:\s*unresolvedDistanceCount/
  );
  expect(source).toMatch(
    /stage: 'stream-network',[\s\S]*?unresolvedCount: network.unresolvedCount/
  );
  expect(
    getGPUAnalysisOperator({
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'addDrainageNetworkRecipe'
    })?.status
  ).toContain('unresolvedCount');
  expect(getGPUAnalysisOperator('addDrainageNetworkRecipe')?.summary).toContain(
    'remaining distance cells at downstream-distance and remaining stream cells at stream-network'
  );
  expect(
    queryGPUAnalysisOperators({status: 'unresolvedCount'}).map(record => record.exportName)
  ).toContain('addDrainageNetworkRecipe');
  expect(
    queryGPUSpatialAnalysisOperators({status: 'unresolvedCount'}).map(record => record.exportName)
  ).toContain('addDrainageNetworkRecipe');
  for (const records of [
    queryGPUAnalysisOperators({status: 'uncertainCount'}),
    queryGPUSpatialAnalysisOperators({status: 'uncertainCount'})
  ]) {
    const names = records.map(record => record.exportName);
    expect(names).toContain('GPUIslandAttachment');
    expect(names).not.toContain('addDrainageNetworkRecipe');
  }
});

it('STA-12 discovers a complete-only recipe and retains affirmative empty audits', () => {
  const source = readFileSync(
    `${SOURCE_ROOT}/gpu-spatial-analysis/urban-morphology/building-adjacency.ts`,
    'utf8'
  );
  expect(source).toMatch(/const sharedPathStatus:[^=]+?=\s*\{\s*complete\s*\};/);
  expect(source).toContain('status: sharedPathStatus');
  const identity = {
    package: '@luma-contributors/analysis/gpu-spatial-analysis',
    exportName: 'addBuildingAdjacencyRecipe'
  } as const;
  expect(getGPUAnalysisOperator(identity)?.status).toEqual(['complete']);
  expect(getGPUAnalysisOperator(identity)?.summary).toContain('shared-path input');
  expect(getGPUSpatialAnalysisOperator(identity.exportName)?.status).toEqual(['complete']);
  expect(
    queryGPUAnalysisOperators({status: 'complete'}).map(record => record.exportName)
  ).toContain(identity.exportName);
  expect(
    queryGPUSpatialAnalysisOperators({status: 'complete'}).map(record => record.exportName)
  ).toContain(identity.exportName);
  expect(getGPUAnalysisOperator('GPUCanonicalPolygonEquality')?.status).toEqual([]);
  expect(
    AUDITED_EMPTY_OPERATOR_STATUSES[`${identity.package}:${identity.exportName}`]
  ).toBeUndefined();
});

it('B2 discovers the confirmed trajectory and dasymetric completeness publishers with scope', () => {
  const trajectorySource = readFileSync(
    `${SOURCE_ROOT}/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-locations.ts`,
    'utf8'
  );
  expect(trajectorySource).toContain('view: output.status.complete');
  expect(trajectorySource).toContain('complete[completeOffset] = select(1u, 0u, changed');
  const recipeSource = readFileSync(
    `${SOURCE_ROOT}/gpu-spatial-analysis/recipes/dasymetric-redistribution-recipe.ts`,
    'utf8'
  );
  expect(recipeSource).toMatch(/status:\s*\{\s*complete,/);
  expect(recipeSource).toMatch(/stage: 'coverage-validation',[\s\S]*?complete: coverageComplete/);
  expect(recipeSource).toMatch(/stage: 'dasymetric-conservation',[\s\S]*?complete,/);
  const complete = queryGPUAnalysisOperators({
    package: '@luma-contributors/analysis/gpu-spatial-analysis',
    status: 'complete'
  });
  for (const exportName of ['GPUTrajectoryLocations', 'addDasymetricRedistributionRecipe']) {
    expect(complete.map(record => record.exportName)).toContain(exportName);
  }
  expect(getGPUAnalysisOperator('GPUTrajectoryLocations')?.summary).toMatch(
    /complete.*settled labels.*whole input/
  );
  expect(getGPUAnalysisOperator('addDasymetricRedistributionRecipe')?.summary).toMatch(
    /complete.*coverage-validation.*dasymetric-conservation/
  );
  expect(
    queryGPUSpatialAnalysisOperators({status: 'complete'}).map(record => record.exportName)
  ).toContain('addDasymetricRedistributionRecipe');
  expect(getGPUSpatialAnalysisOperator('GPUTrajectoryLocations')).toBeUndefined();
});

it('earlier-fix note: drainage registers only statuses actually published by its stages', () => {
  const expected = ['count', 'overflow', 'converged', 'unresolvedCount'];
  expect(getGPUAnalysisOperator('addDrainageNetworkRecipe')?.status).toEqual(expected);
  expect(getGPUSpatialAnalysisOperator('addDrainageNetworkRecipe')?.status).toEqual(expected);
  expect(
    queryGPUAnalysisOperators({status: 'invalidCount'}).map(record => record.exportName)
  ).not.toContain('addDrainageNetworkRecipe');
  expect(
    queryGPUSpatialAnalysisOperators({status: 'invalidCount'}).map(record => record.exportName)
  ).not.toContain('addDrainageNetworkRecipe');
});

it('S1 binds island candidate-loss and optional count to their supplied destinations', () => {
  for (const inherited of [false, true]) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    let serial = 0;
    const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `catalog-islands-view-${serial++}`, format, length);
    const upstreamCandidate = inherited ? view('uint32', 1) : undefined;
    const upstreamOverflow = inherited ? view('uint32', 1) : undefined;
    const candidateOverflow = view('uint32', 1);
    const overflow = view('uint32', 1);
    const count = view('uint32', 1);
    const offsets = view('uint32', 4);
    try {
      const contributor = new GPUIslandAttachment({
        id: 'catalog-islands',
        nearestSearch: 'scan',
        positions: view('float32x2', 3),
        weights: {
          offsets: view('uint32', 4),
          neighbors: view('uint32', 3),
          weights: view('float32', 3),
          ...(upstreamCandidate && upstreamOverflow
            ? {status: {candidateOverflow: upstreamCandidate, overflow: upstreamOverflow}}
            : {})
        },
        output: {
          offsets,
          neighbors: view('uint32', 9),
          weights: view('float32', 9),
          attachment: view('uint32', 3),
          added: view('uint32', 9),
          status: {candidateOverflow, overflow, count, requiredCount: view('uint32', 1)}
        }
      });
      contributor.getCommandNodes(graph);
      const lossRecord = getWGSLKernelNodeRecord(graph, 'catalog-islands-inherited-candidate-0');
      const countRecord = getWGSLKernelNodeRecord(graph, 'catalog-islands-count');
      if (!lossRecord || !countRecord) throw new Error('expected both supplied status writers');
      const assertDestinations = (
        loss: WGSLKernelNodeRecord,
        cardinality: WGSLKernelNodeRecord
      ) => {
        expect(loss.bindings.find(binding => binding.name === 'loss')?.view).toBe(
          candidateOverflow
        );
        expect(loss.bindings.find(binding => binding.name === 'loss')?.access).toBe('read_write');
        expect(loss.body).toContain('var flag = 0u;');
        expect(loss.body).toContain('loss[lossOffset] = flag;');
        const lossInputs = loss.bindings.filter(binding => binding.name.startsWith('lossInput'));
        expect(lossInputs).toHaveLength(inherited ? 2 : 0);
        if (inherited) {
          expect(lossInputs.map(binding => binding.view)).toContain(upstreamCandidate);
          expect(lossInputs.map(binding => binding.view)).toContain(upstreamOverflow);
          for (const binding of lossInputs) {
            expect(binding.access).toBe('read');
            expect(loss.body).toContain(
              `flag |= u32(${binding.name}[${binding.name}Offset] != 0u);`
            );
          }
        }
        expect(cardinality.bindings.find(binding => binding.name === 'count')?.view).toBe(count);
        expect(cardinality.bindings.find(binding => binding.name === 'count')?.access).toBe(
          'read_write'
        );
        expect(cardinality.bindings.find(binding => binding.name === 'offsets')?.view).toBe(
          offsets
        );
        expect(cardinality.body).toContain('count[countOffset] = offsets[offsetsOffset + 3u];');
      };
      assertDestinations(lossRecord, countRecord);
      // Discriminating controls: identical shader text bound to the wrong output must fail.
      expect(() =>
        assertDestinations(
          {
            ...lossRecord,
            bindings: lossRecord.bindings.map(binding =>
              binding.name === 'loss' ? {...binding, view: overflow} : binding
            )
          },
          countRecord
        )
      ).toThrow();
      expect(() =>
        assertDestinations(lossRecord, {
          ...countRecord,
          bindings: countRecord.bindings.map(binding =>
            binding.name === 'count' ? {...binding, view: overflow} : binding
          )
        })
      ).toThrow();
    } finally {
      device.destroy();
    }
  }
});
