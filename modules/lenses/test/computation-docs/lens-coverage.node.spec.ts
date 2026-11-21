// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readdirSync, readFileSync} from 'node:fs';
import {basename, join, relative, resolve, sep} from 'node:path';
import {describe, expect, it} from 'vitest';
import {WEIGHTS_OPERATORS} from '../../src/gpu-spatial-analysis/computation-pack/operators/weights';
import {PENDING_LENS_RECORDS, collectPendingRecordFailures} from './pending-lens-records';
import type {GPUComputationFormulaRef} from '@luma-contributors/analysis/computation-record';
// Registers every scenario descriptor (lazy; no GPU code runs).
import '../../src/computation-lenses/scenarios/index';
import {
  fromRecordedLensFile,
  listLensScenarios,
  type LensScenarioDescriptor
} from '../../src/computation-lenses/index';
import {parseLensDirective} from '../../src/computation-views/twin/static-twin';
import {
  type LensRungId,
  type OperatorScope,
  type OperatorReduction,
  LENS_RUNG_IDS
} from '../../src/computation-views/types';
import {type ViewPackData} from '../../src/computation-views/registry/view-registration';
import {
  DOCS_ROOT,
  getLensElementId,
  REPOSITORY_ROOT,
  listDocsMarkdownFiles,
  listEntryElements,
  createEntryElementDiscovery,
  parseDirectiveAttributes,
  scanLensDirectives,
  scanLensMarkers,
  type LensMarker,
  type ScannedLensDirective
} from './lens-directives';

/**
 * Scenarios removed in the upstream-first split because they document luma.gl-owned operators or
 * features that do not exist in published luma.gl. The prose docs still carry their directives until
 * the docs are rewritten (dev-docs/upstream-first-plan.md); these are ignored here, not checked.
 */
function isDroppedScenario(scenario: string | undefined): boolean {
  return (
    scenario !== undefined &&
    (/^(?:gpu-core|gpu-sql)\./.test(scenario) ||
      /^gpu-dataframe\.data-frame-[a-z-]*query$/.test(scenario) ||
      scenario === 'gpu-graph.graph-strongly-connected-components')
  );
}

// Resume lists for the next round: element → what is still missing (from the stream handoffs).
// Element → what is missing, as `<stream>: <parts>`. Generated from the failing checks and cross-checked with the
// stream handoffs (2026-10-14); 'docs card' means scenario and pack exist but the doc section lacks lenses.
/** Contributors of the spatiotemporal review: documented with an empty slot, lenses not yet authored. */
const SPATIOTEMPORAL_REVIEW_PENDING =
  'lenses: pack operator, formula, scenario, directives and records (spatiotemporal review follow-up)';
const GPU_DATAFRAME_PENDING: Readonly<Record<string, string>> = {
  GPUCubeLayoutTranspose: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUPeriodIndex: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUTimeWordsRelative: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUCellPeriodKeys: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUSpaceTimeDiggleTest: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUSpaceTimeKFunction: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUValueCube: SPATIOTEMPORAL_REVIEW_PENDING
};
const GPU_SPATIAL_ANALYSIS_PENDING: Readonly<Record<string, string>> = {
  addStopValueCubeRecipe:
    'lenses: gpu-spatial-analysis pack operator and formula, stop-value-cube-recipe scenario, documentation marker/directives and recording are absent (CMP-12)',
  GPUChunkedSpatialWeightsAlgebra:
    'lenses: pack operator, formula, supported row-local family scenario, directives and records; the whole-CSR algebra scenario uses unsupported higherOrder',
  GPUEnsembleScores: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUGaussianPredictiveScores: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUJacquezTest: SPATIOTEMPORAL_REVIEW_PENDING,
  GPULocalKnox: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUMobilityIndicators: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUModalSplit: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUNearRepeatKnox: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUSpaceTimeCrossCorrelation: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUSpaceTimeKriging: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUSpaceTimeLocalMoran: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUTrajectoryHampel: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUTrajectoryLocations: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUTrajectoryOutlierFlags: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUTrajectorySegments: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUTrajectorySplit: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUTrajectoryStaypoints: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUTrajectoryThinning: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUTrajectoryTrackRelativeTimes: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUAdaptiveBandwidthKernelDensity: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUAreaOfApplicability: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUBrownianBridgeUtilization: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUCategoryPairCounts: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUColocationQuotient: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUConformalPredictionIntervals: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUExpectationBasedSpaceTimeScan: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUGeoConformalPredictionIntervals: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUGeographicalDetector: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUGeographicallyTemporallyWeightedRegression: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUHealpixCellAggregation: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUHealpixPointToCell: SPATIOTEMPORAL_REVIEW_PENDING,
  GPULocalCategoryDependence: SPATIOTEMPORAL_REVIEW_PENDING,
  GPULocationChangePoints: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUMultiscaleGeographicallyTemporallyWeightedRegression: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUPettittTest: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterFalseDiscoveryRateMask: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURedcapSingleLinkage: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUSinkhornDivergence: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUSpaceTimeHawkes: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUSpaceTimeKernelDensity: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUSpaceTimeRelativeRisk: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUSpatialEntropy: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUTrajectoryRecursions: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUUtilizationVolumeIsopleths: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUVecchiaFisherScoring: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUVecchiaLogLikelihood: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUVecchiaSpaceTimeLogLikelihood: SPATIOTEMPORAL_REVIEW_PENDING,
  addContiguityOrderCorrelogramRecipe: SPATIOTEMPORAL_REVIEW_PENDING
};
const GPU_NETWORK_PENDING: Readonly<Record<string, string>> = {
  GPUKernelFlowSmoothing: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUMapMatchedEdgeVolumes: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUNetworkPotentialPathArea: SPATIOTEMPORAL_REVIEW_PENDING,
  GPUNetworkTravelTimePercentiles: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURadiationFlowModel: SPATIOTEMPORAL_REVIEW_PENDING
};
const GPU_RASTER_PENDING: Readonly<Record<string, string>> = {
  GPUFiniteTimeLyapunovExponent: SPATIOTEMPORAL_REVIEW_PENDING,
  GPULagrangianFlowMap: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterBFASTMonitor: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterBestAvailablePixel: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterCOLDMonitor: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterClimateAnomaly: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterClimateRunAccumulator: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterEOFProjection: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterEmbeddingChange: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterEmbeddingCovariance: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterEmbeddingDequantize: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterEmbeddingKNearestClassifier: SPATIOTEMPORAL_REVIEW_PENDING,
  // These existing adapters previously exposed only addToGraph; getCommandNodes discovery now
  // includes them. They have no family scenario yet, rather than silently counting as covered.
  GPURasterEmbeddingKMeans:
    'lenses: pack operator, formula, scenario, directives and records (existing raster embedding adapter)',
  GPURasterEmbeddingNearestPixels:
    'lenses: pack operator, formula, scenario, directives and records (existing raster embedding adapter)',
  GPURasterEmbeddingLinearProbe: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterEmbeddingProjection: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterEmbeddingPyramidLevel: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterEmbeddingSimilarity: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterEmbeddingZoneMean: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterGeometricMedianComposite: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterHANTS: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterHarmonicRegression: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterLaggedCorrelation: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterMedoidComposite: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterPhenologyMetrics: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterRollingMean: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterSnapshotCovariance: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterStandardizedIndex: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterStandardizedIndexCalibration: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterStreamingAccumulator: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterTemporalResample: SPATIOTEMPORAL_REVIEW_PENDING,
  GPURasterWhittakerSmoother: SPATIOTEMPORAL_REVIEW_PENDING
};
const GPU_TERRAIN_PENDING: Readonly<Record<string, string>> = {};

/**
 * Entries whose coordinators declared lens coverage complete. Wave-2 coordinators append their entry here
 * when their docs, pack and scenarios are in place; from then on every check below is enforced for it.
 */
export const COVERED_ENTRIES: readonly CoveredEntry[] = [
  {
    entry: 'gpu-dataframe',
    barrel: 'modules/analysis/src/dataframe-analysis/index.ts',
    docs: ['docs/api/gpu-dataframe-analysis.md', 'docs/api/variable-standardization.md'],
    loadPack: async () =>
      (await import('../../src/gpu-dataframe/computation-pack/pack')).createDataFramePackData(),
    exempt: {
      GPUWideNumericScan:
        'Low-level word-phase executor shared by exact aggregates and narrowing diagnostics; the domain contracts are documented in reused-analysis-preparation.md.'
    },
    pending: GPU_DATAFRAME_PENDING
  },
  {
    entry: 'gpu-spatial-analysis',
    barrel: 'modules/analysis/src/gpu-spatial-analysis/index.ts',
    docs: ['docs/api/gpu-spatial-analysis'],
    requireVariableKinds: true,
    // These three chunked variants share the demonstrated whole-CSR computation. Algebra remains
    // pending: its whole-CSR example uses higherOrder, outside the chunked row-local contract.
    familyScenarios: {
      addBuildingTessellationReferenceRecipe: {
        scenarioId: 'gpu-spatial-analysis.building-tessellation-recipe',
        reason:
          'The legacy reference lens calls the named reference entry and captures its per-building polygon overlays; it does not demonstrate the local sample-cell-union route.'
      },
      GPUChunkedSpatialWeightsValidation: {
        scenarioId: 'gpu-spatial-analysis.spatial-weights-validation',
        reason: 'Per-chunk dispatch of the whole-CSR validation kernels.'
      },
      GPUChunkedSpatialWeightsTranspose: {
        scenarioId: 'gpu-spatial-analysis.spatial-weights-transpose',
        reason: 'Per-chunk dispatch of the whole-CSR transpose kernels.'
      },
      GPUChunkedSpatialWeightsTransform: {
        scenarioId: 'gpu-spatial-analysis.spatial-weights-transform',
        reason: 'Per-chunk dispatch of the whole-CSR transform kernels.'
      }
    },
    loadPack: async () =>
      (
        await import('../../src/gpu-spatial-analysis/computation-pack/pack')
      ).createSpatialPackData(),
    pending: GPU_SPATIAL_ANALYSIS_PENDING
  },
  {
    entry: 'gpu-network',
    barrel: 'modules/analysis/src/gpu-network/index.ts',
    docs: ['docs/api/gpu-network'],
    loadPack: async () =>
      (await import('../../src/gpu-network/computation-pack/network-pack-data'))
        .GPU_NETWORK_PACK_DATA,
    pending: GPU_NETWORK_PENDING
  },
  {
    entry: 'gpu-raster',
    barrel: 'modules/analysis/src/raster-analysis/index.ts',
    docs: ['docs/api/gpu-raster'],
    loadPack: async () =>
      (await import('../../src/gpu-raster/computation-pack/pack')).getRasterViewPack(),
    pending: GPU_RASTER_PENDING
  },
  {
    entry: 'gpu-terrain',
    barrel: 'modules/analysis/src/gpu-terrain/index.ts',
    docs: ['docs/api/gpu-terrain'],
    loadPack: async () =>
      (await import('../../src/gpu-terrain/computation-pack/pack')).getTerrainViewPack(),
    pending: GPU_TERRAIN_PENDING
  }
];

/**
 * Recorded records are written only by the main session's headless capture run. Until it has run, missing
 * `docs/lens-records/<scenario>.json` files are allowed; set LENS_COVERAGE_REQUIRE_RECORDS=1 to enforce.
 */
const ALLOW_MISSING_RECORDS = process.env['LENS_COVERAGE_REQUIRE_RECORDS'] !== '1';

it('shared discovery retains exact sibling exports, inherited methods and method-specific caches', () => {
  const first = 'modules/lenses/test/computation-docs/discovery-first.fixture.ts';
  const second = 'modules/lenses/test/computation-docs/discovery-second.fixture.ts';
  const discover = createEntryElementDiscovery([first, second]);
  expect(discover(first)).toEqual({
    contributors: ['GPUInherited'],
    recipes: ['addFixtureRecipe'],
    otherClasses: []
  });
  expect(discover(second)).toEqual({contributors: [], recipes: [], otherClasses: ['GPUPass']});
  expect(discover(second, ['getPasses'])).toEqual({
    contributors: ['GPUPass'],
    recipes: [],
    otherClasses: []
  });
  // An exported sibling must never supply a contributor to the first barrel or its method query.
  expect(discover(first, ['getPasses'])).toEqual({
    contributors: [],
    recipes: ['addFixtureRecipe'],
    otherClasses: ['GPUInherited']
  });
  expect(discover(first).contributors).toEqual(['GPUInherited']);
});

// Build lazily, once, so the coverage gate shares inherited type information across barrels.
let discoverCoveredEntries: ReturnType<typeof createEntryElementDiscovery> | undefined;
function listCoveredEntryElements(barrel: string, methods?: readonly string[]) {
  discoverCoveredEntries ??= createEntryElementDiscovery(
    COVERED_ENTRIES.flatMap(entry =>
      typeof entry.barrel === 'string' ? [entry.barrel] : [...entry.barrel]
    )
  );
  return discoverCoveredEntries(barrel, methods);
}

type CoveredEntry = {
  /** Entry id, also the scenario id prefix, e.g. `gpu-spatial-analysis`. */
  entry: string;
  /** Repo-relative barrel(s) whose exports are the elements. */
  barrel: string | readonly string[];
  /**
   * `pack-operators`: the elements are the pack's operator keys instead of every barrel contributor.
   * For entries whose barrel also exports infrastructure that is documented without lenses
   * (gpu-core command graphs and programs, gpu-raster filters and residency).
   */
  scope?: 'pack-operators';
  /** Repo-relative doc pages (directories allowed) that hold the element sections. */
  docs: readonly string[];
  /** Loads the domain pack data (plain data; must not import GPU code). */
  loadPack: () => Promise<PackCoverageData>;
  /** Require `variableKinds` on every operator descriptor (spatial pack). */
  requireVariableKinds?: boolean;
  /** Exported contributor classes intentionally without a lens, with the reason. */
  exempt?: Readonly<Record<string, string>>;
  /** Variants demonstrated by a named family scenario, with an explicit justification. */
  familyScenarios?: Readonly<Record<string, {scenarioId: string; reason: string}>>;
  /** Instance methods that make an exported class a contributor (default `getCommandNodes`). */
  contributorMethods?: readonly string[];
  /**
   * The resume list: element → what is still missing (from the stream handoffs). Pending elements
   * skip checks (b)–(e); (a) still applies unless the reason names the `pack`. A pending element
   * that passes every check fails the spec, so the list only shrinks.
   */
  pending?: Readonly<Record<string, string>>;
};
/** The pack fields the contract reads (ViewPackData plus `formulas`, see V.done). */
type PackCoverageData = Pick<ViewPackData, 'id' | 'operators'> & {
  formulas?: Readonly<Record<string, GPUComputationFormulaRef | {formula: 'none'; reason: string}>>;
};

// Wave 2 may author directives before scenarios land. Once any entry is covered, all must register.
const ALLOW_UNREGISTERED_SCENARIOS = COVERED_ENTRIES.length === 0;
const OPERATOR_SCOPES: readonly OperatorScope[] = [
  'local',
  'focal',
  'block',
  'zonal',
  'global',
  'frontier',
  'pairwise',
  'solver'
];
const OPERATOR_REDUCTIONS: readonly OperatorReduction[] = [
  'none',
  'count',
  'sum',
  'weighted-mean',
  'argmin',
  'histogram',
  'mode',
  'select',
  'scan-sort',
  'solve'
];
const DIRECTIVE_RUNGS: readonly LensRungId[] = LENS_RUNG_IDS;
const ATTRIBUTE_NAMES = ['scenario', 'rung', 'bind', 'datum', 'iteration', 'caption', 'view'];
const SLOT_ENTRIES = ['gpu-spatial-analysis', 'gpu-network', 'gpu-terrain'];

type ScannedPage = {file: string; markers: LensMarker[]; directives: ScannedLensDirective[]};

function isNonEmpty(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0;
}

function collectDirectiveFailures(
  attributes: Record<string, string>,
  scenarios: readonly LensScenarioDescriptor[],
  allowUnregistered: boolean
): string[] {
  const failures: string[] = [];
  for (const name of Object.keys(attributes)) {
    if (!ATTRIBUTE_NAMES.includes(name)) failures.push(`unknown attribute ${name}`);
  }
  const scenario = attributes['scenario'];
  const rung = attributes['rung'];
  if (!scenario || !/^[a-z0-9-]+(\.[a-z0-9-]+){1,2}$/.test(scenario)) {
    failures.push(`invalid scenario ${scenario}`);
  }
  if (!DIRECTIVE_RUNGS.some(candidate => candidate === rung)) failures.push(`invalid rung ${rung}`);
  for (const name of ['datum', 'iteration']) {
    const value = attributes[name];
    if (value !== undefined && (!/^\d+$/.test(value) || !Number.isSafeInteger(Number(value)))) {
      failures.push(`${name} must be a non-negative safe integer`);
    }
  }
  if (failures.length === 0) {
    try {
      parseLensDirective({...attributes, scenario, rung});
    } catch (error) {
      failures.push(String(error));
    }
  }
  const descriptor = scenarios.find(candidate => candidate.id === scenario);
  if (!descriptor && !allowUnregistered && !isDroppedScenario(scenario))
    failures.push(`unregistered scenario ${scenario}`);
  if (descriptor && !descriptor.rungs.some(candidate => candidate === rung)) {
    failures.push(`scenario ${scenario} does not list rung ${rung}`);
  }
  return failures;
}

function collectSlotFailures(
  elements: ReturnType<typeof listEntryElements>,
  pages: readonly ScannedPage[],
  conceptsFile: string,
  pending: Readonly<Record<string, string>> = {}
): string[] {
  const failures: string[] = [];
  const markers = pages.flatMap(page => page.markers.map(marker => ({...marker, file: page.file})));
  const required = [...elements.contributors, ...elements.recipes];
  const exported = new Set([...required, ...elements.otherClasses]);
  for (const name of required) {
    const count = markers.filter(marker => marker.name === name).length;
    // Reuse the element pending list for an explicitly absent marker only. Keep discovery,
    // duplicate markers, unknown markers and all siblings strict; entry coverage retires it.
    if (count === 0 && /\bmarker\b/.test(pending[name] ?? '')) continue;
    if (count !== 1) failures.push(`${name}: expected exactly one marker, found ${count}`);
  }
  for (const marker of markers) {
    if (marker.name === 'execution-model') {
      if (marker.file !== conceptsFile) {
        failures.push(`${marker.file}:${marker.line}: execution-model belongs in concepts.md`);
      }
    } else if (!exported.has(marker.name)) {
      failures.push(`${marker.file}:${marker.line}: marker ${marker.name} is not exported`);
    }
  }
  const conceptsCount = markers.filter(
    marker => marker.name === 'execution-model' && marker.file === conceptsFile
  ).length;
  if (conceptsCount !== 1)
    failures.push(`execution-model: expected one concepts.md marker, found ${conceptsCount}`);
  return failures;
}

function collectEntryFailures(
  entry: CoveredEntry,
  elements: ReturnType<typeof listEntryElements>,
  pack: PackCoverageData,
  pages: readonly ScannedPage[],
  scenarios: readonly LensScenarioDescriptor[],
  allowMissingRecords: boolean,
  hasRecord: (scenario: string) => boolean
): string[] {
  const failures: string[] = [];
  const names = [...elements.contributors, ...elements.recipes];
  for (const [name, coverage] of Object.entries(entry.familyScenarios ?? {})) {
    // This named example executes a cross-row frontier, not the row-local algebra subset.
    if (
      name === 'GPUChunkedSpatialWeightsAlgebra' &&
      coverage.scenarioId === 'gpu-spatial-analysis.spatial-weights-algebra'
    ) {
      failures.push(
        `${name}: family ${coverage.scenarioId} exercises unsupported operation higherOrder`
      );
    }
    const family = scenarios.find(scenario => scenario.id === coverage.scenarioId);
    if (
      !names.includes(name) ||
      !isNonEmpty(coverage.reason) ||
      family?.entry !== entry.entry ||
      !names.includes(family.element)
    ) {
      failures.push(`${name}: invalid family scenario coverage`);
    }
  }
  for (const [name, reason] of Object.entries(entry.exempt ?? {})) {
    if (!elements.contributors.includes(name) || !isNonEmpty(reason)) {
      failures.push(`${name}: invalid exemption (requires an exported contributor and a reason)`);
    }
  }
  for (const [name, reason] of Object.entries(entry.pending ?? {})) {
    if (!names.includes(name) || !isNonEmpty(reason)) {
      failures.push(`${name}: invalid pending entry (requires an element and what is missing)`);
    }
  }
  for (const name of names) {
    if (elements.contributors.includes(name) && isNonEmpty(entry.exempt?.[name])) continue;
    const familyScenarioId = entry.familyScenarios?.[name]?.scenarioId;
    const family = scenarios.find(scenario => scenario.id === familyScenarioId);
    if (familyScenarioId !== undefined && !family) continue; // Reported above.
    const elementFailures = collectElementFailures(
      entry,
      name,
      pack,
      pages,
      family ? [family] : scenarios,
      allowMissingRecords,
      hasRecord,
      family?.element
    ).map(failure => (family ? `${name}: family ${family.id}: ${failure}` : failure));
    const pending = entry.pending?.[name];
    if (pending === undefined) {
      failures.push(...elementFailures);
    } else if (elementFailures.length === 0) {
      // The resume list must shrink honestly: a finished element leaves `pending`.
      failures.push(`${name}: listed as pending (${pending}) but complete; remove it from pending`);
    } else if (!/\bpack\b/.test(pending)) {
      // A pending element whose pack entry landed must keep a valid operator descriptor.
      failures.push(...elementFailures.filter(failure => failure.startsWith(`${name}: (a)`)));
    }
  }
  return failures;
}

/** Serialized scenario-ID slugs kept for exports renamed in wave 6 (NAM-14, UDF-14). */
const RENAMED_EXPORT_SCENARIO_SLUGS: Readonly<Partial<Record<string, string>>> = {
  GPUKeyLookup: 'key-join',
  GPUDBSCAN: 'spatial-clustering',
  GPUHDBSCAN: 'hierarchical-density-clustering'
};

/** Checks (a)–(e) for one element; every failure string starts with `<name>: (<check>)`. */
function collectElementFailures(
  entry: CoveredEntry,
  name: string,
  pack: PackCoverageData,
  pages: readonly ScannedPage[],
  scenarios: readonly LensScenarioDescriptor[],
  allowMissingRecords: boolean,
  hasRecord: (scenario: string) => boolean,
  evidenceElement = name
): string[] {
  const failures: string[] = [];
  const operator = pack.operators?.[name];
  if (!operator) failures.push(`${name}: (a) missing operator`);
  else {
    if (!OPERATOR_SCOPES.includes(operator.scope)) failures.push(`${name}: (a) invalid scope`);
    if (!OPERATOR_REDUCTIONS.includes(operator.reduction))
      failures.push(`${name}: (a) invalid reduction`);
    if (entry.requireVariableKinds && Object.keys(operator.variableKinds ?? {}).length === 0) {
      failures.push(`${name}: (a) missing variableKinds`);
    }
  }
  const elementScenarios = scenarios.filter(
    scenario => scenario.entry === entry.entry && scenario.element === evidenceElement
  );
  const selectedScenarioIds = new Set(elementScenarios.map(scenario => scenario.id));
  const directives = pages.flatMap(page =>
    page.directives.filter(
      directive =>
        directive.sectionClasses.includes(evidenceElement) &&
        selectedScenarioIds.has(directive.attributes['scenario'])
    )
  );
  const formula = pack.formulas?.[name];
  if (!formula) failures.push(`${name}: (b) missing formula`);
  else if ('formula' in formula && formula.formula === 'none') {
    if (!isNonEmpty(formula.reason)) failures.push(`${name}: (b) missing formula exemption reason`);
    if (
      !directives.some(directive => ['iterations', 'graph'].includes(directive.attributes['rung']))
    ) {
      failures.push(`${name}: (b) formula:none needs an iterations or graph directive`);
    }
  } else if (
    !('tex' in formula) ||
    !isNonEmpty(formula.tex) ||
    !isNonEmpty(formula.ascii) ||
    !Array.isArray(formula.symbols) ||
    !['curated', 'derived'].includes(formula.derivation)
  )
    failures.push(`${name}: (b) invalid formula`);
  // `element` is the exported name; the kebab form is only the scenario id's second segment.
  // Scenario IDs are serialized and keep their spelling after an export rename (wave 6).
  const scenarioSlug =
    RENAMED_EXPORT_SCENARIO_SLUGS[evidenceElement] ?? getLensElementId(evidenceElement);
  if (elementScenarios.length === 0) {
    failures.push(`${name}: (c) missing registered scenario`);
  } else if (!elementScenarios.some(scenario => scenario.id.split('.')[1] === scenarioSlug)) {
    failures.push(`${name}: (c) no scenario id ${entry.entry}.${scenarioSlug}[.variant]`);
  }
  const markerCount = pages
    .flatMap(page => page.markers)
    .filter(marker => marker.name === evidenceElement).length;
  if (markerCount !== 1)
    failures.push(`${name}: (d) expected exactly one marker, found ${markerCount}`);
  if (
    !directives.some(directive =>
      elementScenarios.some(
        scenario =>
          scenario.id === directive.attributes['scenario'] && scenario.entry === entry.entry
      )
    )
  )
    failures.push(`${name}: (d) missing directive for its registered scenario`);
  if (!allowMissingRecords) {
    for (const directive of directives) {
      const scenario = directive.attributes['scenario'];
      if (!hasRecord(scenario))
        failures.push(
          `${name}: (e) missing record ${scenario} (${directive.file}:${directive.line})`
        );
    }
  }
  return failures;
}

/** The entry's elements: barrel contributors and recipes, or the pack's operator keys. */
function getCoveredElements(
  entry: CoveredEntry,
  pack: PackCoverageData
): ReturnType<typeof listEntryElements> {
  if (entry.scope === 'pack-operators') {
    return {contributors: Object.keys(pack.operators ?? {}).sort(), recipes: [], otherClasses: []};
  }
  const barrels = typeof entry.barrel === 'string' ? [entry.barrel] : entry.barrel;
  const listed = barrels.map(barrel => listCoveredEntryElements(barrel, entry.contributorMethods));
  return {
    contributors: listed.flatMap(elements => elements.contributors).sort(),
    recipes: listed.flatMap(elements => elements.recipes).sort(),
    otherClasses: listed.flatMap(elements => elements.otherClasses).sort()
  };
}

function selectPages(pages: readonly ScannedPage[], paths: readonly string[]): ScannedPage[] {
  const absolutePaths = paths.map(path => resolve(REPOSITORY_ROOT, path));
  return pages.filter(page =>
    absolutePaths.some(path => page.file === path || page.file.startsWith(`${path}${sep}`))
  );
}

describe('lens coverage contract', () => {
  const files = listDocsMarkdownFiles();
  const scanFailures: string[] = [];
  const pages = files.map(file => {
    const text = readFileSync(file, 'utf8');
    let directives: ScannedLensDirective[] = [];
    try {
      directives = scanLensDirectives(file, text);
    } catch (error) {
      scanFailures.push(String(error));
    }
    return {file, markers: scanLensMarkers(file, text), directives};
  });

  it('validates every authored directive and its registered rungs', () => {
    expect(files.length).toBeGreaterThan(0);
    const failures = [...scanFailures];
    const scenarios = listLensScenarios();
    for (const page of pages) {
      for (const directive of page.directives) {
        const location = `${relative(REPOSITORY_ROOT, page.file)}:${directive.line}`;
        failures.push(
          ...collectDirectiveFailures(
            directive.attributes,
            scenarios,
            ALLOW_UNREGISTERED_SCENARIOS
          ).map(failure => `${location}: ${failure}`)
        );
      }
    }
    expect(failures).toEqual([]);
  });

  it('allows only explicitly pending captures and retires their entries on capture', () => {
    expect(scanFailures).toEqual([]);
    const scenarioIds = new Set(
      pages.flatMap(page => page.directives.map(directive => directive.attributes['scenario']))
    );
    expect(
      collectPendingRecordFailures(scenarioIds, PENDING_LENS_RECORDS, scenarioId =>
        existsSync(join(DOCS_ROOT, 'lens-records', `${scenarioId}.json`))
      )
    ).toEqual([]);
  });

  for (const entry of SLOT_ENTRIES) {
    it(`${entry} has exactly one slot for every contributor and recipe`, () => {
      const directory = join(DOCS_ROOT, 'api', entry);
      const elements = listCoveredEntryElements(`modules/analysis/src/${entry}/index.ts`);
      expect(elements.contributors.length).toBeGreaterThan(0);
      expect(
        collectSlotFailures(
          elements,
          selectPages(pages, [directory]),
          join(directory, 'concepts.md'),
          COVERED_ENTRIES.find(covered => covered.entry === entry)?.pending
        )
      ).toEqual([]);
    }, 120_000);
  }

  for (const entry of COVERED_ENTRIES) {
    it(`${entry.entry} covers every contributor and recipe with a pack, scenario and lens`, async () => {
      const failures = [...scanFailures];
      try {
        const pack = await entry.loadPack();
        const elements = getCoveredElements(entry, pack);
        failures.push(
          ...collectEntryFailures(
            entry,
            elements,
            pack,
            selectPages(pages, entry.docs),
            listLensScenarios(),
            ALLOW_MISSING_RECORDS,
            scenario => existsSync(join(DOCS_ROOT, 'lens-records', `${scenario}.json`))
          )
        );
      } catch (error) {
        failures.push(`${entry.entry}: pack coverage failed: ${String(error)}`);
      }
      expect(failures).toEqual([]);
    }, 120_000);
  }

  it('validates every captured record and its scenario identity', () => {
    const directory = join(DOCS_ROOT, 'lens-records');
    const failures: string[] = [];
    const scenarios = listLensScenarios();
    for (const file of existsSync(directory) ? readdirSync(directory, {withFileTypes: true}) : []) {
      if (!file.isFile() || !file.name.endsWith('.json')) continue;
      try {
        const record = fromRecordedLensFile(
          JSON.parse(readFileSync(join(directory, file.name), 'utf8'))
        );
        if (record.scenarioId !== basename(file.name, '.json'))
          failures.push(`${file.name}: scenarioId does not match filename`);
        if (
          !ALLOW_UNREGISTERED_SCENARIOS &&
          !scenarios.some(scenario => scenario.id === record.scenarioId)
        ) {
          failures.push(`${file.name}: unregistered scenario ${record.scenarioId}`);
        }
      } catch (error) {
        failures.push(`${file.name}: ${String(error)}`);
      }
    }
    expect(failures).toEqual([]);
  });
});

describe('coverage helpers', () => {
  it('registers own-name operators and formulas for the three demonstrated chunked weights variants', () => {
    for (const name of [
      'GPUChunkedSpatialWeightsValidation',
      'GPUChunkedSpatialWeightsTranspose',
      'GPUChunkedSpatialWeightsTransform'
    ]) {
      expect(WEIGHTS_OPERATORS[name], name).toBeDefined();
      expect(WEIGHTS_OPERATORS[name]?.formula, name).toBeDefined();
    }
  });
  it('rejects higherOrder algebra as coverage for the chunked row-local variant', () => {
    expect(
      readFileSync(
        join(
          REPOSITORY_ROOT,
          'modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-weights-algebra.example.ts'
        ),
        'utf8'
      )
    ).toContain("operation: 'higherOrder'");
    const covered = COVERED_ENTRIES.find(entry => entry.entry === 'gpu-spatial-analysis')!;
    const family = listLensScenarios().find(
      scenario => scenario.id === 'gpu-spatial-analysis.spatial-weights-algebra'
    )!;
    expect(
      collectEntryFailures(
        {
          ...covered,
          pending: {},
          exempt: {},
          familyScenarios: {
            GPUChunkedSpatialWeightsAlgebra: {scenarioId: family.id, reason: 'Algebra family'}
          }
        },
        {
          contributors: ['GPUChunkedSpatialWeightsAlgebra', 'GPUSpatialWeightsAlgebra'],
          recipes: [],
          otherClasses: []
        },
        {id: 'fixture'},
        [],
        [family],
        true,
        () => true
      )
    ).toContain(
      'GPUChunkedSpatialWeightsAlgebra: family gpu-spatial-analysis.spatial-weights-algebra exercises unsupported operation higherOrder'
    );
  });
  it('rejects unlisted capture gaps, unexplained entries and obsolete pending entries', () => {
    expect(
      collectPendingRecordFailures(
        new Set(['test.lost', 'test.pending', 'test.captured']),
        {
          'test.pending': '',
          'test.captured': 'Awaiting capture',
          'test.removed': 'Awaiting capture'
        },
        scenarioId => scenarioId === 'test.captured'
      )
    ).toEqual([
      'test.captured: captured; remove pending entry',
      'test.lost: missing record without a documented pending entry',
      'test.pending: missing record without a documented pending entry',
      'test.pending: pending entry needs a reason',
      'test.removed: pending entry has no directive'
    ]);
  });
  const elements = {contributors: ['GPUExample'], recipes: [], otherClasses: []};
  const entry: CoveredEntry = {
    entry: 'test',
    barrel: 'unused.ts',
    docs: ['inline.md'],
    loadPack: async () => ({id: 'test', operators: {}})
  };
  const scenario: LensScenarioDescriptor = {
    id: 'test.example',
    entry: 'test',
    element: 'GPUExample',
    title: 'Example',
    audience: 'analyst',
    rungs: ['graph'],
    defaultRung: 'graph',
    load: async () => {
      throw new Error('Coverage must never load GPU code');
    }
  };
  it('does not borrow a sibling graph for a formula exemption', () => {
    const own = {...scenario, rungs: ['result', 'graph'] as LensRungId[]};
    const sibling = {...scenario, id: 'test.other', element: 'GPUOther'};
    const pack: PackCoverageData = {
      id: 'test',
      operators: {GPUExample: {scope: 'local', reduction: 'none'}},
      formulas: {GPUExample: {formula: 'none', reason: 'Structural operator'}}
    };
    const page = makePage(
      '### Example\n<!-- lens:GPUExample -->\n::lens{scenario=test.example rung=result}\n::lens{scenario=test.other rung=graph}'
    );
    expect(
      collectEntryFailures(entry, elements, pack, [page], [own, sibling], true, () => true)
    ).toContain('GPUExample: (b) formula:none needs an iterations or graph directive');
  });
  function makePage(text: string, file = 'inline.md'): ScannedPage {
    return {file, markers: scanLensMarkers(file, text), directives: scanLensDirectives(file, text)};
  }

  it('reports a missing marker and rejects unknown or duplicate slots', () => {
    const concepts = makePage('## Execution\n<!-- lens:execution-model -->', 'concepts.md');
    expect(collectSlotFailures(elements, [concepts], 'concepts.md')).toEqual([
      'GPUExample: expected exactly one marker, found 0'
    ]);
    const page = makePage(
      '### Example\n<!-- lens:GPUExample -->\n<!-- lens:GPUExample -->\n<!-- lens:Unknown -->'
    );
    expect(collectSlotFailures(elements, [concepts, page], 'concepts.md')).toEqual([
      'GPUExample: expected exactly one marker, found 2',
      'inline.md:4: marker Unknown is not exported'
    ]);
  });

  it('rejects bad rungs, unknown attributes, invalid pins and unregistered scenarios', () => {
    expect(collectDirectiveFailures({scenario: 'test.example', rung: 'bogus'}, [], true)).toContain(
      'invalid rung bogus'
    );
    expect(
      collectDirectiveFailures(
        {scenario: 'test.example', rung: 'graph', datum: '-1', surprise: 'yes'},
        [],
        true
      )
    ).toEqual(['unknown attribute surprise', 'datum must be a non-negative safe integer']);
    expect(collectDirectiveFailures({scenario: 'test.example', rung: 'graph'}, [], false)).toEqual([
      'unregistered scenario test.example'
    ]);
    expect(
      collectDirectiveFailures({scenario: 'test.example', rung: 'datum'}, [scenario], false)
    ).toEqual(['scenario test.example does not list rung datum']);
  });

  it('reports all coverage failures for an empty pack', async () => {
    expect(
      collectEntryFailures(entry, elements, await entry.loadPack(), [], [], true, () => false)
    ).toEqual([
      'GPUExample: (a) missing operator',
      'GPUExample: (b) missing formula',
      'GPUExample: (c) missing registered scenario',
      'GPUExample: (d) expected exactly one marker, found 0',
      'GPUExample: (d) missing directive for its registered scenario'
    ]);
  });

  it('enforces formula fallbacks, variable kinds and the capture gate', () => {
    const pack: PackCoverageData = {
      id: 'test',
      operators: {GPUExample: {scope: 'local', reduction: 'none'}},
      formulas: {GPUExample: {formula: 'none', reason: 'Structural operator'}}
    };
    const page = makePage(
      '### Example\n<!-- lens:GPUExample -->\n::lens{scenario=test.example rung=graph}'
    );
    expect(
      collectEntryFailures(entry, elements, pack, [page], [scenario], true, () => false)
    ).toEqual([]);
    expect(
      collectEntryFailures(
        {...entry, requireVariableKinds: true},
        elements,
        pack,
        [page],
        [scenario],
        false,
        () => false
      )
    ).toEqual([
      'GPUExample: (a) missing variableKinds',
      'GPUExample: (e) missing record test.example (inline.md:3)'
    ]);
    const noFallback = makePage(
      '### Example\n<!-- lens:GPUExample -->\n::lens{scenario=test.example rung=datum}'
    );
    expect(
      collectEntryFailures(entry, elements, pack, [noFallback], [scenario], true, () => false)
    ).toContain('GPUExample: (b) formula:none needs an iterations or graph directive');
  });

  it('rejects a sibling scenario in an element section', () => {
    const pack: PackCoverageData = {
      id: 'test',
      operators: {GPUExample: {scope: 'local', reduction: 'none'}},
      formulas: {GPUExample: {formula: 'none', reason: 'Structural operator'}}
    };
    const sibling = {...scenario, id: 'test.other', element: 'GPUOther'};
    const page = makePage(
      '### Example\n<!-- lens:GPUExample -->\n::lens{scenario=test.other rung=graph}'
    );
    expect(
      collectEntryFailures(entry, elements, pack, [page], [scenario, sibling], true, () => false)
    ).toContain('GPUExample: (d) missing directive for its registered scenario');
  });

  it('covers a variant only through its explicitly named family scenario', () => {
    const familyEntry: CoveredEntry = {
      ...entry,
      familyScenarios: {
        GPUVariant: {scenarioId: scenario.id, reason: 'Same observable family computation'}
      }
    };
    const familyElements = {...elements, contributors: ['GPUExample', 'GPUVariant']};
    const pack: PackCoverageData = {
      id: 'test',
      operators: {
        GPUExample: {scope: 'local', reduction: 'none'},
        GPUVariant: {scope: 'local', reduction: 'none'}
      },
      formulas: {
        GPUExample: {formula: 'none', reason: 'Structural operator'},
        GPUVariant: {formula: 'none', reason: 'Structural variant'}
      }
    };
    const page = makePage(
      '### Example\n<!-- lens:GPUExample -->\n::lens{scenario=test.example rung=graph}'
    );
    expect(
      collectEntryFailures(familyEntry, familyElements, pack, [page], [scenario], true, () => true)
    ).toEqual([]);
    expect(
      collectEntryFailures(
        {
          ...familyEntry,
          familyScenarios: {
            GPUVariant: {scenarioId: 'test.missing', reason: 'Same computation'}
          }
        },
        familyElements,
        pack,
        [page],
        [scenario],
        true,
        () => true
      )
    ).toContain('GPUVariant: invalid family scenario coverage');
    expect(
      collectEntryFailures(familyEntry, familyElements, pack, [], [scenario], true, () => true)
    ).toContain(
      'GPUVariant: family test.example: GPUVariant: (d) missing directive for its registered scenario'
    );
  });

  it('S2: named tessellation routes require their own constructed-pack operator and formula with family evidence intact', async () => {
    const {createSpatialPackData} = await import(
      '../../src/gpu-spatial-analysis/computation-pack/pack'
    );
    const pack = createSpatialPackData();
    const reference = 'addBuildingTessellationReferenceRecipe';
    const legacy = 'addBuildingTessellationRecipe';
    const local = 'addLocalBuildingTessellationRecipe';
    const covered: CoveredEntry = {
      entry: 'gpu-spatial-analysis',
      barrel: 'unused',
      docs: [],
      loadPack: async () => pack,
      requireVariableKinds: true,
      familyScenarios: {
        [reference]: {
          scenarioId: 'gpu-spatial-analysis.building-tessellation-recipe',
          reason: 'Calls the named reference entry'
        }
      }
    };
    const elements = {contributors: [], recipes: [legacy, reference, local], otherClasses: []};
    const pages = [
      makePage(
        `## Reference\n<!-- lens:${legacy} -->\n<!-- lens:${reference} -->\n::lens{scenario=gpu-spatial-analysis.building-tessellation-recipe rung=graph}\n## Local\n<!-- lens:${local} -->\n::lens{scenario=gpu-spatial-analysis.local-building-tessellation-recipe rung=graph}`
      )
    ];
    const registered = listLensScenarios().filter(scenario =>
      [legacy, local].includes(scenario.element)
    );
    expect(
      collectEntryFailures(covered, elements, pack, pages, registered, true, () => true)
    ).toEqual([]);
    const formulas = {...pack.formulas};
    delete formulas[reference];
    expect(formulas[legacy]).toBeDefined();
    expect(
      collectEntryFailures(
        covered,
        elements,
        {...pack, formulas},
        pages,
        registered,
        true,
        () => true
      )
    ).toContain(
      `${reference}: family gpu-spatial-analysis.building-tessellation-recipe: ${reference}: (b) missing formula`
    );
  });

  async function expectStopCubePendingFixtures(spatialEntry: CoveredEntry) {
    const name = 'addStopValueCubeRecipe';
    const productionPack = await spatialEntry.loadPack();
    expect(getCoveredElements(spatialEntry, productionPack).recipes).toContain(name);
    const pack: PackCoverageData = {id: spatialEntry.entry, operators: {}};
    const pending = {
      [name]: 'fixture: missing pack, formula, scenario, marker, directives and recording'
    };
    const stopElements = {contributors: [], recipes: [name], otherClasses: []};
    const concepts = makePage('## Execution\n<!-- lens:execution-model -->', 'concepts.md');
    expect(collectSlotFailures(stopElements, [concepts], 'concepts.md', pending)).toEqual([]);
    expect(
      collectSlotFailures(
        {...stopElements, contributors: ['GPUExample']},
        [concepts],
        'concepts.md',
        pending
      )
    ).toEqual(['GPUExample: expected exactly one marker, found 0']);
    const duplicate = makePage(`## Stop cube\n<!-- lens:${name} -->\n<!-- lens:${name} -->`);
    expect(
      collectSlotFailures(stopElements, [concepts, duplicate], 'concepts.md', pending)
    ).toEqual([`${name}: expected exactly one marker, found 2`]);
    expect(
      collectEntryFailures(
        {...spatialEntry, familyScenarios: {}, pending},
        stopElements,
        pack,
        [],
        [],
        false,
        () => false
      )
    ).toEqual([]);
    const stopScenario = {
      ...scenario,
      id: 'gpu-spatial-analysis.stop-value-cube-recipe',
      entry: spatialEntry.entry,
      element: name
    };
    const completed: PackCoverageData = {
      id: spatialEntry.entry,
      operators: {
        [name]: {
          scope: 'zonal',
          reduction: 'sum',
          variableKinds: {counts: {level: 'ratio', extent: 'extensive'}}
        }
      },
      formulas: {[name]: {formula: 'none', reason: 'Recipe composition shown by its graph'}}
    };
    const page = makePage(
      `### Stop cube\n<!-- lens:${name} -->\n::lens{scenario=${stopScenario.id} rung=graph}`
    );
    const isolatedEntry = {
      ...spatialEntry,
      familyScenarios: {},
      pending
    };
    expect(
      collectEntryFailures(
        isolatedEntry,
        stopElements,
        completed,
        [page],
        [stopScenario],
        false,
        () => true
      )
    ).toEqual([
      `${name}: listed as pending (${pending[name]}) but complete; remove it from pending`
    ]);
    expect(
      collectEntryFailures(
        {...isolatedEntry, pending: {}},
        stopElements,
        completed,
        [page],
        [stopScenario],
        false,
        () => true
      )
    ).toEqual([]);
    expect(
      collectEntryFailures(
        {...isolatedEntry, pending: {}},
        stopElements,
        completed,
        [page],
        [stopScenario],
        false,
        () => false
      )
    ).toEqual([`${name}: (e) missing record ${stopScenario.id} (inline.md:3)`]);
  }

  it('CMP-12: stop-cube pending coverage is element-specific and retires when complete', async () => {
    const spatialEntry = COVERED_ENTRIES.find(
      candidate => candidate.entry === 'gpu-spatial-analysis'
    )!;
    await expectStopCubePendingFixtures(spatialEntry);
    const retiredPending = {...spatialEntry.pending};
    delete retiredPending['addStopValueCubeRecipe'];
    await expectStopCubePendingFixtures({...spatialEntry, pending: retiredPending});
  });

  it('skips pending elements honestly and fails once they are complete', async () => {
    const page = makePage(
      '### Example\n<!-- lens:GPUExample -->\n::lens{scenario=test.example rung=graph}'
    );
    const pack: PackCoverageData = {
      id: 'test',
      operators: {GPUExample: {scope: 'local', reduction: 'none'}},
      formulas: {GPUExample: {formula: 'none', reason: 'Structural operator'}}
    };
    const pending = (reason: string) => ({...entry, pending: {GPUExample: reason}});
    // Missing scenario, docs and formula are skipped; the landed operator is still checked.
    expect(
      collectEntryFailures(
        pending('docs card'),
        elements,
        await entry.loadPack(),
        [],
        [],
        true,
        () => false
      )
    ).toEqual(['GPUExample: (a) missing operator']);
    expect(
      collectEntryFailures(
        pending('pack + formula + scenario'),
        elements,
        await entry.loadPack(),
        [],
        [],
        true,
        () => false
      )
    ).toEqual([]);
    expect(
      collectEntryFailures(
        pending('docs card'),
        elements,
        pack,
        [page],
        [scenario],
        true,
        () => false
      )
    ).toEqual(['GPUExample: listed as pending (docs card) but complete; remove it from pending']);
    expect(
      collectEntryFailures(
        {...entry, pending: {GPUOther: 'docs card'}},
        elements,
        pack,
        [page],
        [scenario],
        true,
        () => false
      )
    ).toEqual(['GPUOther: invalid pending entry (requires an element and what is missing)']);
  });

  it('parses quoted attributes and rejects malformed lists', () => {
    expect(
      parseDirectiveAttributes(
        'scenario=test.example rung=graph caption="A quoted caption" bind=\'page\''
      )
    ).toEqual({scenario: 'test.example', rung: 'graph', caption: 'A quoted caption', bind: 'page'});
    for (const text of [
      'rung=graph trailing',
      'rung="graph',
      'rung=graph rung=datum',
      'caption=unquoted'
    ]) {
      expect(() => parseDirectiveAttributes(text)).toThrow();
    }
  });

  it.each([
    'html',
    'mdx'
  ])('ignores fences and associates %s markers before and after directives', format => {
    const marker = (name: string) =>
      format === 'html' ? `<!-- lens:${name} -->` : `{/* lens:${name} */}`;
    const text = [
      '```md',
      '### Fake',
      marker('Fake'),
      '::lens{rung=bad}',
      '```',
      '~~~md',
      '::lens{rung=bad}',
      '~~~',
      '### Shared',
      '::lens{scenario=test.example rung=graph}',
      marker('GPUExample'),
      '#### How it computes',
      marker('GPUOther'),
      '::lens{scenario=test.example rung=datum}',
      '## Boundary',
      '::lens{scenario=test.example rung=graph}',
      '### Next',
      marker('GPUNext'),
      '::lens{scenario=test.example rung=graph}'
    ].join('\n');
    expect(
      scanLensDirectives('inline.md', text).map(directive => ({
        line: directive.line,
        classes: directive.sectionClasses
      }))
    ).toEqual([
      {line: 10, classes: ['GPUExample', 'GPUOther']},
      {line: 14, classes: ['GPUExample', 'GPUOther']},
      {line: 16, classes: []},
      {line: 19, classes: ['GPUNext']}
    ]);
    expect(scanLensMarkers('inline.md', text)[0]).toEqual({
      name: 'GPUExample',
      line: 11,
      section: {start: 9, end: 14}
    });
  });
});
