// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {evaluateGPUAnalysisParameterCondition} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';

import {
  deriveGPUAnalysisCautions,
  describeGPUAnalysisPermutationResolution,
  formatGPUAnalysisMeasuredValue,
  formatGPUAnalysisNumber,
  getGPUAnalysisCautionText,
  getGPUAnalysisSignificanceAttainability,
  getGPUAnalysisQuestion,
  GPU_ANALYSIS_CAUTIONS,
  summarizeGPUAnalysisResult
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import type {
  GPUAnalysisColumnProfile,
  GPUAnalysisDataProfile,
  GPUAnalysisCaution,
  GPUAnalysisCautionFacts,
  GPUAnalysisMeasurementLevel,
  GPUAnalysisMultipleTestingCorrection,
  GPUAnalysisQuestion,
  GPUAnalysisSidedness,
  GPUAnalysisSummaryTemplate,
  GPUSpatialAnalysisDataKind
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import type {
  CatalogOperationBinding,
  CatalogParameterSpec,
  CatalogRunResult
} from '../analysis/catalog-binding-types';
import type {CatalogDiagnostic, CatalogOutputTable} from '../analysis/catalog-geojson';
import type {KeplerParameterResolution} from '../descriptors/default-parameters';
import {CURATED_RUNTIME_CAUTION_SUMMARY_KEYS} from '../descriptors/descriptors-curated';
import type {
  KeplerOperationDescriptor,
  KeplerOperationOutputDescriptor,
  KeplerOperationParameterDescriptor
} from '../descriptors/operation-descriptor-types';
import {planKeplerResultPresentation} from './result-presentation';
import type {KeplerResultPresentation} from './result-presentation';

// Everything a user reads next to a result: one sentence built from the run's own numbers, the
// parameters that were chosen for them, the cautions that apply, the questions that follow and
// what engine produced it. Pure CPU code. The sentence states only numbers it found in the
// result; when a template needs a number the result does not hold, the sentence is the row count.

// ---------------------------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------------------------

/**
 * The part of a dataset profile (`KeplerDatasetProfile`) the report reads: what the features are,
 * how the coordinate system is known and what each column measures.
 */
export type KeplerReportProfile = {
  kind?: GPUSpatialAnalysisDataKind;
  catalogGeometry?: 'point' | 'line' | 'polygon';
  pointColumns?: {longitude: string; latitude: string};
  /** Retained declarations used when promoting a run to reusable sources. */
  rowSemantics?: GPUAnalysisDataProfile['rowSemantics'];
  coordinates?: {
    model: string;
    source: 'guess' | 'declared';
    confidence?: 'high' | 'low';
    reason?: string;
  };
  columns?: readonly {
    name: string;
    measurement?: GPUAnalysisMeasurementLevel;
    measurementSource?: GPUAnalysisColumnProfile['measurementSource'];
  }[];
};

export type KeplerResultReportContext = {
  /** `resolutions` of `resolveKeplerOperationParameters`. */
  resolutions?: readonly KeplerParameterResolution[];
  /** Profiles by `CatalogInputSpec.name`; they tell the report what the inputs are and how their columns were measured. */
  profiles?: Readonly<Record<string, KeplerReportProfile | undefined>>;
  /**
   * The columns bound to the string parameters (`fieldBindings` of the applicability result or of
   * `resolveKeplerOperationParameters`). With `profiles` they say what each bound column measures,
   * and a count or amount column bound to a role that does not list that level raises
   * `counts-where-rates-expected`.
   */
  fieldBindings?: Readonly<Record<string, string | readonly string[]>>;
  /** Parameters that the curated analysis fixed, so that an `override` of them is not "set by you". */
  presetParameters?: readonly string[];
  /** Host warnings about the inputs, passed through to the report. */
  warnings?: readonly string[];
  /** Rows of each input, by `CatalogInputSpec.name`; used when the result does not say how many were analysed. */
  featureCounts?: Readonly<Record<string, number>>;
  /** Overrides what the profiles say about the coordinate system. */
  guessedCoordinates?: boolean;
  /** Several settings were tried and compared. */
  parameterSearch?: boolean;
};

// ---------------------------------------------------------------------------------------------
// What each binding's result holds
// ---------------------------------------------------------------------------------------------

type KeplerCorrection = GPUAnalysisMultipleTestingCorrection;

type KeplerStatisticRow = {
  /** Table the statistic is a row of. */
  tableId: string;
  /** Column that names the row, and the value that selects it; absent for a one-row table. */
  matchColumn?: string;
  matchValue?: string | number;
  /** Name as shown in the sentence. */
  name: string;
  valueColumn: string;
  pValueColumn: string;
  /** Direction of the p-value, read from the binding's code. */
  sidedness: GPUAnalysisSidedness;
  /** What the p-value is computed against, when it is not a random pattern. */
  nullDescription?: string;
  /** What a positive and a negative value mean. */
  positiveMeaning?: string;
  negativeMeaning?: string;
};

type KeplerResultFactSource = {
  /** Summary key that counts the features that were tested; absent when every row was tested. */
  testedCountKey?: string;
  /** Summary key that counts the features a global statistic was computed on. */
  analysedCountKey?: string;
  /** How the binding corrects for many tests, from a parameter or fixed by the method. */
  correction?:
    | {constant: KeplerCorrection}
    | {parameter: string; values: Readonly<Record<string, KeplerCorrection>>};
  /** Parameter that holds the number of permutations; absent for analytic tests. */
  permutationsParameter?: string;
  /** Parameter that holds the p-value cut-off. */
  significanceParameter?: string;
  /** Column of the table that holds each row's neighbour count (0 for an island). */
  neighborCountColumn?: string;
  /** Whether the binding itself leaves islands out of its classes (`not-tested`). */
  producerExcludesIslands?: boolean;
  /** Where `global-statistic` finds its numbers: a row of a table, not the summary. */
  statistic?: KeplerStatisticRow;
  /** Summary keys of `regression-fit`. */
  regression?: {observationKey: string; rSquaredKey: string; adjustedKey?: string};
  /** Column of the per-area table that holds a count of items, for `count-per-area`. */
  countColumn?: string;
  /** The count column is a count only while this parameter holds this value. */
  countWhen?: {parameter: string; equals: string};
  /** Summary keys and parameters of a p-value adjustment: how many of the tests were rejected. */
  rejected?: {
    testedKey: string;
    rejectedKey: string;
    alphaParameter: string;
    methodParameter: string;
  };
  /** Column a `value-range` sentence reads when the output has no presentation column, by table id. */
  valueColumns?: Readonly<Record<string, string>>;
  /** Sentence template to use for a table instead of the descriptor's, by table id. */
  outputTemplates?: Readonly<Record<string, GPUAnalysisSummaryTemplate>>;
  featureNoun?: {singular: string; plural: string};
  areaNoun?: {singular: string; plural: string};
};

const CELL_NOUN = {singular: 'cell', plural: 'cells'};
const ZONE_NOUN = {singular: 'zone', plural: 'zones'};

/**
 * What the registered bindings write, by binding id. A binding that is not listed gets the generic
 * reading: cluster counts and value ranges from the presentation column, `regression-fit` from the
 * summary keys `rowCount`, `rSquared` and `adjustedRSquared`, and the row count for the rest.
 * Source lines are of the binding's run function and the lines that write the keys and columns.
 */
export const KEPLER_RESULT_FACT_SOURCES: Readonly<Record<string, KeplerResultFactSource>> = {
  // hotspot-gi-star: run `catalog-platform3-b-statistics.ts:544`; classes at `:589` (fixed 1.645,
  // 1.96 and 2.576 thresholds, `:572`); summary `{featureCount}` at `:600`. No parameter for
  // significance or correction, so the sentence says no correction was applied.
  // Islands have neighbor_count 0, null inference and the not-tested class at the producer.
  'hotspot-gi-star': {
    correction: {constant: 'none'},
    neighborCountColumn: 'neighbor_count',
    producerExcludesIslands: true
  },
  // recipe-hot-spot-lattice: `catalog-platform5-recipes-hot-spot.ts:320`; bin column at `:278`,
  // `testedCellCount` at `:313` (cells with a z-score, the rest are bin 0); the parameter
  // `multipleTesting` (`:339`) is `none` or `benjamini-hochberg`.
  'recipe-hot-spot-lattice': {
    testedCountKey: 'testedCellCount',
    correction: {
      parameter: 'multipleTesting',
      values: {none: 'none', 'benjamini-hochberg': 'false-discovery-rate'}
    },
    featureNoun: CELL_NOUN
  },
  // local-moran-statistics: run `catalog-platform3-b-statistics.ts:393`; classes at `:441-455`
  // (`not-tested` for a row without a p-value); summary `{featureCount, permutations}` at `:468`;
  // parameters `permutations` and `significance` (`catalog-platform3-b-bindings.ts:1010`). The
  // multipleTesting parameter selects none, Benjamini-Hochberg (default) or Bonferroni.
  // `neighbor_count` is 0 for an island, which gets no p-value and the class `not-tested`.
  'local-moran-statistics': {
    correction: {
      parameter: 'multipleTesting',
      values: {none: 'none', 'benjamini-hochberg': 'false-discovery-rate', bonferroni: 'bonferroni'}
    },
    permutationsParameter: 'permutations',
    significanceParameter: 'significance',
    neighborCountColumn: 'neighbor_count',
    producerExcludesIslands: true
  },
  // global-spatial-statistics: run `catalog-platform3-b-statistics.ts:247`; the statistics are
  // rows of table `global` (`:331`, column `statistic`), not summary keys; the summary is
  // `{featureCount, linkCount}` (`:348`). Analytic normal approximations, so no permutations.
  'global-spatial-statistics': {
    analysedCountKey: 'featureCount',
    statistic: {
      tableId: 'global',
      matchColumn: 'statistic',
      matchValue: 'morans-i',
      name: "Moran's I",
      valueColumn: 'value',
      pValueColumn: 'p_value',
      // `getNormalTwoSidedPValue` (`catalog-platform3-b-common.ts:265`).
      sidedness: 'two-sided',
      positiveMeaning: 'similar values near each other',
      negativeMeaning: 'dissimilar values near each other'
    }
  },
  // global-join-counts-getis-ord (GPU only): `catalog-platform5-lisa.ts:295`; row `getis-ord-g`
  // of table `global` (`:119`); summary `{featureCount}` (`:157`); analytic p-value.
  'global-join-counts-getis-ord': {
    analysedCountKey: 'featureCount',
    statistic: {
      tableId: 'global',
      matchColumn: 'statistic',
      matchValue: 'getis-ord-g',
      name: 'Getis-Ord General G',
      valueColumn: 'value',
      pValueColumn: 'p_value',
      // The contributor's `alternative` defaults to two-sided (`gpu-global-spatial-statistics.ts:188`).
      sidedness: 'two-sided'
    }
  },
  // knox-test: run `catalog-platform3-b-statistics.ts:766`; one row in table `knox` with
  // `ratio` (`:848`, observed over expected joint close pairs, null when none were expected) and
  // `p_value`; summary `{eventCount, jointClosePairs}` (`:851`); parameter `permutations`.
  'knox-test': {
    analysedCountKey: 'eventCount',
    statistic: {
      tableId: 'knox',
      name: 'The ratio of observed to expected pairs that are close in space and time',
      valueColumn: 'ratio',
      pValueColumn: 'p_value',
      // Upper tail: permutations with at least as many joint close pairs (`catalog-platform3-b-statistics.ts:831`).
      sidedness: 'greater',
      nullDescription: 'no space-time interaction'
    },
    permutationsParameter: 'permutations'
  },
  // regression-ols: `catalog-platform4-regression.ts:302`; summary `{rowCount, predictorCount,
  // rSquared, adjustedRSquared, ...}` at `:236-243`. regression-gwr (`catalog-platform5-
  // regression.ts:371`), regression-probit (`:645`) and regression-sur (`:1015`) write no R
  // squared; see their entries below.
  'regression-ols': {
    regression: {
      observationKey: 'rowCount',
      rSquaredKey: 'rSquared',
      adjustedKey: 'adjustedRSquared'
    }
  },
  // cell-aggregation: `catalog-platform3-b-zones.ts:105`; int32 column `count` (`:120`).
  'cell-aggregation': {countColumn: 'count', areaNoun: CELL_NOUN},
  // zonal-point-statistics: `catalog-platform3-b-zones.ts:272`; column `count`.
  'zonal-point-statistics': {countColumn: 'count', areaNoun: ZONE_NOUN},
  // dggs-cell-aggregation: `catalog-platform4-dggs.ts:554`; column `count`.
  'dggs-cell-aggregation': {countColumn: 'count', areaNoun: CELL_NOUN},
  // recipe-points-in-polygons-choropleth: `catalog-platform5-recipes.ts:267`; `point_count`
  // (`:226`) counts the points of each polygon whatever the statistic in `value`.
  'recipe-points-in-polygons-choropleth': {countColumn: 'point_count'},
  // raster-join-points-in-polygons: `catalog-platform4-raster.ts:224`; column `count`.
  'raster-join-points-in-polygons': {countColumn: 'count'},
  // point-density-grid: `catalog-platform3-b-bindings.ts:423`; the `value` column is a count only
  // when the parameter `statistic` is `count` (`catalog-platform3-b-grids.ts:375`).
  'point-density-grid': {
    countColumn: 'value',
    countWhen: {parameter: 'statistic', equals: 'count'},
    areaNoun: CELL_NOUN
  },
  // multiple-testing-adjustment: `catalog-platform3-b-statistics.ts:689`; summary `{testCount,
  // rejectedCount}` (`:757`); parameters `method` and `alpha`. No template says "N of M rejected",
  // so the sentence is written here (`describeRejectedSentence`).
  'multiple-testing-adjustment': {
    rejected: {
      testedKey: 'testCount',
      rejectedKey: 'rejectedCount',
      alphaParameter: 'alpha',
      methodParameter: 'method'
    }
  },
  // network-k-function: `catalog-platform3-a-bindings.ts:1282`, table `network-k` (`catalog-
  // platform3-a-network.ts:969`). It has no significance test and no p-value ("no edge correction,
  // Monte Carlo envelope or significance test"), so a `global-statistic` sentence would invent one;
  // its descriptor says `value-range` without a presentation column, and the honest sentence is the
  // range of `excess` (K minus the reference) over the radii.
  'network-k-function': {valueColumns: {'network-k': 'excess'}},
  // regression-gwr: summary `{rowCount, bandwidth, residualSumOfSquares, effectiveParameters,
  // sigmaSquared, singularCount}` (`catalog-platform5-regression.ts:371`); it publishes no R
  // squared, so the sentence is the range of the residuals. regression-probit (`:645`) publishes
  // log-likelihood and deviance and regression-sur (`:1015`) equation counts; their fitted
  // probabilities and residuals get the range, their coefficient tables the row count.
  'regression-gwr': {outputTemplates: {'gwr-local': 'value-range'}},
  'regression-probit': {outputTemplates: {'probit-fitted': 'value-range'}},
  'regression-sur': {outputTemplates: {'sur-residuals': 'value-range'}},
  // curated-rate-cluster-map: `curated/bindings/curated-rate-cluster-map-binding.ts:284`; classes in
  // `cluster` (`:249`, `not-tested` for rows without a p-value); summary `{testedCount,
  // highHighCount, ..., permutations, multipleTesting}` at `:261`; parameters `permutations`,
  // `significance` and `multipleTesting` (`:72`, default Benjamini-Hochberg). The Sidak option has
  // no name in the analysis package's sentence vocabulary, so the sentence is written for no
  // correction and its wording for the correction is replaced (see `describeSidakSentence`).
  'curated-rate-cluster-map': {
    correction: {
      parameter: 'multipleTesting',
      values: {
        'benjamini-hochberg': 'false-discovery-rate',
        holm: 'holm',
        bonferroni: 'bonferroni',
        sidak: 'sidak',
        none: 'none'
      }
    },
    permutationsParameter: 'permutations',
    significanceParameter: 'significance',
    neighborCountColumn: 'neighbor_count',
    producerExcludesIslands: true
  },
  // curated-excess-risk: `curated-excess-risk-binding.ts:231`; summary `{areaCount, overallRate,
  // aboveExpectationCount, ...}` at `:213`. Its `value-range` sentence reads the `excess_risk`
  // column; no entry is needed.
  // curated-clusters-and-footprints: `curated-clusters-and-footprints-binding.ts:184`; summary
  // `{pointCount, clusterCount, noiseCount}` at `:168`. The sentence reads the `cluster` column of
  // the points table, which agrees with the summary; no entry is needed.
  // curated-regression-with-diagnostics: `curated-regression-with-diagnostics-binding.ts:543`;
  // summary `{rowCount, rSquared, adjustedRSquared, residualMoransI}` at `:525`, the generic keys.
  // The `diagnostics` table is described as a row count; the row `residual-morans-i` (`:452`)
  // gives the residual Moran's I and its permutation p-value, so that table gets the
  // `global-statistic` sentence as a second sentence.
  'curated-regression-with-diagnostics': {
    analysedCountKey: 'rowCount',
    outputTemplates: {diagnostics: 'global-statistic'},
    statistic: {
      tableId: 'diagnostics',
      matchColumn: 'diagnostic',
      matchValue: 'residual-morans-i',
      name: "The residual Moran's I",
      valueColumn: 'value',
      pValueColumn: 'p_value',
      // "two-sided permutation pseudo p-value" (`curated-regression-with-diagnostics-binding.ts:723`).
      sidedness: 'two-sided',
      positiveMeaning: 'similar residuals near each other',
      negativeMeaning: 'dissimilar residuals near each other'
    },
    permutationsParameter: 'permutations'
  },
  // curated-kriging-with-diagnostics: `curated-kriging-with-diagnostics-binding.ts:483`; the
  // `value-range` sentence reads the `value` column. The leave-one-out error (`rootMeanSquareError`,
  // `:468`) fits no template: `value-range` has no place for it, so it is not in a sentence.
  // curated-contiguity-correlogram: `curated-contiguity-correlogram-binding.ts:326`; table
  // `correlogram` (`:275`), row `order` 1 with `morans_i` and the permutation `pseudo_p_value`
  // (`:285`); summary `{orderOneMoransI, permutations}` at `:312`.
  'curated-contiguity-correlogram': {
    analysedCountKey: 'areaCount',
    statistic: {
      tableId: 'correlogram',
      matchColumn: 'order',
      matchValue: 1,
      name: "Moran's I at neighbour order 1",
      valueColumn: 'morans_i',
      pValueColumn: 'pseudo_p_value',
      // "the pseudo p-value is two-sided" (`curated-contiguity-correlogram-binding.ts:374`).
      sidedness: 'two-sided',
      positiveMeaning: 'similar values near each other',
      negativeMeaning: 'dissimilar values near each other'
    },
    permutationsParameter: 'permutations',
    significanceParameter: 'significance'
  }
};

// ---------------------------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------------------------

/** What one output's sentence is built from. */
export type KeplerResultOutputSummary = {
  tableId: string;
  /** Explicit inference semantics, independent of summary wording and palette. */
  inference?: boolean;
  /** The template of the descriptor, or the fact table's replacement for it. */
  template: GPUAnalysisSummaryTemplate;
  /** The template the sentence actually uses; `row-count` when the numbers were not found. */
  usedTemplate: GPUAnalysisSummaryTemplate;
  sentence: string;
};

export type KeplerResultCaution = {id: GPUAnalysisCaution; title: string; text: string};

export type KeplerResultNextStep = {question: GPUAnalysisQuestion; label: string};

export type KeplerResultReport = {
  /** The sentence of the first output whose sentence is not just a row count, else the first output. */
  sentence: string;
  /** One entry per described output with a table in the result, in descriptor order. */
  outputs: readonly KeplerResultOutputSummary[];
  /** The descriptor's notes: what to read before trusting the result. */
  notes: readonly string[];
  /** One line per active parameter whose value did not come from a plain constant. */
  chosen: readonly string[];
  cautions: readonly KeplerResultCaution[];
  next: readonly KeplerResultNextStep[];
  /** Which engine produced the result, in plain words. */
  engine: string;
  /** Why the GPU was not used, in plain words; absent when it was or was not asked for. */
  fallbackReason?: string;
  warnings: readonly string[];
  diagnostics: readonly CatalogDiagnostic[];
};

const STATISTICAL_TEMPLATES: readonly GPUAnalysisSummaryTemplate[] = [
  'significant-hot-cold-count',
  'local-cluster-count',
  'global-statistic'
];

/** An R squared this far above 1 is rounding in the fit, not a different number. */
const R_SQUARED_ROUNDING_TOLERANCE = 1e-9;

/** Significance level the stock cautions are written for. */
const STOCK_SIGNIFICANCE_LEVEL = 0.05;

/** The hot/cold tally includes the 90%, 95% and 99% classes, so its least strict level is 0.1. */
const LOWEST_HOT_SPOT_SIGNIFICANCE_LEVEL = 0.1;

type Facts = {template: GPUAnalysisSummaryTemplate; sentence: string};

/** What the extractors learn about the run besides the sentence, for the cautions. */
type RunDetails = {
  pValueConstruction?: 'upper-tail-count' | 'doubled-minimum-tail';
  pValueConstructionUnknown?: boolean;
  /** Features the statistic was computed on and classified. */
  testedCount?: number;
  significanceLevel?: number;
  correction?: KeplerCorrection;
  /** Rows with no neighbour. */
  islandRows?: number;
  /** Whether the binding itself leaves those rows out of its classes. */
  producerExcludesIslands?: boolean;
  /** Permutations behind the p-values of the first statistical output with a correction. */
  permutationCount?: number;
  /** Features the statistic was computed on (global statistics). */
  analysedCount?: number;
  /** Permutations of a single global test, which has no correction. */
  singleTestPermutationCount?: number;
};

type ExtractionContext = {
  source: KeplerResultFactSource | undefined;
  result: CatalogRunResult;
  table: CatalogOutputTable;
  output: KeplerOperationOutputDescriptor;
  plan: KeplerResultPresentation | undefined;
  details: RunDetails;
};

function readNumber(
  record: Readonly<Record<string, unknown>>,
  key: string | undefined
): number | undefined {
  if (key === undefined) return undefined;
  const value = record[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function getCorrection(
  source: KeplerResultFactSource | undefined,
  parameters: CatalogRunResult['parameters']
): KeplerCorrection | undefined {
  const correction = source?.correction;
  if (!correction) return undefined;
  if ('constant' in correction) return correction.constant;
  const value = parameters[correction.parameter];
  return typeof value === 'string' ? correction.values[value] : undefined;
}

function sumLegendCounts(plan: KeplerResultPresentation, tones: readonly string[]): number {
  return plan.legend
    .filter(entry => tones.includes(entry.tone))
    .reduce((sum, entry) => sum + entry.count, 0);
}

/** Rows with no neighbour, from the neighbour-count column; undefined when the binding has none. */
function getIslandMask(
  table: CatalogOutputTable,
  source: KeplerResultFactSource | undefined
): boolean[] | undefined {
  const column = source?.neighborCountColumn
    ? table.columns[source.neighborCountColumn]
    : undefined;
  return column ? column.map(value => value === 0) : undefined;
}

function getTestedCount(context: ExtractionContext): number | undefined {
  const {source, result, table} = context;
  const fromSummary = readNumber(result.summary, source?.testedCountKey);
  if (fromSummary !== undefined) {
    return Number.isInteger(fromSummary) && fromSummary >= 0 && fromSummary <= table.rowCount
      ? fromSummary
      : undefined;
  }
  return source?.testedCountKey === undefined ? table.rowCount : undefined;
}

// --- extractors --------------------------------------------------------------------------------

function extractHotCold(context: ExtractionContext): Facts | undefined {
  const {plan, source, result, table, details} = context;
  if (plan?.classScheme !== 'hot-spot-confidence-bins') return undefined;
  const correction = getCorrection(source, result.parameters);
  if (correction === undefined) return undefined;
  const hotTones = ['strong-high', 'high', 'weak-high'];
  const coldTones = ['strong-low', 'low', 'weak-low'];
  let hotCount = sumLegendCounts(plan, hotTones);
  let coldCount = sumLegendCounts(plan, coldTones);
  let featureCount = getTestedCount(context);
  const islands = getIslandMask(table, source);
  if (islands) {
    const islandRows = islands.filter(Boolean).length;
    details.islandRows = islandRows;
    details.producerExcludesIslands = source?.producerExcludesIslands ?? false;
    if (
      details.producerExcludesIslands &&
      featureCount !== undefined &&
      source?.testedCountKey === undefined
    ) {
      featureCount -= islandRows;
    }
    if (!details.producerExcludesIslands && islandRows > 0) {
      // The binding classes an island from its own value: a row with no neighbours is no test.
      const toneByValue = new Map(plan.legend.map(entry => [entry.value, entry.tone]));
      const values = table.columns[plan.column];
      hotCount = 0;
      coldCount = 0;
      let eligible = 0;
      values.forEach((value, row) => {
        if (islands[row] || value === null) return;
        eligible++;
        const tone = toneByValue.get(value);
        if (tone && hotTones.includes(tone)) hotCount++;
        else if (tone && coldTones.includes(tone)) coldCount++;
      });
      featureCount = eligible;
    }
  } else if (source?.testedCountKey) {
    const rowsWithoutStatistic = featureCount === undefined ? 0 : table.rowCount - featureCount;
    if (rowsWithoutStatistic > 0) {
      details.islandRows = rowsWithoutStatistic;
      details.producerExcludesIslands = true;
    }
  }
  if (featureCount === undefined) return undefined;
  details.testedCount = featureCount;
  details.correction = correction;
  const permutationCount = readNumber(
    result.parameters as Record<string, unknown>,
    source?.permutationsParameter
  );
  if (permutationCount !== undefined) details.permutationCount = permutationCount;
  // All confidence classes count, so the lowest class level also governs the caution.
  details.significanceLevel = LOWEST_HOT_SPOT_SIGNIFICANCE_LEVEL;
  const sentence = summarizeGPUAnalysisResult('significant-hot-cold-count', {
    featureCount,
    hotCount,
    coldCount,
    confidencePercent: (1 - LOWEST_HOT_SPOT_SIGNIFICANCE_LEVEL) * 100,
    correction,
    // With permutations the package rewords a result that could not have flagged anything.
    ...(permutationCount !== undefined ? {permutationCount} : {}),
    ...(source?.featureNoun ? {featureNoun: source.featureNoun} : {})
  });
  return {template: 'significant-hot-cold-count', sentence};
}

function extractLocalClusters(context: ExtractionContext): Facts | undefined {
  const {plan, source, result, table, details} = context;
  if (plan?.classScheme !== 'local-moran-clusters') return undefined;
  const parameters = result.parameters as Record<string, unknown>;
  const correction = getCorrection(source, result.parameters);
  const significanceLevel = readNumber(parameters, source?.significanceParameter);
  if (correction === undefined || significanceLevel === undefined) return undefined;
  const permutations = readNumber(parameters, source?.permutationsParameter);
  const construction = result.summary['pValueConstruction'];
  const pValueConstruction =
    construction === 'upper-tail-count' || construction === 'doubled-minimum-tail'
      ? construction
      : construction === undefined &&
          result.operationId === 'local-moran-statistics' &&
          result.engine === 'gpu' &&
          result.dispatchedContributors.includes('GPULocalPermutationTest')
        ? 'doubled-minimum-tail'
        : undefined;
  if (pValueConstruction) details.pValueConstruction = pValueConstruction;
  else details.pValueConstructionUnknown = true;
  const notTested = sumLegendCounts(plan, ['excluded']);
  const analyzedCount = plan.rowCount - (plan.unrecognized ?? 0);
  const featureCount = analyzedCount - notTested;
  const sentence = summarizeGPUAnalysisResult('local-cluster-count', {
    featureCount,
    highHighCount: sumLegendCounts(plan, ['high']),
    lowLowCount: sumLegendCounts(plan, ['low']),
    highLowCount: sumLegendCounts(plan, ['high-outlier']),
    lowHighCount: sumLegendCounts(plan, ['low-outlier']),
    significanceLevel,
    // A binding with a permutation parameter writes pseudo p-values, whatever the count.
    pValueKind: source?.permutationsParameter === undefined ? 'analytic' : 'pseudo',
    ...(pValueConstruction ? {pValueConstruction} : {}),
    correction,
    // The package words a run that could not flag anything, or computed no p-values, itself.
    ...(permutations !== undefined && pValueConstruction ? {permutationCount: permutations} : {}),
    ...(permutations === 0 ? {pValuesComputed: false, analyzedCount} : {})
  });
  details.testedCount = permutations === 0 ? 0 : featureCount;
  details.analysedCount = analyzedCount;
  details.correction = correction;
  if (permutations !== undefined) details.permutationCount = permutations;
  const islands = getIslandMask(table, source);
  if (islands) {
    details.islandRows = islands.filter(Boolean).length;
    details.producerExcludesIslands = source?.producerExcludesIslands ?? false;
  }
  return {template: 'local-cluster-count', sentence};
}

function extractClusterCount(context: ExtractionContext): Facts | undefined {
  const membership = context.plan?.membership;
  if (!context.plan || !membership) return undefined;
  const pointCount = context.plan.rowCount - (context.plan.unrecognized ?? 0);
  const sentence = summarizeGPUAnalysisResult('cluster-count-and-noise', {
    pointCount,
    clusterCount: membership.clusterCount,
    noiseCount: membership.noiseCount,
    ...(membership.clusterCount > 0 ? {largestClusterCount: membership.largestClusterCount} : {})
  });
  return {template: 'cluster-count-and-noise', sentence};
}

function extractGlobalStatistic(context: ExtractionContext): Facts | undefined {
  const {source, result, details} = context;
  const statistic = source?.statistic;
  if (!statistic) return undefined;
  const table = result.tables.find(candidate => candidate.id === statistic.tableId);
  if (!table) return undefined;
  const values = table.columns[statistic.valueColumn];
  const pValues = table.columns[statistic.pValueColumn];
  const rowIndex = statistic.matchColumn
    ? (table.columns[statistic.matchColumn] ?? []).indexOf(statistic.matchValue ?? '')
    : 0;
  if (!values || !pValues || rowIndex < 0) return undefined;
  const value = values[rowIndex];
  const pValue = pValues[rowIndex];
  if (typeof value !== 'number' || !Number.isFinite(value)) return undefined;
  if (typeof pValue !== 'number' || !Number.isFinite(pValue)) return undefined;
  const parameters = result.parameters as Record<string, unknown>;
  const permutations = readNumber(parameters, source.permutationsParameter);
  const significanceLevel =
    readNumber(parameters, source.significanceParameter) ?? STOCK_SIGNIFICANCE_LEVEL;
  let sentence = summarizeGPUAnalysisResult('global-statistic', {
    statisticName: statistic.name,
    value,
    pValue,
    permutationCount: permutations ?? 0,
    significanceLevel,
    sidedness: statistic.sidedness,
    ...(statistic.nullDescription ? {nullDescription: statistic.nullDescription} : {}),
    ...(statistic.positiveMeaning ? {positiveMeaning: statistic.positiveMeaning} : {}),
    ...(statistic.negativeMeaning ? {negativeMeaning: statistic.negativeMeaning} : {})
  });
  const analysed = readNumber(result.summary, source.analysedCountKey);
  if (analysed !== undefined) details.analysedCount = analysed;
  if (permutations !== undefined) {
    details.singleTestPermutationCount = permutations;
    details.permutationCount = permutations;
    if (permutations > 0) {
      // One test, no correction. The sentence template does not reword this case, so the
      // package's own reason is added.
      const attainability = getGPUAnalysisSignificanceAttainability({
        correction: 'none',
        testCount: 1,
        permutationCount: permutations,
        significanceLevel
      });
      if (!attainability.attainable) sentence += ` This is not a finding: ${attainability.reason}.`;
    }
  }
  return {template: 'global-statistic', sentence};
}

function extractRegressionFit(context: ExtractionContext): Facts | undefined {
  const keys = context.source?.regression ?? {
    observationKey: 'rowCount',
    rSquaredKey: 'rSquared',
    adjustedKey: 'adjustedRSquared'
  };
  const summary = context.result.summary;
  let rSquared = readNumber(summary, keys.rSquaredKey);
  const observationCount = readNumber(summary, keys.observationKey);
  if (rSquared === undefined || observationCount === undefined) return undefined;
  if (rSquared > 1 && rSquared <= 1 + R_SQUARED_ROUNDING_TOLERANCE) rSquared = 1;
  const adjustedRSquared = readNumber(summary, keys.adjustedKey);
  const sentence = summarizeGPUAnalysisResult('regression-fit', {
    rSquared,
    observationCount,
    ...(adjustedRSquared !== undefined ? {adjustedRSquared} : {})
  });
  return {template: 'regression-fit', sentence};
}

/** `first_reach_cost` as `First reach cost`. */
function describeColumn(column: string): string {
  const words = column.replace(/[_-]+/g, ' ').trim();
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : column;
}

function extractValueRange(context: ExtractionContext): Facts | undefined {
  const column =
    context.output.presentation?.column ?? context.source?.valueColumns?.[context.output.tableId];
  const values = column ? context.table.columns[column] : undefined;
  if (!column || !values) return undefined;
  let minimum = Infinity;
  let maximum = -Infinity;
  let valueCount = 0;
  for (const value of values) {
    if (typeof value !== 'number' || !Number.isFinite(value)) continue;
    valueCount++;
    if (value < minimum) minimum = value;
    if (value > maximum) maximum = value;
  }
  if (valueCount === 0) return undefined;
  const sentence = summarizeGPUAnalysisResult('value-range', {
    minimum,
    maximum,
    valueCount,
    valueLabel: describeColumn(column),
    ...(context.plan?.significantDigits !== undefined
      ? {significantDigits: context.plan.significantDigits}
      : {})
  });
  return {template: 'value-range', sentence};
}

function extractCountPerArea(context: ExtractionContext): Facts | undefined {
  const {source, result} = context;
  const when = source?.countWhen;
  if (when && result.parameters[when.parameter] !== when.equals) return undefined;
  const column = source?.countColumn;
  const values = column ? context.table.columns[column] : undefined;
  if (!values) return undefined;
  let totalCount = 0;
  let maximumCount = 0;
  for (const value of values) {
    // A count is a non-negative whole number; anything else means the column is not a count.
    if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) return undefined;
    totalCount += value;
    if (value > maximumCount) maximumCount = value;
  }
  // A producer that counts a point in every area containing it reports how many points lie in
  // more than one (`multipleMatchCount`); the total is then of memberships, not of points.
  const multipleMatchCount = result.summary['multipleMatchCount'];
  const countsMemberships =
    source?.areaNoun === ZONE_NOUN ||
    (typeof multipleMatchCount === 'number' && multipleMatchCount > 0);
  const sentence = summarizeGPUAnalysisResult('count-per-area', {
    totalCount,
    areaCount: values.length,
    maximumCount,
    itemNoun: countsMemberships
      ? {singular: 'point membership', plural: 'point memberships'}
      : {singular: 'point', plural: 'points'},
    ...(source?.areaNoun ? {areaNoun: source.areaNoun} : {})
  });
  return {
    template: 'count-per-area',
    sentence:
      source?.areaNoun === ZONE_NOUN
        ? `${sentence} A point in overlapping zones contributes once to each zone.`
        : countsMemberships
          ? `${sentence} A point in overlapping areas contributes once to each area.`
          : sentence
  };
}

const ADJUSTMENT_NAMES: Readonly<Record<string, string>> = {
  bonferroni: 'Bonferroni',
  holm: 'Holm',
  'benjamini-hochberg': 'Benjamini-Hochberg',
  sidak: 'Sidak'
};

/** "N of M p-values are rejected": no template says it, so it is written here from the summary. */
function describeRejectedSentence(context: ExtractionContext): string | undefined {
  const rejected = context.source?.rejected;
  if (!rejected) return undefined;
  const tested = readNumber(context.result.summary, rejected.testedKey);
  const rejectedCount = readNumber(context.result.summary, rejected.rejectedKey);
  const alpha = readNumber(
    context.result.parameters as Record<string, unknown>,
    rejected.alphaParameter
  );
  const method = context.result.parameters[rejected.methodParameter];
  if (tested === undefined || rejectedCount === undefined || alpha === undefined) return undefined;
  if (!Number.isInteger(tested) || !Number.isInteger(rejectedCount) || rejectedCount > tested) {
    return undefined;
  }
  const name = typeof method === 'string' ? ADJUSTMENT_NAMES[method] : undefined;
  return `${formatGPUAnalysisNumber(rejectedCount)} of ${formatGPUAnalysisNumber(tested)} ${tested === 1 ? 'p-value is' : 'p-values are'} at or below ${formatGPUAnalysisMeasuredValue(alpha)} after ${name ? `${name} ` : ''}adjustment.`;
}

function describeRowCount(table: CatalogOutputTable): Facts {
  return {
    template: 'row-count',
    sentence: summarizeGPUAnalysisResult('row-count', {rowCount: table.rowCount})
  };
}

const EXTRACTORS: Readonly<
  Record<GPUAnalysisSummaryTemplate, (context: ExtractionContext) => Facts | undefined>
> = {
  'significant-hot-cold-count': extractHotCold,
  'local-cluster-count': extractLocalClusters,
  'cluster-count-and-noise': extractClusterCount,
  'global-statistic': extractGlobalStatistic,
  'regression-fit': extractRegressionFit,
  'value-range': extractValueRange,
  'count-per-area': extractCountPerArea,
  'row-count': context => {
    const sentence = describeRejectedSentence(context);
    return sentence ? {template: 'row-count', sentence} : describeRowCount(context.table);
  }
};

function summarizeOutput(context: ExtractionContext, template: GPUAnalysisSummaryTemplate): Facts {
  try {
    const facts = EXTRACTORS[template](context);
    if (facts) return facts;
  } catch {
    // The numbers were inconsistent (a count above its total, a p-value outside 0 to 1): the
    // sentence must not state them, so it falls back to the row count.
  }
  return describeRowCount(context.table);
}

// ---------------------------------------------------------------------------------------------
// Chosen parameters
// ---------------------------------------------------------------------------------------------

function describeUnit(
  record: KeplerParameterResolution,
  specification: CatalogParameterSpec | undefined,
  coordinateSystem: CatalogRunResult['parameters'][string]
): string {
  const unit = record.resolution?.unit;
  const isDistance =
    unit === 'metre' ||
    unit === 'degree' ||
    unit === 'coordinate-unit' ||
    specification?.unit === 'crs';
  if (isDistance && coordinateSystem === 'planar') return ' coordinate units';
  if (unit === 'metre') return ' m';
  if (unit === 'degree') return '°';
  if (unit === 'coordinate-unit') return '';
  const specificationUnit = specification?.unit;
  return specificationUnit && specificationUnit !== 'crs' ? ` ${specificationUnit}` : '';
}

/**
 * The explanation as a reason after the label and the value. Two shapes occur. A rule that only
 * names its statistic carries the value in brackets, `Largest nearest-neighbour distance (1,240
 * m), so every point has a neighbour`, and the brackets are dropped because the line already shows
 * the value. A rule with a factor spells out the sum, `... distance (663 m) times 1.01 = 670 m,
 * so ...`; there the result after `=` is dropped and the bracketed statistic kept. The first word
 * becomes lower case unless it is a name or acronym, and a closing full stop is removed. Unless the
 * input is known to hold points, "point" is replaced by "feature": the rule's wording is written
 * for points and the input may be polygons.
 */
function trimExplanation(explanation: string, inputIsPoints: boolean): string {
  let text = explanation.trim().replace(/\.$/, '');
  const withoutResult = text.replace(/ = [^,;]*/, '');
  text =
    withoutResult === text ? text.replace(/ \([^()]*\d[^()]*\)/, '').trim() : withoutResult.trim();
  if (!inputIsPoints)
    text = text.replace(/\bpoints\b/g, 'features').replace(/\bpoint\b/g, 'feature');
  // Lower-case the first word only when it is an ordinary capitalised word, not a name or acronym.
  const startsOrdinaryWord = /^[A-Z][a-z]+\b(?!')/.test(text);
  return startsOrdinaryWord ? text.charAt(0).toLowerCase() + text.slice(1) : text;
}

/** Whether a parameter has an effect in this run; the descriptor's `activeWhen` says. */
function isParameterActive(
  policy: KeplerOperationParameterDescriptor | undefined,
  parameters: CatalogRunResult['parameters']
): boolean {
  return evaluateGPUAnalysisParameterCondition(policy?.activeWhen, parameters);
}

function describeChosenParameters(
  records: readonly KeplerParameterResolution[],
  binding: CatalogOperationBinding,
  descriptor: KeplerOperationDescriptor,
  parameters: CatalogRunResult['parameters'],
  context: KeplerResultReportContext
): string[] {
  const lines: string[] = [];
  const profiles = context.profiles ?? {};
  for (const record of records) {
    const source = record.source;
    if (source === 'constant') continue;
    const policy = descriptor.parameters.find(candidate => candidate.name === record.parameter);
    if (record.active === false || !isParameterActive(policy, parameters)) continue;
    const specification = binding.parameters.find(candidate => candidate.name === record.parameter);
    const label = specification?.label ?? record.parameter;
    const used = parameters[record.parameter];
    let value: string | undefined;
    if (typeof used === 'number' && Number.isFinite(used)) {
      value = formatGPUAnalysisMeasuredValue(used);
    } else if (typeof used === 'string' || typeof used === 'boolean') {
      value = String(used);
    } else if (record.resolution?.value !== undefined && Number.isFinite(record.resolution.value)) {
      value = formatGPUAnalysisMeasuredValue(record.resolution.value);
    }
    const unit =
      typeof used === 'number' || value !== undefined
        ? describeUnit(record, specification, parameters['coordinateSystem'])
        : '';
    const heading = value === undefined ? label : `${label} ${value}${unit}`;
    const presetReason =
      record.reason ??
      (policy?.defaultPolicy.from === 'preset' ? policy.defaultPolicy.reason : undefined);
    const isPreset =
      source === 'preset' ||
      (source === 'override' && (context.presetParameters ?? []).includes(record.parameter));
    if (source === 'field') {
      // A column chosen for a role: named by the role's label, not "set by you".
      const field = descriptor.inputs
        .flatMap(input => input.fields)
        .find(candidate => candidate.parameter === record.parameter);
      lines.push(`${field?.label ?? label}: ${value ?? 'none'}`);
    } else if (isPreset) {
      lines.push(`${heading}: set by this analysis${presetReason ? `: ${presetReason}` : ''}`);
    } else if (source === 'override') {
      lines.push(`${heading}: set by you`);
    } else if (source === 'fallback') {
      lines.push(
        `${heading}: the default could not be derived from your data${record.reason ? ` (${record.reason})` : ''}, so the fixed fallback was used`
      );
    } else if (record.parameter === 'coordinateSystem') {
      const stated = Object.values(profiles).find(candidate => candidate?.coordinates)?.coordinates;
      // The record says when the model is a guess; the profile says how it is known otherwise.
      const isGuess = record.guessed === true || stated?.source === 'guess';
      const how = isGuess
        ? 'guessed from the coordinate values'
        : stated?.source === 'declared'
          ? 'declared for the dataset'
          : 'taken from the dataset profile';
      lines.push(`${heading}: ${how}`);
      if (isGuess && (stated?.confidence === 'low' || /low confidence/.test(record.reason ?? ''))) {
        lines.push(
          `Coordinate system guess has low confidence${stated?.reason ? ` (${stated.reason})` : ''}: confirm it before trusting distances or areas`
        );
      }
    } else {
      const sampled = record.resolution?.sampled;
      const inputName =
        policy?.defaultPolicy.from === 'data' ? policy.defaultPolicy.input : undefined;
      const inputIsPoints = inputName !== undefined && profiles[inputName]?.kind === 'points';
      const unitMismatch =
        parameters['coordinateSystem'] === 'planar' &&
        (record.resolution?.unit === 'metre' || record.resolution?.unit === 'degree');
      const reason = unitMismatch
        ? 'the supplied default was derived in a different coordinate model; choose a value in the coordinate units used by this run'
        : record.resolution && !record.clamped
          ? trimExplanation(record.resolution.explanation, inputIsPoints)
          : 'derived from your data';
      const sampledText = sampled ? ' (computed on a sample of the rows)' : '';
      let clampedText = '';
      if (record.clamped) {
        const derived = formatGPUAnalysisMeasuredValue(record.clamped.from);
        const changed = formatGPUAnalysisMeasuredValue(record.clamped.to);
        clampedText = record.clamped.reason
          ? `; the value derived from your data (${derived}) was changed to ${changed}: ${record.clamped.reason}`
          : record.clamped.to > record.clamped.from
            ? `; the value derived from your data (${derived}) was raised to the smallest this parameter allows`
            : `; the value derived from your data (${derived}) was limited to the largest this parameter allows`;
      }
      lines.push(`${heading}: ${reason}${sampledText}${clampedText}`);
    }
  }
  return lines;
}

// ---------------------------------------------------------------------------------------------
// Cautions
// ---------------------------------------------------------------------------------------------

/** A count or amount column bound to a role that does not list that level. */
function hasMeasurementMismatch(
  descriptor: KeplerOperationDescriptor,
  context: KeplerResultReportContext
): boolean {
  const bindings = context.fieldBindings;
  if (!bindings) return false;
  for (const input of descriptor.inputs) {
    for (const field of input.fields) {
      const bound = bindings[field.parameter];
      const profile = context.profiles?.[input.name];
      const geometry =
        profile?.catalogGeometry ??
        (profile?.pointColumns
          ? 'point'
          : profile?.kind === 'polygons'
            ? 'polygon'
            : profile?.kind === 'points'
              ? 'point'
              : profile?.kind === 'lines'
                ? 'line'
                : undefined);
      const levels =
        (geometry === undefined ? undefined : field.measurementByGeometry?.[geometry]) ??
        field.measurement;
      if (bound === undefined || !levels) continue;
      const names =
        typeof bound === 'string' ? bound.split(',').map(name => name.trim()) : [...bound];
      const columns = context.profiles?.[input.name]?.columns ?? [];
      for (const name of names) {
        const level = columns.find(column => column.name === name)?.measurement;
        if ((level === 'count' || level === 'amount') && !levels.includes(level)) {
          return true;
        }
      }
    }
  }
  return false;
}

/** What the run's summary says about the cautions it can raise on its own. */
type SummaryCautions = {
  raised: Set<GPUAnalysisCaution>;
  smallDenominatorAreaCount?: number;
  expectedCountLimit?: number;
  poorModelFitReason?: string;
  rowOrderDependentInference?: boolean;
};

/**
 * Cautions the run's own summary raises, by the convention of `CURATED_RUNTIME_CAUTION_SUMMARY_KEYS`
 * (`descriptors-curated.ts`): a summary key and a condition in words, `equals 1` or `is above 0`.
 * The keys that carry more: `poorModelFitReasons` (text) beside `poorModelFit`, and
 * `smallDenominatorThreshold` beside `smallDenominatorAreaCount`.
 */
function readSummaryCautions(
  bindingId: string,
  summary: CatalogRunResult['summary']
): SummaryCautions {
  const entries = CURATED_RUNTIME_CAUTION_SUMMARY_KEYS[bindingId] ?? {};
  const found: SummaryCautions = {raised: new Set()};
  for (const [caution, entry] of Object.entries(entries)) {
    const value = summary[entry.key];
    if (typeof value !== 'number') continue;
    const isRaised = entry.raiseWhen.startsWith('equals 1')
      ? value === 1
      : entry.raiseWhen.startsWith('is above 0') && value > 0;
    if (!isRaised) continue;
    found.raised.add(caution as GPUAnalysisCaution);
    if (caution === 'small-denominators') {
      found.smallDenominatorAreaCount = value;
      const limit = summary['smallDenominatorThreshold'];
      if (typeof limit === 'number') found.expectedCountLimit = limit;
    } else if (caution === 'poor-model-fit') {
      const reasons = summary[`${entry.key}Reasons`];
      found.poorModelFitReason =
        typeof reasons === 'string' && reasons.trim() ? reasons : 'a check of the model failed';
    } else if (caution === 'row-order-dependent-inference') {
      found.rowOrderDependentInference = true;
    }
  }
  return found;
}

/** Facts the stock caution texts can state, from the run. */
type CautionTextFacts = {
  pValueConstruction?: 'upper-tail-count' | 'doubled-minimum-tail';
  pValueConstructionUnknown?: boolean;
  permutationCount?: number;
  significanceLevel: number;
  correction?: GPUAnalysisMultipleTestingCorrection;
  testCount?: number;
  rowsExcluded?: boolean;
  smallDenominatorAreaCount?: number;
  expectedCountLimit?: number;
  reason?: string;
};

function describeCaution(id: GPUAnalysisCaution, facts: CautionTextFacts): KeplerResultCaution {
  const stock = getGPUAnalysisCautionText(id, {
    significanceLevel: facts.significanceLevel,
    ...(facts.correction !== undefined ? {correction: facts.correction} : {}),
    ...(facts.testCount !== undefined ? {testCount: facts.testCount} : {}),
    ...(facts.permutationCount !== undefined ? {permutationCount: facts.permutationCount} : {}),
    ...(facts.pValueConstruction ? {pValueConstruction: facts.pValueConstruction} : {}),
    ...(facts.rowsExcluded !== undefined ? {rowsExcluded: facts.rowsExcluded} : {}),
    ...(facts.smallDenominatorAreaCount !== undefined
      ? {smallDenominatorAreaCount: facts.smallDenominatorAreaCount}
      : {}),
    ...(facts.expectedCountLimit !== undefined
      ? {expectedCountLimit: facts.expectedCountLimit}
      : {}),
    ...(facts.reason !== undefined ? {reason: facts.reason} : {})
  });
  let text = stock.text;
  if (id === 'pseudo-p-values' && facts.pValueConstructionUnknown) {
    text =
      'These p-values are estimated from random permutations and vary with the random seed. Their counting convention is unavailable, so a specific attainable p-value floor cannot be stated.';
  } else if (id === 'pseudo-p-values' && facts.permutationCount !== undefined) {
    const resolution = describeGPUAnalysisPermutationResolution(
      facts.permutationCount,
      facts.significanceLevel,
      facts.pValueConstruction
    );
    if (resolution) text = `${text} ${resolution}`;
  }
  return {id, title: stock.title, text};
}

// ---------------------------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------------------------

function describeEngine(result: CatalogRunResult): string {
  if (result.engine === 'gpu') {
    return result.dispatchedContributors.length
      ? `Computed on the GPU (${result.dispatchedContributors.join(', ')}).`
      : 'Computed on the GPU.';
  }
  return 'Computed on the CPU with the reference engine.';
}

function describeFallbackReason(reason: string): string {
  switch (reason) {
    case 'CPU engine requested':
      return 'The CPU engine was requested, so the GPU was not used.';
    case 'This operation has no GPU path':
      return 'This operation has no GPU version, so it ran on the CPU.';
    case 'No GPU device supplied':
      return 'No GPU was available, so the operation ran on the CPU.';
    default:
      return `The GPU could not run this operation (${reason}), so it ran on the CPU.`;
  }
}

// ---------------------------------------------------------------------------------------------
// Public
// ---------------------------------------------------------------------------------------------

/**
 * The report of one catalog run: sentence, chosen parameters, cautions, next questions, engine and
 * the run's diagnostics. The numbers in the sentence are read from the result; a template whose
 * numbers cannot be found gives the row count instead.
 */
export function describeKeplerCatalogResult(
  result: CatalogRunResult,
  binding: CatalogOperationBinding,
  descriptor: KeplerOperationDescriptor,
  context: KeplerResultReportContext = {}
): KeplerResultReport {
  const source = KEPLER_RESULT_FACT_SOURCES[descriptor.bindingId];
  const outputs: KeplerResultOutputSummary[] = [];
  const alwaysOn: GPUAnalysisCaution[] = [];
  const cautionFacts: GPUAnalysisCautionFacts = {};
  const details: RunDetails = {};
  let hasStatisticalOutput = false;

  for (const output of descriptor.outputs) {
    const table = result.tables.find(candidate => candidate.id === output.tableId);
    if (!table || (!output.interpretation && !output.inference)) continue;
    const template =
      source?.outputTemplates?.[output.tableId] ?? output.interpretation?.summary ?? 'row-count';
    const plan = output.presentation ? planKeplerResultPresentation(table, output) : undefined;
    const isFirstStatistical = STATISTICAL_TEMPLATES.includes(template) && !hasStatisticalOutput;
    const outputDetails: RunDetails = isFirstStatistical ? details : {};
    const facts = summarizeOutput(
      {source, result, table, output, plan, details: outputDetails},
      template
    );
    outputs.push({
      tableId: output.tableId,
      template,
      ...(output.inference !== undefined ? {inference: output.inference} : {}),
      usedTemplate: facts.template,
      sentence: facts.sentence
    });
    for (const caution of output.interpretation?.cautions ?? []) {
      if (!alwaysOn.includes(caution)) alwaysOn.push(caution);
    }
    if (isFirstStatistical) hasStatisticalOutput = true;
  }

  const parameters = result.parameters as Record<string, unknown>;
  const permutationCount = details.permutationCount;
  const significanceLevel =
    details.significanceLevel ??
    readNumber(parameters, source?.significanceParameter) ??
    STOCK_SIGNIFICANCE_LEVEL;
  const summaryCautions = readSummaryCautions(descriptor.bindingId, result.summary);
  if (
    permutationCount !== undefined &&
    (permutationCount === 0 || !details.pValueConstructionUnknown)
  )
    cautionFacts.permutationCount = permutationCount;
  if (details.pValueConstruction) cautionFacts.pValueConstruction = details.pValueConstruction;
  cautionFacts.significanceLevel = significanceLevel;
  if (hasStatisticalOutput) {
    const firstInput = descriptor.inputs[0]?.name;
    const featureCount =
      details.analysedCount ??
      details.testedCount ??
      (firstInput !== undefined ? context.featureCounts?.[firstInput] : undefined);
    if (featureCount !== undefined) cautionFacts.featureCount = featureCount;
    if (details.islandRows !== undefined) cautionFacts.rowsWithoutNeighbors = details.islandRows;
    else if (typeof result.summary['islandCount'] === 'number') {
      cautionFacts.rowsWithoutNeighbors = result.summary['islandCount'];
    } else {
      const diagnostic = result.diagnostics.find(candidate => candidate.code === 'island-features');
      if (diagnostic) cautionFacts.rowsWithoutNeighbors = diagnostic.count;
    }
    // A correction over many tests; a single global test has none to state.
    const correction = details.correction ?? getCorrection(source, result.parameters);
    if (correction !== undefined) {
      cautionFacts.correction = correction;
      if (permutationCount === 0) cautionFacts.permutationsRequired = true;
      if (details.testedCount !== undefined) cautionFacts.testCount = details.testedCount;
    }
  }
  const records = context.resolutions ?? [];
  const activeRecords = records.filter(
    record =>
      record.active !== false &&
      isParameterActive(
        descriptor.parameters.find(policy => policy.name === record.parameter),
        result.parameters
      )
  );
  if (activeRecords.some(record => record.resolution?.sampled))
    cautionFacts.sampledStatistics = true;
  if (activeRecords.some(record => record.source === 'fallback'))
    cautionFacts.usedFallbackDefault = true;
  const coordinateRecord = activeRecords.find(record => record.parameter === 'coordinateSystem');
  const explicitCoordinates =
    coordinateRecord?.source === 'override' || coordinateRecord?.source === 'preset';
  const guessed =
    !explicitCoordinates &&
    (context.guessedCoordinates ??
      (coordinateRecord
        ? (coordinateRecord.guessed ??
          (coordinateRecord.source === 'data' &&
            Object.values(context.profiles ?? {}).some(
              profile => profile?.coordinates?.source === 'guess'
            )))
        : parameters['coordinateSystem'] === undefined &&
          Object.values(context.profiles ?? {}).some(
            profile => profile?.coordinates?.source === 'guess'
          )));
  if (guessed) cautionFacts.guessedCoordinates = true;
  if (context.parameterSearch) cautionFacts.parameterSearch = true;
  if (hasMeasurementMismatch(descriptor, context)) cautionFacts.measurementMismatch = true;
  if (summaryCautions.smallDenominatorAreaCount !== undefined) {
    cautionFacts.smallDenominatorAreaCount = summaryCautions.smallDenominatorAreaCount;
  }
  if (summaryCautions.poorModelFitReason !== undefined) {
    cautionFacts.poorModelFitReason = summaryCautions.poorModelFitReason;
  }
  if (summaryCautions.rowOrderDependentInference) cautionFacts.rowOrderDependentInference = true;

  const derived = new Set<GPUAnalysisCaution>(deriveGPUAnalysisCautions(cautionFacts));
  // The run's own flag, for a binding whose test count the report does not know.
  if (summaryCautions.raised.has('significance-not-attainable')) {
    derived.add('significance-not-attainable');
  }
  // A single global permutation test has no correction to hand to the package's derivation.
  let singleTestAttainable = true;
  if (details.singleTestPermutationCount !== undefined && details.singleTestPermutationCount > 0) {
    singleTestAttainable = getGPUAnalysisSignificanceAttainability({
      correction: 'none',
      testCount: 1,
      permutationCount: details.singleTestPermutationCount,
      significanceLevel
    }).attainable;
    if (!singleTestAttainable) derived.add('significance-not-attainable');
  }
  const noTests = permutationCount === 0 || details.testedCount === 0;
  const cautionIds = GPU_ANALYSIS_CAUTIONS.filter(
    id =>
      !(noTests && (id === 'multiple-testing-uncorrected' || id === 'pseudo-p-values')) &&
      (alwaysOn.includes(id) || derived.has(id))
  );
  const unattainableFacts =
    details.singleTestPermutationCount !== undefined
      ? {
          correction: 'none' as const,
          testCount: 1,
          permutationCount: details.singleTestPermutationCount
        }
      : {
          ...(cautionFacts.correction !== undefined ? {correction: cautionFacts.correction} : {}),
          ...(cautionFacts.testCount !== undefined ? {testCount: cautionFacts.testCount} : {}),
          ...(permutationCount !== undefined && !details.pValueConstructionUnknown
            ? {permutationCount}
            : {})
        };
  const cautions = cautionIds.map(id =>
    describeCaution(id, {
      significanceLevel,
      ...(details.pValueConstructionUnknown ? {pValueConstructionUnknown: true} : {}),
      ...(details.pValueConstruction ? {pValueConstruction: details.pValueConstruction} : {}),
      ...unattainableFacts,
      ...(details.producerExcludesIslands !== undefined
        ? {rowsExcluded: details.producerExcludesIslands}
        : {}),
      ...(summaryCautions.smallDenominatorAreaCount !== undefined
        ? {smallDenominatorAreaCount: summaryCautions.smallDenominatorAreaCount}
        : {}),
      ...(summaryCautions.expectedCountLimit !== undefined
        ? {expectedCountLimit: summaryCautions.expectedCountLimit}
        : {}),
      ...(summaryCautions.poorModelFitReason !== undefined
        ? {reason: summaryCautions.poorModelFitReason}
        : {})
    })
  );

  const next: KeplerResultNextStep[] = [];
  for (const output of descriptor.outputs) {
    for (const question of output.next ?? []) {
      if (!next.some(step => step.question === question)) {
        next.push({question, label: getGPUAnalysisQuestion(question).label});
      }
    }
  }

  const primary = outputs.find(output => output.usedTemplate !== 'row-count') ?? outputs[0];
  const fallbackReason = result.fallbackReason;
  return {
    sentence: primary
      ? primary.sentence
      : summarizeGPUAnalysisResult('row-count', {
          rowCount: result.tables.reduce((sum, table) => sum + table.rowCount, 0)
        }),
    outputs,
    notes: [
      ...(descriptor.notes ?? []),
      ...descriptor.outputs.flatMap(output =>
        output.presentationException
          ? [`${output.tableId}: ${output.presentationException.reason}`]
          : []
      )
    ],
    chosen: describeChosenParameters(records, binding, descriptor, result.parameters, context),
    cautions,
    next,
    engine: describeEngine(result),
    ...(fallbackReason ? {fallbackReason: describeFallbackReason(fallbackReason)} : {}),
    warnings: context.warnings ?? [],
    diagnostics: result.diagnostics
  };
}

/**
 * The report as plain text for a result view or a log: the notes, the sentence, what was chosen, the
 * cautions, what to ask next and the engine. It leaves out the legend, which a host draws from the
 * plan.
 */
export function formatKeplerResultReport(report: KeplerResultReport): string {
  const lines: string[] = [];
  if (report.notes.length) {
    for (const note of report.notes) lines.push(note);
    lines.push('');
  }
  lines.push(report.sentence);
  const sentences = new Set([report.sentence]);
  for (const output of report.outputs) {
    if (!sentences.has(output.sentence)) {
      lines.push(output.sentence);
      sentences.add(output.sentence);
    }
  }
  if (report.chosen.length) {
    lines.push('', 'What was chosen');
    for (const line of report.chosen) lines.push(`- ${line}`);
  }
  if (report.cautions.length) {
    lines.push('', 'Read with care');
    for (const caution of report.cautions) lines.push(`- ${caution.title}: ${caution.text}`);
  }
  if (report.warnings.length) {
    lines.push('', 'Warnings');
    for (const warning of report.warnings) lines.push(`- ${warning}`);
  }
  if (report.next.length) {
    lines.push('', 'Ask next');
    for (const step of report.next) lines.push(`- ${step.label}`);
  }
  lines.push('', report.engine);
  if (report.fallbackReason) lines.push(report.fallbackReason);
  return lines.join('\n');
}
