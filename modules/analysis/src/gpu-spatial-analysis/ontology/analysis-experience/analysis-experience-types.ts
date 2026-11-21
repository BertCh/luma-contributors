// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpatialAnalysisDataKind} from '../spatial-analysis-ontology';

// Vocabulary for choosing an analysis from the data a user has: the question an operation
// answers, what a column measures, what must hold before an operation runs, how a parameter
// default is derived from the data, and how a result is presented and read. Hosts (the Kepler
// bindings, the website) describe their operations with these types; nothing here touches a GPU.

/** Questions an analysis answers. The first question of an operation is its primary one. */
export const GPU_ANALYSIS_QUESTIONS = [
  'where-concentrated',
  'where-significant',
  'how-much-per-area',
  'what-is-near-or-reachable',
  'how-related',
  'value-where-unmeasured',
  'what-changed',
  'how-it-moves',
  'shape-of-the-surface',
  'which-groups-or-regions',
  'what-is-at-each-place',
  'prepare-data'
] as const;

export type GPUAnalysisQuestion = (typeof GPU_ANALYSIS_QUESTIONS)[number];

/** One question with the words shown to a user. */
export type GPUAnalysisQuestionRecord = {
  id: GPUAnalysisQuestion;
  /** The question as a user would ask it, ending in a question mark. */
  label: string;
  /** One sentence on what kind of answer the analyses under it give. */
  summary: string;
  /** Two or three concrete examples of the question on real subjects. */
  examples: readonly string[];
};

/** Storage type of a column, as far as analysis is concerned. */
export const GPU_ANALYSIS_COLUMN_TYPES = [
  'number',
  'integer',
  'category',
  'time',
  'boolean',
  'text'
] as const;

export type GPUAnalysisColumnType = (typeof GPU_ANALYSIS_COLUMN_TYPES)[number];

/**
 * What a column measures. `count` and `amount` are totals that add up over areas (extensive);
 * `rate` is a ratio, share or density that does not (intensive); `cyclic` wraps (bearing, hour of
 * day); `identifier` labels a row and carries no magnitude.
 */
export const GPU_ANALYSIS_MEASUREMENT_LEVELS = [
  'count',
  'amount',
  'rate',
  'ordinal',
  'nominal',
  'cyclic',
  'identifier'
] as const;

export type GPUAnalysisMeasurementLevel = (typeof GPU_ANALYSIS_MEASUREMENT_LEVELS)[number];

/**
 * What a row stands for. `events`: each row is an occurrence, and the pattern of locations is the
 * subject. `objects`: each row is a thing with attributes. `surface-samples`: each row is a
 * measurement of a quantity that exists everywhere (rainfall, elevation).
 */
export const GPU_ANALYSIS_ROW_SEMANTICS = ['events', 'objects', 'surface-samples'] as const;

export type GPUAnalysisRowSemantics = (typeof GPU_ANALYSIS_ROW_SEMANTICS)[number];

/** Where a fact about the data came from. A guess must be shown and be correctable. */
export type GPUAnalysisFactSource = 'guess' | 'declared';

/** Summary statistics of one column. Absent members were not computed. */
export type GPUAnalysisColumnStatistics = {
  name: string;
  type: GPUAnalysisColumnType;
  rowCount: number;
  nullCount: number;
  /** Finite minimum and maximum of number, integer and time columns. */
  minimum?: number;
  maximum?: number;
  mean?: number;
  standardDeviation?: number;
  /** Distinct non-null values; an estimate when `distinctCountIsEstimate` is true. */
  distinctCount?: number;
  distinctCountIsEstimate?: boolean;
  /**
   * Guaranteed inclusive bounds on distinct accepted non-null values in the inspected rows.
   * These are deterministic guarantees, never standard errors or confidence intervals. Both
   * members must be supplied together. With sampleSize, the full population has at least the
   * lower bound and at most upperBound + (rowCount - sampleSize) distinct values. An exact
   * sample count therefore cannot silently authorize a capacity for the whole population.
   */
  distinctCountLowerBound?: number;
  distinctCountUpperBound?: number;
  /** True when every finite value of a number column is a whole number. */
  integral?: boolean;
  /** Rows the statistics were computed on, when fewer than `rowCount`. */
  sampleSize?: number;
};

/** A column with what it is believed to measure. */
export type GPUAnalysisColumnProfile = GPUAnalysisColumnStatistics & {
  measurement?: GPUAnalysisMeasurementLevel;
  measurementSource?: GPUAnalysisFactSource;
  /** How sure a guessed level is; a low-confidence guess must not fill a role unasked. */
  measurementConfidence?: 'high' | 'low';
  /** Plain-language reason for a guessed measurement level. */
  measurementReason?: string;
};

export type GPUAnalysisCoordinateModel = 'geographic' | 'planar';

/**
 * How point locations (or polygon and line anchors) are spread. Distances are in metres for
 * geographic coordinates and in coordinate units for planar coordinates.
 */
export type GPUAnalysisSpacingStatistics = {
  pointCount: number;
  /** Rejected source rows, counted before sampling. Absent in legacy profiles. */
  invalidCount?: number;
  /** Points the distance statistics were computed on; equals `pointCount` when exact. */
  sampleSize: number;
  sampled: boolean;
  coordinateModel: GPUAnalysisCoordinateModel;
  unit: 'metre' | 'coordinate-unit';
  /** [minimumX, minimumY, maximumX, maximumY] in source coordinates. */
  extent: readonly [number, number, number, number];
  /** Extent width, height and diagonal in `unit`. */
  extentWidth: number;
  extentHeight: number;
  extentDiagonal: number;
  /** Mean centre in source coordinates. */
  center: readonly [number, number];
  /** Root mean squared distance from the mean centre, in `unit`. */
  standardDistance: number;
  /** Median distance from the mean centre, in `unit`. */
  medianDistanceFromCenter: number;
  /**
   * Distance from each point to its nearest other point, in `unit`; absent below two points.
   * `upperFence` is the upper quartile plus three interquartile ranges, capped at `maximum`;
   * `aboveUpperFenceCount` points are farther than that from every other point (isolated points).
   */
  nearestNeighbor?: {
    minimum: number;
    lowerQuartile: number;
    median: number;
    upperQuartile: number;
    mean: number;
    maximum: number;
    upperFence: number;
    aboveUpperFenceCount: number;
    /**
     * Median, over the distinct locations, of the distance from each to the nearest OTHER
     * distinct location; absent with fewer than two distinct locations. With many repeated
     * locations the plain median is 0, and this is the spacing of the places themselves.
     */
    nonZeroMedian?: number;
  };
  /** Ascending distances to the k-th nearest neighbour, at most 512 evenly spaced quantiles. */
  kDistance?: {k: number; distances: readonly number[]};
  /** Points that share a location with an earlier point. */
  duplicateLocationCount: number;
};

/** What is known about one dataset when choosing an analysis for it. */
export type GPUAnalysisDataProfile = {
  kind: GPUSpatialAnalysisDataKind;
  rowCount: number;
  vertexCount?: number;
  /** Exact counts for the selected network topology; absent until that topology is profiled. */
  network?: {nodeCount: number; edgeCount: number; arcCount: number};
  coordinates?: {
    model: GPUAnalysisCoordinateModel;
    source: GPUAnalysisFactSource;
    reason?: string;
    /** How sure a guessed model is; a low-confidence guess needs confirming before metric work. */
    confidence?: 'high' | 'low';
  };
  /** [minimumX, minimumY, maximumX, maximumY] in source coordinates. */
  extent?: readonly [number, number, number, number];
  columns: readonly GPUAnalysisColumnProfile[];
  rowSemantics?: {
    value: GPUAnalysisRowSemantics;
    source: GPUAnalysisFactSource;
    reason?: string;
    /** How sure a guess is; guessed row semantics are a question to ask, never a fact. */
    confidence?: 'high' | 'low';
  };
  spacing?: GPUAnalysisSpacingStatistics;
  /** Neighbour counts under contiguity, for area data, when they have been computed. */
  contiguity?: {minimum: number; median: number; maximum: number; islandCount: number};
};

/** A scalar comparison; `whenParameter` preserves the original field-role spelling. */
export type GPUAnalysisParameterComparison =
  | {parameter: string; equals: readonly (string | number | boolean)[]}
  | {whenParameter: string; equals: readonly (string | number | boolean)[]};

/** Any branch may activate a setting or require a role, including different controllers. */
export type GPUAnalysisParameterCondition =
  | GPUAnalysisParameterComparison
  | {anyOf: readonly GPUAnalysisParameterCondition[]};

/** A column an operation reads, by the role it plays. */
export type GPUAnalysisFieldRoleSpec = {
  /** Stable role id: `value`, `cases`, `population`, `time`, `category`, `response`, `predictors`. */
  role: string;
  label: string;
  accepts: readonly GPUAnalysisColumnType[];
  /** Measurement levels the method is meant for. Another level is a warning, never a block. */
  measurement?: readonly GPUAnalysisMeasurementLevel[];
  /** True, false, or required while the parameter condition holds (including cross-parameter OR). */
  required: boolean | GPUAnalysisParameterCondition;
  /** Present when the role takes several columns. */
  multiple?: {minimum: number; maximum: number};
};

/**
 * Preconditions that can be stated about data. Those that a profile cannot establish
 * (`connected-weights`, `disjoint-coverage`) always evaluate to `unknown`.
 */
export const GPU_ANALYSIS_REQUIREMENT_IDS = [
  'minimum-rows',
  'maximum-rows',
  'maximum-vertices',
  'maximum-network-edges',
  'maximum-network-nodes',
  'maximum-network-arcs',
  'planar-coordinates',
  'geographic-coordinates',
  'declared-coordinates',
  'row-semantics',
  'positive-values',
  'non-negative-values',
  'numeric-range',
  'no-missing-values',
  'varying-values',
  'minimum-distinct-values',
  'maximum-distinct-values',
  'every-row-has-a-neighbor',
  'no-duplicate-locations',
  'connected-weights',
  'disjoint-coverage'
] as const;

export type GPUAnalysisRequirementId = (typeof GPU_ANALYSIS_REQUIREMENT_IDS)[number];

/** One declared precondition of an operation on one of its inputs. */
export type GPUAnalysisRequirementSpec = {
  id: GPUAnalysisRequirementId;
  /** `block`: the operation would fail or be meaningless. `warn`: it runs, read with care. */
  severity: 'block' | 'warn';
  /** Numeric limit of row/vertex/network counts and minimum/maximum distinct-value requirements. */
  limit?: number;
  /** Field role the requirement is about, for the value requirements. */
  role?: string;
  /** Inclusive bounds and optional integrality for `numeric-range`; at least one is required. */
  minimum?: number;
  maximum?: number;
  integer?: boolean;
  /** Numeric parameter checked instead of a column role, for `numeric-range`. */
  parameter?: string;
  /** Accepted row semantics, for `row-semantics`. */
  expects?: readonly GPUAnalysisRowSemantics[];
  /** What the user can do about a failure: `Aggregate the points to cells first`. */
  remedy?: string;
};

/** How a requirement was settled. `unknown` is a result, not an error. */
export type GPUAnalysisRequirementResult = {
  id: GPUAnalysisRequirementId;
  role?: string;
  severity: 'block' | 'warn';
  state: 'passed' | 'failed' | 'unknown';
  /** What the state rests on. */
  evidence: 'profile' | 'sampled-profile' | 'declaration' | 'guess' | 'not-checkable';
  /** One sentence naming the fact and the limit: `12 rows; at least 30 are needed`. */
  message: string;
  remedy?: string;
};

/**
 * Rules that derive a parameter default from a profile. Each is documented where implemented.
 * The `column-*` rules read the statistics of one column, named by the caller; the others read
 * the extent and spacing of the locations.
 */
export const GPU_ANALYSIS_DEFAULT_RULES = [
  'maximum-nearest-neighbor-distance',
  'median-nearest-neighbor-distance',
  'k-distance-knee',
  'extent-shorter-side-over-250',
  'extent-shorter-side-over-250-in-source-units',
  'extent-diagonal',
  'extent-west',
  'extent-south',
  'extent-east',
  'extent-north',
  'normal-reference-bandwidth-scott',
  'normal-reference-bandwidth-silverman',
  'standard-distance',
  'outlier-resistant-nearest-neighbor-distance',
  'mean-point-spacing',
  'column-variance',
  'column-standard-deviation',
  'column-range',
  'column-sum'
] as const;

export type GPUAnalysisDefaultRule = (typeof GPU_ANALYSIS_DEFAULT_RULES)[number];

/** The value a rule produced, with enough to tell the user why. */
export type GPUAnalysisDefaultResolution = {
  rule: GPUAnalysisDefaultRule;
  /** Rule output times `factor`; undefined when the profile lacks what the rule needs. */
  value: number | undefined;
  factor: number;
  unit: 'metre' | 'coordinate-unit' | 'degree' | 'column-unit';
  /** `Largest nearest-neighbour distance (1,240 m), so every point has a neighbour`. */
  explanation: string;
  /** True when the statistics behind the value came from a sample. */
  sampled: boolean;
};

/** Fixed class schemes for results whose values are categories with an agreed meaning. */
export const GPU_ANALYSIS_CLASS_SCHEMES = [
  'hot-spot-confidence-bins',
  'local-moran-clusters',
  'local-geary-classes',
  'cluster-membership',
  'significance-flag',
  'validity-flag'
] as const;

export type GPUAnalysisClassScheme = (typeof GPU_ANALYSIS_CLASS_SCHEMES)[number];

/**
 * Producers write the same classes differently. `contributor-integer`: the integer codes the
 * analysis contributors write. `catalog-text`: the text labels of the Kepler catalog tables
 * (`hot-99`, `HH`). `catalog-signed-label`: signed integer labels with -1 for noise.
 */
export const GPU_ANALYSIS_CLASS_ENCODINGS = [
  'contributor-integer',
  'catalog-text',
  'catalog-signed-label'
] as const;

export type GPUAnalysisClassEncoding = (typeof GPU_ANALYSIS_CLASS_ENCODINGS)[number];

/** How one output is shown without the user choosing anything. */
export type GPUAnalysisPresentationSpec = {
  /** Output column the map colours by. */
  column: string;
  scale: 'sequential' | 'diverging' | 'categorical' | 'class-scheme';
  classScheme?: GPUAnalysisClassScheme;
  /** How the producer writes the scheme's classes; `contributor-integer` when absent. */
  classEncoding?: GPUAnalysisClassEncoding;
  /** Value a diverging scale is centred on. */
  center?: number;
};

/** One class of a class scheme, as a legend entry. */
export type GPUAnalysisResultClass = {
  /** Value stored in the output column. */
  value: number | string;
  label: string;
  /** Meaning of the class, used to pick a colour consistently across hosts. */
  tone:
    | 'strong-high'
    | 'high'
    | 'weak-high'
    | 'neutral'
    | 'weak-low'
    | 'low'
    | 'strong-low'
    | 'high-outlier'
    | 'low-outlier'
    | 'category'
    | 'excluded';
};

/** Sentence templates that summarise a result. */
export const GPU_ANALYSIS_SUMMARY_TEMPLATES = [
  'significant-hot-cold-count',
  'local-cluster-count',
  'cluster-count-and-noise',
  'global-statistic',
  'regression-fit',
  'value-range',
  'count-per-area',
  'row-count'
] as const;

export type GPUAnalysisSummaryTemplate = (typeof GPU_ANALYSIS_SUMMARY_TEMPLATES)[number];

/** Cautions that travel with a result. */
export const GPU_ANALYSIS_CAUTIONS = [
  'fewer-than-thirty-features',
  'counts-where-rates-expected',
  'pseudo-p-values',
  'multiple-testing-uncorrected',
  'rows-without-neighbors',
  'sampled-statistics',
  'guessed-coordinates',
  'default-from-fallback',
  'exploratory-parameter-search',
  'significance-not-attainable',
  'small-denominators',
  'poor-model-fit',
  'row-order-dependent-inference'
] as const;

export type GPUAnalysisCaution = (typeof GPU_ANALYSIS_CAUTIONS)[number];

/** How one output is read. */
export type GPUAnalysisInterpretationSpec = {
  summary: GPUAnalysisSummaryTemplate;
  /** Cautions that always apply to this output; others are added from the run's facts. */
  cautions: readonly GPUAnalysisCaution[];
};
