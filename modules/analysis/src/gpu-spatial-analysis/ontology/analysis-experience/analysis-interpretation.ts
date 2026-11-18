// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUAnalysisCaution,
  GPUAnalysisClassEncoding,
  GPUAnalysisClassScheme,
  GPUAnalysisResultClass,
  GPUAnalysisSummaryTemplate
} from './analysis-experience-types';

// Host-neutral vocabulary for showing a result: the legend classes of a fixed class scheme, a
// tally of a column against a scheme, one plain sentence per summary template, and the cautions
// a statistician would attach to a run. Pure CPU code with no luma.gl dependency; every sentence
// is built from the numbers it is given and states no more than they support.

// ---------------------------------------------------------------------------------------------
// Number formatting
// ---------------------------------------------------------------------------------------------

/**
 * Largest magnitude formatted. `Number.prototype.toFixed` switches to exponent notation at 1e21;
 * counts and statistics beyond 1e15 carry no meaning at the precision a sentence reports.
 */
const MAXIMUM_FORMATTED_MAGNITUDE = 1e15;

/**
 * Formats a finite number with a comma every three integer digits, `fractionDigits` decimals and
 * an ASCII minus sign. It does not depend on the runtime locale, so the same number gives the same
 * text on every host. With `trimTrailingZeros` the decimals that are zero at the end are dropped
 * (`1.50` becomes `1.5`, `2.00` becomes `2`). A value that rounds to zero is never shown as `-0`.
 */
export function formatGPUAnalysisNumber(
  value: number,
  fractionDigits: number = 0,
  trimTrailingZeros: boolean = false
): string {
  if (!Number.isFinite(value) || Math.abs(value) >= MAXIMUM_FORMATTED_MAGNITUDE) {
    throw new Error(
      `formatGPUAnalysisNumber: argument "value" must be finite with a magnitude below ${MAXIMUM_FORMATTED_MAGNITUDE}; received ${value}.`
    );
  }
  if (!Number.isInteger(fractionDigits) || fractionDigits < 0 || fractionDigits > 12) {
    throw new Error(
      `formatGPUAnalysisNumber: argument "fractionDigits" must be an integer from 0 to 12; received ${fractionDigits}.`
    );
  }
  const fixed = Math.abs(value).toFixed(fractionDigits);
  const [integerPart, fractionPart = ''] = fixed.split('.');
  const groupedInteger = integerPart.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  const keptFraction = trimTrailingZeros ? fractionPart.replace(/0+$/, '') : fractionPart;
  const text = keptFraction ? `${groupedInteger}.${keptFraction}` : groupedInteger;
  const isZero = /^[0,.]*$/.test(text);
  return value < 0 && !isZero ? `-${text}` : text;
}

/** Magnitudes below this are shown in exponent notation; twelve decimals is the formatter's limit. */
const MINIMUM_FIXED_NOTATION_MAGNITUDE = 1e-9;

/** Significant digits a measured value shows when the caller does not ask for more. */
const DEFAULT_SIGNIFICANT_DIGITS = 3;

/** Largest significant-digit count accepted; the fixed-notation formatter stops at 12 decimals. */
const MAXIMUM_SIGNIFICANT_DIGITS = 12;

/**
 * Formats a measured value (a minimum, a statistic, a mean, a legend break) so that it never
 * loses what it measures, and so that one value has one string wherever a report shows it. The
 * rule, with `significantDigits` of 3 unless the caller asks for more:
 * - the number shows `significantDigits` significant digits after the first non-zero digit, but
 *   never rounds away integer digits: 12345.678 prints `12,346`, 1234.5678 prints `1,235`,
 *   12.345 prints `12.3`, 1.2345 prints `1.23`, 0.04567 prints `0.0457`, 0.000213 prints
 *   `0.000213`;
 * - trailing zeros after the decimal point are dropped (`0.5`, `12`), thousands are grouped with
 *   commas, and the sign is an ASCII minus;
 * - two values that differ in their first `significantDigits` significant digits never print alike
 *   (0.00021 and 0.00029 stay apart);
 * - zero prints `0`, and a negative value that rounds to zero never prints `-0`;
 * - `round-trip` uses the shortest string that recovers the exact float, with grouped integers;
 * - magnitudes below 1e-9 or from 1e15 up print in exponent notation (`2.13e-12`), because
 *   fixed notation cannot show them.
 * Counts, percentages, R squared and p-values have their own rules and do not use this.
 */
export function formatGPUAnalysisMeasuredValue(
  value: number,
  significantDigits: number | 'round-trip' = DEFAULT_SIGNIFICANT_DIGITS
): string {
  if (!Number.isFinite(value)) {
    throw new Error(
      `formatGPUAnalysisMeasuredValue: argument "value" must be a finite number; received ${value}.`
    );
  }
  if (significantDigits === 'round-trip') {
    const [coefficient, exponent] = String(value === 0 ? 0 : value).split('e');
    const [integer, fraction] = coefficient.split('.');
    const grouped = integer.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    return `${grouped}${fraction === undefined ? '' : `.${fraction}`}${exponent === undefined ? '' : `e${exponent}`}`;
  }
  if (
    !Number.isInteger(significantDigits) ||
    significantDigits < 1 ||
    significantDigits > MAXIMUM_SIGNIFICANT_DIGITS
  ) {
    throw new Error(
      `formatGPUAnalysisMeasuredValue: argument "significantDigits" must be an integer from 1 to ${MAXIMUM_SIGNIFICANT_DIGITS} or "round-trip"; received ${significantDigits}.`
    );
  }
  const magnitude = Math.abs(value);
  if (magnitude === 0) return '0';
  if (magnitude < MINIMUM_FIXED_NOTATION_MAGNITUDE || magnitude >= MAXIMUM_FORMATTED_MAGNITUDE) {
    return value.toExponential(significantDigits - 1);
  }
  const decimals = Math.min(
    Math.max(significantDigits - 1 - Math.floor(Math.log10(magnitude)), 0),
    MAXIMUM_SIGNIFICANT_DIGITS
  );
  return formatGPUAnalysisNumber(value, decimals, true);
}

/**
 * Strings for a list of values, such as the class breaks of a legend, such that different values
 * never print alike. It uses the fewest significant digits (starting at the default of
 * {@link formatGPUAnalysisMeasuredValue}) that separate every pair, so 1000 and 1000.4 print
 * `1,000` and `1,000.4` while 10 and 20 print `10` and `20`. Equal values print alike. The result
 * has the order of the input. Falls back to shortest round-trip strings if twelve digits collide.
 */
export function formatGPUAnalysisDistinctValues(values: ArrayLike<number>): string[] {
  const precision = getGPUAnalysisColumnPrecision(values);
  return Array.from(values, value => formatGPUAnalysisMeasuredValue(value, precision));
}

/** Choose one precision for a column; round-trip strings separate adjacent finite floats. */
export function getGPUAnalysisColumnPrecision(values: ArrayLike<number>): number | 'round-trip' {
  for (let index = 0; index < values.length; index++) {
    if (!Number.isFinite(values[index])) {
      throw new Error(
        `getGPUAnalysisColumnPrecision: argument "values" must hold finite numbers; received ${values[index]} at index ${index}.`
      );
    }
  }
  for (
    let significantDigits = DEFAULT_SIGNIFICANT_DIGITS;
    significantDigits <= MAXIMUM_SIGNIFICANT_DIGITS;
    significantDigits++
  ) {
    const valueByString = new Map<string, number>();
    let isDistinct = true;
    for (let index = 0; index < values.length; index++) {
      const value = values[index];
      const text = formatGPUAnalysisMeasuredValue(value, significantDigits);
      if (valueByString.has(text) && valueByString.get(text) !== value) isDistinct = false;
      valueByString.set(text, value);
    }
    if (isDistinct) return significantDigits;
  }
  return 'round-trip';
}

/** Singular and plural form of a noun shown in a sentence. */
export type GPUAnalysisNoun = {singular: string; plural: string};

const DEFAULT_FEATURE_NOUN: GPUAnalysisNoun = {singular: 'feature', plural: 'features'};
const DEFAULT_POINT_NOUN: GPUAnalysisNoun = {singular: 'point', plural: 'points'};
const DEFAULT_ROW_NOUN: GPUAnalysisNoun = {singular: 'row', plural: 'rows'};
const DEFAULT_AREA_NOUN: GPUAnalysisNoun = {singular: 'area', plural: 'areas'};
const DEFAULT_ITEM_NOUN: GPUAnalysisNoun = {singular: 'item', plural: 'items'};
const DEFAULT_OBSERVATION_NOUN: GPUAnalysisNoun = {
  singular: 'observation',
  plural: 'observations'
};

function chooseNoun(count: number, noun: GPUAnalysisNoun): string {
  return count === 1 ? noun.singular : noun.plural;
}

function chooseBe(count: number): string {
  return count === 1 ? 'is' : 'are';
}

function joinList(items: readonly string[]): string {
  if (items.length <= 1) return items.join('');
  return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`;
}

/**
 * A share (0 to 1) as a percentage with one decimal. A share that is not zero is never shown as
 * `0.0%`, and one that is not complete is never shown as `100%`.
 */
function formatShare(share: number): string {
  if (share <= 0) return '0%';
  if (share >= 1) return '100%';
  const percent = share * 100;
  if (percent < 0.05) return 'less than 0.1%';
  if (percent >= 99.95) return 'more than 99.9%';
  return `${formatGPUAnalysisNumber(percent, 1)}%`;
}

function validateNonNegativeInteger(functionName: string, name: string, value: number): void {
  if (!Number.isInteger(value) || value < 0 || value > Number.MAX_SAFE_INTEGER) {
    throw new Error(
      `${functionName}: argument "${name}" must be a non-negative integer; received ${value}.`
    );
  }
}

function validateFiniteNumber(functionName: string, name: string, value: number): void {
  if (!Number.isFinite(value)) {
    throw new Error(
      `${functionName}: argument "${name}" must be a finite number; received ${value}.`
    );
  }
}

function validateUnitInterval(functionName: string, name: string, value: number): void {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error(`${functionName}: argument "${name}" must be from 0 to 1; received ${value}.`);
  }
}

function validateOpenUnitInterval(functionName: string, name: string, value: number): void {
  if (!Number.isFinite(value) || value <= 0 || value >= 1) {
    throw new Error(
      `${functionName}: argument "${name}" must be greater than 0 and less than 1; received ${value}.`
    );
  }
}

// ---------------------------------------------------------------------------------------------
// Class schemes
// ---------------------------------------------------------------------------------------------

// `GPUAnalysisClassEncoding` (from the contract) says how a producer writes the class values of a
// scheme; the library has more than one encoding of the same concept:
// - `contributor-integer`: what the GPU contributors write, and what the Kepler lattice recipe
//   table (`recipe-hot-spot-lattice`, column `bin`) passes through.
// - `catalog-text`: the text columns of the CPU-reference Kepler bindings `hotspot-gi-star`
//   (`classification`) and `local-moran-statistics` (`cluster`).
// - `catalog-signed-label`: the Kepler cluster tables (`cluster` column, `int32`), which map the
//   contributors' unsigned noise label to `-1`.

/**
 * Noise label written by the density clustering contributors (`GPUSpatialClustering`,
 * `GPUOptics`, `GPUHierarchicalDensityClustering`, `GPUSpatialKMeans`) as an unsigned 32 bit
 * label. Source: `GPU_SPATIAL_CLUSTERING_NOISE`,
 * `modules/analysis/src/gpu-spatial-analysis/spatial-clustering/spatial-clustering-parameters.ts:14`.
 */
const CONTRIBUTOR_NOISE_LABEL = 0xffffffff;

/** Noise label of the Kepler cluster tables. Source: `catalog-platform3-b-clusters.ts:164`. */
const CATALOG_NOISE_LABEL = -1;

/**
 * Legend classes of a fixed class scheme, in legend order (strongest high class first). Each
 * class `value` is the value the library writes; the source line of every encoding is cited
 * beside it.
 *
 * For `cluster-membership` the classes are the noise class and the first cluster only; the other
 * clusters are numbered consecutively after it and share its tone `category`.
 *
 * @throws when the scheme has no such encoding (for example `catalog-text` for membership).
 */
export function getGPUAnalysisResultClasses(
  scheme: GPUAnalysisClassScheme,
  encoding: GPUAnalysisClassEncoding = 'contributor-integer'
): readonly GPUAnalysisResultClass[] {
  switch (scheme) {
    case 'hot-spot-confidence-bins':
      // Contributor: `bins` is sint32 in -3..3, +-3, +-2, +-1 for 99, 95 and 90% hot (+) or cold
      // (-), 0 otherwise (`gpu-hot-spot-analysis.ts:66-70`, WGSL `select(level, -level, z < 0)` at
      // `:413`). The lattice recipe table writes the same integers
      // (`catalog-platform5-recipes-hot-spot.ts:116-150`, column `bin` at `:278`).
      if (encoding === 'contributor-integer') {
        return [
          {value: 3, label: 'Hot spot, 99% confidence', tone: 'strong-high'},
          {value: 2, label: 'Hot spot, 95% confidence', tone: 'high'},
          {value: 1, label: 'Hot spot, 90% confidence', tone: 'weak-high'},
          {value: 0, label: 'Not significant', tone: 'neutral'},
          {value: -1, label: 'Cold spot, 90% confidence', tone: 'weak-low'},
          {value: -2, label: 'Cold spot, 95% confidence', tone: 'low'},
          {value: -3, label: 'Cold spot, 99% confidence', tone: 'strong-low'}
        ];
      }
      // Kepler `hotspot-gi-star`: text `hot-99|hot-95|hot-90|cold-90|cold-95|cold-99|
      // not-significant` (`catalog-platform3-b-statistics.ts:580-589`).
      if (encoding === 'catalog-text') {
        return [
          {value: 'hot-99', label: 'Hot spot, 99% confidence', tone: 'strong-high'},
          {value: 'hot-95', label: 'Hot spot, 95% confidence', tone: 'high'},
          {value: 'hot-90', label: 'Hot spot, 90% confidence', tone: 'weak-high'},
          {value: 'not-significant', label: 'Not significant', tone: 'neutral'},
          {value: 'cold-90', label: 'Cold spot, 90% confidence', tone: 'weak-low'},
          {value: 'cold-95', label: 'Cold spot, 95% confidence', tone: 'low'},
          {value: 'cold-99', label: 'Cold spot, 99% confidence', tone: 'strong-low'}
        ];
      }
      break;
    // The numbering below is that of local Moran. Local Geary has its own scheme,
    // `local-geary-classes`; the two use the same integers 0 to 4 with different meanings, so a
    // column of those integers cannot be told apart by its values (see
    // `validateGPUAnalysisResultEncoding`) and the descriptor must name the right scheme.
    case 'local-moran-clusters':
      // Contributor: `GPU_LOCAL_MORAN_QUADRANT`, esda numbering, 0 = not significant, invalid or
      // undefined (islands included, no separate code), 1 HH, 2 LH, 3 LL, 4 HL
      // (`spatial-autocorrelation-parameters.ts:71-85`; written by `GPULocalMoran`, doc at
      // `gpu-local-moran.ts:83-87`).
      if (encoding === 'contributor-integer') {
        return [
          {value: 1, label: 'High-high cluster core', tone: 'high'},
          {value: 4, label: 'High value among low values', tone: 'high-outlier'},
          {value: 2, label: 'Low value among high values', tone: 'low-outlier'},
          {value: 3, label: 'Low-low cluster core', tone: 'low'},
          {value: 0, label: 'Not significant or not analysed', tone: 'neutral'}
        ];
      }
      // Kepler `local-moran-statistics`: text `HH|HL|LH|LL|not-significant|not-tested`
      // (`catalog-platform3-b-statistics.ts:441-455`). `not-tested` is a row without a p-value
      // (zero permutations or no neighbours); the contributor folds it into 0.
      if (encoding === 'catalog-text') {
        return [
          {value: 'HH', label: 'High-high cluster core', tone: 'high'},
          {value: 'HL', label: 'High value among low values', tone: 'high-outlier'},
          {value: 'LH', label: 'Low value among high values', tone: 'low-outlier'},
          {value: 'LL', label: 'Low-low cluster core', tone: 'low'},
          {value: 'not-significant', label: 'Not significant', tone: 'neutral'},
          {value: 'not-tested', label: 'Not tested', tone: 'excluded'}
        ];
      }
      break;
    case 'local-geary-classes':
      // Contributor: `GPU_LOCAL_GEARY_CLASS` (Anselin 2019, esda `Geary_Local` `labs`), written
      // to the `classes` output of `GPULocalPermutationTest` for `'localGeary'`
      // (`gpu-local-permutation-test.ts:119-131`): 0 undefined, 1 high-high, 2 low-low,
      // 3 negative association (other), 4 not significant. The numbering differs from local
      // Moran's: 4 is "not significant" here and "high-low" there, and 0 is "undefined" here and
      // "not significant" there. The values are repeated here, not imported, so that the
      // ontology does not load the contributor; the spec pins them to the exported constant.
      if (encoding === 'contributor-integer') {
        return [
          {value: 1, label: 'Similar values, high (positive association)', tone: 'high'},
          {value: 2, label: 'Similar values, low (positive association)', tone: 'low'},
          {value: 3, label: 'Dissimilar values (negative association)', tone: 'high-outlier'},
          {value: 4, label: 'Not significant', tone: 'neutral'},
          {value: 0, label: 'Undefined or not tested', tone: 'excluded'}
        ];
      }
      break;
    case 'cluster-membership':
      // Contributor: labels are uint32, compact cluster ids from 0, and 0xffffffff for noise
      // (`gpu-spatial-clustering.ts:78`, `GPU_SPATIAL_CLUSTERING_NOISE`).
      if (encoding === 'contributor-integer') {
        return [
          {value: CONTRIBUTOR_NOISE_LABEL, label: 'Noise', tone: 'excluded'},
          {value: 0, label: 'Cluster 0 (clusters are numbered from 0)', tone: 'category'}
        ];
      }
      // Kepler cluster tables: int32 `cluster`, first cluster 0, noise -1
      // (`catalog-platform3-b-clusters.ts:103, 164, 191`).
      if (encoding === 'catalog-signed-label') {
        return [
          {value: CATALOG_NOISE_LABEL, label: 'Noise', tone: 'excluded'},
          {value: 0, label: 'Cluster 0 (clusters are numbered from 0)', tone: 'category'}
        ];
      }
      break;
    case 'significance-flag':
      // Contributors write a uint32 mask of 1 (significant) and 0
      // (`gpu-local-permutation-test.ts:222-223`, `gpu-geographically-weighted-regression.ts:102`);
      // the Kepler `local-lisa` table writes a boolean `significant`
      // (`catalog-platform5-lisa.ts:268`), which `classifyGPUAnalysisResultValues` reads as 1 or 0.
      if (encoding === 'contributor-integer') {
        return [
          {value: 1, label: 'Significant', tone: 'high'},
          {value: 0, label: 'Not significant', tone: 'neutral'}
        ];
      }
      break;
    case 'validity-flag':
      // Contributors write a uint32 mask of 1 (kept) and 0 (excluded or invalid)
      // (`point-dbscan.ts:171` effectiveMask, `gpu-quantile-lisa.ts:237` valid-mask,
      // `gpu-network-k-function.ts:519` validFlags). Kepler writes a boolean (`is_core` is a
      // different concept and is not covered here).
      if (encoding === 'contributor-integer') {
        return [
          {value: 1, label: 'Valid', tone: 'neutral'},
          {value: 0, label: 'Excluded or invalid', tone: 'excluded'}
        ];
      }
      break;
  }
  throw new Error(
    `getGPUAnalysisResultClasses: scheme "${scheme}" has no "${encoding}" encoding; ${describeAcceptedEncodings(scheme)}.`
  );
}

function describeAcceptedEncodings(scheme: GPUAnalysisClassScheme): string {
  switch (scheme) {
    case 'hot-spot-confidence-bins':
    case 'local-moran-clusters':
      return 'accepted: "contributor-integer", "catalog-text"';
    case 'cluster-membership':
      return 'accepted: "contributor-integer", "catalog-signed-label"';
    case 'local-geary-classes':
    case 'significance-flag':
    case 'validity-flag':
      return 'accepted: "contributor-integer"';
  }
}

/**
 * Share (0 to 1) of the non-null values of a column that are classes of the scheme under the
 * encoding. A host or a spec uses it to detect a descriptor that names the wrong scheme or
 * encoding: a text status column given `validity-flag` scores 0. NaN and infinities count as
 * non-null and unrecognised. A column with no non-null values scores 0.
 *
 * Limit: it checks the values, not the producer. `local-moran-clusters` and `local-geary-classes`
 * both use the integers 0 to 4 (with different meanings), so a column of one scores 1 under the
 * other and the validator cannot tell them apart; a 0 and 1 column scores 1 under every integer
 * scheme. It does tell a text column from an integer scheme and the reverse.
 */
export function validateGPUAnalysisResultEncoding(
  scheme: GPUAnalysisClassScheme,
  encoding: GPUAnalysisClassEncoding,
  values: ArrayLike<number | string | boolean | null | undefined>
): number {
  const tally = classifyGPUAnalysisResultValues(scheme, values, encoding);
  let nullCount = 0;
  for (let index = 0; index < values.length; index++) {
    if (values[index] === null || values[index] === undefined) nullCount++;
  }
  const nonNullCount = values.length - nullCount;
  if (nonNullCount === 0) return 0;
  return (values.length - tally.unrecognized) / nonNullCount;
}

/** Result of tallying a column against a class scheme. */
export type GPUAnalysisClassTally = {
  /**
   * Rows per class value, in legend order with a zero for every class of the scheme. For
   * `cluster-membership` the keys are the noise label first, then each cluster id found, ascending.
   */
  counts: ReadonlyMap<number | string, number>;
  /** Rows that are null, undefined, not finite or not a value of the scheme. */
  unrecognized: number;
};

/**
 * Normalize a value for both tallying and publication. Flag booleans become 0/1; contributor
 * validity uint32 values become zero/nonzero. Other schemes retain their exact codes.
 */
export function normalizeGPUAnalysisClassValue(
  scheme: GPUAnalysisClassScheme,
  encoding: GPUAnalysisClassEncoding,
  value: number | string | boolean | null | undefined
): number | string | null | undefined {
  if ((scheme === 'significance-flag' || scheme === 'validity-flag') && typeof value === 'boolean')
    value = value ? 1 : 0;
  if (
    scheme === 'validity-flag' &&
    encoding === 'contributor-integer' &&
    typeof value === 'number' &&
    Number.isInteger(value) &&
    value >= 0 &&
    value <= 0xffffffff
  )
    return value === 0 ? 0 : 1;
  return typeof value === 'boolean' ? undefined : value;
}

/** Count normalized classes; null, nonfinite, malformed and unknown values are unrecognized. */
export function classifyGPUAnalysisResultValues(
  scheme: GPUAnalysisClassScheme,
  values: ArrayLike<number | string | boolean | null | undefined>,
  encoding: GPUAnalysisClassEncoding = 'contributor-integer'
): GPUAnalysisClassTally {
  const classes = getGPUAnalysisResultClasses(scheme, encoding);
  const isMembership = scheme === 'cluster-membership';
  const tally = new Map<number | string, number>();
  for (const resultClass of classes) tally.set(resultClass.value, 0);
  const noiseLabel =
    encoding === 'catalog-signed-label' ? CATALOG_NOISE_LABEL : CONTRIBUTOR_NOISE_LABEL;
  const clusterCounts = new Map<number, number>();
  let unrecognized = 0;

  for (let index = 0; index < values.length; index++) {
    const value = normalizeGPUAnalysisClassValue(scheme, encoding, values[index]);
    if (value === null || value === undefined || typeof value === 'boolean') {
      unrecognized++;
    } else if (isMembership) {
      if (typeof value !== 'number' || !Number.isInteger(value)) {
        unrecognized++;
      } else if (value === noiseLabel) {
        tally.set(noiseLabel, (tally.get(noiseLabel) ?? 0) + 1);
      } else if (value >= 0 && value < CONTRIBUTOR_NOISE_LABEL) {
        clusterCounts.set(value, (clusterCounts.get(value) ?? 0) + 1);
      } else {
        unrecognized++;
      }
    } else if (tally.has(value)) {
      tally.set(value, (tally.get(value) as number) + 1);
    } else {
      unrecognized++;
    }
  }

  if (isMembership) {
    const counts = new Map<number | string, number>([[noiseLabel, tally.get(noiseLabel) ?? 0]]);
    for (const clusterId of [...clusterCounts.keys()].sort((a, b) => a - b)) {
      counts.set(clusterId, clusterCounts.get(clusterId) as number);
    }
    return {counts, unrecognized};
  }
  return {counts: tally, unrecognized};
}

// ---------------------------------------------------------------------------------------------
// Summary sentences
// ---------------------------------------------------------------------------------------------

/**
 * Correction for testing many locations at once, as it is named in a sentence. `false-discovery-rate`
 * is the Benjamini-Hochberg procedure; `holm` and `sidak` are the step-down procedures of those
 * names; `bonferroni` divides the level by the number of tests.
 */
export type GPUAnalysisMultipleTestingCorrection =
  | 'none'
  | 'false-discovery-rate'
  | 'bonferroni'
  | 'holm'
  | 'sidak';

function describeCorrection(correction: GPUAnalysisMultipleTestingCorrection): string {
  switch (correction) {
    case 'none':
      return ' without correction for multiple testing';
    case 'false-discovery-rate':
      return ' after false discovery rate correction';
    case 'bonferroni':
      return ' after Bonferroni correction';
    case 'holm':
      return ' after Holm correction';
    case 'sidak':
      return ' after Šidák correction';
  }
}

/** Facts of `significant-hot-cold-count`: Getis-Ord Gi* style classes at one confidence level. */
export type GPUAnalysisSignificantHotColdFacts = {
  /** Features that were tested. */
  featureCount: number;
  /** Tested features that are significant hot spots at `confidencePercent`. */
  hotCount: number;
  /** Tested features that are significant cold spots at `confidencePercent`. */
  coldCount: number;
  /** Confidence level the classes were cut at, for example 95. */
  confidencePercent: number;
  correction: GPUAnalysisMultipleTestingCorrection;
  /**
   * Permutations behind the p-values when they are pseudo p-values; leave it out (or 0) for an
   * analytic p-value. With it, a run whose correction makes significance impossible is reported
   * as such instead of as "none significant".
   */
  permutationCount?: number;
  /** What a feature is called (`cell`, `county`); defaults to feature. */
  featureNoun?: GPUAnalysisNoun;
};

/** Facts of `local-cluster-count`: the four local Moran classes at one significance level. */
export type GPUAnalysisLocalClusterFacts = {
  /** Features that were tested (those with a p-value). */
  featureCount: number;
  highHighCount: number;
  lowLowCount: number;
  highLowCount: number;
  lowHighCount: number;
  /** Cut-off the p-values were compared with, for example 0.05. */
  significanceLevel: number;
  /** Whether the p-values come from permutations (`pseudo`, the default) or a normal approximation. */
  pValueKind?: 'pseudo' | 'analytic';
  correction: GPUAnalysisMultipleTestingCorrection;
  /** Permutations behind the pseudo p-values. */
  permutationCount?: number;
  /**
   * False when the producer computed no p-values at all (zero permutations and no analytic
   * alternative); the sentence then says so instead of "no features to test". Defaults to true,
   * except that `permutationCount` 0 with `pValueKind` `pseudo` (or absent) counts as false.
   */
  pValuesComputed?: boolean;
  /** Features that entered the analysis, tested or not; used only when no p-values were computed. */
  analyzedCount?: number;
  featureNoun?: GPUAnalysisNoun;
};

/** Facts of `cluster-count-and-noise`: a density or partition clustering. */
export type GPUAnalysisClusterCountFacts = {
  pointCount: number;
  clusterCount: number;
  /** Points labelled noise. */
  noiseCount: number;
  /** Points in the largest cluster; required when `clusterCount` is above zero. */
  largestClusterCount?: number;
  pointNoun?: GPUAnalysisNoun;
};

/** Direction a p-value was computed in. */
export type GPUAnalysisSidedness = 'two-sided' | 'greater' | 'less';

/** Facts of `global-statistic`: one statistic for the whole dataset. */
export type GPUAnalysisGlobalStatisticFacts = {
  /** Name as shown, for example `Moran's I`. */
  statisticName: string;
  value: number;
  pValue: number;
  /** Permutations behind a pseudo p-value; 0 means the p-value is analytic. */
  permutationCount: number;
  /** Cut-off the p-value is compared with; defaults to 0.05. */
  significanceLevel?: number;
  /**
   * Direction of the p-value: `two-sided` (values at least as extreme in either direction),
   * `greater` (at least as large) or `less` (at least as small). Defaults to `two-sided`.
   */
  sidedness?: GPUAnalysisSidedness;
  /**
   * What the p-value is computed against, as a short noun phrase: `a random pattern` (default),
   * or for Knox `no space-time interaction`.
   */
  nullDescription?: string;
  /** What a positive value means for this statistic, for example `similar values near each other`. */
  positiveMeaning?: string;
  /** What a negative value means, for example `dissimilar values near each other`. */
  negativeMeaning?: string;
};

/** Facts of `regression-fit`. */
export type GPUAnalysisRegressionFitFacts = {
  rSquared: number;
  observationCount: number;
  /** Adjusted R squared, when the model reports one. */
  adjustedRSquared?: number;
  observationNoun?: GPUAnalysisNoun;
};

/** Facts of `value-range`. */
export type GPUAnalysisValueRangeFacts = {
  minimum: number;
  maximum: number;
  /** Finite values the range was taken over. */
  valueCount: number;
  /** What the values are, for example `Median income`; the sentence names them only when given. */
  valueLabel?: string;
  /** Precision already chosen for the column, shared with its legend. */
  significantDigits?: number | 'round-trip';
};

/** Facts of `count-per-area`. */
export type GPUAnalysisCountPerAreaFacts = {
  totalCount: number;
  areaCount: number;
  /** Largest count found in one area. */
  maximumCount?: number;
  itemNoun?: GPUAnalysisNoun;
  areaNoun?: GPUAnalysisNoun;
};

/** Facts of `row-count`. */
export type GPUAnalysisRowCountFacts = {
  rowCount: number;
  rowNoun?: GPUAnalysisNoun;
};

/** The facts shape each summary template takes. */
export type GPUAnalysisSummaryFactsByTemplate = {
  'significant-hot-cold-count': GPUAnalysisSignificantHotColdFacts;
  'local-cluster-count': GPUAnalysisLocalClusterFacts;
  'cluster-count-and-noise': GPUAnalysisClusterCountFacts;
  'global-statistic': GPUAnalysisGlobalStatisticFacts;
  'regression-fit': GPUAnalysisRegressionFitFacts;
  'value-range': GPUAnalysisValueRangeFacts;
  'count-per-area': GPUAnalysisCountPerAreaFacts;
  'row-count': GPUAnalysisRowCountFacts;
};

/** Default significance level of `global-statistic` when the facts do not give one. */
const DEFAULT_SIGNIFICANCE_LEVEL = 0.05;

/** Tolerance of comparisons with a significance threshold, so 1 / 1000 resolves a level of 0.001. */
const RESOLUTION_RELATIVE_TOLERANCE = 1e-9;

/** Most decimals a p-value is printed with; beyond this it is below any level a user chooses. */
const MAXIMUM_P_VALUE_DIGITS = 12;

/**
 * Digits of a pseudo p-value: enough to show its resolution, `1 / (permutations + 1)`, and never
 * fewer than three so that `0.050` and `0.001` read alike.
 */
function getPseudoPValueDigits(permutationCount: number): number {
  let digits = 0;
  while (10 ** digits < permutationCount + 1 && digits < MAXIMUM_P_VALUE_DIGITS) digits++;
  return Math.max(digits, 3);
}

/**
 * A p-value as text. Pseudo p-values show their resolution; an analytic p-value shows three
 * decimals, or enough to separate it from the level when the level is below 0.001, and one below
 * that precision (including 0) prints as `less than 0.001`. Digits are added until the printed
 * value falls on the same side of the level as the p-value, so two different p-values that a
 * sentence reads differently never print alike (0.0501 at level 0.05 prints `0.0501`, not `0.050`).
 */
function formatPValue(pValue: number, permutationCount: number, significanceLevel: number): string {
  const baseDigits =
    permutationCount > 0
      ? getPseudoPValueDigits(permutationCount)
      : Math.min(
          Math.max(3, Math.ceil(-Math.log10(significanceLevel) - RESOLUTION_RELATIVE_TOLERANCE)),
          MAXIMUM_P_VALUE_DIGITS
        );
  if (permutationCount === 0 && pValue < 10 ** -baseDigits) {
    return `less than ${formatGPUAnalysisNumber(10 ** -baseDigits, baseDigits)}`;
  }
  for (let digits = baseDigits; digits <= MAXIMUM_P_VALUE_DIGITS; digits++) {
    const text = formatGPUAnalysisNumber(pValue, digits);
    const isSameSide = pValue <= significanceLevel === Number(text) <= significanceLevel;
    if (isSameSide || digits === MAXIMUM_P_VALUE_DIGITS) return text;
  }
  return formatGPUAnalysisNumber(pValue, MAXIMUM_P_VALUE_DIGITS);
}

/** A significance level as text with three significant digits (`0.05`, `0.00001`). */
function formatLevel(level: number): string {
  return formatGPUAnalysisMeasuredValue(level);
}

// ---------------------------------------------------------------------------------------------
// Whether significance can be reached
// ---------------------------------------------------------------------------------------------

/** Facts of a permutation run that decide whether any test can be declared significant. */
export type GPUAnalysisSignificanceAttainabilityFacts = {
  correction: GPUAnalysisMultipleTestingCorrection;
  /** Number of tests the correction is applied over; at least 1. Ignored for `none`. */
  testCount: number;
  /** Permutations behind each pseudo p-value. */
  permutationCount: number;
  /** Level before correction, for example 0.05. */
  significanceLevel: number;
};

/** Whether any test of a permutation run can be declared significant, and why not. */
export type GPUAnalysisSignificanceAttainability = {
  /** False when no test can be declared significant whatever the data. */
  attainable: boolean;
  /** Smallest pseudo p-value the run can report: `1 / (permutations + 1)`. */
  smallestPValue: number;
  /**
   * Bound on the smallest attainable p-value for any rejection to be possible: the level for
   * `none` and `false-discovery-rate` (BH can reach its last rank), level / tests for
   * `bonferroni` and `holm`, and `1 - (1 - level)^(1 / tests)` for `sidak`.
   */
  threshold: number;
  /** Permutations that would make significance reachable; present when `attainable` is false. */
  requiredPermutationCount?: number;
  /**
   * `false-discovery-rate` only: this many tests tied at the smallest attainable p-value suffice
   * for a rejection, even if all remaining p-values are larger. A sufficient configuration,
   * not a necessary condition: larger p-values can pass at later ranks.
   */
  testsNeededAtMinimum?: number;
  /** A clause starting in lower case that states the finding in words. */
  reason: string;
};

function getRequiredPermutationCount(threshold: number): number {
  return Math.max(0, Math.ceil((1 / threshold) * (1 - RESOLUTION_RELATIVE_TOLERANCE)) - 1);
}

function describeCorrectionClause(
  correction: GPUAnalysisMultipleTestingCorrection,
  testCount: number,
  significanceLevel: number
): string {
  const tests = formatGPUAnalysisNumber(testCount);
  const level = formatLevel(significanceLevel);
  switch (correction) {
    case 'none':
      return '';
    case 'bonferroni':
      return ` under Bonferroni correction over ${tests} tests (${level} / ${tests})`;
    case 'holm':
      return ` under Holm correction over ${tests} tests (its first step compares the smallest p-value with ${level} / ${tests})`;
    case 'sidak':
      return ` under Šidák correction over ${tests} tests`;
    case 'false-discovery-rate':
      return ' under false discovery rate correction (no test can be significant unless the smallest p-value is at most the level)';
  }
}

/**
 * Whether any test of a permutation run can be declared significant. The smallest pseudo p-value
 * is `1 / (permutations + 1)`, so with a correction that demands a small first-step threshold, a
 * run can be unable to find anything however strong the pattern is: 999 permutations give 0.001,
 * which can never reach `0.05 / 144` under Bonferroni.
 *
 * - `none`: the smallest p-value must be at most the level.
 * - `bonferroni`, `holm`: at most `level / testCount` (Holm's first step is the Bonferroni test).
 *   Permutations needed: `ceil(testCount / level) - 1`.
 * - `sidak`: at most `1 - (1 - level)^(1 / testCount)`.
 * - `false-discovery-rate` (Benjamini-Hochberg): a test is significant when its p-value is at most
 *   `rank * level / testCount` for some rank reached; with `k` tests tied at the smallest
 *   attainable p-value `pMin` the rank `k` is reached when `pMin <= k * level / testCount`. It is
 *   possible when `pMin <= level` (all tests tied), and `ceil(pMin * testCount / level)` tests tied at `pMin` suffice, reported as
 *   `testsNeededAtMinimum`. This is sufficient, not necessary: larger p-values can pass at later ranks.
 *
 * @throws when `testCount` is not a positive integer, `permutationCount` is not a non-negative
 * integer, or `significanceLevel` is not between 0 and 1.
 */
export function getGPUAnalysisSignificanceAttainability(
  facts: GPUAnalysisSignificanceAttainabilityFacts
): GPUAnalysisSignificanceAttainability {
  const name = 'getGPUAnalysisSignificanceAttainability';
  validateNonNegativeInteger(name, 'permutationCount', facts.permutationCount);
  validateOpenUnitInterval(name, 'significanceLevel', facts.significanceLevel);
  if (!Number.isInteger(facts.testCount) || facts.testCount < 1) {
    throw new Error(
      `${name}: argument "testCount" must be a positive integer; received ${facts.testCount}.`
    );
  }
  const {correction, permutationCount, significanceLevel} = facts;
  const testCount = correction === 'none' ? 1 : facts.testCount;
  const smallestPValue = 1 / (permutationCount + 1);
  const level = significanceLevel;
  let threshold: number;
  switch (correction) {
    case 'bonferroni':
    case 'holm':
      threshold = level / testCount;
      break;
    case 'sidak':
      threshold = -Math.expm1(Math.log1p(-level) / testCount);
      break;
    default:
      threshold = level;
  }
  const isAttainable = smallestPValue <= threshold * (1 + RESOLUTION_RELATIVE_TOLERANCE);
  const permutationsText = `${formatGPUAnalysisNumber(permutationCount)} ${permutationCount === 1 ? 'permutation' : 'permutations'}`;
  const clause = describeCorrectionClause(correction, testCount, level);

  if (!isAttainable) {
    const requiredPermutationCount = getRequiredPermutationCount(threshold);
    const required = `at least ${formatGPUAnalysisNumber(requiredPermutationCount)} permutations are needed`;
    const reason =
      permutationCount === 0
        ? `with 0 permutations no pseudo p-value is computed, so no test can be declared significant${clause}; ${required}`
        : `with ${permutationsText} the smallest pseudo p-value is ${formatGPUAnalysisMeasuredValue(smallestPValue)}, but a test needs ${formatGPUAnalysisMeasuredValue(threshold)} or less to be declared significant${clause}, so none can be; ${required}`;
    return {attainable: false, smallestPValue, threshold, requiredPermutationCount, reason};
  }

  if (correction === 'false-discovery-rate') {
    const testsNeededAtMinimum = Math.max(
      1,
      Math.ceil((smallestPValue * testCount) / level - RESOLUTION_RELATIVE_TOLERANCE)
    );
    const reason = `with ${permutationsText} the smallest pseudo p-value is ${formatGPUAnalysisMeasuredValue(smallestPValue)}; Benjamini-Hochberg over ${formatGPUAnalysisNumber(testCount)} tests rejects through the largest rank k whose k-th smallest p-value is at most k times ${formatLevel(level)} / ${formatGPUAnalysisNumber(testCount)}; ${formatGPUAnalysisNumber(testsNeededAtMinimum)} ${testsNeededAtMinimum === 1 ? 'test' : 'tests'} tied at the minimum would suffice, even if all remaining p-values were larger`;
    return {attainable: true, smallestPValue, threshold, testsNeededAtMinimum, reason};
  }
  const reason = `with ${permutationsText} the smallest pseudo p-value is ${formatGPUAnalysisMeasuredValue(smallestPValue)}, which is at most the threshold ${formatGPUAnalysisMeasuredValue(threshold)}${clause}`;
  return {attainable: true, smallestPValue, threshold, reason};
}

// ---------------------------------------------------------------------------------------------
// Sentences that use it
// ---------------------------------------------------------------------------------------------

function summarizeSignificantHotCold(facts: GPUAnalysisSignificantHotColdFacts): string {
  const name = 'summarizeGPUAnalysisResult';
  validateNonNegativeInteger(name, 'featureCount', facts.featureCount);
  validateNonNegativeInteger(name, 'hotCount', facts.hotCount);
  validateNonNegativeInteger(name, 'coldCount', facts.coldCount);
  validateFiniteNumber(name, 'confidencePercent', facts.confidencePercent);
  if (facts.permutationCount !== undefined) {
    validateNonNegativeInteger(name, 'permutationCount', facts.permutationCount);
  }
  if (facts.hotCount + facts.coldCount > facts.featureCount) {
    throw new Error(
      `${name}: arguments "hotCount" and "coldCount" must sum to at most "featureCount" (${facts.featureCount}); received ${facts.hotCount} and ${facts.coldCount}.`
    );
  }
  const noun = facts.featureNoun ?? DEFAULT_FEATURE_NOUN;
  const confidence = `${formatGPUAnalysisNumber(facts.confidencePercent, 1, true)}% confidence`;
  const correction = describeCorrection(facts.correction);
  const total = formatGPUAnalysisNumber(facts.featureCount);
  if (facts.featureCount === 0) return `There are no ${noun.plural} to test.`;

  if (facts.hotCount === 0 && facts.coldCount === 0) {
    let demandingNote = '';
    if (facts.permutationCount !== undefined && facts.permutationCount > 0) {
      const level = (100 - facts.confidencePercent) / 100;
      if (level > 0 && level < 1) {
        const attainability = getGPUAnalysisSignificanceAttainability({
          correction: facts.correction,
          testCount: facts.featureCount,
          permutationCount: facts.permutationCount,
          significanceLevel: level
        });
        if (!attainability.attainable) {
          return `No ${noun.singular} could be a significant hot spot or cold spot at ${confidence}${correction}: ${attainability.reason}.`;
        }
        demandingNote =
          attainability.testsNeededAtMinimum !== undefined && attainability.testsNeededAtMinimum > 1
            ? `; ${attainability.reason}`
            : '';
      }
    }
    return facts.featureCount === 1
      ? `The only ${noun.singular} is not a significant hot spot or cold spot at ${confidence}${correction}${demandingNote}.`
      : `None of the ${total} ${noun.plural} is a significant hot spot or cold spot at ${confidence}${correction}${demandingNote}.`;
  }
  if (facts.featureCount === 1) {
    return `The only ${noun.singular} is a significant ${facts.hotCount === 1 ? 'hot' : 'cold'} spot at ${confidence}${correction}.`;
  }
  const hotPart = facts.hotCount === 1 ? 'is a significant hot spot' : 'are significant hot spots';
  const coldPart =
    facts.coldCount === 1 ? 'is a significant cold spot' : 'are significant cold spots';
  return `${formatGPUAnalysisNumber(facts.hotCount)} of ${total} ${noun.plural} ${hotPart} and ${formatGPUAnalysisNumber(facts.coldCount)} ${coldPart} at ${confidence}${correction}.`;
}

function summarizeLocalClusters(facts: GPUAnalysisLocalClusterFacts): string {
  const name = 'summarizeGPUAnalysisResult';
  validateNonNegativeInteger(name, 'featureCount', facts.featureCount);
  validateNonNegativeInteger(name, 'highHighCount', facts.highHighCount);
  validateNonNegativeInteger(name, 'lowLowCount', facts.lowLowCount);
  validateNonNegativeInteger(name, 'highLowCount', facts.highLowCount);
  validateNonNegativeInteger(name, 'lowHighCount', facts.lowHighCount);
  validateOpenUnitInterval(name, 'significanceLevel', facts.significanceLevel);
  if (facts.permutationCount !== undefined) {
    validateNonNegativeInteger(name, 'permutationCount', facts.permutationCount);
  }
  if (facts.analyzedCount !== undefined) {
    validateNonNegativeInteger(name, 'analyzedCount', facts.analyzedCount);
  }
  const significantCount =
    facts.highHighCount + facts.lowLowCount + facts.highLowCount + facts.lowHighCount;
  if (significantCount > facts.featureCount) {
    throw new Error(
      `${name}: the four local cluster counts must sum to at most "featureCount" (${facts.featureCount}); received ${significantCount}.`
    );
  }
  const noun = facts.featureNoun ?? DEFAULT_FEATURE_NOUN;
  const isPseudo = facts.pValueKind !== 'analytic';
  const pValueText = isPseudo ? 'pseudo p-value' : 'p-value';
  const levelText = formatLevel(facts.significanceLevel);
  const correctionText = describeCorrection(facts.correction);
  const level = `a ${pValueText} of ${levelText} or less${correctionText}`;
  const total = formatGPUAnalysisNumber(facts.featureCount);

  const hasNoPValues =
    facts.pValuesComputed === false || (isPseudo && facts.permutationCount === 0);
  if (hasNoPValues) {
    const analyzed = facts.analyzedCount ?? facts.featureCount;
    const attainability = getGPUAnalysisSignificanceAttainability({
      correction: facts.correction,
      testCount: Math.max(analyzed, 1),
      permutationCount: 0,
      significanceLevel: facts.significanceLevel
    });
    const subject =
      analyzed === 0
        ? `no ${noun.singular} was tested`
        : analyzed === 1
          ? `the only ${noun.singular} was not tested`
          : `none of the ${formatGPUAnalysisNumber(analyzed)} ${noun.plural} was tested`;
    return `No p-values were computed because the number of permutations is 0, so ${subject}; at least ${formatGPUAnalysisNumber(attainability.requiredPermutationCount ?? 0)} permutations are needed to reach ${level}.`;
  }
  if (facts.featureCount === 0) return `There are no ${noun.plural} to test.`;
  if (significantCount === 0) {
    let demandingNote = '';
    if (isPseudo && facts.permutationCount !== undefined && facts.permutationCount > 0) {
      const attainability = getGPUAnalysisSignificanceAttainability({
        correction: facts.correction,
        testCount: facts.featureCount,
        permutationCount: facts.permutationCount,
        significanceLevel: facts.significanceLevel
      });
      if (!attainability.attainable) {
        return `No ${noun.singular} could be significant at ${level}: ${attainability.reason}.`;
      }
      demandingNote =
        attainability.testsNeededAtMinimum !== undefined && attainability.testsNeededAtMinimum > 1
          ? `; ${attainability.reason}`
          : '';
    }
    return facts.featureCount === 1
      ? `The only ${noun.singular} is not significant at ${level}${demandingNote}.`
      : `None of the ${total} ${noun.plural} is significant at ${level}${demandingNote}.`;
  }
  const parts: string[] = [];
  const addPart = (count: number, singular: string, plural: string): void => {
    if (count > 0)
      parts.push(`${formatGPUAnalysisNumber(count)} ${count === 1 ? singular : plural}`);
  };
  addPart(facts.highHighCount, 'high-high core', 'high-high cores');
  addPart(facts.lowLowCount, 'low-low core', 'low-low cores');
  addPart(facts.highLowCount, 'high-low outlier', 'high-low outliers');
  addPart(facts.lowHighCount, 'low-high outlier', 'low-high outliers');
  return `${formatGPUAnalysisNumber(significantCount)} of ${total} ${chooseNoun(facts.featureCount, noun)} ${chooseBe(significantCount)} significant at ${level}: ${joinList(parts)}.`;
}

function summarizeClusterCount(facts: GPUAnalysisClusterCountFacts): string {
  const name = 'summarizeGPUAnalysisResult';
  validateNonNegativeInteger(name, 'pointCount', facts.pointCount);
  validateNonNegativeInteger(name, 'clusterCount', facts.clusterCount);
  validateNonNegativeInteger(name, 'noiseCount', facts.noiseCount);
  if (facts.noiseCount > facts.pointCount) {
    throw new Error(
      `${name}: argument "noiseCount" must be at most "pointCount" (${facts.pointCount}); received ${facts.noiseCount}.`
    );
  }
  const noun = facts.pointNoun ?? DEFAULT_POINT_NOUN;
  if (facts.pointCount === 0) return `There are no ${noun.plural} to cluster.`;
  const total = formatGPUAnalysisNumber(facts.pointCount);
  if (facts.clusterCount === 0) {
    if (facts.noiseCount !== facts.pointCount) {
      throw new Error(
        `${name}: with "clusterCount" 0 every point is noise, so "noiseCount" must equal "pointCount" (${facts.pointCount}); received ${facts.noiseCount}.`
      );
    }
    return facts.pointCount === 1
      ? `Found no clusters; the only ${noun.singular} is noise.`
      : `Found no clusters; all ${total} ${noun.plural} are noise.`;
  }
  const largest = facts.largestClusterCount;
  if (largest === undefined) {
    throw new Error(
      `${name}: argument "largestClusterCount" is required when "clusterCount" is above 0.`
    );
  }
  validateNonNegativeInteger(name, 'largestClusterCount', largest);
  if (largest < 1 || largest > facts.pointCount - facts.noiseCount) {
    throw new Error(
      `${name}: argument "largestClusterCount" must be from 1 to the clustered points (${facts.pointCount - facts.noiseCount}); received ${largest}.`
    );
  }
  const clusterText = `${formatGPUAnalysisNumber(facts.clusterCount)} ${facts.clusterCount === 1 ? 'cluster' : 'clusters'}`;
  const noiseText =
    facts.noiseCount === 0
      ? `no ${noun.plural} are noise`
      : `${formatGPUAnalysisNumber(facts.noiseCount)} of ${total} ${chooseNoun(facts.pointCount, noun)} (${formatShare(facts.noiseCount / facts.pointCount)}) ${chooseBe(facts.noiseCount)} noise`;
  const largestText =
    facts.clusterCount === 1
      ? `it holds ${formatShare(largest / facts.pointCount)} of all ${noun.plural}`
      : `the largest cluster holds ${formatShare(largest / facts.pointCount)} of all ${noun.plural}`;
  return `Found ${clusterText}; ${noiseText} and ${largestText}.`;
}

function describeExtremeness(sidedness: GPUAnalysisSidedness): string {
  switch (sidedness) {
    case 'greater':
      return 'at least as large as the observed one';
    case 'less':
      return 'at least as small as the observed one';
    case 'two-sided':
      return 'at least as extreme as the observed one in either direction';
  }
}

function summarizeGlobalStatistic(facts: GPUAnalysisGlobalStatisticFacts): string {
  const name = 'summarizeGPUAnalysisResult';
  validateFiniteNumber(name, 'value', facts.value);
  validateUnitInterval(name, 'pValue', facts.pValue);
  validateNonNegativeInteger(name, 'permutationCount', facts.permutationCount);
  const level = facts.significanceLevel ?? DEFAULT_SIGNIFICANCE_LEVEL;
  validateOpenUnitInterval(name, 'significanceLevel', level);
  const sidedness = facts.sidedness ?? 'two-sided';
  const nullDescription = facts.nullDescription ?? 'a random pattern';
  if (!nullDescription.trim()) {
    throw new Error(`${name}: argument "nullDescription" must not be empty.`);
  }
  const extremeness = describeExtremeness(sidedness);
  const pText =
    facts.permutationCount > 0
      ? `pseudo p-value ${formatPValue(facts.pValue, facts.permutationCount, level)} from ${formatGPUAnalysisNumber(facts.permutationCount)} ${facts.permutationCount === 1 ? 'permutation' : 'permutations'} (the number of permutations with a value ${extremeness}, plus one, divided by ${formatGPUAnalysisNumber(facts.permutationCount)} plus one)`
      : `p-value ${formatPValue(facts.pValue, 0, level)} (the chance under ${nullDescription} of a value ${extremeness})`;
  const levelText = formatLevel(level);
  const reading =
    facts.pValue <= level
      ? `a result this extreme would be rare under ${nullDescription} at the ${levelText} level`
      : `this is consistent with ${nullDescription} at the ${levelText} level`;
  let signText = '';
  if (facts.value > 0 && facts.positiveMeaning) {
    signText = ` (a positive value means ${facts.positiveMeaning})`;
  } else if (facts.value < 0 && facts.negativeMeaning) {
    signText = ` (a negative value means ${facts.negativeMeaning})`;
  }
  return `${facts.statisticName} is ${formatGPUAnalysisMeasuredValue(facts.value)}${signText}, with ${pText}; ${reading}.`;
}

/** R squared with two decimals; a value below 1 is never rounded up to `1.00`. */
function formatRSquared(value: number): string {
  if (value < 1 && value >= 0.995) return 'just under 1';
  return formatGPUAnalysisNumber(value, 2);
}

function summarizeRegressionFit(facts: GPUAnalysisRegressionFitFacts): string {
  const name = 'summarizeGPUAnalysisResult';
  validateFiniteNumber(name, 'rSquared', facts.rSquared);
  validateNonNegativeInteger(name, 'observationCount', facts.observationCount);
  if (facts.rSquared > 1) {
    throw new Error(`${name}: argument "rSquared" must be at most 1; received ${facts.rSquared}.`);
  }
  const noun = facts.observationNoun ?? DEFAULT_OBSERVATION_NOUN;
  if (facts.observationCount === 0) {
    return `No ${noun.plural} were fitted, so R² is not defined.`;
  }
  let adjustedText = '';
  if (facts.adjustedRSquared !== undefined) {
    validateFiniteNumber(name, 'adjustedRSquared', facts.adjustedRSquared);
    adjustedText = ` (adjusted R² ${formatRSquared(facts.adjustedRSquared)})`;
  }
  return `R² is ${formatRSquared(facts.rSquared)} on ${formatGPUAnalysisNumber(facts.observationCount)} ${chooseNoun(facts.observationCount, noun)}${adjustedText}, a measure of how closely the model fits the data rather than of cause.`;
}

function summarizeValueRange(facts: GPUAnalysisValueRangeFacts): string {
  const name = 'summarizeGPUAnalysisResult';
  validateNonNegativeInteger(name, 'valueCount', facts.valueCount);
  if (facts.valueCount === 0) return 'There are no values to summarize.';
  validateFiniteNumber(name, 'minimum', facts.minimum);
  validateFiniteNumber(name, 'maximum', facts.maximum);
  if (facts.minimum > facts.maximum) {
    throw new Error(
      `${name}: argument "minimum" must be at most "maximum"; received ${facts.minimum} and ${facts.maximum}.`
    );
  }
  const precision =
    facts.significantDigits ?? getGPUAnalysisColumnPrecision([facts.minimum, facts.maximum]);
  const minimumText = formatGPUAnalysisMeasuredValue(facts.minimum, precision);
  const maximumText = formatGPUAnalysisMeasuredValue(facts.maximum, precision);
  if (facts.valueCount === 1 && facts.minimum !== facts.maximum) {
    throw new Error(
      `${name}: with "valueCount" 1 the "minimum" and "maximum" must be equal; received ${facts.minimum} and ${facts.maximum}.`
    );
  }
  const label = facts.valueLabel;
  const count = formatGPUAnalysisNumber(facts.valueCount);
  const valueWord = facts.valueCount === 1 ? 'value' : 'values';
  if (facts.minimum === facts.maximum) {
    if (label === undefined) {
      return facts.valueCount === 1
        ? `The only value is ${minimumText}.`
        : `All ${count} values are ${minimumText}.`;
    }
    return facts.valueCount === 1
      ? `${label} is ${minimumText} in the only value.`
      : `${label} is ${minimumText} in all ${count} values.`;
  }
  return label === undefined
    ? `The ${count} values range from ${minimumText} to ${maximumText}.`
    : `${label} ranges from ${minimumText} to ${maximumText} across ${count} ${valueWord}.`;
}

function summarizeCountPerArea(facts: GPUAnalysisCountPerAreaFacts): string {
  const name = 'summarizeGPUAnalysisResult';
  validateNonNegativeInteger(name, 'totalCount', facts.totalCount);
  validateNonNegativeInteger(name, 'areaCount', facts.areaCount);
  const item = facts.itemNoun ?? DEFAULT_ITEM_NOUN;
  const area = facts.areaNoun ?? DEFAULT_AREA_NOUN;
  if (facts.areaCount === 0) return `There are no ${area.plural} to count in.`;
  if (facts.maximumCount !== undefined) {
    validateNonNegativeInteger(name, 'maximumCount', facts.maximumCount);
    if (facts.maximumCount > facts.totalCount) {
      throw new Error(
        `${name}: argument "maximumCount" must be at most "totalCount" (${facts.totalCount}); received ${facts.maximumCount}.`
      );
    }
  }
  const total = formatGPUAnalysisNumber(facts.totalCount);
  const areas = formatGPUAnalysisNumber(facts.areaCount);
  if (facts.totalCount === 0) {
    return facts.areaCount === 1
      ? `No ${item.plural} fall in the only ${area.singular}.`
      : `No ${item.plural} fall in any of the ${areas} ${area.plural}.`;
  }
  if (facts.areaCount === 1) {
    return `${total} ${chooseNoun(facts.totalCount, item)} in 1 ${area.singular}.`;
  }
  const mean = formatGPUAnalysisMeasuredValue(facts.totalCount / facts.areaCount);
  const maximumText =
    facts.maximumCount === undefined
      ? ''
      : `, up to ${formatGPUAnalysisNumber(facts.maximumCount)} in one ${area.singular}`;
  return `${total} ${chooseNoun(facts.totalCount, item)} across ${areas} ${area.plural}: ${mean} per ${area.singular} on average${maximumText}.`;
}

function summarizeRowCount(facts: GPUAnalysisRowCountFacts): string {
  validateNonNegativeInteger('summarizeGPUAnalysisResult', 'rowCount', facts.rowCount);
  const noun = facts.rowNoun ?? DEFAULT_ROW_NOUN;
  if (facts.rowCount === 0) return `The result has no ${noun.plural}.`;
  return `The result has ${formatGPUAnalysisNumber(facts.rowCount)} ${chooseNoun(facts.rowCount, noun)}.`;
}

/**
 * One plain sentence for a result, built only from the facts given. Throws when the facts are
 * inconsistent (counts that exceed their total, a p-value outside 0 to 1), so a sentence never
 * states a number the data cannot hold.
 */
export function summarizeGPUAnalysisResult<Template extends GPUAnalysisSummaryTemplate>(
  template: Template,
  facts: GPUAnalysisSummaryFactsByTemplate[Template]
): string {
  switch (template) {
    case 'significant-hot-cold-count':
      return summarizeSignificantHotCold(facts as GPUAnalysisSignificantHotColdFacts);
    case 'local-cluster-count':
      return summarizeLocalClusters(facts as GPUAnalysisLocalClusterFacts);
    case 'cluster-count-and-noise':
      return summarizeClusterCount(facts as GPUAnalysisClusterCountFacts);
    case 'global-statistic':
      return summarizeGlobalStatistic(facts as GPUAnalysisGlobalStatisticFacts);
    case 'regression-fit':
      return summarizeRegressionFit(facts as GPUAnalysisRegressionFitFacts);
    case 'value-range':
      return summarizeValueRange(facts as GPUAnalysisValueRangeFacts);
    case 'count-per-area':
      return summarizeCountPerArea(facts as GPUAnalysisCountPerAreaFacts);
    case 'row-count':
      return summarizeRowCount(facts as GPUAnalysisRowCountFacts);
    default:
      throw new Error(
        `summarizeGPUAnalysisResult: argument "template" must be one of the summary templates; received ${String(template)}.`
      );
  }
}

// ---------------------------------------------------------------------------------------------
// Cautions
// ---------------------------------------------------------------------------------------------

/** Heading and body of a caution, for display next to a result. */
export type GPUAnalysisCautionText = {title: string; text: string};

/** Facts that sharpen the text of a caution; every member is optional. */
export type GPUAnalysisCautionTextFacts = {
  /** Level of the uncorrected tests (`multiple-testing-uncorrected`), for example 0.05 or 0.1. */
  significanceLevel?: number;
  /** For `significance-not-attainable`: the run's correction, test count and permutations. */
  correction?: GPUAnalysisMultipleTestingCorrection;
  testCount?: number;
  permutationCount?: number;
  /**
   * For `rows-without-neighbors`: true when the producer leaves such rows out of the classes,
   * false when it classifies them anyway (from their own value alone).
   */
  rowsExcluded?: boolean;
  /** For `small-denominators`: areas below the expected-count limit, and the limit. */
  smallDenominatorAreaCount?: number;
  expectedCountLimit?: number;
  /** For `poor-model-fit`: a short reason from the model's own diagnostics. */
  reason?: string;
};

const CAUTION_TEXTS: Readonly<Record<GPUAnalysisCaution, GPUAnalysisCautionText>> = {
  'fewer-than-thirty-features': {
    title: 'Fewer than 30 features',
    text: 'Esri gives 30 features as the minimum for reliable results in its hot spot analysis (Getis-Ord Gi*) and cluster and outlier analysis (local Moran’s I). It is a rule of thumb, and it is applied here to other tests by analogy. The reason usually given is that these tests rest on a normal approximation, which needs enough features and neighbours. Treat the result as a rough guide.'
  },
  'counts-where-rates-expected': {
    title: 'Counts where a rate was expected',
    text: 'The analysed column is a count or an amount, where this method is meant for rates, shares or densities. A count grows with the size or population of each area, so high values mostly show where people or activity are, not where risk is. Divide by population or area first if a rate is what you mean.'
  },
  'pseudo-p-values': {
    title: 'Pseudo p-values',
    text: 'These p-values are estimated from random permutations of the values, so they vary with the random seed, and they cannot be smaller than 1 / (permutations + 1). A level below that cannot be reached however strong the pattern is.'
  },
  'multiple-testing-uncorrected': {
    title: 'No correction for multiple testing',
    text: 'Locations were tested separately. For calibrated marginal tests, the expected share of false positives among locations with no pattern is the chosen level, even when tests depend on each other. Dependence changes the joint behaviour, including the chance of at least one false positive and how false positives cluster. Where the tool offers one, use a correction such as false discovery rate, or a stricter level, and read significant locations as cores of clusters, not whole clusters.'
  },
  'rows-without-neighbors': {
    title: 'Rows without neighbours',
    text: 'Some rows have no neighbours under the chosen spatial weights. Depending on the method they are either left out of the classes or classified from their own value alone, which says nothing about clustering. Widen the distance band or use more neighbours if they matter.'
  },
  'sampled-statistics': {
    title: 'Defaults computed on a sample',
    text: 'The statistics that suggested the default settings were computed on a sample of the rows, so the defaults can differ a little from those of the full data. Check them before relying on the result.'
  },
  'guessed-coordinates': {
    title: 'Coordinate system was guessed',
    text: 'The coordinate system was inferred from the values, not declared by the data. If the guess is wrong every distance and area is wrong, so confirm it before trusting distances or neighbours.'
  },
  'default-from-fallback': {
    title: 'Default is a fixed fallback',
    text: 'A default setting fell back to a fixed constant because the data did not support a derived value, so it was not derived from this data. Check it and change it if it does not suit your data.'
  },
  'exploratory-parameter-search': {
    title: 'Many settings tried',
    text: 'Comparing many parameter settings and keeping the most striking one raises the chance of a spurious finding. False discovery rate correction within one map does not control this, because it counts only the tests inside that map, not the settings you tried.'
  },
  'significance-not-attainable': {
    title: 'Significance cannot be reached',
    text: 'With this permutation resolution no feature can be declared significant at the chosen level. A result of none significant is then not a finding about the data. Increase the number of permutations.'
  },
  'small-denominators': {
    title: 'Small expected counts',
    text: 'Ratios of observed to expected counts are unstable where the expected count is small: one more or one fewer event changes the ratio a lot, so extreme ratios there are often noise. Treat extreme values in areas with small expected counts with care, or smooth the rates first.'
  },
  'poor-model-fit': {
    title: 'The model fits poorly',
    text: 'The fitted model’s own diagnostics say it does not describe the data well, so its coefficients, predictions and any map of its residuals should not be relied on.'
  },
  'row-order-dependent-inference': {
    title: 'P-values depend on row order',
    text: 'For this operation the permutation p-values depend on the order of the input rows, so reordering the same rows can change them and which locations are significant. Treat borderline locations with care and do not compare p-values between differently ordered inputs.'
  }
};

function capitalizeFirst(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/**
 * What a reader should keep in mind about a caution, and why, in one or two sentences. Without
 * `facts` the text is generic and names no number; with them it states the figures of the run:
 * the level of the uncorrected tests, why significance cannot be reached, whether rows without
 * neighbours were excluded, how many areas have small expected counts, or why a model fits poorly.
 */
export function getGPUAnalysisCautionText(
  caution: GPUAnalysisCaution,
  facts?: GPUAnalysisCautionTextFacts
): GPUAnalysisCautionText {
  const base = CAUTION_TEXTS[caution];
  if (!base) {
    throw new Error(
      `getGPUAnalysisCautionText: argument "caution" must be one of the caution ids; received ${String(caution)}.`
    );
  }
  let text = base.text;
  const name = 'getGPUAnalysisCautionText';
  if (facts?.significanceLevel !== undefined) {
    validateOpenUnitInterval(name, 'significanceLevel', facts.significanceLevel);
  }
  switch (caution) {
    case 'multiple-testing-uncorrected':
      if (facts?.significanceLevel !== undefined) {
        const level = formatGPUAnalysisMeasuredValue(facts.significanceLevel);
        const share = formatGPUAnalysisMeasuredValue(facts.significanceLevel * 100);
        text = `Locations were tested separately. For calibrated marginal tests at the ${level} level, about ${share} in 100 locations with no pattern would be marked by chance on average, even when tests depend on each other. Dependence changes the joint behaviour, including the chance of at least one false positive and how false positives cluster. Where the tool offers one, use a correction such as false discovery rate, or a stricter level, and read significant locations as cores of clusters, not whole clusters.`;
      }
      break;
    case 'rows-without-neighbors':
      if (facts?.rowsExcluded === true) {
        text =
          'Some rows have no neighbours under the chosen spatial weights, so they get no statistic and are left out of the classes. Widen the distance band or use more neighbours if they matter.';
      } else if (facts?.rowsExcluded === false) {
        text =
          'Some rows have no neighbours under the chosen spatial weights. They are still classified, from their own value alone, which says nothing about clustering. Widen the distance band or use more neighbours if they matter.';
      }
      break;
    case 'significance-not-attainable':
      if (
        facts?.correction !== undefined &&
        facts.testCount !== undefined &&
        (facts.testCount > 0 || facts.correction === 'none') &&
        facts.permutationCount !== undefined &&
        facts.significanceLevel !== undefined
      ) {
        const attainability = getGPUAnalysisSignificanceAttainability({
          correction: facts.correction,
          testCount: Math.max(1, facts.testCount),
          permutationCount: facts.permutationCount,
          significanceLevel: facts.significanceLevel
        });
        text = `${capitalizeFirst(attainability.reason)}. A result of none significant is then not a finding about the data.`;
      }
      break;
    case 'small-denominators':
      if (facts?.smallDenominatorAreaCount !== undefined) {
        validateNonNegativeInteger(
          name,
          'smallDenominatorAreaCount',
          facts.smallDenominatorAreaCount
        );
        const limit =
          facts.expectedCountLimit === undefined
            ? 'the stated limit'
            : formatGPUAnalysisMeasuredValue(facts.expectedCountLimit);
        const count = facts.smallDenominatorAreaCount;
        text = `${formatGPUAnalysisNumber(count)} ${count === 1 ? 'area has an expected count' : 'areas have an expected count'} below ${limit}. ${text}`;
      }
      break;
    case 'poor-model-fit':
      if (facts?.reason?.trim()) text = `${text} Reason given: ${facts.reason.trim()}.`;
      break;
    default:
      break;
  }
  return {title: base.title, text};
}

/**
 * Smallest pseudo p-value a run of `permutationCount` permutations can report: with the observed
 * value counted among the permuted ones, `1 / (permutationCount + 1)`.
 */
export function getGPUAnalysisSmallestPseudoPValue(permutationCount: number): number {
  validateNonNegativeInteger(
    'getGPUAnalysisSmallestPseudoPValue',
    'permutationCount',
    permutationCount
  );
  return 1 / (permutationCount + 1);
}

/**
 * Whether `permutationCount` permutations can report a pseudo p-value at or below
 * `significanceLevel`. 999 permutations resolve 0.001 (the smallest value is exactly 0.001) but not
 * 0.0005. When this is false no location can be marked significant at that level. For a run with a
 * correction use {@link getGPUAnalysisSignificanceAttainability}.
 */
export function canResolveGPUAnalysisSignificanceLevel(
  permutationCount: number,
  significanceLevel: number
): boolean {
  validateOpenUnitInterval(
    'canResolveGPUAnalysisSignificanceLevel',
    'significanceLevel',
    significanceLevel
  );
  return (
    getGPUAnalysisSmallestPseudoPValue(permutationCount) <=
    significanceLevel * (1 + RESOLUTION_RELATIVE_TOLERANCE)
  );
}

/**
 * A sentence for the case where the permutation count cannot resolve the significance level, with
 * the count that would; `undefined` when it can.
 */
export function describeGPUAnalysisPermutationResolution(
  permutationCount: number,
  significanceLevel: number
): string | undefined {
  if (canResolveGPUAnalysisSignificanceLevel(permutationCount, significanceLevel)) return undefined;
  const requiredPermutations = Math.ceil(1 / significanceLevel - RESOLUTION_RELATIVE_TOLERANCE) - 1;
  const smallest = getGPUAnalysisSmallestPseudoPValue(permutationCount);
  return `${formatGPUAnalysisNumber(permutationCount)} permutations cannot give a pseudo p-value below ${formatGPUAnalysisNumber(smallest, getPseudoPValueDigits(permutationCount), true)}, so no location can be significant at ${formatGPUAnalysisNumber(significanceLevel, 6, true)}; at least ${formatGPUAnalysisNumber(requiredPermutations)} permutations are needed.`;
}

/** Facts of one run that decide which cautions apply. A member that is absent is unknown. */
export type GPUAnalysisCautionFacts = {
  /**
   * Features analysed (rows that entered the analysis), not the count that was tested; below 30
   * raises `fewer-than-thirty-features`.
   */
  featureCount?: number;
  /** False when many locations were tested without correction; undefined is not assumed. */
  corrected?: boolean;
  /** The correction of the run; `none` raises `multiple-testing-uncorrected` like `corrected: false`. */
  correction?: GPUAnalysisMultipleTestingCorrection;
  /** Tests the correction is applied over; needed with `correction` to judge attainability. */
  testCount?: number;
  /** Permutations behind the p-values; above 0 raises `pseudo-p-values`. */
  permutationCount?: number;
  /**
   * True when the producer has no p-values but permutation p-values; with `permutationCount` 0
   * that raises `significance-not-attainable` (no p-value is computed, so nothing can be tested).
   */
  permutationsRequired?: boolean;
  /** Cut-off the permutation count is checked against; defaults to 0.05. */
  significanceLevel?: number;
  /** Rows that got no statistic for want of a neighbour. */
  rowsWithoutNeighbors?: number;
  sampledStatistics?: boolean;
  guessedCoordinates?: boolean;
  usedFallbackDefault?: boolean;
  /**
   * The measurement level of the analysed column does not match what the method is meant for: a
   * count or an amount where rates, shares or densities are meant. It says nothing about what the
   * rows stand for (events, objects or surface samples), which is a different mismatch for which
   * the contract has no caution.
   */
  measurementMismatch?: boolean;
  /** Several parameter settings were tried and compared. */
  parameterSearch?: boolean;
  /** Areas whose expected count is below the stated limit; above 0 raises `small-denominators`. */
  smallDenominatorAreaCount?: number;
  /** The model's own diagnostics say it fits poorly; a short reason, non-empty, raises `poor-model-fit`. */
  poorModelFitReason?: string;
  /** The producer's permutation p-values depend on the order of the input rows. */
  rowOrderDependentInference?: boolean;
};

/** Number of features below which Gi* results are unreliable (Esri, How Hot Spot Analysis works). */
const MINIMUM_RELIABLE_FEATURE_COUNT = 30;

/**
 * The cautions that follow from the facts of a run, in the order of `GPU_ANALYSIS_CAUTIONS`.
 * Cautions that always apply to an output belong in its interpretation spec, not here.
 *
 * `significance-not-attainable` is raised when the run's permutations and correction make it
 * impossible for any test to be significant (see
 * {@link getGPUAnalysisSignificanceAttainability}); it needs `permutationCount` above 0 with
 * `correction` (or `corrected: false`, read as `none`) and, for a correction, `testCount`; or
 * `permutationsRequired` with `permutationCount` 0. `significanceLevel` defaults to 0.05.
 */
export function deriveGPUAnalysisCautions(
  facts: GPUAnalysisCautionFacts
): readonly GPUAnalysisCaution[] {
  const name = 'deriveGPUAnalysisCautions';
  if (facts.featureCount !== undefined)
    validateNonNegativeInteger(name, 'featureCount', facts.featureCount);
  if (facts.testCount !== undefined) validateNonNegativeInteger(name, 'testCount', facts.testCount);
  if (facts.permutationCount !== undefined) {
    validateNonNegativeInteger(name, 'permutationCount', facts.permutationCount);
  }
  if (facts.rowsWithoutNeighbors !== undefined) {
    validateNonNegativeInteger(name, 'rowsWithoutNeighbors', facts.rowsWithoutNeighbors);
  }
  if (facts.smallDenominatorAreaCount !== undefined) {
    validateNonNegativeInteger(name, 'smallDenominatorAreaCount', facts.smallDenominatorAreaCount);
  }
  if (facts.significanceLevel !== undefined) {
    validateOpenUnitInterval(name, 'significanceLevel', facts.significanceLevel);
  }

  const level = facts.significanceLevel ?? DEFAULT_SIGNIFICANCE_LEVEL;
  const correction: GPUAnalysisMultipleTestingCorrection | undefined =
    facts.correction ?? (facts.corrected === false ? 'none' : undefined);
  let isSignificanceUnattainable = false;
  if (facts.permutationCount === 0 && facts.permutationsRequired === true) {
    isSignificanceUnattainable = true;
  } else if (
    facts.permutationCount !== undefined &&
    facts.permutationCount > 0 &&
    correction !== undefined &&
    (correction === 'none' || (facts.testCount !== undefined && facts.testCount >= 1))
  ) {
    isSignificanceUnattainable = !getGPUAnalysisSignificanceAttainability({
      correction,
      testCount: facts.testCount ?? 1,
      permutationCount: facts.permutationCount,
      significanceLevel: level
    }).attainable;
  }

  const cautions: GPUAnalysisCaution[] = [];
  if (facts.featureCount !== undefined && facts.featureCount < MINIMUM_RELIABLE_FEATURE_COUNT) {
    cautions.push('fewer-than-thirty-features');
  }
  if (facts.measurementMismatch) cautions.push('counts-where-rates-expected');
  if (facts.permutationCount !== undefined && facts.permutationCount > 0) {
    cautions.push('pseudo-p-values');
  }
  if (
    (facts.corrected === false || facts.correction === 'none') &&
    facts.testCount !== 0 &&
    !(facts.permutationsRequired && facts.permutationCount === 0)
  ) {
    cautions.push('multiple-testing-uncorrected');
  }
  if (facts.rowsWithoutNeighbors !== undefined && facts.rowsWithoutNeighbors > 0) {
    cautions.push('rows-without-neighbors');
  }
  if (facts.sampledStatistics) cautions.push('sampled-statistics');
  if (facts.guessedCoordinates) cautions.push('guessed-coordinates');
  if (facts.usedFallbackDefault) cautions.push('default-from-fallback');
  if (facts.parameterSearch) cautions.push('exploratory-parameter-search');
  if (isSignificanceUnattainable) cautions.push('significance-not-attainable');
  if (facts.smallDenominatorAreaCount !== undefined && facts.smallDenominatorAreaCount > 0) {
    cautions.push('small-denominators');
  }
  if (facts.poorModelFitReason !== undefined && facts.poorModelFitReason.trim() !== '') {
    cautions.push('poor-model-fit');
  }
  if (facts.rowOrderDependentInference) cautions.push('row-order-dependent-inference');
  return cautions;
}
