// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Cooperative trend phase; emitted text and barrier order are shared with the contributor. @internal */
export function getCooperativeMannKendallBody(options: {
  sliceCount: number;
  usesAdjustment: boolean;
  finalSlice: 'require' | 'last-valid';
  hasWeightsLoss: boolean;
}): string {
  const {sliceCount, usesAdjustment, finalSlice, hasWeightsLoss} = options;
  return `let cell = index / MANN_KENDALL_LANES;
  let lane = localInvocationIndex;
  let isActiveCell = cell < CELL_COUNT;
  let base = cell * SLICE_COUNT;
  for (var slice = lane; slice < SLICE_COUNT; slice += MANN_KENDALL_LANES) {
    var value = getQuietNaN(index);
    if (isActiveCell) {
      value = giZScores[giZScoresOffset + base + slice];
    }
    sharedSeries[slice] = value;
    tieCandidates[slice] = 0u;
  }
  for (var word = lane; word < ${Math.ceil(sliceCount / 32)}u; word += MANN_KENDALL_LANES) {
    atomicStore(&equalBeforeBits[word], 0u);
  }
  workgroupBarrier();
  var laneStatistic = 0;
  for (var first = lane; first < SLICE_COUNT; first += MANN_KENDALL_LANES) {
    let firstValue = sharedSeries[first];
    if (isFiniteFloat(firstValue)) {
      var groupSize = 1u;
      for (var second = first + 1u; second < SLICE_COUNT; second++) {
        let secondValue = sharedSeries[second];
        if (isFiniteFloat(secondValue)) {
          laneStatistic += select(select(0, -1, secondValue < firstValue), 1, secondValue > firstValue);
          if (secondValue == firstValue) {
            groupSize++;
            atomicOr(&equalBeforeBits[second / 32u], 1u << (second % 32u));
          }
        }
      }
      if (groupSize > 1u) {
        tieCandidates[first] = groupSize * (groupSize - 1u) * (2u * groupSize + 5u);
      }
    }
  }
  partialStatistics[lane] = laneStatistic;
  workgroupBarrier();
  var laneTieTerm = 0u;
  for (var first = lane; first < SLICE_COUNT; first += MANN_KENDALL_LANES) {
    let hasEqualBefore = (atomicLoad(&equalBeforeBits[first / 32u]) & (1u << (first % 32u))) != 0u;
    if (!hasEqualBefore) {
      laneTieTerm += tieCandidates[first];
    }
  }
  partialTieTerms[lane] = laneTieTerm;
  workgroupBarrier();
  for (var stride = MANN_KENDALL_LANES / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      partialStatistics[lane] += partialStatistics[lane + stride];
      partialTieTerms[lane] += partialTieTerms[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u && isActiveCell) {
  let statistic = partialStatistics[0];
  let tieTerm = partialTieTerms[0];
  ${getSliceCountingWGSL({
    valueExpression: 'sharedSeries[slice]',
    adjustedExpression: 'adjustedPValues[adjustedPValuesOffset + base + slice]',
    usesAdjustment,
    collectSeries: false
  })}
  ${getClassificationWGSL({cellName: 'cell', finalSlice, hasWeightsLoss})}
  }`;
}

/**
 * WGSL `getSliceState`: 1 for a hot bin, 2 for a cold bin, 0 otherwise. Without a correction it
 * compares z with the critical z (the pre-correction behavior); with one, a bin is hot or cold
 * when its adjusted p-value is at most the significance level, with the sign of z.
 */
export function getSliceStateWGSL(usesAdjustment: boolean): string {
  return usesAdjustment
    ? `fn getSliceState(z: f32, adjustedP: f32, level: f32) -> u32 {
  if (adjustedP <= level) {
    if (z > 0.0) {
      return 1u;
    }
    if (z < 0.0) {
      return 2u;
    }
  }
  return 0u;
}`
    : `fn getSliceState(z: f32, criticalZ: f32) -> u32 {
  if (z >= criticalZ) {
    return 1u;
  }
  if (z <= -criticalZ) {
    return 2u;
  }
  return 0u;
}`;
}

/**
 * WGSL pass over one cell's slices that declares `valid`, `hot`, `cold`, `trailingHot`,
 * `trailingCold`, `finalState` and `lastValid` (the last finite slice, `0xffffffff` for none), and
 * `series` when `collectSeries` is set. Reads the parameters it needs itself.
 */
export function getSliceCountingWGSL(options: {
  valueExpression: string;
  adjustedExpression: string;
  usesAdjustment: boolean;
  collectSeries: boolean;
}): string {
  const state = options.usesAdjustment
    ? `getSliceState(value, ${options.adjustedExpression}, significanceLevel)`
    : 'getSliceState(value, criticalZ)';
  return /* wgsl */ `let criticalZ = parameters[parametersOffset + 2u];
  let significanceLevel = parameters[parametersOffset + 6u];
  var valid = 0u;
  var hot = 0u;
  var cold = 0u;
  var trailingHot = 0u;
  var trailingCold = 0u;
  var finalState = 0u;
  var lastValid = 0xffffffffu;
  for (var slice = 0u; slice < SLICE_COUNT; slice++) {
    let value = ${options.valueExpression};
    if (isFiniteFloat(value)) {
      ${options.collectSeries ? 'series[valid] = value;' : ''}
      valid++;
      lastValid = slice;
      let state = ${state};
      if (state == 1u) {
        hot++;
        trailingHot++;
        trailingCold = 0u;
        finalState = 1u;
      } else if (state == 2u) {
        cold++;
        trailingCold++;
        trailingHot = 0u;
        finalState = 2u;
      } else {
        trailingHot = 0u;
        trailingCold = 0u;
        finalState = 0u;
      }
    }
  }`;
}

/**
 * WGSL tail shared by both Mann-Kendall paths: from `statistic`, `tieTerm` and the slice counts it
 * writes the trend outputs, the slice counts and the category. See the GPUEmergingHotSpots class contract for
 * the rules; `UNDETERMINED` (17) replaces Intensifying/Persistent/Diminishing for cells with fewer
 * than `minimumTrendSliceCount` finite slices, and every pattern of a cell whose last slice is
 * missing when `finalSlice` is `'require'`.
 */
function getClassificationWGSL(options: {
  cellName: string;
  finalSlice: 'require' | 'last-valid';
  applyVarianceCorrection?: boolean;
  hasWeightsLoss?: boolean;
}): string {
  const {cellName} = options;
  return /* wgsl */ `let trendLevel = parameters[parametersOffset + 3u];
  let fraction = parameters[parametersOffset + 4u];
  let minimumTrendSliceCount = parameters[parametersOffset + 5u];
  let isLegacySporadic = parameters[parametersOffset + 7u] != 0.0;
  var numerator = 0u;
  if (valid >= 2u) {
    numerator = valid * (valid - 1u) * (2u * valid + 5u) - tieTerm;
  }
  var trendZScore = 0.0;
  var trendPValue = 1.0;
  if (numerator > 0u) {
    let deviation = sqrt(f32(numerator) / 18.0${options.applyVarianceCorrection ? ' * varianceCorrection' : ''});
    if (statistic > 0) {
      trendZScore = f32(statistic - 1) / deviation;
    } else if (statistic < 0) {
      trendZScore = f32(statistic + 1) / deviation;
    }
    trendPValue = getTwoSidedPValue(trendZScore);
  }
  // Too few finite slices for a Mann-Kendall decision: no trend z or p, and no trend category.
  let isTrendKnown = f32(valid) >= minimumTrendSliceCount;
  if (!isTrendKnown${options.hasWeightsLoss ? ' || weightsLoss[weightsLossOffset] != 0u' : ''}) {
    trendZScore = getQuietNaN(${cellName});
    trendPValue = getQuietNaN(${cellName});
  }
  trendZ[trendZOffset + ${cellName}] = trendZScore;
  trendP[trendPOffset + ${cellName}] = trendPValue;
  trendS[trendSOffset + ${cellName}] = statistic;
  let threshold = fraction * f32(valid);
  let hotPersistent = f32(hot) >= threshold;
  let coldPersistent = f32(cold) >= threshold;
  let trendSignificant = trendPValue <= trendLevel;
  let trendUp = trendSignificant && trendZScore > 0.0;
  let trendDown = trendSignificant && trendZScore < 0.0;
  var result = 0u;
  if (valid > 0u) {
    if (finalState == 1u) {
      if (hot == 1u) {
        result = 1u;
      } else if (trailingHot >= 2u && trailingHot == hot && !hotPersistent) {
        result = 2u;
      } else if (hotPersistent) {
        result = select(17u, select(select(4u, 5u, trendDown), 3u, trendUp), isTrendKnown);
      } else {
        result = select(7u, 6u, cold == 0u);
      }
    } else if (finalState == 2u) {
      if (cold == 1u) {
        result = 9u;
      } else if (trailingCold >= 2u && trailingCold == cold && !coldPersistent) {
        result = 10u;
      } else if (coldPersistent) {
        result = select(17u, select(select(12u, 13u, trendUp), 11u, trendDown), isTrendKnown);
      } else {
        result = select(15u, 14u, hot == 0u);
      }
    } else if (hot > 0u && hotPersistent) {
      result = 8u;
    } else if (cold > 0u && coldPersistent) {
      result = 16u;
    } else if (isLegacySporadic && hot > 0u && cold == 0u) {
      result = 6u;
    } else if (isLegacySporadic && cold > 0u && hot == 0u) {
      result = 14u;
    }
    ${
      options.finalSlice === 'require'
        ? `// A missing final slice makes the pattern undecidable instead of silently shifting it.
    if (lastValid != SLICE_COUNT - 1u) {
      result = 17u;
    }`
        : ''
    }
  }
  ${options.hasWeightsLoss ? 'if (weightsLoss[weightsLossOffset] != 0u) { result = 17u; }\n  ' : ''}category[categoryOffset + ${cellName}] = result;`;
}

/** Serial exact pass retained for short series where cooperative setup would dominate. */
export function getSerialMannKendallBody(options: {
  sliceCount: number;
  hasWeightsLoss?: boolean;
  usesAdjustment: boolean;
  finalSlice: 'require' | 'last-valid';
  trendCorrection: 'none' | 'hamed-rao' | 'yue-wang';
  maximumLag: number;
  criticalZ: number;
}): string {
  const {sliceCount, trendCorrection} = options;
  const applyVarianceCorrection = trendCorrection !== 'none';
  return /* wgsl */ `let base = index * SLICE_COUNT;
  var series: array<f32, SLICE_COUNT>;
  var hasEqualBefore: array<u32, ${Math.ceil(sliceCount / 32)}>;
  ${getSliceCountingWGSL({
    valueExpression: 'giZScores[giZScoresOffset + base + slice]',
    adjustedExpression: 'adjustedPValues[adjustedPValuesOffset + base + slice]',
    usesAdjustment: options.usesAdjustment,
    collectSeries: true
  })}
  var statistic = 0;
  var tieTerm = 0u;
  for (var first = 0u; first < valid; first++) {
    let firstValue = series[first];
    var groupSize = 1u;
    for (var second = first + 1u; second < valid; second++) {
      let secondValue = series[second];
      statistic += select(select(0, -1, secondValue < firstValue), 1, secondValue > firstValue);
      if (secondValue == firstValue) {
        groupSize++;
        hasEqualBefore[second / 32u] |= 1u << (second % 32u);
      }
    }
    let isFirstMember = (hasEqualBefore[first / 32u] & (1u << (first % 32u))) == 0u;
    if (isFirstMember && groupSize > 1u) {
      tieTerm += groupSize * (groupSize - 1u) * (2u * groupSize + 5u);
    }
  }
  ${applyVarianceCorrection ? getVarianceCorrectionWGSL(sliceCount, trendCorrection, options.maximumLag, options.criticalZ) : ''}
  ${getClassificationWGSL({
    cellName: 'index',
    finalSlice: options.finalSlice,
    applyVarianceCorrection,
    hasWeightsLoss: options.hasWeightsLoss
  })}`;
}

/**
 * WGSL statements declaring `varianceCorrection` from `series` and `valid`: the factor of
 * `Var(S)` of the modified Mann-Kendall tests over the Sen-detrended finite series, with `n` the
 * finite slice count, `r_k` the lag autocorrelation with denominator `n` over the series
 * deviations, positions being positions in the finite series.
 * Hamed-Rao: `1 + 2 / (n (n - 1) (n - 2)) * sum (n - k)(n - k - 1)(n - k - 2) r_k` of the ranks,
 * lags with `|r_k| > criticalZ / sqrt(n)` only. Yue-Wang: `1 + 2 * sum (1 - k / n) r_k` of the
 * detrended values.
 */
function getVarianceCorrectionWGSL(
  sliceCount: number,
  trendCorrection: 'hamed-rao' | 'yue-wang',
  maximumLag: number,
  criticalZ: number
): string {
  const isHamedRao = trendCorrection === 'hamed-rao';
  const seriesWGSL = isHamedRao
    ? `  var ranks: array<f32, ${sliceCount}>;
    for (var first = 0u; first < valid; first++) {
      var lower = 0u;
      var equal = 0u;
      for (var second = 0u; second < valid; second++) {
        if (detrended[second] < detrended[first]) {
          lower++;
        } else if (detrended[second] == detrended[first]) {
          equal++;
        }
      }
      ranks[first] = f32(lower) + 0.5 * f32(equal + 1u);
    }`
    : '';
  const seriesName = isHamedRao ? 'ranks' : 'detrended';
  const lagSumWGSL = isHamedRao
    ? `  let lagFilter = ${getWGSLFloatLiteral(criticalZ)} / sqrt(lengthValue);
      if (abs(correlation) > lagFilter) {
        let gap = lengthValue - f32(lag);
        lagSum += gap * (gap - 1.0) * (gap - 2.0) * correlation;
      }`
    : `  lagSum += (1.0 - f32(lag) / lengthValue) * correlation;`;
  const scale = isHamedRao
    ? '2.0 / (lengthValue * (lengthValue - 1.0) * (lengthValue - 2.0))'
    : '2.0';
  return /* wgsl */ `var varianceCorrection = 1.0;
  if (valid >= 3u) {
    let lengthValue = f32(valid);
    let slope = getTheilSenSlope(&series, valid);
    var detrended: array<f32, ${sliceCount}>;
    for (var row = 0u; row < valid; row++) {
      detrended[row] = series[row] - slope * f32(row + 1u);
    }
    ${seriesWGSL}
    var seriesMean = 0.0;
    var hasVariation = false;
    for (var row = 0u; row < valid; row++) {
      seriesMean += ${seriesName}[row];
      hasVariation = hasVariation || ${seriesName}[row] != ${seriesName}[0];
    }
    seriesMean /= lengthValue;
    var variance = 0.0;
    for (var row = 0u; row < valid; row++) {
      variance += (${seriesName}[row] - seriesMean) * (${seriesName}[row] - seriesMean);
    }
    // Mean division may round even when every signal value is identical (Metal fast math).
    // Such equal nonzero residuals would invent r_k = (n-k)/n. A constant signal has no
    // autocorrelation correction, independently of that arithmetic rounding.
    if (hasVariation && variance > 0.0) {
      var lagSum = 0.0;
      let lastLag = ${maximumLag > 0 ? `min(${maximumLag}u, valid - 1u)` : 'valid - 1u'};
      for (var lag = 1u; lag <= lastLag; lag++) {
        var covariance = 0.0;
        for (var row = 0u; row + lag < valid; row++) {
          covariance += (${seriesName}[row] - seriesMean) * (${seriesName}[row + lag] - seriesMean);
        }
        let correlation = covariance / variance;
        ${lagSumWGSL}
      }
      let factor = 1.0 + ${scale} * lagSum;
      if (factor > 0.0) {
        varianceCorrection = factor;
      }
    }
  }`;
}
