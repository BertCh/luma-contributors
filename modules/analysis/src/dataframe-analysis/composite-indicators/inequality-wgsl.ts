// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GET_NAN_WGSL,
  GET_ORDERED_KEY_WGSL,
  IS_FINITE_BITS_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';
import {ATKINSON_WEIGHT_NORMALIZATION_THRESHOLD} from './inequality-parameters';

/** Floats per zone in the interleaved statistics buffer. */
export const INEQUALITY_STATS_STRIDE = 10;

/** Slots of one zone's interleaved statistics. */
export const INEQUALITY_STAT = {
  MEAN: 0,
  GINI: 1,
  THEIL_T: 2,
  THEIL_L: 3,
  ATKINSON: 4,
  HOOVER: 5,
  PALMA: 6,
  TOTAL_WEIGHT: 7,
  TOTAL_INCOME: 8,
  WEIGHTED_X_LOG_X: 9
} as const;

/** Shared WGSL helpers. */
export const INEQUALITY_HELPERS_WGSL = /* wgsl */ `
${SCALED_QUOTIENT_WGSL}
${GET_NAN_WGSL}

${IS_FINITE_BITS_WGSL}

${GET_ORDERED_KEY_WGSL}

// Stable divided differences: keep the epsilon-near-one signal before float32 rounding.
// The short Taylor branches avoid subtracting nearly equal exp/log values.
fn getAtkinsonExpm1OverArgument(argument: f32) -> f32 {
  if (abs(argument) < 0.01) {
    return 1.0 + argument * (0.5 + argument * (1.0 / 6.0 + argument * (1.0 / 24.0 + argument / 120.0)));
  }
  return (exp(argument) - 1.0) / argument;
}
fn getAtkinsonLog1pOverArgument(argument: f32) -> f32 {
  if (abs(argument) < 0.01) {
    return 1.0 + argument * (-0.5 + argument * (1.0 / 3.0 + argument * (-0.25 + argument / 5.0)));
  }
  return log(1.0 + argument) / argument;
}
fn getAtkinsonCenteredTerm(logRatio: f32, power: f32) -> f32 {
  return logRatio * getAtkinsonExpm1OverArgument(power * logRatio);
}
fn shouldNormalizeAtkinsonWeights(population: f32) -> bool {
  // Below this threshold even the largest logarithmic and centered terms have exponent headroom.
  // Above it, reduce fractional population contributions directly.
  return isFiniteBits(population) && population > ${ATKINSON_WEIGHT_NORMALIZATION_THRESHOLD};
}
// value and mean are positive here. Forming their quotient can underflow even when both
// logarithms are finite; retain the original ratio arithmetic on the ordinary path.
fn getAtkinsonLogRatio(value: f32, mean: f32, ratio: f32) -> f32 {
  if (ratio > 0.0 && isFiniteBits(ratio)) { return log(ratio); }
  return log(value) - log(mean);
}
fn getAtkinsonFromCenteredMean(positiveCenteredMean: f32, power: f32, zeroFraction: f32) -> f32 {
  // Positive rows contribute (r^power - 1)/power; each zero contributes -1/power.
  // Keep the zero population fraction separate until after the positive reduction.
  let centeredMean = positiveCenteredMean - zeroFraction / power;
  let logarithm = centeredMean * getAtkinsonLog1pOverArgument(power * positiveCenteredMean - zeroFraction);
  if (logarithm < -80.0) { return 1.0; }
  return -logarithm * getAtkinsonExpm1OverArgument(logarithm);
}

fn interpolateLorenz(cut: f32, previousP: f32, nextP: f32, previousL: f32, nextL: f32) -> f32 {
  if (cut >= nextP) {
    return nextL;
  }
  return previousL + getScaledQuotient(cut - previousP, nextP - previousP, 1.0, 0) * (nextL - previousL);
}
`;

/** Body of the row validation and key generation kernel. `weightCheck` is empty without weights. */
export function getInequalityKeysBody(options: {hasMask: boolean; hasWeights: boolean}): string {
  return /* wgsl */ `let value = values[valuesOffset + index];
  let zone = zoneIds[zoneIdsOffset + index];
  var isValid = isFiniteBits(value) && value >= 0.0 && zone < ZONE_COUNT;
  ${options.hasMask ? 'isValid = isValid && rowMask[rowMaskOffset + index] != 0u;' : ''}
  ${
    options.hasWeights
      ? `let weight = weights[weightsOffset + index];
  isValid = isValid && isFiniteBits(weight) && weight > 0.0;`
      : ''
  }
  // Valid keys are below 0xff800000, so invalid rows sort after every valid row; -0 sorts as +0.
  valueKeys[valueKeysOffset + index] = select(0xffffffffu, getOrderedKey(select(value, 0.0, value == 0.0)), isValid);
  rowIndices[rowIndicesOffset + index] = index;`;
}

/** Body of the kernel that keys the value-sorted rows by zone. */
export const INEQUALITY_ZONE_KEYS_BODY = /* wgsl */ `let row = sortedRows[sortedRowsOffset + index];
  let isValid = sortedValueKeys[sortedValueKeysOffset + index] != 0xffffffffu;
  zoneKeys[zoneKeysOffset + index] = select(ZONE_COUNT, zoneIds[zoneIdsOffset + row], isValid);`;

/**
 * Legacy reference walk for the other indices (one thread per zone). Its Gini and zero-boundary
 * Atkinson scratch slots are replaced by parallel kernels before GPUInequality publishes outputs.
 */
export function getInequalityZoneStatsBody(weightExpression: string): string {
  return /* wgsl */ `let zone = index;
  var searchLow = 0u;
  var searchHigh = ROW_COUNT;
  while (searchLow < searchHigh) {
    let middle = (searchLow + searchHigh) / 2u;
    if (sortedZones[sortedZonesOffset + middle] < zone) {
      searchLow = middle + 1u;
    } else {
      searchHigh = middle;
    }
  }
  let rangeStart = searchLow;
  searchHigh = ROW_COUNT;
  while (searchLow < searchHigh) {
    let middle = (searchLow + searchHigh) / 2u;
    if (sortedZones[sortedZonesOffset + middle] <= zone) {
      searchLow = middle + 1u;
    } else {
      searchHigh = middle;
    }
  }
  let rangeEnd = searchLow;
  let count = rangeEnd - rangeStart;
  counts[countsOffset + zone] = count;
  let nan = getNaN();
  let base = statsOffset + zone * STATS_STRIDE;
  let knotBase = lorenzOffset + zone * KNOT_COUNT;
  for (var slot = 0u; slot < STATS_STRIDE; slot++) {
    stats[base + slot] = nan;
  }
  for (var knot = 0u; knot < KNOT_COUNT; knot++) {
    lorenz[knotBase + knot] = nan;
  }
  stats[base + 7u] = 0.0;
  stats[base + 8u] = 0.0;
  stats[base + 9u] = 0.0;
  if (count == 0u) {
    return;
  }

  // Pass 1: totals in sorted order.
  var totalWeight = 0.0;
  var totalIncome = 0.0;
  var weightedXLogX = 0.0;
  var hasZero = false;
  var zeroWeight = 0.0;
  for (var position = rangeStart; position < rangeEnd; position++) {
    let row = sortedRows[sortedRowsOffset + position];
    let x = values[valuesOffset + row];
    let w = ${weightExpression};
    totalWeight = totalWeight + w;
    totalIncome = totalIncome + w * x;
    if (x > 0.0) {
      weightedXLogX = weightedXLogX + w * x * log(x);
    } else {
      hasZero = true;
      zeroWeight = zeroWeight + w;
    }
  }
  let mean = getScaledQuotient(totalIncome, totalWeight, 1.0, 0);
  stats[base] = mean;
  stats[base + 7u] = totalWeight;
  stats[base + 8u] = totalIncome;
  stats[base + 9u] = weightedXLogX;
  if (!(totalIncome > 0.0)) {
    return;
  }

  // Pass 2: Lorenz walk and the mean-relative sums, in the same order.
  let epsilon = params[paramsOffset];
  let palmaTop = params[paramsOffset + 1u];
  let palmaBottom = params[paramsOffset + 2u];
  let power = 1.0 - epsilon;
  let usePower = epsilon != 1.0;
  let useCenteredPower = usePower && abs(power) <= 0.01 && (power > 0.0 || !hasZero);
  let normalizeWeights = shouldNormalizeAtkinsonWeights(totalWeight);
  let normalizeLogWeights = !usePower && normalizeWeights;
  let palmaBottomTarget = palmaBottom;
  let palmaTopTarget = 1.0 - palmaTop;
  var cumulativeWeight = 0.0;
  var cumulativeIncome = 0.0;
  var previousP = 0.0;
  var previousL = 0.0;
  var trapezoidArea = 0.0;
  var sumTheilT = 0.0;
  var sumLog = 0.0;
  var sumPower = 0.0;
  var sumAbsolute = 0.0;
  var nextKnot = 1u;
  var bottomL = nan;
  var topCutL = nan;
  var isBottomDone = false;
  var isTopDone = false;
  lorenz[knotBase] = 0.0;
  for (var position = rangeStart; position < rangeEnd; position++) {
    let row = sortedRows[sortedRowsOffset + position];
    let x = values[valuesOffset + row];
    let w = ${weightExpression};
    cumulativeWeight = cumulativeWeight + w;
    cumulativeIncome = cumulativeIncome + w * x;
    let nextP = getScaledQuotient(cumulativeWeight, totalWeight, 1.0, 0);
    let nextL = getScaledQuotient(cumulativeIncome, totalIncome, 1.0, 0);
    trapezoidArea = trapezoidArea + (nextP - previousP) * (previousL + nextL);
    sumAbsolute = sumAbsolute + w * abs(x - mean);
    if (x > 0.0) {
      let ratio = getScaledQuotient(x, mean, 1.0, 0);
      let logRatio = log(ratio);
      sumTheilT = sumTheilT + w * (getScaledQuotient(x, mean, 1.0, 0)) * logRatio;
      if (normalizeLogWeights) {
        let weightFraction = getScaledQuotient(w, totalWeight, 1.0, 0);
        sumLog = sumLog + weightFraction * getAtkinsonLogRatio(x, mean, ratio);
      } else {
        sumLog = sumLog + w * logRatio;
      }
      if (usePower) {
        if (useCenteredPower) {
          var atkinsonWeight = w;
          if (normalizeWeights) {
            atkinsonWeight = getScaledQuotient(w, totalWeight, 1.0, 0);
          }
          sumPower = sumPower + atkinsonWeight * getAtkinsonCenteredTerm(getAtkinsonLogRatio(x, mean, ratio), power);
        } else {
          sumPower = sumPower + w * exp(power * logRatio);
        }
      }
    }
    while (nextKnot < KNOT_COUNT && f32(nextKnot) / f32(KNOT_COUNT - 1u) <= nextP) {
      lorenz[knotBase + nextKnot] = interpolateLorenz(f32(nextKnot) / f32(KNOT_COUNT - 1u), previousP, nextP, previousL, nextL);
      nextKnot = nextKnot + 1u;
    }
    if (!isBottomDone && palmaBottomTarget <= nextP) {
      bottomL = interpolateLorenz(palmaBottomTarget, previousP, nextP, previousL, nextL);
      isBottomDone = true;
    }
    if (!isTopDone && palmaTopTarget <= nextP) {
      topCutL = interpolateLorenz(palmaTopTarget, previousP, nextP, previousL, nextL);
      isTopDone = true;
    }
    previousP = nextP;
    previousL = nextL;
  }
  // Rounding can leave the last cumulative share a hair below a cut of exactly 1.
  while (nextKnot < KNOT_COUNT) {
    lorenz[knotBase + nextKnot] = previousL;
    nextKnot = nextKnot + 1u;
  }
  if (!isBottomDone) {
    bottomL = previousL;
  }
  if (!isTopDone) {
    topCutL = previousL;
  }

  stats[base + 1u] = 1.0 - trapezoidArea;
  stats[base + 2u] = getScaledQuotient(sumTheilT, totalWeight, 1.0, 0);
  var weightedLogMean = sumLog;
  if (!normalizeLogWeights) {
    weightedLogMean = getScaledQuotient(sumLog, totalWeight, 1.0, 0);
  }
  stats[base + 3u] = select(-weightedLogMean, nan, hasZero);
  var atkinson = nan;
  if (!usePower) {
    if (!hasZero) {
      atkinson = 1.0 - exp(weightedLogMean);
    }
  } else if (power > 0.0 || !hasZero) {
    if (useCenteredPower) {
      var centeredMean = sumPower;
      if (!normalizeWeights) {
        centeredMean = getScaledQuotient(sumPower, totalWeight, 1.0, 0);
      }
      atkinson = getAtkinsonFromCenteredMean(centeredMean, power, getScaledQuotient(zeroWeight, totalWeight, 1.0, 0));
    } else {
      let powerMean = getScaledQuotient(sumPower, totalWeight, 1.0, 0);
      atkinson = 1.0 - pow(powerMean, 1.0 / power);
    }
  }
  stats[base + 4u] = select(nan, atkinson, isFiniteBits(atkinson));
  stats[base + 5u] = getScaledQuotient(sumAbsolute, totalIncome, 2.0, 0);
  stats[base + 6u] = select(nan, getScaledQuotient(1.0 - topCutL, bottomL, 1.0, 0), bottomL > 0.0);`;
}

/** Lanes of the cooperative workgroup used by the global summary and pooled-Gini finish kernels. */
export const INEQUALITY_REDUCTION_LANES = 256;

/** Workgroup memory the global summary kernel needs; append to the kernel's declarations. */
export const INEQUALITY_GLOBAL_SUMS_DECLARATIONS = /* wgsl */ `
var<workgroup> laneFirst: array<f32, ${INEQUALITY_REDUCTION_LANES}>;
var<workgroup> laneSecond: array<f32, ${INEQUALITY_REDUCTION_LANES}>;
var<workgroup> laneThird: array<f32, ${INEQUALITY_REDUCTION_LANES}>;
var<workgroup> laneIncluded: array<u32, ${INEQUALITY_REDUCTION_LANES}>;`;

/**
 * Body of the global summary kernel that combines zone totals. One 256-lane workgroup: pass A
 * strides the zones (lane l takes zones l, l + 256, ...) and a fixed binary tree combines the lane
 * partials into the totals, which every lane then reads from workgroup memory; pass B does the same
 * for the between and within terms. Lane 0 writes the summary rows. Depth is
 * O(zones / 256 + log2 256) instead of two serial passes over all zones.
 */
export const INEQUALITY_GLOBAL_SUMS_BODY = /* wgsl */ `let lane = localInvocationIndex;
  var laneWeight = 0.0;
  var laneIncome = 0.0;
  var laneXLogX = 0.0;
  var laneCount = 0u;
  for (var zone = lane; zone < ZONE_COUNT; zone += ${INEQUALITY_REDUCTION_LANES}u) {
    let zoneCount = counts[countsOffset + zone];
    if (zoneCount == 0u) {
      continue;
    }
    let base = statsOffset + zone * STATS_STRIDE;
    laneWeight = laneWeight + stats[base + 7u];
    laneIncome = laneIncome + stats[base + 8u];
    laneXLogX = laneXLogX + stats[base + 9u];
    laneCount = laneCount + zoneCount;
  }
  laneFirst[lane] = laneWeight;
  laneSecond[lane] = laneIncome;
  laneThird[lane] = laneXLogX;
  laneIncluded[lane] = laneCount;
  workgroupBarrier();
  for (var stride = ${INEQUALITY_REDUCTION_LANES / 2}u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) {
      laneFirst[lane] = laneFirst[lane] + laneFirst[lane + stride];
      laneSecond[lane] = laneSecond[lane] + laneSecond[lane + stride];
      laneThird[lane] = laneThird[lane] + laneThird[lane + stride];
      laneIncluded[lane] = laneIncluded[lane] + laneIncluded[lane + stride];
    }
    workgroupBarrier();
  }
  // Every lane reads the totals, then all lanes pass a barrier before the arrays are reused.
  let totalWeight = laneFirst[0];
  let totalIncome = laneSecond[0];
  let weightedXLogX = laneThird[0];
  let includedCount = laneIncluded[0];
  workgroupBarrier();
  // Lanes contribute nothing to pass B when the pooled income is not positive.
  let hasIncome = totalIncome > 0.0;
  var laneBetween = 0.0;
  var laneWithin = 0.0;
  for (var zone = lane; zone < ZONE_COUNT; zone += ${INEQUALITY_REDUCTION_LANES}u) {
    if (!hasIncome || counts[countsOffset + zone] == 0u) {
      continue;
    }
    let base = statsOffset + zone * STATS_STRIDE;
    let zoneIncome = stats[base + 8u];
    if (!(zoneIncome > 0.0)) {
      continue;
    }
    let incomeShare = getScaledQuotient(zoneIncome, totalIncome, 1.0, 0);
    let populationShare = getScaledQuotient(stats[base + 7u], totalWeight, 1.0, 0);
    laneBetween = laneBetween + incomeShare * log(getScaledQuotient(incomeShare, populationShare, 1.0, 0));
    laneWithin = laneWithin + incomeShare * stats[base + 2u];
  }
  laneFirst[lane] = laneBetween;
  laneSecond[lane] = laneWithin;
  workgroupBarrier();
  for (var stride = ${INEQUALITY_REDUCTION_LANES / 2}u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) {
      laneFirst[lane] = laneFirst[lane] + laneFirst[lane + stride];
      laneSecond[lane] = laneSecond[lane] + laneSecond[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    let nan = getNaN();
    summary[summaryOffset + 4u] = f32(includedCount);
    summary[summaryOffset + 5u] = select(nan, getScaledQuotient(totalIncome, totalWeight, 1.0, 0), includedCount != 0u);
    summary[summaryOffset + 6u] = totalWeight;
    summary[summaryOffset + 7u] = totalIncome;
    if (!hasIncome) {
      summary[summaryOffset] = nan;
      summary[summaryOffset + 1u] = nan;
      summary[summaryOffset + 2u] = nan;
    } else {
      // The direct x*log(x) moment can overflow while normalized Theil terms remain finite.
      var pooledTheil = laneFirst[0] + laneSecond[0];
      if (isFiniteBits(weightedXLogX)) {
        pooledTheil = getScaledQuotient(weightedXLogX, totalIncome, 1.0, 0) - log(getScaledQuotient(totalIncome, totalWeight, 1.0, 0));
      }
      summary[summaryOffset] = pooledTheil;
      summary[summaryOffset + 1u] = laneFirst[0];
      summary[summaryOffset + 2u] = laneSecond[0];
    }
  }`;

/** Rows per tile of the retained legacy tiled pooled-Gini helpers (not used for publication). */
export const INEQUALITY_GINI_TILE_ROWS = 256;

/**
 * Body of the pooled-Gini tile totals kernel (one thread per tile of the globally value-sorted
 * rows): sums weight and weighted income of the tile's valid rows. Valid rows are a prefix of the
 * sorted order, so a tile past the prefix is empty.
 */
export function getInequalityGiniTileTotalsBody(weightExpression: string): string {
  return /* wgsl */ `let firstPosition = index * GINI_TILE_ROWS;
  let endPosition = min(firstPosition + GINI_TILE_ROWS, ROW_COUNT);
  var tileWeight = 0.0;
  var tileIncome = 0.0;
  for (var position = firstPosition; position < endPosition; position++) {
    if (sortedValueKeys[sortedValueKeysOffset + position] == 0xffffffffu) {
      break;
    }
    let row = sortedRows[sortedRowsOffset + position];
    let w = ${weightExpression};
    tileWeight = tileWeight + w;
    tileIncome = tileIncome + w * values[valuesOffset + row];
  }
  tileTotals[tileTotalsOffset + 2u * index] = tileWeight;
  tileTotals[tileTotalsOffset + 2u * index + 1u] = tileIncome;`;
}

/** Workgroup memory the tile prefix kernel needs; append to the kernel's declarations. */
export const INEQUALITY_GINI_TILE_PREFIX_DECLARATIONS = /* wgsl */ `
var<workgroup> scanWeight: array<f32, ${INEQUALITY_REDUCTION_LANES}>;
var<workgroup> scanIncome: array<f32, ${INEQUALITY_REDUCTION_LANES}>;`;

/**
 * Body of the 256-lane kernel that turns the tile totals into exclusive tile prefixes and the
 * pooled totals. Lane l owns the contiguous chunk of tiles [l * C, min((l + 1) * C, T)) with
 * C = ceil(T / 256), sums it in order, and a fixed-order Hillis-Steele scan of the 256 chunk sums
 * gives each lane its starting offset. Only the owning lane touches its chunk, so the rewrite of
 * `tileTotals` in place is race free. Depth is O(T / 256 + log2 256), not O(T).
 */
export const INEQUALITY_GINI_TILE_PREFIX_BODY = /* wgsl */ `let lane = localInvocationIndex;
  const chunkSize: u32 = (GINI_TILE_COUNT + ${INEQUALITY_REDUCTION_LANES - 1}u) / ${INEQUALITY_REDUCTION_LANES}u;
  let firstTile = min(lane * chunkSize, GINI_TILE_COUNT);
  let endTile = min(firstTile + chunkSize, GINI_TILE_COUNT);
  var chunkWeight = 0.0;
  var chunkIncome = 0.0;
  for (var tile = firstTile; tile < endTile; tile++) {
    chunkWeight = chunkWeight + tileTotals[tileTotalsOffset + 2u * tile];
    chunkIncome = chunkIncome + tileTotals[tileTotalsOffset + 2u * tile + 1u];
  }
  scanWeight[lane] = chunkWeight;
  scanIncome[lane] = chunkIncome;
  workgroupBarrier();
  for (var offset = 1u; offset < ${INEQUALITY_REDUCTION_LANES}u; offset = offset << 1u) {
    var addWeight = 0.0;
    var addIncome = 0.0;
    if (lane >= offset) {
      addWeight = scanWeight[lane - offset];
      addIncome = scanIncome[lane - offset];
    }
    workgroupBarrier();
    scanWeight[lane] = scanWeight[lane] + addWeight;
    scanIncome[lane] = scanIncome[lane] + addIncome;
    workgroupBarrier();
  }
  // The scan is inclusive; this lane starts after the chunks before it.
  var cumulativeWeight = 0.0;
  var cumulativeIncome = 0.0;
  if (lane > 0u) {
    cumulativeWeight = scanWeight[lane - 1u];
    cumulativeIncome = scanIncome[lane - 1u];
  }
  for (var tile = firstTile; tile < endTile; tile++) {
    let base = tileTotalsOffset + 2u * tile;
    let tileWeight = tileTotals[base];
    let tileIncome = tileTotals[base + 1u];
    tileTotals[base] = cumulativeWeight;
    tileTotals[base + 1u] = cumulativeIncome;
    cumulativeWeight = cumulativeWeight + tileWeight;
    cumulativeIncome = cumulativeIncome + tileIncome;
  }
  if (lane == ${INEQUALITY_REDUCTION_LANES - 1}u) {
    pooledTotals[pooledTotalsOffset] = scanWeight[lane];
    pooledTotals[pooledTotalsOffset + 1u] = scanIncome[lane];
  }`;

/**
 * Body of the pooled-Gini tile walk (one thread per tile): the trapezoid area of the Lorenz curve
 * over the tile's rows, starting from the exclusive prefix of the earlier tiles.
 */
export function getInequalityGiniTileAreaBody(weightExpression: string): string {
  return /* wgsl */ `let totalWeight = pooledTotals[pooledTotalsOffset];
  let totalIncome = pooledTotals[pooledTotalsOffset + 1u];
  var area = 0.0;
  if (totalIncome > 0.0) {
    let firstPosition = index * GINI_TILE_ROWS;
    let endPosition = min(firstPosition + GINI_TILE_ROWS, ROW_COUNT);
    var cumulativeWeight = tileTotals[tileTotalsOffset + 2u * index];
    var cumulativeIncome = tileTotals[tileTotalsOffset + 2u * index + 1u];
    var previousP = getScaledQuotient(cumulativeWeight, totalWeight, 1.0, 0);
    var previousL = getScaledQuotient(cumulativeIncome, totalIncome, 1.0, 0);
    for (var position = firstPosition; position < endPosition; position++) {
      if (sortedValueKeys[sortedValueKeysOffset + position] == 0xffffffffu) {
        break;
      }
      let row = sortedRows[sortedRowsOffset + position];
      let w = ${weightExpression};
      cumulativeWeight = cumulativeWeight + w;
      cumulativeIncome = cumulativeIncome + w * values[valuesOffset + row];
      let nextP = getScaledQuotient(cumulativeWeight, totalWeight, 1.0, 0);
      let nextL = getScaledQuotient(cumulativeIncome, totalIncome, 1.0, 0);
      area = area + (nextP - previousP) * (previousL + nextL);
      previousP = nextP;
      previousL = nextL;
    }
  }
  tileAreas[tileAreasOffset + index] = area;`;
}

/** Workgroup memory the pooled-Gini finish kernel needs; append to the kernel's declarations. */
export const INEQUALITY_GINI_FINISH_DECLARATIONS = /* wgsl */ `
var<workgroup> laneAreas: array<f32, ${INEQUALITY_REDUCTION_LANES}>;`;

/**
 * Body of the 256-lane kernel that sums the tile areas into the pooled Gini: lanes stride the
 * tiles, a fixed binary tree combines the partials, and lane 0 writes `1 - area` (or NaN when the
 * pooled income is not positive; every lane still passes every barrier).
 */
export const INEQUALITY_GINI_FINISH_BODY = /* wgsl */ `let lane = localInvocationIndex;
  var laneArea = 0.0;
  for (var tile = lane; tile < GINI_TILE_COUNT; tile += ${INEQUALITY_REDUCTION_LANES}u) {
    laneArea = laneArea + tileAreas[tileAreasOffset + tile];
  }
  laneAreas[lane] = laneArea;
  workgroupBarrier();
  for (var stride = ${INEQUALITY_REDUCTION_LANES / 2}u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) {
      laneAreas[lane] = laneAreas[lane] + laneAreas[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    if (!(pooledTotals[pooledTotalsOffset + 1u] > 0.0)) {
      summary[summaryOffset + 3u] = getNaN();
    } else {
      summary[summaryOffset + 3u] = 1.0 - laneAreas[0];
    }
  }`;
