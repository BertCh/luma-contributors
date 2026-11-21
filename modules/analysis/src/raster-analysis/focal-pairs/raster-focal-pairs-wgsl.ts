// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterFocalPairsProps} from './gpu-raster-focal-pairs';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';

/** @internal */
export function getFocalPairsDeclarations(props: GPURasterFocalPairsProps): string {
  return /* wgsl */ `
${IS_FINITE_VALUE_WGSL}
${SCALED_QUOTIENT_WGSL}
const WIDTH: i32 = ${props.grid.width};
const HEIGHT: i32 = ${props.grid.height};
const RADIUS: i32 = ${props.radius};
fn reflectCoordinate(value: i32, length: i32) -> i32 {
  if (length <= 1) { return 0; }
  let period = (length - 1) * 2;
  let reflected = ((value % period) + period) % period;
  return select(reflected, period - reflected, reflected >= length);
}
fn getSampleIndex(column: i32, row: i32) -> i32 {
  ${props.borderMode === 'clamp' ? 'return clamp(row, 0, HEIGHT - 1) * WIDTH + clamp(column, 0, WIDTH - 1);' : props.borderMode === 'reflect' ? 'return reflectCoordinate(row, HEIGHT) * WIDTH + reflectCoordinate(column, WIDTH);' : 'if (column < 0 || column >= WIDTH || row < 0 || row >= HEIGHT) { return -1; }\n  return row * WIDTH + column;'}
}
fn getPair(sampleIndex: i32) -> vec2<f32> {
  if (sampleIndex < 0) { return vec2<f32>(bitcast<f32>(0x7fc00000u | (u32(sampleIndex) & 0u))); }
  return vec2<f32>(predictor[predictorOffset + u32(sampleIndex)], response[responseOffset + u32(sampleIndex)]);
}
fn isPairValid(pair: vec2<f32>) -> bool {
  return isFiniteValue(pair.x) && isFiniteValue(pair.y);
}`;
}

/** @internal */
export function getFocalPairsBody(props: GPURasterFocalPairsProps): string {
  return /* wgsl */ `
let column = i32(index % u32(WIDTH));
let row = i32(index / u32(WIDTH));
let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
var count = 0u;
var missingPair = ${props.propagateCenterNoData ? '!isPairValid(getPair(i32(index)))' : 'false'};
var anchor = vec2<f32>(0.0);
var totalDifference = vec2<f32>(0.0);
for (var rowOffset = -RADIUS; rowOffset <= RADIUS; rowOffset++) {
  for (var columnOffset = -RADIUS; columnOffset <= RADIUS; columnOffset++) {
    let pair = getPair(getSampleIndex(column + columnOffset, row + rowOffset));
    if (!isPairValid(pair)) { ${props.noDataPolicy === 'propagate' ? 'missingPair = true;' : ''} continue; }
    if (count == 0u) { anchor = pair; }
    totalDifference += pair - anchor;
    count++;
  }
}
pairedCount[pairedCountOffset + index] = count;
var code = select(0u, 1u, count < ${props.minimumCount ?? 2}u) | select(0u, 2u, missingPair);
var covariance = invalid;
var correlation = invalid;
var slope = invalid;
var intercept = invalid;
var rSquared = invalid;
var residualSumSquares = invalid;
if (code == 0u) {
  let meanDifference = totalDifference / f32(count);
  var predictorSquares = 0.0;
  var responseSquares = 0.0;
  var crossProducts = 0.0;
  var centeredTotals = vec2<f32>(0.0);
  for (var rowOffset = -RADIUS; rowOffset <= RADIUS; rowOffset++) {
    for (var columnOffset = -RADIUS; columnOffset <= RADIUS; columnOffset++) {
      let pair = getPair(getSampleIndex(column + columnOffset, row + rowOffset));
      if (!isPairValid(pair)) { continue; }
      let centered = (pair - anchor) - meanDifference;
      centeredTotals += centered;
      predictorSquares += centered.x * centered.x;
      responseSquares += centered.y * centered.y;
      crossProducts += centered.x * centered.y;
    }
  }
  // A backend may reassociate (pair - anchor) - meanDifference into pair minus a
  // rounded large centroid. Correct its nonzero residual mean using only small values.
  let correctedPredictorSquares = predictorSquares - centeredTotals.x * (centeredTotals.x / f32(count));
  let correctedResponseSquares = responseSquares - centeredTotals.y * (centeredTotals.y / f32(count));
  let correctedCrossProducts = crossProducts - centeredTotals.x * (centeredTotals.y / f32(count));
  if (!isFiniteValue(predictorSquares) || !isFiniteValue(responseSquares) || !isFiniteValue(crossProducts) ||
      !isPairValid(centeredTotals) || !isFiniteValue(correctedPredictorSquares) ||
      !isFiniteValue(correctedResponseSquares) || !isFiniteValue(correctedCrossProducts)) {
    code |= 16u;
  } else {
    predictorSquares = max(0.0, correctedPredictorSquares);
    responseSquares = max(0.0, correctedResponseSquares);
    crossProducts = correctedCrossProducts;
    covariance = crossProducts / f32(count${props.covarianceNormalization === 'population' ? '' : ' - 1u'});
    if (predictorSquares == 0.0) { code |= 4u; }
    if (responseSquares == 0.0) { code |= 8u; }
    if (predictorSquares > 0.0) {
      slope = getScaledQuotient(crossProducts, predictorSquares, 1.0, 0);
      intercept = (anchor.y - slope * anchor.x) + (meanDifference.y - slope * meanDifference.x);
      residualSumSquares = 0.0;
      var residualTotal = 0.0;
      for (var rowOffset = -RADIUS; rowOffset <= RADIUS; rowOffset++) {
        for (var columnOffset = -RADIUS; columnOffset <= RADIUS; columnOffset++) {
          let pair = getPair(getSampleIndex(column + columnOffset, row + rowOffset));
          if (!isPairValid(pair)) { continue; }
          let centered = (pair - anchor) - meanDifference;
          let residual = centered.y - slope * centered.x;
          residualTotal += residual;
          residualSumSquares += residual * residual;
        }
      }
      // The same centroid rounding adds a constant to every fitted residual.
      // Remove its squared mean without rebuilding any large original-coordinate mean.
      let correctedResidualSquares = residualSumSquares - residualTotal * (residualTotal / f32(count));
      if (!isFiniteValue(residualSumSquares) || !isFiniteValue(correctedResidualSquares)) {
        code |= 16u;
      } else {
        residualSumSquares = max(0.0, correctedResidualSquares);
      }
      if (responseSquares > 0.0) {
        // Divide successively to avoid overflowing the product of the two squared sums.
        correlation = clamp(getScaledQuotient(crossProducts, sqrt(predictorSquares), sqrt(responseSquares), 0), -1.0, 1.0);
        rSquared = clamp(1.0 - getScaledQuotient(residualSumSquares, responseSquares, 1.0, 0), 0.0, 1.0);
      }
      if (!isFiniteValue(slope) || !isFiniteValue(intercept) || !isFiniteValue(residualSumSquares) ||
          (responseSquares > 0.0 && (!isFiniteValue(correlation) || !isFiniteValue(rSquared)))) {
        code |= 16u;
      }
    }
  }
}
if ((code & 16u) != 0u) {
  covariance = invalid; correlation = invalid; slope = invalid;
  intercept = invalid; rSquared = invalid; residualSumSquares = invalid;
}
let destination = metricsOffset + index * 6u;
metrics[destination] = covariance;
metrics[destination + 1u] = correlation;
metrics[destination + 2u] = slope;
metrics[destination + 3u] = intercept;
metrics[destination + 4u] = rSquared;
metrics[destination + 5u] = residualSumSquares;
status[statusOffset + index] = code;`;
}
