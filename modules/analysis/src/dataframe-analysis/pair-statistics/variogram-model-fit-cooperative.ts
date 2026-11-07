// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Fixed-size workgroup reductions for one model's ordered range refinement. */
export const COOPERATIVE_VARIOGRAM_FIT_WGSL = /* wgsl */ `
struct FitSums {
  moments: vec4f,
  weightedCross: f32,
}
var<workgroup> fitSums: array<FitSums, 64>;
var<workgroup> sharedBest: vec4f;
var<workgroup> sharedBestStep: u32;
var<workgroup> sharedValid: u32;
var<workgroup> sharedLimits: vec2f;

fn reduceFitSums(lane: u32) -> FitSums {
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      fitSums[lane].moments += fitSums[lane + stride].moments;
      fitSums[lane].weightedCross += fitSums[lane + stride].weightedCross;
    }
    workgroupBarrier();
  }
  let total = workgroupUniformLoad(&fitSums[0]);
  workgroupBarrier();
  return total;
}

fn fitLinearCooperative(range: f32, lane: u32) -> vec4f {
  // Pass 0: total weight. Cressie weights are 1/max(fitted^2, 1e-20) and can reach 1e20, so the
  // normal-equation products weightSum * squareSum would overflow f32. Every weight is divided by
  // the total, which bounds the normalised weightSum by 1 and the determinant by 1.
  var totalWeight = 0.0;
  for (var bin = lane; bin < BIN_COUNT; bin += 64u) {
    if (usable(bin)) {
      totalWeight += weight(bin);
    }
  }
  fitSums[lane] = FitSums(vec4f(totalWeight, 0.0, 0.0, 0.0), 0.0);
  let weightTotal = reduceFitSums(lane).moments.x;
  let weightScale = select(1.0, 1.0 / weightTotal, weightTotal > 0.0 && weightTotal < 3.0e38);
  var moments = vec4f(0.0);
  var weightedCross = 0.0;
  for (var bin = lane; bin < BIN_COUNT; bin += 64u) {
    if (usable(bin)) {
      let binWeight = weight(bin) * weightScale;
      let binShape = shape(distances[distancesOffset + bin] / range);
      let value = semivariances[semivariancesOffset + bin];
      moments += vec4f(binWeight, binWeight * binShape, binWeight * binShape * binShape, binWeight * value);
      weightedCross += binWeight * binShape * value;
    }
  }
  fitSums[lane] = FitSums(moments, weightedCross);
  let sums = reduceFitSums(lane);
  let weightSum = sums.moments.x;
  let shapeSum = sums.moments.y;
  let squareSum = sums.moments.z;
  let valueSum = sums.moments.w;
  let crossSum = sums.weightedCross;
  let nuggetOnly = max(valueSum / weightSum, 0.0);
  let sillOnly = max(crossSum / max(squareSum, 1e-30), 0.0);
  let determinant = weightSum * squareSum - shapeSum * shapeSum;
  var nugget = 0.0;
  var sill = 0.0;
  var jointValid = false;
  if (determinant > 1e-6 * weightSum * squareSum) {
    nugget = (squareSum * valueSum - shapeSum * crossSum) / determinant;
    sill = (weightSum * crossSum - shapeSum * valueSum) / determinant;
    jointValid = nugget >= 0.0 && sill >= 0.0;
  }
  var errors = vec3f(0.0);
  for (var bin = lane; bin < BIN_COUNT; bin += 64u) {
    if (usable(bin)) {
      let binWeight = weight(bin) * weightScale;
      let binShape = shape(distances[distancesOffset + bin] / range);
      let value = semivariances[semivariancesOffset + bin];
      let differences = vec3f(value - nuggetOnly, value - sillOnly * binShape, value - nugget - sill * binShape);
      errors += binWeight * differences * differences;
    }
  }
  fitSums[lane] = FitSums(vec4f(errors, 0.0), 0.0);
  let residuals = reduceFitSums(lane).moments;
  var best = vec4f(nuggetOnly, 0.0, range, residuals.x);
  if (residuals.y < best.w) { best = vec4f(0.0, sillOnly, range, residuals.y); }
  if (jointValid && residuals.z < best.w) { best = vec4f(nugget, sill, range, residuals.z); }
  // Report the residual in the caller's unnormalised weights.
  best.w = best.w / weightScale;
  return best;
}
`;
