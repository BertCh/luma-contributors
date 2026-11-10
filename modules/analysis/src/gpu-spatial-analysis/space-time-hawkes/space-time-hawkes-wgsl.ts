// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/**
 * Floor of the background rate before it enters a logarithm: the smallest normal float32, because
 * WGSL implementations may flush denormals. It plays the part of the floor on tiny sums of Ko,
 * Suchard and Holbrook (2024; 1e-40 there). It must not be larger: a floor of 1e-30 distorted fits
 * whose rates are below it (experiments/stm08-hawkes-float32.mjs, position scale 1e17).
 */
export const GPU_SPACE_TIME_HAWKES_INTENSITY_FLOOR = 1.1754944e-38;

/** Floor of the dimensionless branching ratio before it enters a logarithm. */
export const GPU_SPACE_TIME_HAWKES_BRANCHING_RATIO_FLOOR = 1e-30;

/** Slots of the float32 state buffer of {@link GPUSpaceTimeHawkes}. */
export const SPACE_TIME_HAWKES_STATE_SLOTS = {
  background: 0,
  branchingRatio: 1,
  decayRate: 2,
  spatialVariance: 3,
  /** First of the eight merged sums, see {@link SPACE_TIME_HAWKES_SUM_COUNT}. */
  sums: 4,
  done: 12,
  iterationCount: 13,
  converged: 14,
  iterationLimitReached: 15,
  logLikelihood: 16
} as const;

/**
 * Sums merged per iteration, in order: background responsibility, triggered responsibility (the
 * sum over events of one minus the background responsibility, summed directly so it keeps its
 * precision when little is triggered), responsibility-weighted lag, responsibility-weighted
 * squared distance, log intensity, events with a cut parent window, events in the fit, and the
 * time-kernel mass inside the observation window.
 */
export const SPACE_TIME_HAWKES_SUM_COUNT = 8;

/** Length of the float32 state buffer. */
export const SPACE_TIME_HAWKES_STATE_LENGTH = 17;

/**
 * WGSL of the E-step for one event of the time-sorted table. Expects bindings `events` (four
 * float32 per sorted row: position, time, unused; a NaN time marks an excluded row),
 * `state` and `params`.
 *
 * With theta = (mu, omega, beta, variance) the intensity at event j is
 * `lambda_j = mu + sum_i omega beta exp(-beta lag) / (2 pi variance) exp(-d^2 / (2 variance))` over
 * parents i before j (lag > 0) within the lag and distance bounds and the last `maximumParents`
 * candidates in time. It is accumulated in the log domain in one pass: the running maximum `m` of
 * the log terms, the background weight and parent weights `exp(term - m)` (the background is always
 * present, so their sum `s` is at least one), and the same sums weighted by lag and squared distance. When a larger term appears the
 * sums are rescaled by `exp(m - term)`. Then `log lambda = m + log s`, the background
 * responsibility is the background weight over `s`, and the responsibility-weighted lag and squared
 * distance are `sLag / s` and `sDistance / s`, with no division by a tiny lambda.
 */
export function getSpaceTimeHawkesEventWGSL(maximumParents: number): string {
  const slots = SPACE_TIME_HAWKES_STATE_SLOTS;
  return `const MAXIMUM_PARENTS: u32 = ${maximumParents}u;
const INTENSITY_FLOOR: f32 = ${getWGSLFloatLiteral(GPU_SPACE_TIME_HAWKES_INTENSITY_FLOOR)};
const BRANCHING_RATIO_FLOOR: f32 = ${getWGSLFloatLiteral(GPU_SPACE_TIME_HAWKES_BRANCHING_RATIO_FLOOR)};
struct EventStatistics {
  background: f32,
  triggered: f32,
  lag: f32,
  distanceSquared: f32,
  logIntensity: f32,
  cut: f32
}
fn evaluateEvent(row: u32) -> EventStatistics {
  var result = EventStatistics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  let time = events[eventsOffset + 4u * row + 2u];
  let position = vec2f(events[eventsOffset + 4u * row], events[eventsOffset + 4u * row + 1u]);
  let background = state[stateOffset + ${slots.background}u];
  let branchingRatio = state[stateOffset + ${slots.branchingRatio}u];
  let decayRate = state[stateOffset + ${slots.decayRate}u];
  let variance = state[stateOffset + ${slots.spatialVariance}u];
  let logBackground = log(max(background, INTENSITY_FLOOR));
  let logProductivity = log(max(branchingRatio, BRANCHING_RATIO_FLOOR)) + log(decayRate) - log(6.2831853 * variance);
  let inverseTwoVariance = 0.5 / variance;
  let maximumDistance = params[paramsOffset + 3u];
  let maximumDistanceSquared = maximumDistance * maximumDistance;
  // First sorted row whose time is within the maximum lag of this event.
  let earliest = time - params[paramsOffset + 2u];
  var low = 0u;
  var high = row;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (events[eventsOffset + 4u * middle + 2u] < earliest) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (row - low > MAXIMUM_PARENTS) {
    low = row - MAXIMUM_PARENTS;
    result.cut = 1.0;
  }
  var maximumLog = logBackground;
  var backgroundWeight = 1.0;
  var parentSum = 0.0;
  var lagSum = 0.0;
  var distanceSum = 0.0;
  for (var parent = low; parent < row; parent++) {
    let lag = time - events[eventsOffset + 4u * parent + 2u];
    let delta = position - vec2f(events[eventsOffset + 4u * parent], events[eventsOffset + 4u * parent + 1u]);
    let distanceSquared = dot(delta, delta);
    if (!(lag > 0.0) || distanceSquared > maximumDistanceSquared) {
      continue;
    }
    let term = logProductivity - decayRate * lag - distanceSquared * inverseTwoVariance;
    if (term > maximumLog) {
      let scale = exp(maximumLog - term);
      backgroundWeight = backgroundWeight * scale;
      parentSum = parentSum * scale + 1.0;
      lagSum = lagSum * scale + lag;
      distanceSum = distanceSum * scale + distanceSquared;
      maximumLog = term;
    } else {
      let weight = exp(term - maximumLog);
      parentSum += weight;
      lagSum += weight * lag;
      distanceSum += weight * distanceSquared;
    }
  }
  let sum = backgroundWeight + parentSum;
  result.background = backgroundWeight / sum;
  result.triggered = parentSum / sum;
  result.lag = lagSum / sum;
  result.distanceSquared = distanceSum / sum;
  result.logIntensity = maximumLog + log(sum);
  return result;
}`;
}
