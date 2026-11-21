// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {FIXED_POINT_WGSL} from '../../analysis-core/kernels/fixed-point-accumulator';
import {HOP_WORD_COUNT} from './map-matching-search-wgsl';

/** Module declarations of the forward Viterbi kernel: candidate access and model constants. */
export function getViterbiDeclarations(candidateCount: number): string {
  return `fn readPoint(point: u32) -> vec2<f32> {
  return vec2<f32>(points[pointsOffset + point * 2u], points[pointsOffset + point * 2u + 1u]);
}
fn getCandidateEdge(point: u32, slot: u32) -> u32 {
  return candidates[candidatesOffset + (point * K + slot) * 3u];
}
fn getCandidateFraction(point: u32, slot: u32) -> f32 {
  return bitcast<f32>(candidates[candidatesOffset + (point * K + slot) * 3u + 1u]);
}
fn getCandidateDistance(point: u32, slot: u32) -> f32 {
  return bitcast<f32>(candidates[candidatesOffset + (point * K + slot) * 3u + 2u]);
}

const LOG_TWO_PI_HALF: f32 = 0.9189385;
const CANDIDATE_COUNT: u32 = ${candidateCount}u;`;
}

/** Body of the forward Viterbi kernel: one thread per track. */
export function getForwardBody(hasFixFilter: boolean): string {
  return `let first = trackOffsets[trackOffsetsOffset + index];
  let last = trackOffsets[trackOffsetsOffset + index + 1u];
  let sigma = max(parameters[parametersOffset], 1.0e-6);
  let beta = max(parameters[parametersOffset + 1u], 1.0e-6);
  let emissionConstant = -log(sigma) - LOG_TWO_PI_HALF;
  let transitionConstant = -log(beta);
  for (var point = first; point < last; point++) {
    var newScore: array<f32, CANDIDATE_COUNT>;
    var newBack: array<u32, CANDIDATE_COUNT>;
    for (var slot = 0u; slot < K; slot++) {
      newScore[slot] = NEG;
      newBack[slot] = NONE;
    }
    var isReachable = false;
    var isBudgetLimited = false;
    ${
      hasFixFilter
        ? `let previous = predecessors[predecessorsOffset + point];
    if (previous == DROPPED_FIX) {
      // A dropped fix is skipped: neither a source nor a target of a transition.
      for (var slot = 0u; slot < K; slot++) {
        scores[scoresOffset + point * SCORE_STRIDE + slot] = NEG;
        backPointers[backPointersOffset + point * K + slot] = NONE;
      }
      scores[scoresOffset + point * SCORE_STRIDE + K] = 0.0;
      continue;
    }
    if (previous != NONE) {`
        : `let previous = point - 1u;
    if (point > first) {`
    }
      let straight = networkDistance(readPoint(point), readPoint(previous));
      for (var fromSlot = 0u; fromSlot < K; fromSlot++) {
        let fromEdge = getCandidateEdge(previous, fromSlot);
        let fromScore = scores[scoresOffset + previous * SCORE_STRIDE + fromSlot];
        if (fromEdge == NONE || fromScore < -1.0e29) {
          continue;
        }
        let routeRow = routesOffset + (point * K + fromSlot) * ROUTE_STRIDE;
        isBudgetLimited = isBudgetLimited || routes[routeRow + K] > 0.0;
        for (var to = 0u; to < K; to++) {
          let route = routes[routeRow + to];
          if (route < 0.0) {
            continue;
          }
          let score = fromScore + transitionConstant - getScaledQuotient(abs(straight - route), beta, 1.0, 0);
          if (score > newScore[to]) {
            newScore[to] = score;
            newBack[to] = fromSlot;
            isReachable = true;
          }
        }
      }
    }
    for (var slot = 0u; slot < K; slot++) {
      if (getCandidateEdge(point, slot) == NONE) {
        newScore[slot] = NEG;
        newBack[slot] = NONE;
        continue;
      }
      let normalized = getScaledQuotient(getCandidateDistance(point, slot), sigma, 1.0, 0);
      let emission = emissionConstant - 0.5 * normalized * normalized;
      if (isReachable) {
        if (newScore[slot] > -1.0e29) {
          newScore[slot] += emission;
        }
      } else {
        // First point of the track or a break: restart from the emission alone. A break where a
        // route search dropped nodes is marked, since the missing route may be the budget's doing.
        newScore[slot] = emission;
        newBack[slot] = select(NONE, BUDGET_BREAK, isBudgetLimited);
      }
    }
    // Renormalize: subtract the best score of the step from every valid score. Every score of the
    // step moves by the same amount, so the best path is unchanged, and the magnitudes stay small
    // over any track length. The subtracted amount is kept for the log-likelihood.
    var maximum = NEG;
    for (var slot = 0u; slot < K; slot++) {
      if (newScore[slot] > -1.0e29) {
        maximum = max(maximum, newScore[slot]);
      }
    }
    let offset = select(0.0, maximum, maximum > -1.0e29);
    for (var slot = 0u; slot < K; slot++) {
      if (newScore[slot] > -1.0e29) {
        newScore[slot] -= offset;
      }
      scores[scoresOffset + point * SCORE_STRIDE + slot] = newScore[slot];
      backPointers[backPointersOffset + point * K + slot] = newBack[slot];
    }
    scores[scoresOffset + point * SCORE_STRIDE + K] = offset;
  }`;
}

/** Body of the backward pass: one thread per track traces the best path of every unbroken stretch. */
export function getBackwardBody(hasLogLikelihood: boolean, hasFixFilter: boolean): string {
  return `let first = trackOffsets[trackOffsetsOffset + index];
  let last = trackOffsets[trackOffsetsOffset + index + 1u];
  likelihoodSum = vec3<u32>(0u);
  ${
    hasLogLikelihood
      ? `// Choose a shared scale from the actual finite terms, excluding sentinel scores.
  var maximumExponent = 0u;
  for (var point = first; point < last; point++) {
    ${hasFixFilter ? 'if (predecessors[predecessorsOffset + point] == DROPPED_FIX) { continue; }' : ''}
    let row = scoresOffset + point * SCORE_STRIDE;
    maximumExponent = max(maximumExponent, getFixedPointExponentBits(scores[row + K]));
    for (var slot = 0u; slot < K; slot++) {
      let score = scores[row + slot];
      if (score > -1.0e29) { maximumExponent = max(maximumExponent, getFixedPointExponentBits(score)); }
    }
  }
  // At most two terms per uint32-indexed point: 33 bits of integer headroom cannot wrap.
  likelihoodUnitExponent = getFixedPointUnitExponent(maximumExponent, 33u);`
      : ''
  }
  var current = NONE;
  var isPicking = true;
  for (var step = 0u; step < last - first; step++) {
    let point = last - 1u - step;
    ${
      hasFixFilter
        ? `if (predecessors[predecessorsOffset + point] == DROPPED_FIX) {
      // Skipped: the walk continues with the slot chosen for the next retained fix.
      chosen[chosenOffset + point] = NONE;
      breaks[breaksOffset + point] = 0u;
      continue;
    }`
        : ''
    }
    ${
      hasLogLikelihood
        ? `// The scores of a stretch are renormalized per step: its log-likelihood is its final score
    // plus the amounts subtracted at its points.
    addToSum(scores[scoresOffset + point * SCORE_STRIDE + K]);`
        : ''
    }
    if (isPicking) {
      var bestScore = -1.0e29;
      current = NONE;
      for (var slot = 0u; slot < K; slot++) {
        let score = scores[scoresOffset + point * SCORE_STRIDE + slot];
        if (score > bestScore) {
          bestScore = score;
          current = slot;
        }
      }
      if (current != NONE) {
        ${hasLogLikelihood ? 'addToSum(bestScore);' : ''}
      }
    }
    chosen[chosenOffset + point] = current;
    var isBreak = current == NONE;
    var isBudgetBreak = false;
    isPicking = true;
    if (current != NONE) {
      let previous = backPointers[backPointersOffset + point * K + current];
      if (previous < K) {
        current = previous;
        isPicking = false;
      } else {
        isBreak = true;
        isBudgetBreak = previous == BUDGET_BREAK;
      }
    }
    breaks[breaksOffset + point] = select(0u, select(1u, 2u, isBudgetBreak), isBreak && point > first);
  }
  ${hasLogLikelihood ? 'logLikelihoods[logLikelihoodsOffset + index] = getSum();' : ''}`;
}

/**
 * A shared 96-bit fixed-point sum avoids both signed whole-part wrap and reassociated float
 * compensation on Metal. The per-track maximum exponent and 33 headroom bits cover at most
 * two terms per uint32-indexed point. Decode rounds once to float32; quantization error is at
 * most half a fixed-point unit per term (see FIXED_POINT_WGSL).
 */
export function getSumDeclarations(): string {
  return `${FIXED_POINT_WGSL}
var<private> likelihoodSum: vec3<u32> = vec3<u32>(0u);
var<private> likelihoodUnitExponent: i32 = 0;
fn addToSum(term: f32) {
  likelihoodSum = addFixedPoint(likelihoodSum, quantizeFixedPoint(term, likelihoodUnitExponent));
}
fn getSum() -> f32 {
  return getFixedPointFloat(likelihoodSum, likelihoodUnitExponent);
}`;
}

/**
 * Body of the hop kernel: one thread per point. Word 0 of a point's hop is 0 for a point with no
 * route to record (dropped or unmatched), 1 for the first matched point of a sub-track, 2 for a
 * point reached from the previous retained fix over the network and 3 for one reached along the
 * same edge. Words 1 to 3 of a kind 2 hop are the start node, the target node and the search
 * allowance (the `remaining` of the route kernel) and word 4 is the point's matched edge.
 */
export function getHopsBody(hasFixFilter: boolean): string {
  return `let base = hopsOffset + index * ${HOP_WORD_COUNT}u;
  hops[base] = 0u;
  let slot = chosen[chosenOffset + index];
  if (slot == NONE) {
    return;
  }
  let toEdge = getCandidateEdge(index, slot);
  hops[base + 4u] = toEdge;
  ${
    hasFixFilter
      ? `let previous = predecessors[predecessorsOffset + index];
  if (previous >= DROPPED_FIX || breaks[breaksOffset + index] != 0u) {`
      : `let previous = index - 1u;
  if (index == 0u || pointStarts[pointStartsOffset + index] != 0u || breaks[breaksOffset + index] != 0u) {`
  }
    hops[base] = 1u;
    return;
  }
  let fromSlot = chosen[chosenOffset + previous];
  let fromEdge = getCandidateEdge(previous, fromSlot);
  let fromFraction = getCandidateFraction(previous, fromSlot);
  if (toEdge == fromEdge && getCandidateFraction(index, slot) >= fromFraction) {
    hops[base] = 3u;
    return;
  }
  let straight = networkDistance(readPoint(index), readPoint(previous));
  let bound = straight * parameters[parametersOffset + 3u] + parameters[parametersOffset + 4u];
  let tail = (1.0 - fromFraction) * edgeLength(fromEdge);
  hops[base] = 2u;
  hops[base + 1u] = edgeTarget(fromEdge);
  hops[base + 2u] = edgeSource(toEdge);
  hops[base + 3u] = bitcast<u32>(bound - tail);`;
}
