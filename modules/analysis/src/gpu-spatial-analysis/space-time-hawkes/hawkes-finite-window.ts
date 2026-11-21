// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Smallest fitted beta*T: represents the finite-window zero-decay boundary. @internal */
export const HAWKES_MINIMUM_SCALED_DECAY = 1e-6;
/**
 * Relative rounding slack of the likelihood safeguard: an update is undone only when the fitted
 * log likelihood falls below the entering one by more than this fraction of the summed magnitudes
 * (log intensities, compensator, entering value). Shared by the CPU and GPU forms. @internal
 */
export const HAWKES_LIKELIHOOD_DECREASE_SLACK = 2 ** -16;
/** Newton refinements per GPU EM update; each refinement uses a tiled window reduction. @internal */
export const HAWKES_DECAY_REFINEMENTS = 8;

/** Integrals of u^k exp(-rate*u), k=0,1,2, over [0, remaining], in units of T. @internal */
export function getHawkesWindowIntegrals(
  rate: number,
  remaining: number
): [number, number, number] {
  const exponent = rate * remaining;
  if (exponent < 0.5) {
    const integrals: [number, number, number] = [0, 0, 0];
    let coefficient = 1;
    for (let term = 0; term < 12; term++) {
      for (let moment = 0; moment < 3; moment++)
        integrals[moment] += coefficient / (term + moment + 1);
      coefficient *= -exponent / (term + 1);
    }
    return [remaining * integrals[0], remaining ** 2 * integrals[1], remaining ** 3 * integrals[2]];
  }
  const exponential = Math.exp(-exponent);
  const zeroth = -Math.expm1(-exponent) / rate;
  const first = (zeroth - remaining * exponential) / rate;
  return [zeroth, first, (2 * first - remaining ** 2 * exponential) / rate];
}

/**
 * Maximize the finite-window profile Q(beta) with omega refit at every trial. The profile score
 * S E_w[u] - L/T decreases strictly (its slope is -S Var_w[u]), so its bracketed root is the unique
 * maximizer. The result is the last evaluated iterate, never a choice among iterates by
 * comparing Q values: near the root Q is flat to below its rounding, so such a choice would pick
 * a rate anywhere in the rounding plateau. The actual-likelihood safeguard of the callers guards
 * the step as a whole. @internal
 */
export function fitHawkesFiniteWindowDecay(
  remaining: number[],
  duration: number,
  triggered: number,
  lag: number,
  previousDecay: number,
  previousBranching: number,
  branchingFloor: number
): [number, number] {
  const integralsAt = (rate: number) => {
    const sums = [0, 0, 0];
    for (const time of remaining) {
      const integrals = getHawkesWindowIntegrals(rate, time / duration);
      for (let moment = 0; moment < 3; moment++) sums[moment] += integrals[moment];
    }
    return sums;
  };
  const scaledLag = lag / duration;
  let fittedRate = previousDecay * duration;
  let fittedBranching = previousBranching;
  let lower = HAWKES_MINIMUM_SCALED_DECAY;
  let upper = Math.max(lower, triggered / scaledLag);
  // Start at the entering decay, as the GPU does; the bracket keeps every step safe.
  let rate = Math.min(Math.max(fittedRate, lower), upper);
  for (let step = 0; step < 64; step++) {
    const [zeroth, first, second] = integralsAt(rate);
    if (!(zeroth > 0)) break;
    const mass = rate * zeroth;
    const branching = Math.max(triggered / mass, branchingFloor);
    fittedRate = rate;
    fittedBranching = branching;
    // The unconstrained profile score is S E[u] - lag/T, and its slope is -S Var[u].
    // With omega at its floor, differentiate the unprofiled compensator instead.
    const score = triggered / rate - scaledLag - branching * (zeroth - rate * first);
    const curvature = -triggered / (rate * rate) + branching * (2 * first - rate * second);
    // Avoid cancellation between S/rate and omega*K0 at the zero-decay boundary.
    const profileScore =
      branching > branchingFloor ? (triggered * first) / zeroth - scaledLag : score;
    const profileCurvature =
      branching > branchingFloor
        ? -triggered * Math.max(0, second / zeroth - (first / zeroth) ** 2)
        : curvature;
    // Both score terms are about L/T at the root, so the stopping test is relative to L/T.
    if (
      (rate === lower && lower === HAWKES_MINIMUM_SCALED_DECAY && profileScore <= 0) ||
      Math.abs(profileScore) <= 1e-12 * scaledLag
    )
      break;
    if (profileScore > 0) lower = rate;
    else upper = rate;
    const next = rate - profileScore / profileCurvature;
    // As on the GPU: a Newton step below the untested zero-decay boundary evaluates it.
    const boundary = profileScore < 0 && lower === HAWKES_MINIMUM_SCALED_DECAY && rate !== lower;
    rate =
      Number.isFinite(next) && next > lower && next < upper
        ? next
        : boundary
          ? lower
          : (lower + upper) / 2;
  }
  return [fittedRate / duration, fittedBranching];
}

/** Stable float32 integrals, using a fixed Taylor polynomial below z=.5. @internal */
export const HAWKES_WINDOW_INTEGRALS_WGSL = /* wgsl */ `
fn hawkesWindowIntegrals(rate: f32, remaining: f32) -> vec3f {
  let exponent = rate * remaining;
  if (exponent < 0.5) {
    var integrals = vec3f(0.0);
    var coefficient = 1.0;
    for (var term=0u; term<12u; term++) {
      integrals += coefficient / vec3f(f32(term+1u), f32(term+2u), f32(term+3u));
      coefficient *= -exponent / f32(term+1u);
    }
    return vec3f(remaining, remaining*remaining, remaining*remaining*remaining)*integrals;
  }
  let exponential = exp(-exponent);
  let zeroth = (1.0-exponential)/rate;
  let first = (zeroth-remaining*exponential)/rate;
  return vec3f(zeroth, first, (2.0*first-remaining*remaining*exponential)/rate);
}
`;
