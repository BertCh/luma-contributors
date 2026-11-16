// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Binary64 reference for `GPUSpaceTimeHawkes` (STM-08), written from the definition in the class
 * documentation and not from the kernel: direct (not log-domain) sums, plain loops, one pass for
 * the intensity and a second for the responsibilities. The ETAS R package (0.7.2) is the oracle
 * the review names; it is GPL, R is not installed here, and it fits magnitude terms this model does
 * not have, so no ETAS fixture exists yet and this file is the reference. EXTERNAL FIXTURE PENDING.
 */

/** One event. */
export type HawkesEvent = {x: number; y: number; time: number};

/** Parameter vector of the model. */
export type HawkesParameters = {
  background: number;
  branchingRatio: number;
  decayRate: number;
  spatialStandardDeviation: number;
};

/** Observation window and parent bounds. */
export type HawkesWindow = {
  area: number;
  duration: number;
  maximumLag: number;
  maximumDistance: number;
  maximumParents: number;
};

/** Result of {@link fitHawkesReference}. */
export type HawkesReferenceFit = {
  estimates: HawkesParameters;
  iterationCount: number;
  converged: boolean;
  /** Log-likelihood at the parameters that entered the last iteration. */
  logLikelihood: number;
  /** Background probability per input event, at the final estimates. */
  backgroundProbability: number[];
  /** Events whose candidate window was cut at `maximumParents`. */
  cutCount: number;
};

/** Seeded generator (mulberry32). */
export function createHawkesRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function drawStandardNormal(random: () => number): number {
  const radius = Math.sqrt(-2 * Math.log(1 - random()));
  return radius * Math.cos(2 * Math.PI * random());
}

function drawPoisson(random: () => number, mean: number): number {
  // Sum of exponential gaps until the mean is passed; exact, O(mean).
  let count = 0;
  let total = -Math.log(1 - random());
  while (total < mean) {
    count++;
    total += -Math.log(1 - random());
  }
  return count;
}

/** A simulated catalogue with the generating label of every event. */
export type SimulatedHawkesCatalogue = {
  events: HawkesEvent[];
  /** True for an event drawn from the background, false for an offspring. */
  isBackground: boolean[];
};

/**
 * Simulates the branching process on the square `[0, side]^2` for `[0, duration]`: Poisson
 * background, then every event has Poisson(branching ratio) offspring with exponential lags and
 * Gaussian displacements; offspring outside the window are dropped. Events are returned in
 * generation order (not time order).
 */
export function simulateHawkes(
  truth: HawkesParameters,
  side: number,
  duration: number,
  seed: number
): SimulatedHawkesCatalogue {
  const random = createHawkesRandom(seed);
  const events: HawkesEvent[] = [];
  const isBackground: boolean[] = [];
  const backgroundCount = drawPoisson(random, truth.background * side * side * duration);
  for (let index = 0; index < backgroundCount; index++) {
    events.push({x: random() * side, y: random() * side, time: random() * duration});
    isBackground.push(true);
  }
  for (let cursor = 0; cursor < events.length; cursor++) {
    const parent = events[cursor];
    const childCount = drawPoisson(random, truth.branchingRatio);
    for (let child = 0; child < childCount; child++) {
      const time = parent.time - Math.log(1 - random()) / truth.decayRate;
      const x = parent.x + truth.spatialStandardDeviation * drawStandardNormal(random);
      const y = parent.y + truth.spatialStandardDeviation * drawStandardNormal(random);
      if (time <= duration && x >= 0 && x <= side && y >= 0 && y <= side) {
        events.push({x, y, time});
        isBackground.push(false);
      }
    }
  }
  return {events, isBackground};
}

type EventStatistics = {
  background: number;
  triggered: number;
  lag: number;
  distanceSquared: number;
  logIntensity: number;
  cut: boolean;
};

function evaluateEvent(
  sorted: readonly HawkesEvent[],
  row: number,
  parameters: HawkesParameters,
  window: HawkesWindow
): EventStatistics {
  const {background, branchingRatio, decayRate, spatialStandardDeviation} = parameters;
  const variance = spatialStandardDeviation ** 2;
  const event = sorted[row];
  let first = 0;
  while (first < row && sorted[first].time < event.time - window.maximumLag) {
    first++;
  }
  let parentEnd = row;
  while (parentEnd > first && sorted[parentEnd - 1].time === event.time) parentEnd--;
  let cut = false;
  if (parentEnd - first > window.maximumParents) {
    first = parentEnd - window.maximumParents;
    cut = true;
  }
  let intensity = background;
  let weightedLag = 0;
  let weightedDistance = 0;
  let parentTotal = 0;
  for (let parent = first; parent < parentEnd; parent++) {
    const lag = event.time - sorted[parent].time;
    const distanceSquared = (event.x - sorted[parent].x) ** 2 + (event.y - sorted[parent].y) ** 2;
    if (!(lag > 0) || distanceSquared > window.maximumDistance ** 2) {
      continue;
    }
    const term =
      ((branchingRatio * decayRate * Math.exp(-decayRate * lag)) / (2 * Math.PI * variance)) *
      Math.exp(-distanceSquared / (2 * variance));
    intensity += term;
    parentTotal += term;
    weightedLag += term * lag;
    weightedDistance += term * distanceSquared;
  }
  return {
    background: background / intensity,
    triggered: parentTotal / intensity,
    lag: weightedLag / intensity,
    distanceSquared: weightedDistance / intensity,
    logIntensity: Math.log(intensity),
    cut
  };
}

/**
 * Runs the EM of the class documentation for at most `maximumIterations` iterations or until the
 * largest relative change of an estimate is at most `tolerance`. Mirrors the kernel's published
 * semantics: `iterationCount` counts executed iterations, and `logLikelihood` is evaluated at the
 * parameters that entered the last executed iteration.
 */
export function fitHawkesReference(
  events: readonly HawkesEvent[],
  window: HawkesWindow,
  initial: HawkesParameters,
  options: {maximumIterations: number; tolerance: number}
): HawkesReferenceFit {
  const order = events
    .map((event, row) => ({event, row}))
    .filter(
      ({event}) => Number.isFinite(event.time) && event.time >= 0 && event.time <= window.duration
    )
    .sort((first, second) => first.event.time - second.event.time || first.row - second.row);
  const sorted = order.map(({event}) => event);
  let parameters = {...initial};
  let iterationCount = 0;
  let converged = false;
  let logLikelihood = Number.NaN;
  let cutCount = 0;
  while (iterationCount < options.maximumIterations && !converged) {
    iterationCount++;
    let backgroundSum = 0;
    let triggeredSum = 0;
    let lagSum = 0;
    let distanceSum = 0;
    let logSum = 0;
    let windowMass = 0;
    cutCount = 0;
    for (let row = 0; row < sorted.length; row++) {
      const statistics = evaluateEvent(sorted, row, parameters, window);
      backgroundSum += statistics.background;
      triggeredSum += statistics.triggered;
      lagSum += statistics.lag;
      distanceSum += statistics.distanceSquared;
      logSum += statistics.logIntensity;
      cutCount += statistics.cut ? 1 : 0;
      windowMass += 1 - Math.exp(-parameters.decayRate * (window.duration - sorted[row].time));
    }
    logLikelihood =
      logSum -
      (parameters.background * window.area * window.duration +
        parameters.branchingRatio * windowMass);
    const next: HawkesParameters = {
      background: backgroundSum / (window.area * window.duration),
      branchingRatio: triggeredSum / windowMass,
      decayRate: parameters.decayRate,
      spatialStandardDeviation: parameters.spatialStandardDeviation
    };
    if (triggeredSum > 1e-6 && lagSum > 0 && distanceSum > 0) {
      next.decayRate = triggeredSum / lagSum;
      next.spatialStandardDeviation = Math.sqrt(distanceSum / (2 * triggeredSum));
    }
    const change = Math.max(
      ...(['background', 'branchingRatio', 'decayRate', 'spatialStandardDeviation'] as const).map(
        name => Math.abs(next[name] - parameters[name]) / Math.abs(parameters[name])
      )
    );
    parameters = next;
    converged = change <= options.tolerance;
  }
  const backgroundProbability = new Array<number>(events.length).fill(Number.NaN);
  sorted.forEach((_, row) => {
    backgroundProbability[order[row].row] = evaluateEvent(
      sorted,
      row,
      parameters,
      window
    ).background;
  });
  return {
    estimates: parameters,
    iterationCount,
    converged,
    logLikelihood,
    backgroundProbability,
    cutCount
  };
}

/** Sampling distribution of the estimator for a configuration, from replicate simulations. */
export type HawkesSamplingDistribution = {
  replicateCount: number;
  mean: HawkesParameters;
  standardDeviation: HawkesParameters;
};

/**
 * Fits `replicateCount` independent simulations of `truth` with the reference and returns the mean
 * and the standard deviation of every estimate. The spread is the sampling variability a recovery
 * tolerance is derived from; the mean minus the truth is the estimator's bias at this
 * configuration (the unmodelled spatial edge and the finite parent bounds).
 */
export function measureHawkesSamplingDistribution(
  truth: HawkesParameters,
  side: number,
  window: HawkesWindow,
  initial: HawkesParameters,
  replicateCount: number,
  seedBase: number,
  options: {maximumIterations: number; tolerance: number}
): HawkesSamplingDistribution {
  const names = ['background', 'branchingRatio', 'decayRate', 'spatialStandardDeviation'] as const;
  const samples = names.map(() => [] as number[]);
  for (let replicate = 0; replicate < replicateCount; replicate++) {
    const {events} = simulateHawkes(truth, side, window.duration, seedBase + replicate);
    const fit = fitHawkesReference(events, window, initial, options);
    names.forEach((name, index) => samples[index].push(fit.estimates[name]));
  }
  const mean = {} as HawkesParameters;
  const standardDeviation = {} as HawkesParameters;
  names.forEach((name, index) => {
    const average = samples[index].reduce((sum, value) => sum + value, 0) / replicateCount;
    mean[name] = average;
    standardDeviation[name] = Math.sqrt(
      samples[index].reduce((sum, value) => sum + (value - average) ** 2, 0) / (replicateCount - 1)
    );
  });
  return {replicateCount, mean, standardDeviation};
}
