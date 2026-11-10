// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Experiment for STM-08 (GPUSpaceTimeHawkes): what do the float32 techniques of Ko, Suchard and
// Holbrook (2024) buy for a space-time Hawkes EM at one million events?
//
// Binary32 arithmetic is emulated with Math.fround after every operation (exp and log are
// correctly rounded here; a GPU's are within a few ulp, so real errors are somewhat larger).
// Three E-step variants are compared with binary64 on the same catalogue:
//   direct   plain float32 sum of the offspring terms, responsibilities as term / lambda;
//   logsum   the kernel's log-domain running-maximum accumulation (space-time-hawkes-wgsl.ts),
//            with the background floor at the smallest normal float32;
//   logsum1e-30  the same with a floor of 1e-30, the first choice, kept to show why it was moved;
//   (the sums over events are tiled with Neumaier compensation in both, as the kernel does).
// Then the reduction of the log-likelihood terms is compared three ways, and the effect of the
// floor on the background rate is shown.
//
// The binary64 reference runs on the float32-rounded inputs, so the printed errors are arithmetic
// only; the effect of rounding the inputs (long windows lose time resolution) is printed apart.
//
// Usage: node stm08-hawkes-float32.mjs [eventCount] [iterations]
// The catalogue is a branching process on the unit square with constant event density, so a
// million events need a long window but only about 20 parent candidates per event.

const targetEventCount = Number(process.argv[2] ?? 1000000);
const iterationCount = Number(process.argv[3] ?? 6);
const SMALLEST_NORMAL = 1.1754943508222875e-38;
// Float32 rounding with denormals flushed to zero, as WGSL implementations may do.
const f = value => {
  const rounded = Math.fround(value);
  return Math.abs(rounded) < SMALLEST_NORMAL ? 0 : rounded;
};

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const random = mulberry32(20261109);
const gaussian = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
function poisson(mean) {
  let count = 0;
  let total = -Math.log(1 - random());
  while (total < mean) {
    count++;
    total += -Math.log(1 - random());
  }
  return count;
}

const TRUTH = {mu: 2, omega: 0.5, beta: 2, sigma: 0.01};
const MAXIMUM_LAG = 5;
const MAXIMUM_DISTANCE = 0.06;
const MAXIMUM_PARENTS = 256;
// Expected events = mu * A * T / (1 - omega).
const DURATION = (targetEventCount * (1 - TRUTH.omega)) / TRUTH.mu;

function simulate() {
  const x = [];
  const y = [];
  const t = [];
  const backgroundCount = poisson(TRUTH.mu * DURATION);
  for (let i = 0; i < backgroundCount; i++) {
    x.push(random());
    y.push(random());
    t.push(random() * DURATION);
  }
  for (let cursor = 0; cursor < x.length; cursor++) {
    const children = poisson(TRUTH.omega);
    for (let c = 0; c < children; c++) {
      const time = t[cursor] - Math.log(1 - random()) / TRUTH.beta;
      const cx = x[cursor] + TRUTH.sigma * gaussian();
      const cy = y[cursor] + TRUTH.sigma * gaussian();
      if (time <= DURATION && cx >= 0 && cx <= 1 && cy >= 0 && cy <= 1) {
        x.push(cx);
        y.push(cy);
        t.push(time);
      }
    }
  }
  const order = Array.from(t.keys()).sort((a, b) => t[a] - t[b]);
  return {
    x: Float64Array.from(order, i => x[i]),
    y: Float64Array.from(order, i => y[i]),
    t: Float64Array.from(order, i => t[i])
  };
}

// First candidate parent of every event: the lag bound, and the capacity.
function getFirstParents(t) {
  const first = new Int32Array(t.length);
  let low = 0;
  for (let j = 0; j < t.length; j++) {
    while (t[low] < t[j] - MAXIMUM_LAG) low++;
    first[j] = Math.max(low, j - MAXIMUM_PARENTS);
  }
  return first;
}

// Sums over events in tiles of 244 events (plain float32 inside, Neumaier across), as the kernel.
function tiledSum(values, count, compensated = true) {
  const tile = 244;
  let sum = 0;
  let compensation = 0;
  for (let start = 0; start < count; start += tile) {
    let partial = 0;
    for (let i = start; i < Math.min(start + tile, count); i++) partial = f(partial + values[i]);
    if (!compensated) {
      sum = f(sum + partial);
      continue;
    }
    const total = f(sum + partial);
    compensation = f(
      compensation +
        (Math.abs(sum) >= Math.abs(partial) ? f(f(sum - total) + partial) : f(f(partial - total) + sum))
    );
    sum = total;
  }
  return sum + compensation;
}

// One E-step. Returns per-event background responsibility, the four sums of the M-step, and the
// per-event log intensity. `precision` is 'f64', 'direct' or 'logsum'.
function eStep(data, firstParents, theta, precision, distanceBound = MAXIMUM_DISTANCE, backgroundFloor = SMALLEST_NORMAL) {
  const count = data.t.length;
  const background = new Float64Array(count);
  const triggered = new Float64Array(count);
  const lag = new Float64Array(count);
  const distance = new Float64Array(count);
  const logIntensity = new Float64Array(count);
  const single = precision !== 'f64';
  const r = single ? f : value => value;
  const variance = r(theta.sigma * theta.sigma);
  const productivity = r(r(r(theta.omega * theta.beta) / r(r(2 * Math.PI) * variance)));
  const inverseTwoVariance = r(0.5 / variance);
  const logBackground = Math.log(Math.max(theta.mu, precision === 'logsum1e-30' ? 1e-30 : backgroundFloor));
  const logProductivity = r(
    r(Math.log(Math.max(theta.omega, 1e-30)) + Math.log(theta.beta)) -
      r(Math.log(r(r(2 * Math.PI) * variance)))
  );
  const maximumDistanceSquared = r(distanceBound * distanceBound);
  for (let j = 0; j < count; j++) {
    let intensity = r(theta.mu);
    let parentTotal = 0;
    let lagTotal = 0;
    let distanceTotal = 0;
    // logsum state
    let maximum = r(logBackground);
    let backgroundWeight = 1;
    let parentSum = 0;
    for (let i = firstParents[j]; i < j; i++) {
      const delay = r(data.t[j] - data.t[i]);
      const dx = r(data.x[j] - data.x[i]);
      const dy = r(data.y[j] - data.y[i]);
      const d2 = r(r(dx * dx) + r(dy * dy));
      if (!(delay > 0) || d2 > maximumDistanceSquared) continue;
      if (precision.startsWith('logsum')) {
        const term = r(r(logProductivity - r(theta.beta * delay)) - r(d2 * inverseTwoVariance));
        if (term > maximum) {
          const scale = r(Math.exp(r(maximum - term)));
          backgroundWeight = r(backgroundWeight * scale);
          parentSum = r(r(parentSum * scale) + 1);
          lagTotal = r(r(lagTotal * scale) + delay);
          distanceTotal = r(r(distanceTotal * scale) + d2);
          maximum = term;
        } else {
          const weight = r(Math.exp(r(term - maximum)));
          parentSum = r(parentSum + weight);
          lagTotal = r(lagTotal + r(weight * delay));
          distanceTotal = r(distanceTotal + r(weight * d2));
        }
      } else {
        const term = r(
          r(productivity * r(Math.exp(r(-theta.beta * delay)))) *
            r(Math.exp(r(-d2 * inverseTwoVariance)))
        );
        intensity = r(intensity + term);
        parentTotal = r(parentTotal + term);
        lagTotal = r(lagTotal + r(term * delay));
        distanceTotal = r(distanceTotal + r(term * d2));
      }
    }
    if (precision.startsWith('logsum')) {
      const sum = r(backgroundWeight + parentSum);
      background[j] = r(backgroundWeight / sum);
      triggered[j] = r(parentSum / sum);
      lag[j] = r(lagTotal / sum);
      distance[j] = r(distanceTotal / sum);
      logIntensity[j] = r(maximum + r(Math.log(sum)));
    } else {
      background[j] = r(r(theta.mu) / intensity);
      triggered[j] = r(parentTotal / intensity);
      lag[j] = r(lagTotal / intensity);
      distance[j] = r(distanceTotal / intensity);
      logIntensity[j] = r(Math.log(intensity));
    }
  }
  return {background, triggered, lag, distance, logIntensity};
}

function mStep(data, theta, e, compensated, area) {
  const count = data.t.length;
  const edge = new Float64Array(count);
  for (let j = 0; j < count; j++) {
    edge[j] = f(1 - f(Math.exp(f(-theta.beta * f(DURATION - data.t[j])))));
  }
  const sum = values => tiledSum(values, count, compensated);
  const triggered = sum(e.triggered);
  return {
    mu: sum(e.background) / (area * DURATION),
    omega: triggered / sum(edge),
    beta: triggered / sum(e.lag),
    sigma: Math.sqrt(sum(e.distance) / (2 * triggered)),
    logLikelihood: sum(e.logIntensity)
  };
}

function mStep64(data, theta, e, area) {
  const count = data.t.length;
  let background = 0;
  let triggered = 0;
  let lag = 0;
  let distance = 0;
  let edge = 0;
  let logSum = 0;
  for (let j = 0; j < count; j++) {
    background += e.background[j];
    triggered += e.triggered[j];
    lag += e.lag[j];
    distance += e.distance[j];
    edge += 1 - Math.exp(-theta.beta * (DURATION - data.t[j]));
    logSum += e.logIntensity[j];
  }
  return {
    mu: background / (area * DURATION),
    omega: triggered / edge,
    beta: triggered / lag,
    sigma: Math.sqrt(distance / (2 * triggered)),
    logLikelihood: logSum
  };
}

const rel = (a, b) => Math.abs(a - b) / Math.abs(b);
const NAMES = ['mu', 'omega', 'beta', 'sigma'];

console.log('simulating...');
const base = simulate();
const count = base.t.length;
console.log(`events ${count}, duration ${DURATION.toFixed(0)}`);

// The unit scale s multiplies positions; mu scales as s^-2, sigma and the distance bound as s.
// s = 1e17 to 1e19 put the background rate and the offspring terms at and below the smallest
// normal float32 (1.2e-38); they are not physical units, they probe the range where a plain sum underflows.
const START = {mu: TRUTH.mu / 2, omega: 0.3, beta: 1, sigma: 0.015};
for (const scale of [1, 1e12, 1e17, 1e18, 1e19]) {
  const data = {
    x: base.x.map(v => v * scale),
    y: base.y.map(v => v * scale),
    t: base.t
  };
  const firstParents = getFirstParents(data.t);
  // Positions enter a float32 kernel rounded.
  const rounded = {x: data.x.map(f), y: data.y.map(f), t: data.t.map(f)};
  const bound = MAXIMUM_DISTANCE * scale;
  console.log(`\nposition scale ${scale}`);
  let theta64 = {...START, mu: START.mu / scale ** 2, sigma: START.sigma * scale};
  const variants = {direct: {...theta64}, logsum: {...theta64}, 'logsum1e-30': {...theta64}};
  {
    // Input rounding alone: binary64 arithmetic on exact inputs against binary64 on float32 inputs.
    const exact = mStep64(data, theta64, eStep(data, firstParents, theta64, 'f64', bound), scale ** 2);
    const onRounded = mStep64(rounded, theta64, eStep(rounded, firstParents, theta64, 'f64', bound), scale ** 2);
    console.log(
      `  rounding the inputs to float32 alone moves one EM step by ${Math.max(...NAMES.map(n => rel(onRounded[n], exact[n]))).toExponential(2)} (relative, largest estimate)`
    );
  }
  const started = Date.now();
  for (let iteration = 1; iteration <= iterationCount; iteration++) {
    const e64 = eStep(rounded, firstParents, theta64, 'f64', bound);
    const next64 = mStep64(rounded, theta64, e64, scale ** 2);
    const row = [`  iteration ${iteration}`];
    for (const name of ['direct', 'logsum', 'logsum1e-30']) {
      const theta = variants[name];
      const e = eStep(rounded, firstParents, theta, name, bound);
      const next = mStep(rounded, theta, e, true, scale ** 2);
      // Error of this variant's sums against binary64 at the same theta (one E and M step).
      const reference = mStep64(rounded, theta, eStep(rounded, firstParents, theta, 'f64', bound), scale ** 2);
      row.push(
        `${name}: max rel error ${Math.max(...NAMES.map(n => rel(next[n], reference[n]))).toExponential(2)}`
      );
      variants[name] = {mu: next.mu, omega: next.omega, beta: next.beta, sigma: next.sigma};
    }
    theta64 = {mu: next64.mu, omega: next64.omega, beta: next64.beta, sigma: next64.sigma};
    console.log(`${row.join('   ')}   (${((Date.now() - started) / 1000).toFixed(0)} s)`);
  }
  console.log('  final estimates (relative to the binary64 iterate):');
  for (const name of ['direct', 'logsum', 'logsum1e-30']) {
    console.log(
      `    ${name}: ${NAMES.map(n => `${n} ${rel(variants[name][n], theta64[n]).toExponential(2)}`).join('  ')}`
    );
  }
  // Responsibility error at the converged-ish binary64 theta.
  const e64 = eStep(rounded, firstParents, theta64, 'f64', bound);
  for (const name of ['direct', 'logsum', 'logsum1e-30']) {
    const e = eStep(rounded, firstParents, theta64, name, bound);
    let worst = 0;
    let zeroCount = 0;
    for (let j = 0; j < count; j++) {
      worst = Math.max(worst, Math.abs(e.background[j] - e64.background[j]));
      if (!(e.logIntensity[j] > -Infinity) || Number.isNaN(e.logIntensity[j])) zeroCount++;
    }
    console.log(
      `    ${name}: max abs error of the background responsibility ${worst.toExponential(2)}, events with a non-finite log intensity ${zeroCount}`
    );
  }
}

// Reduction of the log intensity terms.
{
  const data = {x: base.x, y: base.y, t: base.t};
  const firstParents = getFirstParents(data.t);
  const e = eStep({x: data.x.map(f), y: data.y.map(f), t: data.t.map(f)}, firstParents, TRUTH.mu === 2 ? {mu: 2, omega: 0.5, beta: 2, sigma: 0.01} : TRUTH, 'logsum');
  let exact = 0;
  for (let j = 0; j < count; j++) exact += e.logIntensity[j];
  let sequential = 0;
  for (let j = 0; j < count; j++) sequential = f(sequential + e.logIntensity[j]);
  const tiledPlain = tiledSum(e.logIntensity, count, false);
  const tiledCompensated = tiledSum(e.logIntensity, count, true);
  console.log(`\nLog-likelihood reduction of ${count} terms (exact sum of the same float32 terms ${exact.toFixed(3)}):`);
  console.log(`  one float32 accumulator over all events: relative error ${rel(sequential, exact).toExponential(2)}`);
  console.log(`  tiles of 244, tile partials summed in float32: ${rel(tiledPlain, exact).toExponential(2)}`);
  console.log(`  tiles of 244, Neumaier over tiles (the kernel): ${rel(tiledCompensated, exact).toExponential(2)}`);
}

// The floor on the background rate: a start with mu = 0 (a log of zero).
{
  const few = 2000;
  const data = {x: base.x.slice(0, few), y: base.y.slice(0, few), t: base.t.slice(0, few)};
  const firstParents = getFirstParents(data.t);
  const theta = {mu: 0, omega: 0.5, beta: 2, sigma: 0.01};
  for (const [label, floorValue] of [['no floor', 0], ['floor 1e-30', 1e-30]]) {
    const logBackground = Math.log(Math.max(theta.mu, floorValue));
    let finiteCount = 0;
    for (let j = 0; j < few; j++) {
      // A background-only event (no parent inside the bounds) has log lambda = log mu.
      const hasParent = firstParents[j] < j && data.t[j] - data.t[j - 1] <= MAXIMUM_LAG;
      if (!hasParent && Number.isFinite(logBackground)) finiteCount++;
      if (hasParent) finiteCount++;
    }
    console.log(`\nmu = 0 start, ${label}: log mu = ${logBackground}; events with a finite log intensity ${finiteCount} of ${few}`);
  }
}
