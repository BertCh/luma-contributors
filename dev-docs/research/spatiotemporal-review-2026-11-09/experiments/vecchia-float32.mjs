// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Experiment: is a Vecchia (nearest-neighbour Gaussian process) log-likelihood usable in float32?
// Emulates binary32 arithmetic with Math.fround after every operation and compares against binary64.
// Usage: node vecchia-float32.mjs [n] [m]

const n = Number(process.argv[2] ?? 10000);
const m = Number(process.argv[3] ?? 30);

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
function gaussian() {
  const u = Math.max(random(), 1e-300);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
}

// Points in a 100 km square, centred on the origin, metres. Random order is the Vecchia ordering.
const SIDE = 100000;
const xs = new Float64Array(n);
const ys = new Float64Array(n);
for (let i = 0; i < n; i++) {
  xs[i] = (random() - 0.5) * SIDE;
  ys[i] = (random() - 0.5) * SIDE;
}

// m nearest previous neighbours, brute force.
const neighbours = new Int32Array(n * m).fill(-1);
const counts = new Int32Array(n);
{
  const best = new Float64Array(m);
  const bestIndex = new Int32Array(m);
  for (let i = 1; i < n; i++) {
    let count = 0;
    for (let j = 0; j < i; j++) {
      const dx = xs[i] - xs[j];
      const dy = ys[i] - ys[j];
      const d = dx * dx + dy * dy;
      if (count < m) {
        let k = count++;
        while (k > 0 && best[k - 1] > d) {
          best[k] = best[k - 1];
          bestIndex[k] = bestIndex[k - 1];
          k--;
        }
        best[k] = d;
        bestIndex[k] = j;
      } else if (d < best[m - 1]) {
        let k = m - 1;
        while (k > 0 && best[k - 1] > d) {
          best[k] = best[k - 1];
          bestIndex[k] = bestIndex[k - 1];
          k--;
        }
        best[k] = d;
        bestIndex[k] = j;
      }
    }
    counts[i] = count;
    for (let k = 0; k < count; k++) neighbours[i * m + k] = bestIndex[k];
  }
}

const identity = v => v;
const f = Math.fround;

/** Correlation as a function of distance over range, each step rounded by `r`. */
const KERNELS = {
  exponential: (h, r) => r(Math.exp(-h)),
  matern32: (h, r) => {
    const s = r(1.7320508075688772 * h);
    return r(r(1 + s) * r(Math.exp(-s)));
  },
  matern52: (h, r) => {
    const s = r(2.23606797749979 * h);
    return r(r(r(1 + s) + r(r(s * s) / 3)) * r(Math.exp(-s)));
  },
  gaussian: (h, r) => r(Math.exp(-r(h * h)))
};

/**
 * Per-site conditional mean and variance through a Cholesky of the (k+1) x (k+1) covariance with the
 * site last. `r` rounds every operation. Returns {failures, terms(z, logVariance)} through a callback.
 */
function forEachSite(kernel, range, nugget, values, r, visit) {
  const size = m + 1;
  const a = new Float64Array(size * size);
  const b = new Float64Array(size);
  const px = new Float64Array(size);
  const py = new Float64Array(size);
  let failures = 0;
  for (let i = 0; i < n; i++) {
    const k = counts[i];
    const dim = k + 1;
    for (let q = 0; q < k; q++) {
      const j = neighbours[i * m + q];
      px[q] = r(xs[j]);
      py[q] = r(ys[j]);
      b[q] = values ? r(values[j]) : 0;
    }
    px[k] = r(xs[i]);
    py[k] = r(ys[i]);
    b[k] = values ? r(values[i]) : 0;
    for (let row = 0; row < dim; row++) {
      for (let column = 0; column <= row; column++) {
        if (row === column) {
          a[row * size + column] = r(1 + nugget);
        } else {
          const dx = r(px[row] - px[column]);
          const dy = r(py[row] - py[column]);
          const h = r(r(Math.sqrt(r(r(dx * dx) + r(dy * dy)))) / range);
          a[row * size + column] = kernel(h, r);
        }
      }
    }
    let ok = true;
    for (let column = 0; column < dim && ok; column++) {
      let diagonal = a[column * size + column];
      for (let q = 0; q < column; q++) {
        const value = a[column * size + q];
        diagonal = r(diagonal - r(value * value));
      }
      if (!(diagonal > 0)) {
        ok = false;
        break;
      }
      const pivot = r(Math.sqrt(diagonal));
      a[column * size + column] = pivot;
      for (let row = column + 1; row < dim; row++) {
        let value = a[row * size + column];
        for (let q = 0; q < column; q++) {
          value = r(value - r(a[row * size + q] * a[column * size + q]));
        }
        a[row * size + column] = r(value / pivot);
      }
    }
    if (!ok) {
      failures++;
      visit(i, NaN, NaN, a, size, k);
      continue;
    }
    // Forward solve L z = b; only the last element is needed for the likelihood term.
    for (let row = 0; row < dim; row++) {
      let value = b[row];
      for (let q = 0; q < row; q++) value = r(value - r(a[row * size + q] * b[q]));
      b[row] = r(value / a[row * size + row]);
    }
    visit(i, b[k], a[k * size + k], a, size, k);
  }
  return failures;
}

/** Sequential simulation from the Vecchia model in binary64. */
function simulate(kernel, range, nugget) {
  const values = new Float64Array(n);
  const size = m + 1;
  const b = new Float64Array(size);
  forEachSiteSimulate(kernel, range, nugget, values, size, b);
  return values;
}
function forEachSiteSimulate(kernel, range, nugget, values, size, b) {
  const a = new Float64Array(size * size);
  for (let i = 0; i < n; i++) {
    const k = counts[i];
    const dim = k + 1;
    const index = q => (q < k ? neighbours[i * m + q] : i);
    for (let row = 0; row < dim; row++) {
      for (let column = 0; column <= row; column++) {
        if (row === column) a[row * size + column] = 1 + nugget;
        else {
          const dx = xs[index(row)] - xs[index(column)];
          const dy = ys[index(row)] - ys[index(column)];
          a[row * size + column] = kernel(Math.sqrt(dx * dx + dy * dy) / range, identity);
        }
      }
    }
    for (let column = 0; column < dim; column++) {
      let diagonal = a[column * size + column];
      for (let q = 0; q < column; q++) diagonal -= a[column * size + q] ** 2;
      const pivot = Math.sqrt(Math.max(diagonal, 1e-300));
      a[column * size + column] = pivot;
      for (let row = column + 1; row < dim; row++) {
        let value = a[row * size + column];
        for (let q = 0; q < column; q++) value -= a[row * size + q] * a[column * size + q];
        a[row * size + column] = value / pivot;
      }
    }
    // y_i = sum_q L[k][q] z_q + L[k][k] e, with z = L^-1 y_neighbours.
    let mean = 0;
    for (let row = 0; row < k; row++) {
      let value = values[index(row)];
      for (let q = 0; q < row; q++) value -= a[row * size + q] * b[q];
      b[row] = value / a[row * size + row];
      mean += a[k * size + row] * b[row];
    }
    values[i] = mean + a[k * size + k] * gaussian();
  }
}

const LOG_TWO_PI = Math.log(2 * Math.PI);

function logLikelihood(kernel, range, nugget, values, mode) {
  const r = mode === 'f64' ? identity : f;
  let sum64 = 0;
  let sum32 = 0;
  const failures = forEachSite(kernel, r(range), r(nugget), values, r, (i, z, pivot) => {
    if (Number.isNaN(z)) return;
    const term = r(-0.5 * r(r(LOG_TWO_PI + r(2 * r(Math.log(pivot)))) + r(z * z)));
    sum64 += term;
    sum32 = f(sum32 + term);
  });
  return {value: mode === 'f32-naive-sum' ? sum32 : sum64, failures};
}

const TRUE_RANGE = 10000; // metres
const rangeGrid = [0.5, 0.7, 0.85, 0.95, 1, 1.05, 1.15, 1.3, 1.5, 2].map(v => v * TRUE_RANGE);
const rows = [];
for (const [name, kernel] of Object.entries(KERNELS)) {
  for (const nugget of [0, 1e-4, 1e-3, 1e-2, 1e-1]) {
    const values = simulate(kernel, TRUE_RANGE, nugget === 0 ? 0 : nugget);
    const curve64 = [];
    const curve32 = [];
    const curveNaive = [];
    let failures = 0;
    for (const range of rangeGrid) {
      const exact = logLikelihood(kernel, range, nugget, values, 'f64');
      const single = logLikelihood(kernel, range, nugget, values, 'f32');
      const naive = logLikelihood(kernel, range, nugget, values, 'f32-naive-sum');
      curve64.push(exact.value);
      curve32.push(single.value);
      curveNaive.push(naive.value);
      failures = Math.max(failures, single.failures);
      if (exact.failures) failures = Math.max(failures, -exact.failures);
    }
    const argmax = curve => curve.indexOf(Math.max(...curve.filter(Number.isFinite)));
    const maximumError = Math.max(...curve64.map((v, q) => Math.abs(v - curve32[q])));
    const maximumNaiveError = Math.max(...curve64.map((v, q) => Math.abs(v - curveNaive[q])));
    // Error in likelihood differences relative to the best grid point: what an optimizer sees.
    const best = argmax(curve64);
    const differenceError = Math.max(
      ...curve64.map((v, q) => Math.abs(v - curve64[best] - (curve32[q] - curve32[best])))
    );
    const neighbourGap = Math.min(
      ...curve64.map((v, q) => (q === best ? Infinity : curve64[best] - v))
    );
    rows.push({
      kernel: name,
      nugget,
      failures,
      'argmax f64': rangeGrid[best] / TRUE_RANGE,
      'argmax f32': rangeGrid[argmax(curve32)] / TRUE_RANGE,
      '|ll32-ll64| max': Number(maximumError.toPrecision(3)),
      'naive f32 sum err': Number(maximumNaiveError.toPrecision(3)),
      'diff err': Number(differenceError.toPrecision(3)),
      'gap to next grid pt': Number(neighbourGap.toPrecision(3)),
      'll64 at best': Number(curve64[best].toFixed(1))
    });
  }
}
console.log(`n=${n} m=${m} side=${SIDE} m range=${TRUE_RANGE} m`);
console.table(rows);
