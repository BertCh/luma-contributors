// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Binary64 reference for `GPUSinkhornDivergence` (STM-09), written from the definition: a dense
 * cost matrix between all cell centers (no separability), log-domain alternating Sinkhorn updates,
 * and the transport cost evaluated from the coupling `P`, `<P, C> + epsilon KL(P | a x b)`, not from
 * the dual potentials the kernel uses. POT 0.9.7.post1 (MIT) cross-checks the same definition in
 * `test/oracles/stm09-sinkhorn-pot.py`.
 */

/** A grid problem. */
export type SinkhornGridProblem = {
  width: number;
  height: number;
  cellWidth: number;
  cellHeight: number;
  /** Raster row order. */
  source: readonly number[];
  target: readonly number[];
  epsilon: number;
};

/** Entropic transport cost and its coupling statistics. */
export type SinkhornCost = {
  /** `<P, C> + epsilon KL(P | a x b)`. */
  cost: number;
  /** `<P, C>`. */
  transport: number;
  /** L1 distance of the coupling's marginals from the inputs. */
  marginalError: number;
  iterationCount: number;
};

function normalize(masses: readonly number[]): number[] {
  const total = masses.reduce((sum, value) => sum + (value > 0 ? value : 0), 0);
  return masses.map(value => (value > 0 ? value / total : 0));
}

function logSumExp(values: readonly number[]): number {
  const maximum = Math.max(...values);
  if (maximum === -Infinity) {
    return -Infinity;
  }
  return maximum + Math.log(values.reduce((sum, value) => sum + Math.exp(value - maximum), 0));
}

/** Dense squared-distance matrix between cell centers, row-major `[i * count + j]`. */
function getCostMatrix(problem: SinkhornGridProblem): Float64Array {
  const {width, height, cellWidth, cellHeight} = problem;
  const count = width * height;
  const cost = new Float64Array(count * count);
  for (let i = 0; i < count; i++) {
    for (let j = 0; j < count; j++) {
      const dx = ((i % width) - (j % width)) * cellWidth;
      const dy = (Math.floor(i / width) - Math.floor(j / width)) * cellHeight;
      cost[i * count + j] = dx * dx + dy * dy;
    }
  }
  return cost;
}

/**
 * Entropic optimal transport cost between normalized `first` and `second` on the grid, by
 * log-domain alternating updates until the L1 marginal error is below `tolerance`.
 */
export function computeSinkhornCost(
  problem: SinkhornGridProblem,
  first: readonly number[],
  second: readonly number[],
  cost: Float64Array,
  tolerance = 1e-12,
  maximumIterations = 200000
): SinkhornCost {
  const a = normalize(first);
  const b = normalize(second);
  const count = a.length;
  const {epsilon} = problem;
  const logA = a.map(value => (value > 0 ? Math.log(value) : -Infinity));
  const logB = b.map(value => (value > 0 ? Math.log(value) : -Infinity));
  let f = new Array<number>(count).fill(0);
  let g = new Array<number>(count).fill(0);
  let iterationCount = 0;
  let marginalError = Infinity;
  while (iterationCount < maximumIterations && marginalError > tolerance) {
    iterationCount++;
    g = g.map(
      (_, j) =>
        -epsilon *
        logSumExp(f.map((value, i) => value / epsilon + logA[i] - cost[i * count + j] / epsilon))
    );
    f = f.map(
      (_, i) =>
        -epsilon *
        logSumExp(g.map((value, j) => value / epsilon + logB[j] - cost[i * count + j] / epsilon))
    );
    // Row marginals are exact after the f update; measure the column marginals.
    marginalError = 0;
    for (let j = 0; j < count; j++) {
      let column = 0;
      for (let i = 0; i < count; i++) {
        column +=
          a[i] > 0 && b[j] > 0
            ? a[i] * b[j] * Math.exp((f[i] + g[j] - cost[i * count + j]) / epsilon)
            : 0;
      }
      marginalError += Math.abs(column - b[j]);
    }
  }
  // The coupling P_ij = a_i b_j exp((f_i + g_j - C_ij) / epsilon); cost from its definition.
  let transport = 0;
  let divergenceTerm = 0;
  for (let i = 0; i < count; i++) {
    for (let j = 0; j < count; j++) {
      if (!(a[i] > 0 && b[j] > 0)) {
        continue;
      }
      const ratio = Math.exp((f[i] + g[j] - cost[i * count + j]) / epsilon);
      const coupling = a[i] * b[j] * ratio;
      transport += coupling * cost[i * count + j];
      // KL(P | a x b) = sum P log(P / (a b)) = sum P (f_i + g_j - C_ij) / epsilon.
      divergenceTerm += coupling * Math.log(ratio);
    }
  }
  return {
    cost: transport + epsilon * divergenceTerm,
    transport,
    marginalError,
    iterationCount
  };
}

/** The three costs and the debiased divergence of a grid problem. */
export type SinkhornDivergenceReference = {
  crossCost: number;
  sourceCost: number;
  targetCost: number;
  divergence: number;
  iterationCount: number;
};

/** `S = OT(a, b) - OT(a, a) / 2 - OT(b, b) / 2`, all three in binary64. */
export function computeSinkhornDivergenceReference(
  problem: SinkhornGridProblem,
  tolerance = 1e-12
): SinkhornDivergenceReference {
  const cost = getCostMatrix(problem);
  const cross = computeSinkhornCost(problem, problem.source, problem.target, cost, tolerance);
  const source = computeSinkhornCost(problem, problem.source, problem.source, cost, tolerance);
  const target = computeSinkhornCost(problem, problem.target, problem.target, cost, tolerance);
  return {
    crossCost: cross.cost,
    sourceCost: source.cost,
    targetCost: target.cost,
    divergence: cross.cost - source.cost / 2 - target.cost / 2,
    iterationCount: cross.iterationCount
  };
}

/** A Gaussian blob of masses on the grid, centred at cell coordinates `(centerX, centerY)`. */
export function makeGaussianBlob(
  width: number,
  height: number,
  centerX: number,
  centerY: number,
  deviation: number,
  floor = 0
): number[] {
  return Array.from({length: width * height}, (_, cell) => {
    const dx = (cell % width) - centerX;
    const dy = Math.floor(cell / width) - centerY;
    const value = Math.exp(-(dx * dx + dy * dy) / (2 * deviation * deviation));
    return value < floor ? 0 : value;
  });
}
