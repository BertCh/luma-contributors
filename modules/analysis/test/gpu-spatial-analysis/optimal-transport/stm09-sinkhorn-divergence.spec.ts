// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import fixtureJson from '../../oracles/fixtures/stm09-sinkhorn-pot.json';
import {requireTestDevice} from '../../utils/require-test-device';
import {runSinkhornGPU} from './stm09-sinkhorn-harness';
import {
  computeSinkhornDivergenceReference,
  makeGaussianBlob,
  type SinkhornGridProblem
} from './stm09-sinkhorn-reference';

// Oracle: POT 0.9.7.post1 (MIT), test/oracles/stm09-sinkhorn-pot.py, the entropic cost taken from
// POT's coupling. A binary64 dense reference (stm09-sinkhorn-reference.ts) covers the cases the
// fixture does not.

type FixtureCase = SinkhornGridProblem & {
  name: string;
  maximumCostOverEpsilon: number;
  crossCost: number;
  sourceCost: number;
  targetCost: number;
  divergence: number;
  potPrimalOnlyDivergence: number;
};
const FIXTURE = decodeOracleFixture<{cases: FixtureCase[]}>(fixtureJson);

/** Float32 unit roundoff. */
const FLOAT32_EPSILON = 2 ** -24;
/**
 * Relative agreement (to the transport cost) of a converged float32 run with binary64, for grids
 * whose largest exponent `maximum cost / epsilon` is at most 200. Measured 3e-7 at most (about two
 * float32 roundings of the cost scale); 1e-6 leaves a factor of three.
 */
const CONVERGED_RELATIVE_TOLERANCE = 1e-6;
/**
 * Bound of the divergence of identical inputs and of its negativity, relative to `OT(a, b)`. The
 * marginal-error test cannot see changes below the rounding of the exponents, about
 * `FLOAT32_EPSILON * maximum cost / epsilon` (1.8e-5 at 150), and the divergence of identical
 * inputs is the difference of two iterations stopped at that level. Measured worst case 1e-5
 * with the default stopping tolerance and 400 iterations; the bound is `FLOAT32_EPSILON * 150 * 1.1`.
 */
const MAXIMUM_EXPONENT_RANGE = 150;
const ZERO_RELATIVE_TOLERANCE = 1.1 * FLOAT32_EPSILON * MAXIMUM_EXPONENT_RANGE;

function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/** Sparse random masses on a small grid, with an exponent range of at most 150. */
function makeRandomProblem(seed: number) {
  const random = createRandom(seed);
  const width = 5 + Math.floor(random() * 3);
  const height = 5 + Math.floor(random() * 3);
  const cell = 0.5 + random();
  const makeMasses = () =>
    Array.from({length: width * height}, () =>
      random() < 0.4 ? 0 : Math.fround(random() ** 3 * 10)
    );
  const source = makeMasses();
  const target = makeMasses();
  source[0] += 1;
  target[width * height - 1] += 1;
  const maximumCost = ((width - 1) * cell) ** 2 + ((height - 1) * cell * 0.8) ** 2;
  return {
    width,
    height,
    cellWidth: cell,
    cellHeight: cell * 0.8,
    source,
    target,
    epsilon: maximumCost / (30 + random() * 120)
  };
}

const RUN = {maximumIterations: 400};

it('GPUSinkhornDivergence matches POT on the fixture grids, with the components', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const entry of FIXTURE.cases) {
    const result = await runSinkhornGPU(device, entry, {maximumIterations: 400, tolerance: 1e-6});
    expect(result.converged, entry.name).toBe(1);
    expect(result.iterationLimitReached).toBe(0);
    expect(result.approximation).toBe(0);
    expect(result.invalidCount).toBe(0);
    const scale = Math.max(entry.crossCost, 1);
    expect(Math.abs(result.divergence - entry.divergence), entry.name).toBeLessThan(
      CONVERGED_RELATIVE_TOLERANCE * scale
    );
    [entry.crossCost, entry.sourceCost, entry.targetCost].forEach((expected, index) =>
      expect(
        Math.abs(result.components[index] - expected),
        `${entry.name} component ${index}`
      ).toBeLessThan(CONVERGED_RELATIVE_TOLERANCE * scale)
    );
    expect(result.marginalError).toBeLessThan(1e-5);
  }
  // Teeth: POT's own empirical_sinkhorn_divergence omits the entropic term and is a different
  // number; agreement with it would not distinguish the definitions.
  const blobs = FIXTURE.cases.find(entry => entry.name === 'blobs-8x8-epsilon-0.5')!;
  expect(Math.abs(blobs.potPrimalOnlyDivergence - blobs.divergence)).toBeGreaterThan(0.05);
}, 240000);

it('GPUSinkhornDivergence of two point masses is the squared distance, for any epsilon', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Closed form, no oracle: one coupling cell, KL(P | a x b) = 0, OT(a, a) = 0.
  for (const [cellWidth, cellHeight, epsilon] of [
    [1, 1, 2],
    [2.5, 0.5, 0.3],
    [1, 1, 50]
  ]) {
    const source = new Array(48).fill(0);
    const target = new Array(48).fill(0);
    source[1 * 8 + 1] = 5;
    target[4 * 8 + 6] = 0.25;
    const result = await runSinkhornGPU(device, {
      width: 8,
      height: 6,
      cellWidth,
      cellHeight,
      source,
      target,
      epsilon
    });
    const expected = (5 * cellWidth) ** 2 + (3 * cellHeight) ** 2;
    expect(result.divergence).toBeCloseTo(expected, 4);
    expect(result.components[1]).toBe(0);
    expect(result.components[2]).toBe(0);
  }
}, 120000);

it('GPUSinkhornDivergence is zero for identical inputs and not negative, within the float32 bound', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  let smallest = Infinity;
  for (let seed = 1; seed <= 10; seed++) {
    const problem = makeRandomProblem(seed);
    const identical = await runSinkhornGPU(device, {...problem, target: problem.source}, RUN);
    const different = await runSinkhornGPU(device, problem, RUN);
    const reversed = await runSinkhornGPU(
      device,
      {...problem, source: problem.target, target: problem.source},
      RUN
    );
    // Teeth: the plain entropic cost of identical inputs is far from zero (the entropic bias), so
    // the debiasing is what makes the divergence vanish; and distinct inputs are clearly positive.
    expect(identical.components[0]).toBeGreaterThan(0.1);
    expect(Math.abs(identical.divergence), `seed ${seed} identical`).toBeLessThan(
      ZERO_RELATIVE_TOLERANCE * identical.components[0]
    );
    expect(different.divergence, `seed ${seed} distinct`).toBeGreaterThan(
      1e-3 * different.components[0]
    );
    smallest = Math.min(smallest, identical.divergence / identical.components[0]);
    expect(
      Math.abs(reversed.divergence - different.divergence),
      `seed ${seed} symmetry`
    ).toBeLessThan(ZERO_RELATIVE_TOLERANCE * different.components[0]);
    // The components differ from the plain cost by the self terms.
    expect(different.components[1]).toBeGreaterThan(0);
  }
  expect(smallest).toBeGreaterThan(-ZERO_RELATIVE_TOLERANCE);
}, 240000);

it('GPUSinkhornDivergence matches the binary64 dense reference on random sparse masses', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const seed of [21, 22, 23]) {
    const problem = makeRandomProblem(seed);
    const reference = computeSinkhornDivergenceReference(problem, 1e-10);
    const result = await runSinkhornGPU(device, problem, {maximumIterations: 800, tolerance: 1e-5});
    expect(result.converged, `seed ${seed}`).toBe(1);
    expect(Math.abs(result.divergence - reference.divergence), `seed ${seed}`).toBeLessThan(
      CONVERGED_RELATIVE_TOLERANCE * reference.crossCost
    );
  }
}, 240000);

it('GPUSinkhornDivergence normalizes masses, counts invalid cells and refuses unusable input', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const problem = makeRandomProblem(5);
  const base = await runSinkhornGPU(device, problem, RUN);
  const scaled = await runSinkhornGPU(
    device,
    {
      ...problem,
      source: problem.source.map(value => value * 7.5),
      target: problem.target.map(value => value * 0.125)
    },
    RUN
  );
  expect(Math.abs(scaled.divergence - base.divergence)).toBeLessThan(2e-6 * base.components[0]);

  const dirty = {
    ...problem,
    source: problem.source.map((value, cell) => (cell === 3 ? Number.NaN : value)),
    target: problem.target.map((value, cell) => (cell === 4 ? -2 : value))
  };
  const zeroed = {
    ...problem,
    source: problem.source.map((value, cell) => (cell === 3 ? 0 : value)),
    target: problem.target.map((value, cell) => (cell === 4 ? 0 : value))
  };
  const withInvalid = await runSinkhornGPU(device, dirty, RUN);
  const reference = await runSinkhornGPU(device, zeroed, RUN);
  expect(withInvalid.invalidCount).toBe(2);
  expect(withInvalid.divergence).toBe(reference.divergence);

  const empty = await runSinkhornGPU(
    device,
    {...problem, target: problem.target.map(() => 0)},
    RUN
  );
  expect(empty.divergence).toBeNaN();
  expect(empty.converged).toBe(0);
  expect(empty.iterationLimitReached).toBe(0);
  expect(empty.iterationCount).toBe(0);
  const badParameters = await runSinkhornGPU(device, problem, {
    ...RUN,
    parameterOverride: {2: -1}
  });
  expect(badParameters.divergence).toBeNaN();
}, 120000);

it('GPUSinkhornDivergence reports the iteration budget and applies the epsilon schedule', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const problem = makeRandomProblem(7);
  const limited = await runSinkhornGPU(device, problem, {maximumIterations: 2});
  expect(limited.iterationCount).toBe(2);
  expect(limited.iterationLimitReached).toBe(1);
  expect(limited.converged).toBe(0);
  expect(Number.isFinite(limited.divergence)).toBe(true);
  // Teeth: two iterations are not converged, so they differ from the converged value.
  const converged = await runSinkhornGPU(device, problem, {
    maximumIterations: 800,
    tolerance: 1e-5
  });
  expect(converged.converged).toBe(1);
  expect(Math.abs(limited.divergence - converged.divergence)).toBeGreaterThan(
    1e-4 * converged.components[0]
  );
  // A schedule that starts at ten times epsilon reaches the same value.
  const scaled = await runSinkhornGPU(device, problem, {
    maximumIterations: 800,
    tolerance: 1e-5,
    epsilonStart: 10 * problem.epsilon,
    scalingFactor: 0.7
  });
  expect(scaled.converged).toBe(1);
  expect(Math.abs(scaled.divergence - converged.divergence)).toBeLessThan(
    CONVERGED_RELATIVE_TOLERANCE * converged.components[0]
  );
}, 240000);

it('GPUSinkhornDivergence with a short blur window flags the approximation and stays close when the kernel is negligible', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 16;
  const problem = {
    width,
    height: width,
    cellWidth: 1,
    cellHeight: 1,
    epsilon: 2,
    source: makeGaussianBlob(width, width, 4, 5, 1.5, 1e-6).map(Math.fround),
    target: makeGaussianBlob(width, width, 7, 6, 1.8, 1e-6).map(Math.fround)
  };
  const exact = await runSinkhornGPU(device, problem, {maximumIterations: 400, tolerance: 1e-6});
  const windowed = await runSinkhornGPU(device, problem, {
    maximumIterations: 400,
    tolerance: 1e-6,
    blurRadius: 12
  });
  expect(exact.approximation).toBe(0);
  expect(windowed.approximation).toBe(1);
  // exp(-12^2 / 2) is 5e-32 of the nearest term: the window drops nothing that float32 keeps.
  expect(Math.abs(windowed.divergence - exact.divergence)).toBeLessThan(2e-6 * exact.components[0]);
  // Teeth: a window of three cells drops terms the plan needs.
  const narrow = await runSinkhornGPU(device, problem, {
    maximumIterations: 400,
    tolerance: 1e-6,
    blurRadius: 3
  });
  expect(Math.abs(narrow.divergence - exact.divergence)).toBeGreaterThan(
    1e-3 * exact.components[0]
  );
}, 240000);

it('GPUSinkhornDivergence grows as a distribution moves away and is bitwise repeatable', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 14;
  const source = makeGaussianBlob(width, width, 3, 6, 1.5, 1e-6).map(Math.fround);
  let previous = -1;
  for (const centerX of [3, 5, 7, 9]) {
    const target = makeGaussianBlob(width, width, centerX, 6, 1.5, 1e-6).map(Math.fround);
    const result = await runSinkhornGPU(
      device,
      {
        width,
        height: width,
        cellWidth: 1,
        cellHeight: 1,
        epsilon: 3,
        source,
        target
      },
      RUN
    );
    expect(result.divergence).toBeGreaterThan(previous);
    previous = result.divergence;
    if (centerX === 9) {
      // Two same-shape blobs six cells apart: the divergence is close to the squared shift.
      expect(result.divergence).toBeGreaterThan(30);
      expect(result.divergence).toBeLessThan(40);
      const again = await runSinkhornGPU(
        device,
        {
          width,
          height: width,
          cellWidth: 1,
          cellHeight: 1,
          epsilon: 3,
          source,
          target
        },
        RUN
      );
      expect(again.divergence).toBe(result.divergence);
      expect(again.components).toEqual(result.components);
    }
  }
}, 240000);

it('GPUSinkhornDivergence sums a million cells to float32 accuracy (no compensated float sum to fold away)', async () => {
  // Math correctness review 2026-11-10, the note on compensated float sums: a Kahan or Neumaier
  // correction is algebraically zero and the test GPU's shader compiler removes it, leaving a
  // plain serial float32 sum. A serial float32 sum of a million equal terms is about 1 percent
  // off (0.1 added a million times gives 100958.34, not 100000). This checks the two sums of the
  // operator at that size: the mass totals that normalize the inputs, and the dual objective.
  //
  // Closed form, independent of the kernel: with epsilon far below the squared cell size the
  // Gibbs kernel exp(-C / epsilon) is the identity to float32 (exp(-40) = 4e-18), the optimal
  // plan of OT(a, a) is diag(a) with zero transport cost, and
  // OT(a, a) = epsilon KL(diag(a) | a x a) = -epsilon sum a_i log(a_i) = epsilon log(N)
  // for N equal masses. Every mass is 0.1, so the masses must first be divided by their sum.
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 1000;
  const cellCount = width * width;
  const masses = new Float32Array(cellCount).fill(0.1);
  const epsilon = 1 / 40;
  const result = await runSinkhornGPU(
    device,
    {width, height: width, cellWidth: 1, cellHeight: 1, epsilon, source: masses, target: masses},
    {blurRadius: 1, maximumIterations: 4}
  );
  expect(result.invalidCount).toBe(0);
  expect(result.converged).toBe(1);
  const expected = epsilon * Math.log(cellCount);
  for (const component of result.components) {
    expect(Math.abs(component / expected - 1)).toBeLessThan(1e-5);
  }
  expect(Math.abs(result.divergence)).toBeLessThan(1e-5 * expected);
  expect(result.marginalError).toBeLessThan(1e-5);
}, 240000);
