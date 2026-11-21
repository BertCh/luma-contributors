// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getSymmetricEigenWGSL} from '../../src/analysis-core/kernels/symmetric-eigen-wgsl';

// CORE-106: a Jacobi rotation formed theta = (a_uu - a_ll) / (2 a_lu). With diagonals of opposite
// sign near the float32 limit the difference overflowed, the tangent became 0 and the coupling was
// cleared without a rotation; with a coupling above 1.7e38 twice the coupling overflowed and theta
// became 0 (a 45 degree rotation). A first fix gated on isFiniteValue(difference) did not hold on
// Metal (fast math); the kernel now gates on the input magnitude (above 2^126) and scales the
// operands by a runtime power of two. This is a float32 transcription (Math.fround at every f32
// operation) of one rotation of getSymmetricEigen_N on a 2 by 2 matrix, which a single rotation
// diagonalizes; the wgsl_reflect interpreter does not run this module (struct array members
// through pointers). Reference eigenvalues: numpy.linalg.eigh in float64 of the float32 entries
// (/usr/bin/python3): [[-3e38, 1e37], [1e37, 3e38]] -> +-3.0016662094333818e38;
// [[0, 2e38], [2e38, 1e38]] -> -1.5615527628837527e38, 2.561552730912322e38.
// The headless twin, on the real kernel, is symmetric-eigen-wgsl.spec.ts.

const toFloat32 = Math.fround;

/** One rotation; `scaled` is the CORE-106 form. Returns sorted diagonals and the vectors. */
function rotate(entries: readonly [number, number, number], scaled: boolean) {
  const [lowerDiagonal, coupling, upperDiagonal] = entries.map(toFloat32);
  const magnitude = Math.max(Math.abs(upperDiagonal), Math.abs(lowerDiagonal), Math.abs(coupling));
  let theta: number;
  if (!scaled || magnitude <= 2 ** 126) {
    theta = toFloat32(toFloat32(upperDiagonal - lowerDiagonal) / toFloat32(2 * coupling));
  } else {
    // frexp exponent: magnitude = fraction * 2^exponent with fraction in [0.5, 1).
    const exponent = Math.floor(Math.log2(magnitude)) + 1;
    const scale = 2 ** -exponent;
    const scaledCoupling = toFloat32(coupling * scale);
    theta =
      scaledCoupling === 0
        ? 1e30
        : toFloat32(
            toFloat32(toFloat32(upperDiagonal * scale) - toFloat32(lowerDiagonal * scale)) /
              toFloat32(2 * scaledCoupling)
          );
  }
  const direction = theta >= 0 ? 1 : -1;
  const tangent =
    Math.abs(theta) > 1e18
      ? toFloat32(0.5 / theta)
      : toFloat32(
          direction /
            toFloat32(
              Math.abs(theta) + toFloat32(Math.sqrt(toFloat32(toFloat32(theta * theta) + 1)))
            )
        );
  const cosine = toFloat32(1 / toFloat32(Math.sqrt(toFloat32(toFloat32(tangent * tangent) + 1))));
  const sine = toFloat32(tangent * cosine);
  const matrix = [
    [lowerDiagonal, coupling],
    [coupling, upperDiagonal]
  ];
  const rotatePair = (first: number, second: number): [number, number] => [
    toFloat32(toFloat32(cosine * first) - toFloat32(sine * second)),
    toFloat32(toFloat32(sine * first) + toFloat32(cosine * second))
  ];
  for (let k = 0; k < 2; k++) {
    [matrix[k][0], matrix[k][1]] = rotatePair(matrix[k][0], matrix[k][1]);
  }
  for (let k = 0; k < 2; k++) {
    [matrix[0][k], matrix[1][k]] = rotatePair(matrix[0][k], matrix[1][k]);
  }
  // Columns of the rotation: (cosine, -sine) for the lower diagonal, (sine, cosine) for the upper.
  const pairs = [
    {value: matrix[0][0], vector: [cosine, -sine]},
    {value: matrix[1][1], vector: [sine, cosine]}
  ].sort((left, right) => left.value - right.value);
  return {theta, tangent, pairs};
}

function getRelativeResidual(
  entries: readonly [number, number, number],
  value: number,
  vector: number[]
): number {
  const [first, cross, second] = entries.map(toFloat32);
  const [x, y] = vector;
  return (
    Math.hypot(first * x + cross * y - value * x, cross * x + second * y - value * y) /
    Math.abs(value)
  );
}

const CASES: {entries: [number, number, number]; values: [number, number]}[] = [
  {entries: [-3e38, 1e37, 3e38], values: [-3.0016662094333818e38, 3.0016662094333818e38]},
  {entries: [0, 2e38, 1e38], values: [-1.5615527628837527e38, 2.561552730912322e38]}
];

it('CORE-106 the previous rotation lost the coupling when its operands overflowed', () => {
  const [opposite, wideCoupling] = CASES.map(({entries}) => rotate(entries, false));
  expect(opposite.theta).toBe(Infinity);
  expect(opposite.tangent).toBe(0);
  expect(getRelativeResidual(CASES[0].entries, ...unpack(opposite.pairs[1]))).toBeGreaterThan(0.03);
  expect(wideCoupling.theta).toBe(0);
  expect(wideCoupling.tangent).toBe(1);
});

it('CORE-106 scaled operands give the eigenpairs when the difference or coupling overflows', () => {
  for (const {entries, values} of CASES) {
    const {pairs} = rotate(entries, true);
    pairs.forEach(({value, vector}, index) => {
      expect(Number.isFinite(value)).toBe(true);
      expect(Math.abs(value - values[index])).toBeLessThanOrEqual(1e-6 * Math.abs(values[1]));
      expect(getRelativeResidual(entries, value, vector)).toBeLessThan(1e-6);
    });
  }
  // Ordinary operands keep the previous arithmetic bit for bit.
  const ordinary: [number, number, number] = [1, 0.5, 3];
  expect(rotate(ordinary, true)).toEqual(rotate(ordinary, false));
});

it('CORE-106 the kernel source forms theta as the transcription does', () => {
  const source = getSymmetricEigenWGSL(2).replace(/\s+/g, ' ');
  expect(source).toContain('if (magnitude <= 0x1p126) {');
  expect(source).toContain('theta = (upperDiagonal - lowerDiagonal) / (2.0 * coupling);');
  expect(source).toContain('let scaledCoupling = ldexp(coupling, -exponent);');
  expect(source).toContain(
    'theta = (ldexp(upperDiagonal, -exponent) - ldexp(lowerDiagonal, -exponent)) / (2.0 * scaledCoupling);'
  );
});

function unpack(pair: {value: number; vector: number[]}): [number, number[]] {
  return [pair.value, pair.vector];
}
