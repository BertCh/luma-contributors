// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Integer polynomials preserve cancellation in determinants and in the profile derivative.
// Their coefficients encode the float64 rows exactly; no estimator arithmetic is f32.
type Polynomial = bigint[];
type RationalPolynomial = {numerator: Polynomial; denominator: Polynomial};

function trimPolynomial(values: Polynomial): Polynomial {
  while (values.length > 1 && values[values.length - 1] === 0n) values.pop();
  return values;
}

function addPolynomials(first: Polynomial, second: Polynomial, scale = 1n): Polynomial {
  return trimPolynomial(
    Array.from(
      {length: Math.max(first.length, second.length)},
      (_, index) => (first[index] ?? 0n) + scale * (second[index] ?? 0n)
    )
  );
}

function multiplyPolynomials(first: Polynomial, second: Polynomial): Polynomial {
  const result = new Array<bigint>(first.length + second.length - 1).fill(0n);
  for (let row = 0; row < first.length; row++) {
    if (first[row] === 0n) continue;
    for (let column = 0; column < second.length; column++)
      result[row + column] += first[row] * second[column];
  }
  return trimPolynomial(result);
}

function differentiatePolynomial(values: Polynomial): Polynomial {
  return values.length === 1
    ? [0n]
    : values.slice(1).map((value, index) => value * BigInt(index + 1));
}

function dividePolynomials(numerator: Polynomial, denominator: Polynomial): Polynomial {
  if (numerator.length === 1 && numerator[0] === 0n) return [0n];
  const remainder = numerator.slice();
  const quotient = new Array<bigint>(numerator.length - denominator.length + 1).fill(0n);
  for (let degree = quotient.length - 1; degree >= 0; degree--) {
    const leading = remainder[degree + denominator.length - 1];
    const divisor = denominator[denominator.length - 1];
    if (leading % divisor !== 0n) throw new Error('spatial error likelihood determinant division');
    const coefficient = leading / divisor;
    quotient[degree] = coefficient;
    for (let index = 0; index < denominator.length; index++)
      remainder[degree + index] -= coefficient * denominator[index];
  }
  if (remainder.some(value => value !== 0n))
    throw new Error('spatial error likelihood determinant remainder');
  return trimPolynomial(quotient);
}

function getGreatestCommonDivisor(first: bigint, second: bigint): bigint {
  let left = first < 0n ? -first : first;
  let right = second < 0n ? -second : second;
  while (right !== 0n) [left, right] = [right, left % right];
  return left;
}

function reducePolynomial(values: Polynomial): Polynomial {
  const divisor = values.reduce(getGreatestCommonDivisor, 0n);
  return divisor > 1n ? values.map(value => value / divisor) : values;
}

// A finite binary64 is an integer divided by a power of two, including subnormals.
function getDyadic(value: number): [bigint, bigint] {
  if (value === 0) return [0n, 1n];
  const storage = new DataView(new ArrayBuffer(8));
  storage.setFloat64(0, value);
  const bits = storage.getBigUint64(0);
  const exponent = Number((bits >> 52n) & 2047n);
  let numerator = (bits & ((1n << 52n) - 1n)) | (exponent === 0 ? 0n : 1n << 52n);
  if (bits >> 63n !== 0n) numerator = -numerator;
  const shift = (exponent === 0 ? -1022 : exponent - 1023) - 52;
  let denominator = 1n;
  if (shift >= 0) numerator <<= BigInt(shift);
  else denominator <<= BigInt(-shift);
  const divisor = getGreatestCommonDivisor(numerator, denominator);
  return [numerator / divisor, denominator / divisor];
}

function getIntegerCoefficients(values: readonly number[]): [Polynomial, bigint] {
  const dyadics = values.map(getDyadic);
  const denominator = dyadics.reduce(
    (largest, value) => (value[1] > largest ? value[1] : largest),
    1n
  );
  return [dyadics.map(([numerator, divisor]) => numerator * (denominator / divisor)), denominator];
}

function getDeterminant(matrix: Polynomial[][], reduce = true): Polynomial {
  const working = matrix.map(row => row.map(value => value.slice()));
  let previous: Polynomial = [1n];
  // Bareiss elimination: every division is exact, including polynomial cancellation.
  for (let pivot = 0; pivot + 1 < working.length; pivot++) {
    for (let row = pivot + 1; row < working.length; row++) {
      for (let column = pivot + 1; column < working.length; column++) {
        working[row][column] = dividePolynomials(
          addPolynomials(
            multiplyPolynomials(working[row][column], working[pivot][pivot]),
            multiplyPolynomials(working[row][pivot], working[pivot][column]),
            -1n
          ),
          previous
        );
      }
    }
    previous = working[pivot][pivot];
  }
  const determinant = working[working.length - 1][working.length - 1];
  return reduce ? reducePolynomial(determinant) : determinant;
}

function addRationalPolynomials(
  first: RationalPolynomial,
  second: RationalPolynomial
): RationalPolynomial {
  return {
    numerator: addPolynomials(
      multiplyPolynomials(first.numerator, second.denominator),
      multiplyPolynomials(second.numerator, first.denominator)
    ),
    denominator: multiplyPolynomials(first.denominator, second.denominator)
  };
}

function getLogDerivative(
  traces: readonly number[],
  eigenvalues: readonly number[] | undefined,
  tail:
    | {level: number; ratio: number; alternatingLevel: number; alternatingRatio: number}
    | undefined
): RationalPolynomial {
  let result: RationalPolynomial;
  if (eigenvalues) {
    result = {numerator: [0n], denominator: [1n]};
    const multiplicities = new Map<number, number>();
    for (const value of eigenvalues)
      multiplicities.set(value, (multiplicities.get(value) ?? 0) + 1);
    for (const [value, multiplicity] of multiplicities) {
      if (value === 0) continue;
      const [numerator, denominator] = getDyadic(value);
      result = addRationalPolynomials(result, {
        numerator: [-numerator * BigInt(multiplicity)],
        denominator: [denominator, -numerator]
      });
    }
  } else {
    const [coefficients, denominator] = getIntegerCoefficients(traces.map(value => -value));
    result = {numerator: trimPolynomial(coefficients), denominator: [denominator]};
    if (tail) {
      for (const [level, ratio] of [
        [tail.level, tail.ratio],
        [tail.alternatingLevel, tail.alternatingRatio]
      ]) {
        if (level === 0 || ratio === 0) continue;
        const [levelNumerator, levelDenominator] = getDyadic(level);
        const [ratioNumerator, ratioDenominator] = getDyadic(ratio);
        result = addRationalPolynomials(result, {
          numerator: [
            ...new Array<bigint>(traces.length).fill(0n),
            -levelNumerator * ratioNumerator
          ],
          denominator: [levelDenominator * ratioDenominator, -levelDenominator * ratioNumerator]
        });
      }
    }
  }
  return result;
}

function evaluatePolynomial(values: Polynomial, argument: number): bigint {
  const [numerator, denominator] = getDyadic(argument);
  let result = values[values.length - 1];
  let power = denominator;
  for (let degree = values.length - 2; degree >= 0; degree--) {
    result = result * numerator + values[degree] * power;
    power *= denominator;
  }
  return result;
}

function countSignChanges(values: Polynomial): number {
  let previous = 0n;
  let changes = 0;
  for (const value of values) {
    if (value === 0n) continue;
    if (previous !== 0n && value > 0n !== previous > 0n) changes++;
    previous = value;
  }
  return changes;
}

function getBinomials(degree: number): bigint[] {
  const values = [1n];
  for (let index = 1; index <= degree; index++)
    values.push((values[index - 1] * BigInt(degree + 1 - index)) / BigInt(index));
  return values;
}

function getBernsteinCoefficients(values: Polynomial, lower: number, upper: number): Polynomial {
  const degree = values.length - 1;
  const [lowerNumerator, lowerDenominator] = getDyadic(lower);
  const [upperNumerator, upperDenominator] = getDyadic(upper);
  const denominator = lowerDenominator > upperDenominator ? lowerDenominator : upperDenominator;
  const start = lowerNumerator * (denominator / lowerDenominator);
  const difference = upperNumerator * (denominator / upperDenominator) - start;
  let composed: Polynomial = [values[degree]];
  let power = denominator;
  for (let index = degree - 1; index >= 0; index--) {
    composed = multiplyPolynomials(composed, [start, difference]);
    composed[0] += values[index] * power;
    power *= denominator;
  }
  const binomials = getBinomials(degree);
  const common = binomials.reduce(
    (product, value) => (product / getGreatestCommonDivisor(product, value)) * value,
    1n
  );
  const result = new Array<bigint>(degree + 1).fill(0n);
  for (let index = 0; index <= degree; index++) {
    const coefficients = getBinomials(index);
    for (let inner = 0; inner <= index; inner++)
      result[index] += (composed[inner] ?? 0n) * coefficients[inner] * (common / binomials[inner]);
  }
  return reducePolynomial(result);
}

function splitBernsteinCoefficients(values: Polynomial): [Polynomial, Polynomial] {
  const degree = values.length - 1;
  const working = values.slice();
  const lower = [values[0] << BigInt(degree)];
  const upper = [values[degree] << BigInt(degree)];
  for (let level = 1; level <= degree; level++) {
    for (let index = 0; index <= degree - level; index++) working[index] += working[index + 1];
    lower.push(working[0] << BigInt(degree - level));
    upper.push(working[degree - level] << BigInt(degree - level));
  }
  return [reducePolynomial(lower), reducePolynomial(upper.reverse())];
}

function isExactMidpoint(lower: number, middle: number, upper: number): boolean {
  const [lowerNumerator, lowerDenominator] = getDyadic(lower);
  const [middleNumerator, middleDenominator] = getDyadic(middle);
  const [upperNumerator, upperDenominator] = getDyadic(upper);
  return (
    2n * middleNumerator * lowerDenominator * upperDenominator ===
    (lowerNumerator * upperDenominator + upperNumerator * lowerDenominator) * middleDenominator
  );
}

/**
 * Exact centered covariance certificate for original dyadic observations with an intercept
 * in column zero. Integer moments avoid rounding non-dyadic means or suppressing weak slopes.
 * @internal
 */
export function hasCPUSpatialErrorCenteredOrthogonality(
  design: readonly number[][],
  response: readonly number[]
): boolean {
  const width = design[0].length;
  const stride = width + 1;
  const [integers] = getIntegerCoefficients(
    design.flatMap((row, position) => [...row, response[position]])
  );
  const count = BigInt(response.length);
  let responseTotal = 0n;
  for (let row = 0; row < response.length; row++) responseTotal += integers[row * stride + width];
  // Column zero is the intercept. All moments use original dyadic observations, before
  // subtracting a mean such as -2/7 that has no exact binary representation.
  for (let column = 1; column < width; column++) {
    let predictorTotal = 0n;
    let crossProduct = 0n;
    for (let row = 0; row < response.length; row++) {
      const value = integers[row * stride + column];
      predictorTotal += value;
      crossProduct += value * integers[row * stride + width];
    }
    if (count * crossProduct !== predictorTotal * responseTotal) return false;
  }
  return true;
}

function getExactAugmentedGram(design: readonly number[][], response: readonly number[]) {
  const size = design[0].length + 1;
  const [integers, denominator] = getIntegerCoefficients(
    design.flatMap((row, position) => [...row, response[position]])
  );
  const gram = Array.from({length: size}, () => new Array<bigint>(size).fill(0n));
  for (let position = 0; position < design.length; position++) {
    for (let row = 0; row < size; row++) {
      for (let column = 0; column <= row; column++)
        gram[row][column] += integers[position * size + row] * integers[position * size + column];
    }
  }
  return {
    matrix: gram.map((row, position) =>
      row.map((value, column) => [column <= position ? value : gram[column][position]])
    ),
    denominator
  };
}

/** Exact uncentered zero-moment and augmented-Gram rank certificates. @internal */
export function getCPUSpatialErrorResponseRelation(
  design: readonly number[][],
  response: readonly number[]
): {orthogonal: boolean; exactFit: boolean} {
  const {matrix} = getExactAugmentedGram(design, response);
  const width = matrix.length - 1;
  const orthogonal = matrix[width].slice(0, width).every(entry => entry[0] === 0n);
  if (orthogonal) return {orthogonal, exactFit: matrix[width][width][0] === 0n};
  return {orthogonal, exactFit: getDeterminant(matrix)[0] === 0n};
}

/** Exact dyadic Gram determinants give RSS via the Schur complement. Call with full rank design. @internal */
export function getCPUDyadicLeastSquaresResidualSquares(
  design: readonly number[][],
  response: readonly number[]
): number {
  const {matrix, denominator} = getExactAugmentedGram(design, response);
  const numerator = getDeterminant(matrix, false)[0];
  if (numerator === 0n) return 0;
  const normalMatrix = matrix.slice(0, -1).map(row => row.slice(0, -1));
  const divisor = getDeterminant(normalMatrix, false)[0] * denominator ** 2n;
  // Convert only the leading bits so large determinants never become Infinity before division.
  const numeratorShift = Math.max(0, numerator.toString(2).length - 54);
  const divisorShift = Math.max(0, divisor.toString(2).length - 54);
  return (
    (Number(numerator >> BigInt(numeratorShift)) / Number(divisor >> BigInt(divisorShift))) *
    2 ** (numeratorShift - divisorShift)
  );
}

/**
 * Every maximum of the rational filtered least-squares profile, without a sampling grid.
 * Gram determinants give SSE=N/D. Its log derivative combined with the series/eigenvalue log
 * determinant has a polynomial numerator. Exact integer Bareiss determinants and Bernstein
 * sign variations isolate its roots; bisection ends at adjacent binary64 parameters. Repeated
 * roots remain together until binary64 resolution; only positive-to-negative crossings count.
 * The returned count includes eligible interval boundaries, in ascending parameter order.
 * @internal
 */
export function getCPUSpatialErrorLikelihoodCandidates(
  rows: readonly number[][],
  laggedRows: readonly number[][],
  traces: readonly number[],
  eigenvalues: readonly number[] | undefined,
  tail:
    | {level: number; ratio: number; alternatingLevel: number; alternatingRatio: number}
    | undefined,
  rowCount: number,
  bound: number
): {parameter: number; atBoundary: boolean}[] {
  const size = rows[0].length;
  const entries = [...rows.flat(), ...laggedRows.flat()];
  const [integers] = getIntegerCoefficients(entries);
  const matrix = Array.from({length: size}, () => Array.from({length: size}, () => [0n, 0n, 0n]));
  for (let position = 0; position < rows.length; position++) {
    for (let row = 0; row < size; row++) {
      const first = integers[position * size + row];
      const firstLag = integers[(rows.length + position) * size + row];
      for (let column = 0; column <= row; column++) {
        const second = integers[position * size + column];
        const secondLag = integers[(rows.length + position) * size + column];
        matrix[row][column][0] += first * second;
        matrix[row][column][1] -= first * secondLag + firstLag * second;
        matrix[row][column][2] += firstLag * secondLag;
      }
    }
  }
  for (let row = 0; row < size; row++) {
    for (let column = 0; column <= row; column++) {
      trimPolynomial(matrix[row][column]);
      matrix[column][row] = matrix[row][column];
    }
  }
  const numerator = getDeterminant(matrix);
  const denominator = getDeterminant(matrix.slice(0, -1).map(row => row.slice(0, -1)));
  const logDerivative = getLogDerivative(traces, eigenvalues, tail);
  const profileDerivative = addPolynomials(
    multiplyPolynomials(differentiatePolynomial(numerator), denominator),
    multiplyPolynomials(numerator, differentiatePolynomial(denominator)),
    -1n
  );
  const stationary = reducePolynomial(
    addPolynomials(
      multiplyPolynomials(multiplyPolynomials(numerator, denominator), logDerivative.numerator).map(
        value => 2n * value
      ),
      multiplyPolynomials(profileDerivative, logDerivative.denominator),
      -BigInt(rowCount)
    )
  );
  if (stationary.length === 1 && stationary[0] === 0n)
    return [{parameter: -bound, atBoundary: true}];
  const candidates: {parameter: number; atBoundary: boolean}[] = [];
  const initialCoefficients = getBernsteinCoefficients(stationary, -bound, bound);
  const lowerSlope = evaluatePolynomial(stationary, -bound);
  if (
    lowerSlope < 0n ||
    (lowerSlope === 0n && initialCoefficients.find(value => value !== 0n)! < 0n)
  )
    candidates.push({parameter: -bound, atBoundary: true});
  const stack = [{lower: -bound, upper: bound, coefficients: initialCoefficients}];
  while (stack.length) {
    const interval = stack.pop()!;
    const changes = countSignChanges(interval.coefficients);
    if (changes === 0) continue;
    let {lower, upper} = interval;
    if (changes === 1) {
      const firstSign = interval.coefficients.find(value => value !== 0n)!;
      const lastSign = interval.coefficients
        .slice()
        .reverse()
        .find(value => value !== 0n)!;
      if (!(firstSign > 0n && lastSign < 0n)) continue;
      while (true) {
        const middle = lower + (upper - lower) / 2;
        if (middle === lower || middle === upper) break;
        const value = evaluatePolynomial(stationary, middle);
        if (value === 0n) {
          lower = middle;
          upper = middle;
          break;
        }
        if (value > 0n) lower = middle;
        else upper = middle;
      }
      const parameter = lower + (upper - lower) / 2;
      candidates.push({parameter, atBoundary: parameter === -bound || parameter === bound});
      continue;
    }
    const middle = lower + (upper - lower) / 2;
    if (middle === lower || middle === upper) {
      if (evaluatePolynomial(stationary, lower) > 0n && evaluatePolynomial(stationary, upper) < 0n)
        candidates.push({parameter: middle, atBoundary: false});
      continue;
    }
    if (evaluatePolynomial(stationary, middle) === 0n) {
      let derivative = stationary;
      let order = 0;
      let value = 0n;
      while (value === 0n && derivative.length > 1) {
        derivative = differentiatePolynomial(derivative);
        order++;
        value = evaluatePolynomial(derivative, middle);
      }
      if (order % 2 === 1 && value < 0n) candidates.push({parameter: middle, atBoundary: false});
    }
    const [left, right] = isExactMidpoint(lower, middle, upper)
      ? splitBernsteinCoefficients(interval.coefficients)
      : [
          getBernsteinCoefficients(stationary, lower, middle),
          getBernsteinCoefficients(stationary, middle, upper)
        ];
    stack.push(
      {lower: middle, upper, coefficients: right},
      {lower, upper: middle, coefficients: left}
    );
  }
  const upperSlope = evaluatePolynomial(stationary, bound);
  if (upperSlope > 0n) candidates.push({parameter: bound, atBoundary: true});
  else if (
    upperSlope === 0n &&
    initialCoefficients
      .slice()
      .reverse()
      .find(value => value !== 0n)! > 0n
  )
    candidates.push({parameter: bound, atBoundary: true});
  return candidates.sort((first, second) => first.parameter - second.parameter);
}
