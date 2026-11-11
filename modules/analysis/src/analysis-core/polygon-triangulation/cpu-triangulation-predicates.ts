// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Exact integer on the common binary64 scale 2^-1074. */
function getCoordinateInteger(value: number): bigint {
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, value);
  const bits = view.getBigUint64(0);
  const exponent = Number((bits >> 52n) & 0x7ffn);
  const fraction = bits & 0xfffffffffffffn;
  const magnitude = exponent === 0 ? fraction : (fraction | (1n << 52n)) << BigInt(exponent - 1);
  return bits >> 63n ? -magnitude : magnitude;
}

/** Float64-filtered exact orientation for triangulation; 2 denotes nonfinite input. @internal */
export function computeTriangulationOrientation(
  first: readonly number[],
  second: readonly number[],
  third: readonly number[]
): number {
  if (![...first, ...second, ...third].every(Number.isFinite)) return 2;
  const positive = (second[0] - first[0]) * (third[1] - first[1]);
  const negative = (second[1] - first[1]) * (third[0] - first[0]);
  const determinant = positive - negative;
  if (Math.abs(determinant) > 16 * Number.EPSILON * (Math.abs(positive) + Math.abs(negative))) {
    return Math.sign(determinant);
  }
  const firstHorizontal = getCoordinateInteger(first[0]);
  const firstVertical = getCoordinateInteger(first[1]);
  const exact =
    (getCoordinateInteger(second[0]) - firstHorizontal) *
      (getCoordinateInteger(third[1]) - firstVertical) -
    (getCoordinateInteger(second[1]) - firstVertical) *
      (getCoordinateInteger(third[0]) - firstHorizontal);
  return exact > 0n ? 1 : exact < 0n ? -1 : 0;
}

/** Exact incircle sign, positive inside a counterclockwise circle; no row-order tie. @internal */
export function computeTriangulationIncircle(
  first: readonly number[],
  second: readonly number[],
  third: readonly number[],
  point: readonly number[]
): number {
  if (![...first, ...second, ...third, ...point].every(Number.isFinite)) return 0;
  const firstHorizontal = first[0] - point[0];
  const firstVertical = first[1] - point[1];
  const secondHorizontal = second[0] - point[0];
  const secondVertical = second[1] - point[1];
  const thirdHorizontal = third[0] - point[0];
  const thirdVertical = third[1] - point[1];
  const firstLift = firstHorizontal ** 2 + firstVertical ** 2;
  const secondLift = secondHorizontal ** 2 + secondVertical ** 2;
  const thirdLift = thirdHorizontal ** 2 + thirdVertical ** 2;
  const determinant =
    firstLift * (secondHorizontal * thirdVertical - thirdHorizontal * secondVertical) -
    secondLift * (firstHorizontal * thirdVertical - thirdHorizontal * firstVertical) +
    thirdLift * (firstHorizontal * secondVertical - secondHorizontal * firstVertical);
  const permanent =
    firstLift *
      (Math.abs(secondHorizontal * thirdVertical) + Math.abs(thirdHorizontal * secondVertical)) +
    secondLift *
      (Math.abs(firstHorizontal * thirdVertical) + Math.abs(thirdHorizontal * firstVertical)) +
    thirdLift *
      (Math.abs(firstHorizontal * secondVertical) + Math.abs(secondHorizontal * firstVertical));
  if (Number.isFinite(determinant) && Math.abs(determinant) > 1e-12 * permanent)
    return Math.sign(determinant);
  const pointHorizontal = getCoordinateInteger(point[0]);
  const pointVertical = getCoordinateInteger(point[1]);
  const firstExactHorizontal = getCoordinateInteger(first[0]) - pointHorizontal;
  const firstExactVertical = getCoordinateInteger(first[1]) - pointVertical;
  const secondExactHorizontal = getCoordinateInteger(second[0]) - pointHorizontal;
  const secondExactVertical = getCoordinateInteger(second[1]) - pointVertical;
  const thirdExactHorizontal = getCoordinateInteger(third[0]) - pointHorizontal;
  const thirdExactVertical = getCoordinateInteger(third[1]) - pointVertical;
  const exact =
    (firstExactHorizontal ** 2n + firstExactVertical ** 2n) *
      (secondExactHorizontal * thirdExactVertical - thirdExactHorizontal * secondExactVertical) -
    (secondExactHorizontal ** 2n + secondExactVertical ** 2n) *
      (firstExactHorizontal * thirdExactVertical - thirdExactHorizontal * firstExactVertical) +
    (thirdExactHorizontal ** 2n + thirdExactVertical ** 2n) *
      (firstExactHorizontal * secondExactVertical - secondExactHorizontal * firstExactVertical);
  return exact > 0n ? 1 : exact < 0n ? -1 : 0;
}
