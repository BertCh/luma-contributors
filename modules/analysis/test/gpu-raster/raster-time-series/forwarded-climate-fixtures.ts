// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// For [8388612, 8388613] the exact mean is 8388612.5 and the residuals are ±0.5.
// Two observations have M2=(b-a)^2/2 and sample SD=|b-a|/sqrt(2), so z=±1/sqrt(2).
// The float32 mean rounds to the lower even integer. Residuals use the unrounded mean.
export const ADJACENT_CLIMATE_OBSERVATIONS = [8388612, 8388613] as const;

export const POWER_BOUNDARY_CLIMATE_OBSERVATIONS = [16777214, 16777215] as const;
export const POWER_BOUNDARY_CLIMATE_CASE = {
  name: 'query across a power of two',
  count: 2,
  // The retained mean is 16777214.5 and the sample deviation is 1/sqrt(2).
  // Query 2^24 is 1.5 above that mean, although the mean publishes as 16777214.
  sum: 33554429,
  moment: 0.5,
  sumUnit: -69,
  momentUnit: -43,
  values: [16777216],
  mean: 16777214,
  deviation: Math.SQRT1_2,
  differences: [1.5],
  anomalies: [1.5 * Math.SQRT2]
} as const;

const MAXIMUM_FINITE_FLOAT32 = 2 ** 128 - 2 ** 104;

// Words are exact integers in the stated power-of-two units, independent of production decoders.
export const CLIMATE_CASES = [
  {
    name: 'adjacent',
    count: 2,
    sum: 16777225,
    moment: 0.5,
    sumUnit: -44,
    momentUnit: -20,
    values: ADJACENT_CLIMATE_OBSERVATIONS,
    mean: 8388612,
    deviation: Math.SQRT1_2,
    differences: [-0.5, 0.5],
    anomalies: [-Math.SQRT1_2, Math.SQRT1_2]
  },
  POWER_BOUNDARY_CLIMATE_CASE,
  {
    name: 'large moment',
    count: 2,
    sum: 0,
    moment: 2 ** 137,
    sumUnit: 0,
    momentUnit: 60,
    values: [2 ** 68, -(2 ** 68)],
    mean: 0,
    deviation: Math.SQRT2 * 2 ** 68,
    differences: [2 ** 68, -(2 ** 68)],
    anomalies: [Math.SQRT1_2, -Math.SQRT1_2]
  },
  {
    name: 'distant query',
    count: 2,
    sum: 3,
    moment: 0.5,
    sumUnit: -44,
    momentUnit: -20,
    values: [2 ** 100],
    mean: 1.5,
    deviation: Math.SQRT1_2,
    differences: [2 ** 100],
    anomalies: [Math.SQRT2 * 2 ** 100]
  },
  {
    name: 'overflowing physical deviation',
    count: 2,
    sum: 0,
    // Baseline is [M,-M] for the largest finite float32 M; its exact M2 is 2*M*M.
    moment: 2 * (2 ** 128 - 2 ** 104) ** 2,
    sumUnit: 40,
    momentUnit: 170,
    values: [2 ** 128 - 2 ** 104],
    mean: 0,
    deviation: Infinity,
    differences: [2 ** 128 - 2 ** 104],
    anomalies: [Math.SQRT1_2]
  },
  {
    name: 'overflowing physical sum',
    count: 2,
    // Adjacent float32 observations with an ULP of 2^104. Their exact sum exceeds float32,
    // but the mean rounds to 2^127 (ties to even). Each unrounded residual has magnitude 2^103.
    sum: 2 ** 128 + 2 ** 104,
    moment: 2 ** 207,
    sumUnit: 35,
    momentUnit: 164,
    values: [2 ** 127, 2 ** 127 + 2 ** 104],
    mean: 2 ** 127,
    deviation: Math.SQRT1_2 * 2 ** 104,
    differences: [-(2 ** 103), 2 ** 103],
    anomalies: [-Math.SQRT1_2, Math.SQRT1_2]
  },
  {
    name: 'five equal observations',
    count: 5,
    sum: 5 * 16777215,
    moment: 0,
    sumUnit: -68,
    momentUnit: -42,
    values: [16777215],
    mean: 16777215,
    deviation: 0,
    differences: [0],
    anomalies: [NaN]
  },
  {
    name: 'seven maximum-finite observations',
    count: 7,
    sum: 7 * MAXIMUM_FINITE_FLOAT32,
    moment: 0,
    sumUnit: 36,
    momentUnit: 166,
    values: [MAXIMUM_FINITE_FLOAT32],
    mean: MAXIMUM_FINITE_FLOAT32,
    deviation: 0,
    differences: [0],
    anomalies: [NaN]
  }
] as const;

export function climateWords(value: number, unit: number): Uint32Array {
  const encoded = BigInt.asUintN(96, BigInt(value / 2 ** unit));
  return Uint32Array.of(
    Number(encoded & 0xffffffffn),
    Number((encoded >> 32n) & 0xffffffffn),
    Number(encoded >> 64n)
  );
}
