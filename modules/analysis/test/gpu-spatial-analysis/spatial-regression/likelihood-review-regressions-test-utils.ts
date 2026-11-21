// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Closed form for x=0..7, y=x^2, W=3I, rho=-.33. OLS beta=(-7,7), SSE=168.
 * Lag beta and residuals scale by 1.99, sigma2=21*1.99^2. All joint inference is undefined.
 * LL=-4*(log(2*pi*21)+1), AIC=6-2LL, BIC=3log(8)-2LL; corr(y,x)^2=49/53.
 * Centered y has orthogonal fitted/residual components with squares 2058 and 168.
 * The lag fitted response is OLSfit-.99*OLSresidual, so its covariance is 1891.68 and
 * variance 2222.6568, giving pseudoR2=1891.68^2/(2226*2222.6568).
 * Every power of W/||W||_inf is I, so both trace/total power sequences are identically 8.
 */
export function getScalarLagExpected(maximumPower: number) {
  return {
    table: [-13.93, NaN, NaN, NaN, 13.93, NaN, NaN, NaN, -0.33, NaN, NaN, NaN],
    summary: [
      8,
      -0.33,
      83.1621,
      -23.529598016531075,
      53.05919603306215,
      53.29752065810166,
      1891.68 ** 2 / (2226 * 2222.6568),
      49 / 53,
      665.2968,
      3.3532670900025026,
      1,
      1,
      0,
      0.33,
      0,
      0,
      1,
      0,
      0,
      0,
      1,
      5.505077109891208
    ],
    residuals: [13.93, 1.99, -5.97, -9.95, -9.95, -5.97, 1.99, 13.93],
    powerTraces: new Array<number>(maximumPower).fill(8),
    powerSums: new Array<number>(maximumPower).fill(8),
    infinityNorm: [3],
    powerTraceStandardErrors: new Array<number>(maximumPower).fill(0),
    status: [0],
    converged: [0]
  };
}

export function getReviewPanelInput() {
  return {
    offsets: {format: 'uint32' as const, values: Uint32Array.from({length: 15}, (_, row) => row)},
    neighbors: {
      format: 'uint32' as const,
      values: Uint32Array.from({length: 14}, (_, row) => row ^ 1)
    },
    weights: {
      format: 'float32' as const,
      values: Float32Array.from({length: 14}, (_, row) => (row < 8 && row % 2 ? -1 : 1))
    },
    predictors: {
      format: 'float32' as const,
      values: Float32Array.from({length: 28}, (_, row) => (row < 14 ? 1 : -1) * (row % 14))
    },
    response: {
      format: 'float32' as const,
      values: Float32Array.from(
        {length: 28},
        (_, row) => (row < 14 ? 1 : -1) * ((4 * (row % 14) + 2) % 5)
      )
    }
  };
}
