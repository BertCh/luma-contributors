// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Independent finite-window likelihood, directly from lambda(t,x) and its compensator.
// Four-event review witness after catalog origin subtraction; area=(1+2*4)^2=81, T=1.
export const HAWKES_WITNESS_POSITIONS = [
  [0, 0],
  [1, 0],
  [0, 1],
  [1, 1]
];
export const HAWKES_WITNESS_TIMES = [0, 0.15, 0.3, 0.45];

export function getWitnessLogLikelihood(estimates: readonly number[]): number {
  const [background, branchingRatio, decayRate, deviation] = estimates;
  const variance = deviation ** 2;
  let result = -background * 81;
  for (let row = 0; row < 4; row++) {
    let intensity = background;
    for (let parent = 0; parent < row; parent++) {
      const distanceSquared = HAWKES_WITNESS_POSITIONS[row].reduce(
        (total, coordinate, axis) =>
          total + (coordinate - HAWKES_WITNESS_POSITIONS[parent][axis]) ** 2,
        0
      );
      const lag = HAWKES_WITNESS_TIMES[row] - HAWKES_WITNESS_TIMES[parent];
      intensity +=
        (branchingRatio *
          decayRate *
          Math.exp(-decayRate * lag - distanceSquared / (2 * variance))) /
        (2 * Math.PI * variance);
    }
    result +=
      Math.log(intensity) -
      branchingRatio * -Math.expm1(-decayRate * (1 - HAWKES_WITNESS_TIMES[row]));
  }
  return result;
}

/** Numerical derivative in log-parameter coordinates; finite, positive estimates are interior. */
export function getWitnessLikelihoodScore(estimates: readonly number[]): number[] {
  const step = 1e-4;
  return estimates.map((_, parameter) => {
    const above = estimates.map((value, index) => value * Math.exp(index === parameter ? step : 0));
    const below = estimates.map(
      (value, index) => value * Math.exp(index === parameter ? -step : 0)
    );
    return (getWitnessLogLikelihood(above) - getWitnessLogLikelihood(below)) / (2 * step);
  });
}

// /usr/bin/python3 + scipy.optimize.minimize_scalar, profiled EM:
// E-step weights = offspring / lambda; W=sum weights, L=sum weights*lag,
// M(beta)=sum -expm1(-beta*(1-time)); maximize W*(log(beta)-log(M(beta)))-beta*L;
// omega=W/M(beta), mu=sum background weights/81, sigma²=sum weights*d²/(2W).
export const HAWKES_WITNESS_FIRST_UPDATE = [
  0.0440773252107819, 0.11927531144515108, 3.1557284732777506, 0.8108455038070963
];
export const HAWKES_WITNESS_FINAL = [
  0.014690617073561668, 0.7362965224050944, 4.293032191411873, 0.7654518353341745
];
