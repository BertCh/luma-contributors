// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Closed form, evaluated with /usr/bin/python3 math (binary64): two events (0,0,0), (1,0,1).
// A=1,T=2, mu=1, omega=.5, beta=sigma=1. Only one parent contributes:
// g=.5*exp(-3/2)/(2*pi); event intensities [1,1+g]; temporal mass M=2-exp(-2)-exp(-1).
// r=g/(1+g). The finite-window Q profile decreases from beta=0, so the fitted boundary
// beta*T=1e-6 gives beta=5e-7, omega=r/[-expm1(-2*beta)-expm1(-beta)].
// mu=(2-r)/2, sigma=sqrt(1/2). Entering density: log(1+g)-2-.5*M.
// Directly evaluate the fitted intensities and full temporal compensator at that boundary.
export const TWO_EVENT_DENSITY = {
  estimates: [0.9912768229158258, 11630.907625109201, 5e-7, 0.7071067811865476],
  logLikelihood: [-2.730792302422022],
  fittedLogLikelihood: [-2.016836149295383],
  backgroundProbability: [1, 0.999313491627981],
  converged: [0],
  iterationLimitReached: [1],
  iterationCount: [1],
  candidateOverflow: [0],
  invalidCount: [0],
  approximation: [1]
} as const;
