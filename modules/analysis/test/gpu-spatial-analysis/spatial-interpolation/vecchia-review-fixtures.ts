// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Independent NumPy: C=exp(-norm(x[:,None]-x,axis=2)), y=[1,2,-1,.5].
// For each selected row set r, w=solve(C[r,r],C[r,3]), m=w@y[r], v=1-w@C[r,3],
// L=-.5*(log(2*pi*v)+(y[3]-m)**2/v). The unsorted row distinguishes all three policies.
export const OVER_WIDTH_SCENE = {
  positions: [99, 99, 0, 0, 1, 1, 3, 1, 4, 0],
  values: [1000, 1, 2, -1, 0.5],
  validity: [0, 1, 1, 1, 1],
  offsets: Uint32Array.of(0, 0, 0, 0, 0, 5),
  // Padding and an excluded observation precede the three original usable rows.
  neighbors: Uint32Array.of(0xffffffff, 0, 1, 3, 2),
  neighborCount: 1
};
export const OVER_WIDTH_EXPECTED = {
  first: {mean: 0.01831563888873418, variance: 0.9996645373720975, likelihood: -1.0348196156696936},
  nearest: {
    mean: -0.2431167344342142,
    variance: 0.9408942534380438,
    likelihood: -1.1819324594987886
  },
  all: {mean: -0.2193996010568421, variance: 0.9407686451167654, likelihood: -1.1634695919095086}
};

// Three equispaced exponential sites: rho=e^-1, y=[1,2,-1]. Gaussian Markov conditionals
// (m,v) = (0,1), (rho,1-rho^2), (2*rho,1-rho^2). Sum the three log densities.
export const TRANSLATED_VECCHIA_EXPECTED = {
  likelihood: -6.393990355493488,
  means: [0, 0.36787944117144233, 0.7357588823428847],
  variances: [1, 0.8646647167633873, 0.8646647167633873]
};
