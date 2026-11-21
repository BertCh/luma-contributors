// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import fixture from './likelihood-mathematical-audit-fixture.json';

export {fixture};

export function getLikelihoodAuditInput(transformed: boolean) {
  return {
    weights: {
      offsets: Uint32Array.from(fixture.weights.offsets),
      neighbors: Uint32Array.from(fixture.weights.neighbors),
      weights: Float32Array.from(fixture.weights.weights, value => value * (transformed ? 2 : 1))
    },
    predictors: Float32Array.from(
      fixture.predictors,
      (value, index) => value + (transformed ? (index % 2 === 0 ? 4 : -2) : 0)
    ),
    response: Float32Array.from(fixture.response, value => value * (transformed ? 4 : 1))
  };
}
