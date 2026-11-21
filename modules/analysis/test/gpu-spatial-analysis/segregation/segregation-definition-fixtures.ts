// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import fixture from './segregation-definition.json';

export const DEFINITION_COUNTS = Float32Array.of(
  8,
  2,
  0,
  2,
  8,
  0,
  5,
  5,
  0,
  0,
  0,
  0,
  1,
  3,
  6,
  4,
  1,
  5,
  0,
  4,
  6,
  3,
  3,
  4
);
export const DEFINITION_SMOOTHED_COUNTS = Float32Array.of(
  2,
  8,
  0,
  8,
  2,
  0,
  2.5,
  2,
  5.5,
  5,
  5,
  0,
  8,
  2,
  10,
  0.25,
  3.75,
  6,
  4,
  1,
  5,
  0,
  0,
  0
);
export const DEFINITION_WEIGHTS = {
  offsets: [0, 1, 2, 4, 5, 6, 8, 9, 9],
  neighbors: [1, 0, 4, 5, 2, 5, 4, 6, 5],
  weights: [1, 1, 0.5, 0.5, 1, 2, 0.25, 0.75, 1]
};

export function getSegregationDefinition(form: 'environment' | 'smoothed-population') {
  const source = fixture.fixtures[form];
  const decode = (values: (number | null)[]) =>
    values.map(value => (value === null ? Number.NaN : value));
  return {
    indices: decode(source.indices),
    localEnvironment: decode(source.localEnvironment),
    localEntropy: decode(source.localEntropy),
    localDissimilarity: decode(source.localDissimilarity),
    localTheil: decode(source.localTheil)
  };
}
