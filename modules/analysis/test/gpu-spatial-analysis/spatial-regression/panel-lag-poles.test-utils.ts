// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import edges from './spatial-panel-lag-maximum-likelihood-operation-edge-fixtures.json';

// Fraction oracle in panel-lag-poles.oracle.py: y = X beta + .5 W y, beta=1.
// Symmetric dyadic W has eigenvalues [-1,-.25,.25,1], including its sign convention.
export const weightedPanelPole = {
  name: 'weightedPole',
  entityCount: 4,
  periodCount: 2,
  predictorCount: 1,
  predictors: [-0.5625, 0.5625, 2.0625, -2.0625, 0.5625, -0.5625, -2.0625, 2.0625],
  response: [-1, 1, 2, -2, 1, -1, -2, 2],
  weights: {
    offsets: [0, 2, 4, 6, 8],
    neighbors: [1, 3, 0, 2, 1, 3, 0, 2],
    weights: [0.375, 0.625, 0.375, 0.625, 0.625, 0.375, 0.625, 0.375]
  }
};
export const panelPoleScenes = [
  ...edges.reviewCases.filter(scene => ['responseCopy', 'originalPole'].includes(scene.name)),
  weightedPanelPole,
  {
    ...weightedPanelPole,
    name: 'weightedRationalPole',
    predictors: weightedPanelPole.predictors.map(value => 3 * value)
  },
  {
    name: 'rationalSlopePole',
    entityCount: 4,
    periodCount: 2,
    predictorCount: 1,
    response: [5, 1, -5, -1, -3, -6, 3, 6],
    predictors: [15, 3, -15, -3, -9, -18, 9, 18],
    weights: {
      offsets: [0, 2, 4, 6, 8],
      neighbors: [1, 3, 0, 2, 1, 3, 0, 2],
      weights: new Array(8).fill(0.5)
    }
  }
];
