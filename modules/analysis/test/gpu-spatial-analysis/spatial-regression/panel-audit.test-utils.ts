// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

// Literals from panel-audit.oracle.py, NumPy SVD / nested SciPy likelihood optimization.
export const PANEL_AUDIT_PREDICTORS = [-2, 1, 1, 3, 0, -1, 2, -2, 1, 3, -2, 0, 2, -2, 0, 1];
export const PANEL_AUDIT_RESPONSE = [1, 4, 2, 8, 3, -1, 6, 1, 4, 5, -2, 2, 7, -3, 4, 3];
export const PANEL_AUDIT_WEIGHTS = {
  offsets: [0, 2, 4, 6, 8],
  neighbors: [1, 3, 0, 2, 1, 3, 0, 2],
  weights: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
};
export const PANEL_AUDIT_EXPECTED = {
  unit: {
    coefficients: [NaN, 1.36408019054767, -0.260977587441426, 0],
    covariance: [
      NaN,
      NaN,
      NaN,
      NaN,
      0.0247050615839583,
      0.00966559354621773,
      NaN,
      0.00966559354621773,
      0.0136137908107394
    ],
    summary: [0.80737903886482, -43.4541890401031],
    fitted: [
      -4.01522418320743, 1.44714872250307, 0.0770163884355846, 3.83428905596149, 0.278801722536469,
      -2.06394442091655, 3.00696210363181, -3.76904465910114, 0.729460357039149, 4.30579789731912,
      -3.36278021460386, -0.127462721960805, 3.00696210363181, -3.68900219890564, 0.278801722536469,
      0.062218325100449
    ],
    residuals: [
      1.26522418320743, 1.30285127749693, -0.577016388435585, 0.665710944038507, -1.02880172253647,
      -0.186055579083453, 0.493037896368192, 1.26904465910114, -0.479460357039149,
      -0.555797897319121, -1.13721978539614, -1.37253727803919, 0.243037896368191,
      -0.560997801094357, 1.22119827746353, -0.562218325100449
    ]
  },
  random: {
    coefficients: [3.03977159448002, 1.3686368604542, -0.260898404862529, 0.586728472972251],
    covariance: [
      0.344283590093099, -0.0425221022731332, -0.0492848890978387, -0.0425221022731332,
      0.032166876550679, 0.0118072557640167, -0.0492848890978387, 0.0118072557640167,
      0.0165800442442122
    ],
    summary: [1.07660557186371, -23.5755029691733],
    fitted: [
      -2.40447129541695, 2.9563485194595, 1.70143928594564, 5.5522175790513, 1.89819285466662,
      -0.56362041603648, 4.63546657557501, -2.07366193780727, 2.35368529810196, 5.82407144279916,
      -1.75222528326063, 1.57675620011998, 4.63546657557501, -2.1931556813532, 1.89819285466662,
      1.77135023869279
    ],
    residuals: [
      1.85470306906289, 0.527062071755814, -0.734618103515011, 1.00133207635158, -0.447961081020676,
      -0.952968992748207, 0.33135460685562, 1.62721159321015, 0.0965464755439802, -1.34066085158384,
      -1.28095353430875, -1.0232065447171, 0.814765198070933, -1.32343372743148, 1.06862832776401,
      -0.217800583289913
    ],
    hausman: [0.00342455351747909, 3, 0.999946755027463]
  }
};

export function createPanelAuditInput(shifted = false, island = false) {
  return {
    predictors: Float32Array.from(PANEL_AUDIT_PREDICTORS, value => value + (shifted ? 16 : 0)),
    response: Float32Array.from(PANEL_AUDIT_RESPONSE, value => (shifted ? 2 * value + 16 : value)),
    weights: {
      offsets: Uint32Array.from(PANEL_AUDIT_WEIGHTS.offsets),
      neighbors: Uint32Array.from(PANEL_AUDIT_WEIGHTS.neighbors),
      weights: Float32Array.from(PANEL_AUDIT_WEIGHTS.weights, (value, index) =>
        island && index < 2 ? 0 : value
      )
    }
  };
}

export function expectPanelAuditRows(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tolerance: number
) {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    if (Number.isNaN(expected[index])) expect(actual[index], `slot ${index}`).toBeNaN();
    else {
      expect(Number.isFinite(actual[index]), `finite slot ${index}`).toBe(true);
      expect(Math.abs(actual[index] - expected[index]), `slot ${index}`).toBeLessThanOrEqual(
        tolerance * Math.max(1, Math.abs(expected[index]))
      );
    }
  }
}

export function getPanelAuditExpected(effects: 'unit' | 'random', shifted: boolean) {
  const original = PANEL_AUDIT_EXPECTED[effects];
  if (!shifted) return original;
  // Affine equivariance derived from y'=2y+16 and x'=x+16, with W1=1.
  const [intercept, slope, rho, phi] = original.coefficients;
  const transform = [
    [2, -32, -16],
    [0, 2, 0],
    [0, 0, 1]
  ];
  const covariance = original.covariance.map((_, index) => {
    const row = Math.floor(index / 3),
      column = index % 3;
    if (effects === 'unit') {
      if (row === 0 || column === 0) return NaN;
      return original.covariance[index] * (row === 1 ? 2 : 1) * (column === 1 ? 2 : 1);
    }
    let value = 0;
    for (let first = 0; first < 3; first++)
      for (let second = 0; second < 3; second++)
        value +=
          transform[row][first] *
          original.covariance[first * 3 + second] *
          transform[column][second];
    return value;
  });
  return {
    ...original,
    coefficients: [2 * intercept + 16 * (1 - rho) - 32 * slope, 2 * slope, rho, phi],
    covariance,
    summary: [4 * original.summary[0], original.summary[1] - 16 * Math.log(2)],
    fitted: original.fitted.map(value => 2 * value + 16 * phi),
    residuals: original.residuals.map(value => 2 * value)
  };
}
