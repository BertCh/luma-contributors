// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Simultaneous block solve, not backfitting: multiscale-gwr-audit.oracle.py. */
const REFERENCES = {
  gaussian: {
    coefficients: [
      [2.210337873925625, 0.33299208289355353],
      [2.289662126074375, 0.16700791710644652]
    ],
    fitted: [1.8773457910320714, 2.5433299568191785, 2.122654208967928, 2.4566700431808215],
    conditionalStandardErrors: [0.4133540346796827, 0.41074013982818525],
    effectiveParameters: [1.158648504297499, 1.110656110524738],
    'residual-df': {
      summary: [2.269304614822237, 2.0374141227122964, 2.658566317026909, 9.730695385177762],
      standardErrors: [0.47657464417247697, 0.4735609660557943]
    },
    'smoother-df': {
      summary: [2.269304614822237, 2.0374141227122964, 2.7234688240793696, 9.498804893067822],
      standardErrors: [0.4823567799919604, 0.47930653783142874]
    }
  },
  bisquare: {
    coefficients: [
      [2.057365439093484, 0.6462264150943395],
      [2.442634560906516, -0.1462264150943396]
    ],
    fitted: [1.4111390239991448, 2.703591854187824, 2.5888609760008556, 2.296408145812176],
    conditionalStandardErrors: [0.5153848375874637, 0.46171832748052244],
    effectiveParameters: [1.7705382436260628, 1.5283018867924525],
    'residual-df': {
      summary: [3.2988401304185153, 2.872832068478803, 2.4756881655581853, 8.701159869581485],
      standardErrors: [0.5734091385547861, 0.5137006157474178]
    },
    'smoother-df': {
      summary: [3.2988401304185153, 2.872832068478803, 2.603137563695218, 8.275151807641773],
      standardErrors: [0.5879835816546666, 0.5267574365951884]
    }
  }
} as const;

/** R6: orthogonal to intercept and +/-1; RSS = 8*y², residual df = 8-2. */
export function createMultiscaleOverflowFixture() {
  return {
    positions: new Float32Array(16),
    predictors: Float32Array.of(-1, 1, -1, 1, -1, 1, -1, 1),
    response: Float32Array.of(1e20, 1e20, -1e20, -1e20, 1e20, 1e20, -1e20, -1e20),
    parameters: Float32Array.of(1e-5, 2, 3)
  };
}

export function createMultiscaleAuditFixture(
  kernel: 'gaussian' | 'bisquare',
  rescaled = false,
  varianceConvention?: 'residual-df' | 'smoother-df'
) {
  const reference = REFERENCES[kernel];
  const predictorScale = rescaled ? 2 : 1;
  const positions = [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 1, 1].flatMap(value =>
    rescaled ? [32 + 4 * value, -16] : [value, 0]
  );
  const expected: Record<string, number[]> = {
    coefficients: [],
    fitted: [],
    codes: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    iterationCount: [2],
    selectedBandwidths: rescaled ? [10, 12] : [2.5, 3],
    componentSearchIterationLimitReached: [0, 0],
    conditionalStandardErrors: []
  };
  if (varianceConvention)
    Object.assign(expected, {
      inferenceStandardErrors: [],
      inferenceEffectiveParameters: [...reference.effectiveParameters],
      inferenceSummary: [...reference[varianceConvention].summary],
      inferenceStatus: [0]
    });
  for (let row = 0; row < 14; row++) {
    const coefficients = reference.coefficients[row < 6 ? 0 : 1];
    expected['coefficients'].push(
      ...(row < 12 ? [coefficients[0], coefficients[1] / predictorScale] : [NaN, NaN])
    );
    expected['fitted'].push(row < 12 ? reference.fitted[Math.floor(row / 3)] : NaN);
    expected['conditionalStandardErrors'].push(
      ...(row < 12
        ? [
            reference.conditionalStandardErrors[0],
            reference.conditionalStandardErrors[1] / predictorScale
          ]
        : [NaN, NaN])
    );
    if (varianceConvention)
      expected['inferenceStandardErrors'].push(
        ...(row < 12
          ? [
              reference[varianceConvention].standardErrors[0],
              reference[varianceConvention].standardErrors[1] / predictorScale
            ]
          : [NaN, NaN])
      );
  }
  return {
    input: {
      positions: Float32Array.from(positions),
      predictors: Float32Array.from(
        [-1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1],
        value => predictorScale * value
      ),
      response: Float32Array.of(0, 1, 2, 1, 3, 5, 2, 3, 4, 0, 2, 4, NaN, NaN),
      mask: Uint32Array.from(expected['codes'], code => (code === 0 ? 1 : 0)),
      parameters: Float32Array.from([1e-5, ...expected['selectedBandwidths']]),
      noiseVariance: Float32Array.of(2)
    },
    expected
  };
}
