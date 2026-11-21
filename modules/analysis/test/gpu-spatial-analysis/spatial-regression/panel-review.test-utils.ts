// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {createPanelAuditInput, expectPanelAuditRows} from './panel-audit.test-utils';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/graph/gpu-core';

export function createConstantPanelLagInput() {
  const input = createPanelAuditInput();
  input.response = input.response.slice(0, 12);
  input.predictors = Float32Array.from({length: 12}, (_, row) => [0.1, -0.1, -0.1, 0.1][row % 4]);
  return input;
}
export function createExtremeFirstPanelLagInput() {
  const input = createConstantPanelLagInput();
  input.predictors = Float32Array.from(
    {length: 12},
    (_, row) => (row % 2 ? -1 : 1) * [2 ** 25, -(2 ** 25), 1][Math.floor(row / 4)]
  );
  return input;
}

/** Wy=-y/2 period by period. The concentrated negative log likelihood is
 * constant + 4 log(1+rho/2)-4 log(1-rho), strictly increasing on [-.999,.999]. */
export function createBoundaryPanelLagInput() {
  const input = createPanelAuditInput();
  return {
    ...input,
    predictors: input.predictors.slice(0, 12),
    response: Float32Array.of(1, 2, -3, 2, 1, -3, -2, 3, -1, 0, -1, 1),
    weights: {
      offsets: Uint32Array.of(0, 2, 4, 6),
      neighbors: Uint32Array.of(1, 2, 0, 2, 0, 1),
      weights: new Float32Array(6).fill(0.5)
    }
  };
}

export const BOUNDARY_PANEL_LAG_EXPECTED = {
  coefficients: [NaN, 0.014165094339622614, -0.999, 0],
  covariance: [
    NaN,
    NaN,
    NaN,
    NaN,
    0.02244409093804228,
    0.0007577221676257787,
    NaN,
    0.0007577221676257786,
    0.05564662286812784
  ],
  summary: [0.5944949918926886, -31.58303759520847],
  fitted: [
    0.3250471698113208, 0.39941391509433954, -0.7386261792452831, 0.895372641509434,
    -0.11425117924528304, -0.7669563679245283, -1.1167924528301887, 0.8564186320754718,
    0.26037382075471693, -0.10362735849056608, -1.1415813679245281, 1.2452087264150944
  ],
  residuals: [
    0.4249528301886792, 0.35058608490566046, -0.7613738207547169, 0.854627358490566,
    -0.13574882075471695, -0.7330436320754717, -1.1332075471698113, 0.8935813679245282,
    0.23962617924528307, -0.1463726415094339, -1.1084186320754719, 1.2547912735849056
  ]
};

export function expectBoundaryPanelLag(
  result: Record<
    'coefficients' | 'covariance' | 'summary' | 'fitted' | 'residuals' | 'status',
    ArrayLike<number>
  >,
  budget: number
) {
  expect(Array.from(result.status)).toEqual([budget === 1 ? 3 : 0]);
  if (budget === 1) {
    expect(Array.from(result.covariance).every(Number.isNaN)).toBe(true);
    expect(result.summary[4]).toBeGreaterThan(1);
    return;
  }
  for (const port of ['coefficients', 'covariance', 'fitted', 'residuals'] as const)
    expectPanelAuditRows(result[port], BOUNDARY_PANEL_LAG_EXPECTED[port], 2e-5);
  expectPanelAuditRows(
    Array.from(result.summary).slice(0, 2),
    BOUNDARY_PANEL_LAG_EXPECTED.summary,
    2e-5
  );
  expect(result.summary[2]).toBe(1);
  expect(result.summary[3]).toBe(0);
  expect(result.summary[4]).toBeGreaterThanOrEqual(0);
  expect(result.summary[4]).toBeLessThan(1e-5);
  expect(result.summary[5]).toBe(1);
}

export function createPanelReviewLagProps(
  graph: GPUCommandGraph,
  input: ReturnType<typeof createPanelAuditInput>,
  periodCount: number,
  predictorCount: number
) {
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `review-${serial++}`, format, length);
  return {
    effects: 'unit' as const,
    periodCount,
    predictorCount,
    predictors: view('float32', input.predictors.length),
    response: view('float32', input.response.length),
    weights: {
      offsets: view('uint32', input.weights.offsets.length),
      neighbors: view('uint32', input.weights.neighbors.length),
      weights: view('float32', input.weights.weights.length)
    },
    output: {
      coefficients: view('float32', predictorCount + 3),
      covariance: view('float32', (predictorCount + 2) ** 2),
      summary: view('float32', 6),
      status: view('uint32', 1),
      fitted: view('float32', input.response.length),
      residuals: view('float32', input.response.length)
    }
  };
}

export function createOffsetPanelLagInput() {
  const input = createPanelAuditInput();
  input.predictors = Float32Array.from({length: 32}, (_, index) => {
    const row = Math.floor(index / 2),
      period = Math.floor(row / 4);
    return (row % 2 ? 2048 : -2048) + (index % 2 ? [1, 1, -1, -1][period] : [1, -1, 1, -1][period]);
  });
  return input;
}

export function createExtremeOffsetPanelLagInput() {
  const input = createPanelAuditInput();
  input.predictors = Float32Array.from({length: 32}, (_, index) => {
    const row = Math.floor(index / 2),
      entity = row % 4,
      period = Math.floor(row / 4);
    return entity === 0
      ? 2 ** 48
      : entity === 1
        ? -(2 ** 48)
        : index % 2
          ? [1, 1, -1, -1][period]
          : [1, -1, 1, -1][period];
  });
  return input;
}
export const EXTREME_OFFSET_PANEL_LAG_EXPECTED = {
  coefficients: [NaN, -0.3575716848258928, 1.3924283151741066, -0.5697132606964284, 0],
  covariance: [
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    0.45996402650542806,
    -0.0010768428139482289,
    0.002427894386116151,
    NaN,
    -0.0010768428139482286,
    0.4638808552427744,
    -0.009454520681989917,
    NaN,
    0.0024278943861161507,
    -0.009454520681989917,
    0.021316553716005895
  ],
  summary: [3.677499970335438, -56.87123359564799],
  fitted: [
    -2.065210570024553, 0.925784048631696, -1.030353939676339, 1.9606406789799098,
    1.3530689941540173, -0.783355733457589, 3.1030689941540164, 0.9666442665424103,
    -0.6409274182834819, 1.2106406789799102, -2.390927418283481, -0.5393593210200891,
    1.3530689941540173, -1.3530689941540173, 0.3182123638058034, -2.3879256245022313
  ],
  residuals: [
    -0.6847894299754471, 1.8242159513683038, 0.530353939676339, 2.53935932102009,
    -2.1030689941540173, -1.4666442665424109, 0.3969310058459836, -3.4666442665424104,
    0.8909274182834819, 2.5393593210200898, -2.109072581716519, -0.9606406789799109,
    1.8969310058459827, -2.8969310058459827, 1.1817876361941966, 1.8879256245022313
  ]
};

// Independent NumPy SVD / Gaussian information from panel-review.oracle.py, R4.
export const OFFSET_PANEL_LAG_EXPECTED = {
  coefficients: [NaN, 0.38626520242451057, 0.3862652024245104, -0.5450608096980418, 0],
  covariance: [
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    0.28183886153361454,
    0.0014879466063253725,
    -0.00595178642530149,
    NaN,
    0.0014879466063253725,
    0.28183886153361454,
    -0.00595178642530149,
    NaN,
    -0.005951786425301491,
    -0.005951786425301491,
    0.023807145701205963
  ],
  summary: [4.485614638836626, -58.30084353610479],
  fitted: [
    -1.2033150303063807, 1.6582542206083388, -1.2033150303063807, 1.6582542206083388,
    1.2945194230328492, -0.7494586133348078, 1.2945194230328492, -0.7494586133348078,
    -0.6131934109102968, 1.158254220608339, -0.6131934109102968, 1.158254220608339,
    0.5219890181838285, -2.0670498278818705, 0.5219890181838285, -2.0670498278818705
  ],
  residuals: [
    -1.5466849696936193, 1.0917457793916612, 0.7033150303063807, 2.841745779391661,
    -2.044519423032849, -1.5005413866651922, 2.205480576967151, -1.7505413866651922,
    0.8631934109102968, 2.591745779391661, -3.8868065890897032, -2.658254220608339,
    2.7280109818161717, -2.1829501721181295, 0.9780109818161715, 1.5670498278818705
  ]
};

export function expectOffsetPanelLag(
  result: Record<
    'coefficients' | 'covariance' | 'summary' | 'fitted' | 'residuals' | 'status',
    ArrayLike<number>
  >,
  expected = OFFSET_PANEL_LAG_EXPECTED
) {
  expect(Array.from(result.status)).toEqual([0]);
  for (const port of ['coefficients', 'covariance', 'fitted', 'residuals'] as const)
    expectPanelAuditRows(result[port], expected[port], 2e-5);
  expectPanelAuditRows(Array.from(result.summary).slice(0, 2), expected.summary, 2e-5);
  expect(result.summary[2]).toBe(1);
  expect(result.summary[3]).toBe(0);
  expect(result.summary[4]).toBeGreaterThanOrEqual(0);
  expect(result.summary[4]).toBeLessThan(1e-5);
  expect(result.summary[5]).toBe(0);
}
