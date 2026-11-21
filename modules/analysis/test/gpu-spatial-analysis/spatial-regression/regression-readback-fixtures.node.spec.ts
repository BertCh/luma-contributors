// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {close, getMaximumLikelihoodReadback} from './regression-readback-test-utils';

it('converts plain GPU readback arrays to the exact ML reader array types', () => {
  const readback = getMaximumLikelihoodReadback([1, 2], [3, 4, NaN], [0]);
  expect(readback.table).toBeInstanceOf(Float32Array);
  expect(readback.summary).toBeInstanceOf(Float32Array);
  expect(readback.status).toBeInstanceOf(Uint32Array);
  expect(Array.from(readback.table)).toEqual([1, 2]);
  expect(Array.from(readback.summary)).toEqual([3, 4, NaN]);
  expect(Array.from(readback.status)).toEqual([0]);
});

it('rejects trailing actual values instead of silently ignoring them', () => {
  expect(() => close([1, 2], [1])).toThrow();
});

it('rejects missing values and nonfinite numeric mismatches', () => {
  expect(() => close([1], [1, 2])).toThrow();
  expect(() => close([NaN], [1])).toThrow();
  expect(() => close([Infinity], [1])).toThrow();
  expect(() => close([1], [1])).not.toThrow();
  expect(() => close([NaN], [NaN])).not.toThrow();
});

import {getGPUOrdinaryLeastSquaresModelResult} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-statistics';
import {getGPUSpatialTwoStageLeastSquaresModelResult} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-two-stage-least-squares-model-result';
import {
  getGPUSpatialLagMaximumLikelihoodModelResult,
  getGPUSpatialErrorMaximumLikelihoodModelResult
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-model-result';
import {getGPUSpatialErrorGMModelResult} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-error-gm-model-result';
import {getGPUSpatialPanelErrorMaximumLikelihoodModelResult} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-panel-error-maximum-likelihood-model-result';

it('UST-17 rejects empty primary status in all six named model readers before coefficient access', () => {
  const readback = {
    table: new Float32Array(12),
    summary: new Float32Array(64),
    status: new Uint32Array(0)
  };
  readback.summary[0] = 8;
  const olsReadback = {
    ...readback,
    coefficients: new Float32Array(2),
    standardErrors: new Float32Array(2),
    tStatistics: new Float32Array(2)
  };
  const readers = [
    [
      'getGPUOrdinaryLeastSquaresModelResult',
      () => getGPUOrdinaryLeastSquaresModelResult(olsReadback)
    ],
    [
      'getGPUSpatialTwoStageLeastSquaresModelResult',
      () => getGPUSpatialTwoStageLeastSquaresModelResult(readback)
    ],
    [
      'getGPUSpatialLagMaximumLikelihoodModelResult',
      () => getGPUSpatialLagMaximumLikelihoodModelResult(readback)
    ],
    [
      'getGPUSpatialErrorMaximumLikelihoodModelResult',
      () => getGPUSpatialErrorMaximumLikelihoodModelResult(readback)
    ],
    ['getGPUSpatialErrorGMModelResult', () => getGPUSpatialErrorGMModelResult(readback)],
    [
      'getGPUSpatialPanelErrorMaximumLikelihoodModelResult',
      () => getGPUSpatialPanelErrorMaximumLikelihoodModelResult(readback, {entityCount: 2})
    ]
  ] as const;
  for (const [name, read] of readers) {
    expect(read).toThrow(new RegExp(`${name} status must hold at least one uint32 code`));
  }
});

it('UST-17 validates covariance status symmetrically before table access', () => {
  expect(() =>
    getGPUSpatialTwoStageLeastSquaresModelResult({
      table: new Float32Array(0),
      summary: new Float32Array(64),
      status: new Uint32Array([0]),
      robust: {method: 'HC0', status: new Uint32Array(0)}
    })
  ).toThrow(
    /getGPUSpatialTwoStageLeastSquaresModelResult robust.status must hold at least one uint32 code/
  );
});

const REVIEW_READERS = [
  [
    'getGPUOrdinaryLeastSquaresModelResult',
    (status: Uint32Array | Uint8Array | ArrayBuffer) =>
      getGPUOrdinaryLeastSquaresModelResult({
        status,
        coefficients: new Float32Array(2),
        standardErrors: new Float32Array(2),
        tStatistics: new Float32Array(2),
        summary: new Float32Array(64).fill(8)
      })
  ],
  [
    'getGPUSpatialTwoStageLeastSquaresModelResult',
    (status: Uint32Array | Uint8Array | ArrayBuffer) =>
      getGPUSpatialTwoStageLeastSquaresModelResult({
        status,
        table: new Float32Array(12),
        summary: new Float32Array(64).fill(8)
      })
  ],
  [
    'getGPUSpatialLagMaximumLikelihoodModelResult',
    (status: Uint32Array | Uint8Array | ArrayBuffer) =>
      getGPUSpatialLagMaximumLikelihoodModelResult({
        status,
        table: new Float32Array(12),
        summary: new Float32Array(64).fill(8)
      })
  ],
  [
    'getGPUSpatialErrorMaximumLikelihoodModelResult',
    (status: Uint32Array | Uint8Array | ArrayBuffer) =>
      getGPUSpatialErrorMaximumLikelihoodModelResult({
        status,
        table: new Float32Array(12),
        summary: new Float32Array(64).fill(8)
      })
  ],
  [
    'getGPUSpatialErrorGMModelResult',
    (status: Uint32Array | Uint8Array | ArrayBuffer) =>
      getGPUSpatialErrorGMModelResult({
        status,
        table: new Float32Array(12),
        summary: new Float32Array(64).fill(8)
      })
  ],
  [
    'getGPUSpatialPanelErrorMaximumLikelihoodModelResult',
    (status: Uint32Array | Uint8Array | ArrayBuffer) =>
      getGPUSpatialPanelErrorMaximumLikelihoodModelResult(
        {status, table: new Float32Array(12), summary: new Float32Array(64).fill(8)},
        {entityCount: 2}
      )
  ]
] as const;

for (const [reader, read] of REVIEW_READERS) {
  it(`B2 ${reader} decodes whole offset byte words and ArrayBuffers`, () => {
    expect(() => read(Uint8Array.of(0))).toThrow(new RegExp(`${reader} status`));
    const backing = Uint8Array.of(99, 0, 1, 0, 0, 99);
    expect(read(backing.subarray(1, 5)).status).toMatchObject({code: 256, ok: false});
    expect(read(Uint8Array.of(0, 0, 0, 0).buffer).status).toMatchObject({code: 0, ok: true});
  });
}

it('S1 validates primary status before touching coefficient/table getters', () => {
  const readback = {
    status: new Uint32Array(0),
    get coefficients(): Float32Array {
      throw new Error('coefficient getter touched');
    },
    standardErrors: new Float32Array(2),
    tStatistics: new Float32Array(2),
    summary: new Float32Array(64)
  };
  expect(() => getGPUOrdinaryLeastSquaresModelResult(readback)).toThrow(
    /getGPUOrdinaryLeastSquaresModelResult status/
  );
  const spatial = {
    status: new Uint32Array(0),
    get table(): Float32Array {
      throw new Error('table getter touched');
    },
    summary: new Float32Array(64)
  };
  for (const read of [
    getGPUSpatialTwoStageLeastSquaresModelResult,
    getGPUSpatialLagMaximumLikelihoodModelResult,
    getGPUSpatialErrorMaximumLikelihoodModelResult,
    getGPUSpatialErrorGMModelResult
  ])
    expect(() => read(spatial)).toThrow(/status must hold/);
  expect(() =>
    getGPUSpatialPanelErrorMaximumLikelihoodModelResult(spatial, {entityCount: 2})
  ).toThrow(/status must hold/);
});

it('S1 captures covariance status exactly once and retains its failure', () => {
  let reads = 0;
  const result = getGPUSpatialTwoStageLeastSquaresModelResult({
    status: new Uint32Array([0]),
    table: new Float32Array(12),
    summary: new Float32Array(64).fill(8),
    robust: {
      method: 'HC0',
      get status() {
        reads++;
        return reads === 1 ? new Uint32Array([5]) : new Uint32Array(0);
      }
    }
  });
  expect(result.covarianceInference.status).toEqual({code: 5, ok: false});
  expect(reads).toBe(1);
});

it('B2 covariance byte status has the same whole-word rules', () => {
  const readback = {
    status: new Uint32Array([0]),
    table: new Float32Array(12),
    summary: new Float32Array(64).fill(8)
  };
  expect(() =>
    getGPUSpatialTwoStageLeastSquaresModelResult({
      ...readback,
      robust: {method: 'HC0', status: Uint8Array.of(0)}
    })
  ).toThrow(/robust.status/);
  const result = getGPUSpatialTwoStageLeastSquaresModelResult({
    ...readback,
    robust: {method: 'HC0', status: Uint8Array.of(0, 1, 0, 0).buffer}
  });
  expect(result.covarianceInference.status).toEqual({code: 256, ok: false});
  const offsetResult = getGPUSpatialTwoStageLeastSquaresModelResult({
    ...readback,
    robust: {method: 'HC0', status: Uint8Array.of(99, 5, 0, 0, 0, 99).subarray(1, 5)}
  });
  expect(offsetResult.covarianceInference.status).toEqual({code: 5, ok: false});
});
