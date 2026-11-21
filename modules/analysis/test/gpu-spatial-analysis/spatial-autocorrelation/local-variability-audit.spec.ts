// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPULocalSpatialHeteroscedasticity,
  GPUMultivariateLocalGeary,
  getGPULocalSpatialHeteroscedasticityParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {AUDIT_VALUES, AUDIT_MASK, AUDIT_WEIGHTS} from './local-autocorrelation-audit-fixture';

// Pasted exact-rational expectations from local-variability-audit.oracle.py.
const LOSH_REFERENCE = [
  {
    power: 1,
    statistic: [
      0.5130890052356021,
      0.5314136125654451,
      0.7329842931937173,
      0.4397905759162304,
      0.6596858638743456,
      2.7853403141361257,
      0.34205933682373474,
      NaN,
      NaN,
      NaN
    ],
    normalizedResiduals: [
      0.2931937172774869,
      0.6596858638743456,
      0.4397905759162304,
      0.7696335078534031,
      0.4397905759162304,
      0.4397905759162304,
      3.9581151832460733,
      NaN,
      NaN,
      NaN
    ]
  },
  {
    power: 2,
    statistic: [
      0.11048355899419729,
      0.1367504835589942,
      0.21771760154738878,
      0.07798839458413927,
      0.18522243713733075,
      4.237369439071567,
      0.049103803997421015,
      NaN,
      NaN,
      NaN
    ],
    normalizedResiduals: [
      0.03466150870406189,
      0.17547388781431333,
      0.07798839458413927,
      0.2388394584139265,
      0.07798839458413927,
      0.07798839458413927,
      6.31705996131528,
      NaN,
      NaN,
      NaN
    ]
  }
] as const;
const MULTIVARIATE_REFERENCE = [
  0.4856828193832599,
  0.4810900740463024,
  NaN,
  0.7463211172556004,
  0.6024150966975974,
  2.5811228793701377,
  6.932366357359328,
  NaN,
  NaN,
  NaN
];
const SECOND_VALUES = [2, 5, NaN, 7, 4, 3, 11, 999, 9, NaN];

function compare(actual: readonly number[], expected: readonly number[], label: string) {
  expect(actual.length).toBe(expected.length);
  for (const [index, value] of expected.entries()) {
    if (Number.isNaN(value)) expect(actual[index], `${label}[${index}]`).toBeNaN();
    else {
      expect(Number.isFinite(actual[index]), `${label}[${index}]`).toBe(true);
      // Two local weighted means or Welford population moments plus f32 quotient/sqrt;
      // 3e-5 absolute + 1e-5 relative is below any competing divisor or missing-island result.
      expect(Math.abs(actual[index] - value), `${label}[${index}]`).toBeLessThanOrEqual(
        3e-5 + 1e-5 * Math.abs(value)
      );
    }
  }
}

function uploadWeights(rig: WeightsRig, chunked: boolean) {
  if (!chunked) return rig.uploadWeights(AUDIT_WEIGHTS);
  return {
    rowCount: 10,
    chunks: [0, 3].map((rowStart, index) => {
      const rowEnd = index === 0 ? 3 : 10;
      const first = AUDIT_WEIGHTS.offsets[rowStart];
      const last = AUDIT_WEIGHTS.offsets[rowEnd];
      return {
        rowStart,
        rowCount: rowEnd - rowStart,
        weights: rig.uploadWeights({
          offsets: AUDIT_WEIGHTS.offsets.slice(rowStart, rowEnd + 1).map(offset => offset - first),
          neighbors: AUDIT_WEIGHTS.neighbors.slice(first, last),
          weights: AUDIT_WEIGHTS.weights.slice(first, last),
          distances: []
        })
      };
    })
  };
}

for (const chunked of [false, true]) {
  it(`LOSH ${chunked ? 'chunked' : 'whole'} matches exact weighted residual definitions on every port`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    const values = rig.input(Float32Array.from(AUDIT_VALUES), 'float32', 10);
    const power = rig.input(Float32Array.of(2), 'float32', 1);
    const inputOverflow = rig.input(Uint32Array.of(0), 'uint32', 1);
    const inputCandidateOverflow = rig.input(Uint32Array.of(0), 'uint32', 1);
    const statistic = rig.output('float32', 10);
    const status = rig.output('uint32', 10);
    const normalized = rig.output('float32', 10);
    const count = rig.output('uint32', 1);
    rig.graph.add(
      new GPULocalSpatialHeteroscedasticity({
        weights: {
          ...uploadWeights(rig, chunked),
          status: {overflow: inputOverflow, candidateOverflow: inputCandidateOverflow}
        },
        values,
        mask: rig.input(Uint32Array.from(AUDIT_MASK), 'uint32', 10),
        residualPower: power,
        output: {
          statistic: statistic.view,
          status: status.view,
          normalizedResiduals: normalized.view,
          residualCount: count.view
        }
      })
    );
    const compiled = rig.graph.compile();
    try {
      for (const reference of LOSH_REFERENCE) {
        rig
          .bufferOf(power)
          .write(
            getGPULocalSpatialHeteroscedasticityParameterValues({residualPower: reference.power})
          );
        for (const scale of [1, 2]) {
          // Residual powers scale by scale**power; both numerator and denominator scale equally.
          rig
            .bufferOf(values)
            .write(
              Float32Array.from(AUDIT_VALUES, value => value * scale + (scale === 2 ? 32 : 0))
            );
          submitGraph(device, compiled, undefined);
          compare(await readFloat32(statistic.buffer, 10), reference.statistic, 'LOSH statistic');
          compare(
            await readFloat32(normalized.buffer, 10),
            reference.normalizedResiduals,
            'LOSH normalized'
          );
          expect(await readUint32(status.buffer, 10)).toEqual([0, 0, 0, 0, 0, 0, 0, 1, 1, 1]);
          expect(await readUint32(count.buffer, 1)).toEqual([7]);
        }
      }
      for (const flags of [
        [9, 0],
        [0, 7],
        [0, 0]
      ]) {
        rig.bufferOf(inputOverflow).write(Uint32Array.of(flags[0]));
        rig.bufferOf(inputCandidateOverflow).write(Uint32Array.of(flags[1]));
        submitGraph(device, compiled, undefined);
        const incomplete = flags.some(Boolean);
        compare(
          await readFloat32(statistic.buffer, 10),
          incomplete ? new Array(10).fill(NaN) : LOSH_REFERENCE[1].statistic,
          'LOSH loss'
        );
        compare(
          await readFloat32(normalized.buffer, 10),
          incomplete ? new Array(10).fill(NaN) : LOSH_REFERENCE[1].normalizedResiduals,
          'LOSH normalized loss'
        );
        expect(await readUint32(status.buffer, 10)).toEqual(
          incomplete ? new Array(10).fill(3) : [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        );
        expect(await readUint32(count.buffer, 1)).toEqual([incomplete ? 0 : 7]);
      }
      // Exact Fraction oracle: every neighbor mean is the constant itself, hence all residuals
      // are zero for either power. Zero normalization mean leaves H and normalized r undefined.
      // The non-dyadic value also covers weighted sums whose multiplication rounds in f32.
      for (const residualPower of [1, 2] as const) {
        rig
          .bufferOf(power)
          .write(getGPULocalSpatialHeteroscedasticityParameterValues({residualPower}));
        for (const constant of [4, 0.1, 0, 2 ** 20]) {
          rig
            .bufferOf(values)
            .write(
              Float32Array.from(AUDIT_VALUES, value => (Number.isFinite(value) ? constant : NaN))
            );
          submitGraph(device, compiled, undefined);
          compare(
            await readFloat32(statistic.buffer, 10),
            new Array(10).fill(NaN),
            `zero residual statistic ${constant} power ${residualPower}`
          );
          compare(
            await readFloat32(normalized.buffer, 10),
            new Array(10).fill(NaN),
            'zero normalized'
          );
          expect(await readUint32(status.buffer, 10)).toEqual([2, 2, 2, 2, 2, 2, 2, 1, 1, 1]);
          expect(await readUint32(count.buffer, 1)).toEqual([7]);
        }
      }
    } finally {
      compiled.destroy();
      rig.destroy();
    }
  });
}

for (const mode of ['tiled', 'reference'] as const) {
  it(`multivariate Geary ${mode} matches exact joint complete-case distances including island moments`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    const makeValues = (affine: boolean, constant = false) =>
      Float32Array.from(
        AUDIT_VALUES.flatMap((value, row) => [
          affine ? 2 * value + 32 : value,
          constant && Number.isFinite(SECOND_VALUES[row])
            ? 3
            : affine
              ? SECOND_VALUES[row] * 0.5 - 8
              : SECOND_VALUES[row]
        ])
      );
    const values = rig.input(makeValues(false), 'float32', 20);
    const statistic = rig.output('float32', 10);
    const status = rig.output('uint32', 10);
    const count = rig.output('uint32', 1);
    const inputOverflow = rig.input(Uint32Array.of(0), 'uint32', 1);
    const inputCandidateOverflow = rig.input(Uint32Array.of(0), 'uint32', 1);
    rig.graph.add(
      new GPUMultivariateLocalGeary({
        mode,
        variableCount: 2,
        values,
        weights: {
          ...uploadWeights(rig, mode === 'tiled'),
          status: {overflow: inputOverflow, candidateOverflow: inputCandidateOverflow}
        },
        mask: rig.input(Uint32Array.from(AUDIT_MASK), 'uint32', 10),
        output: {
          statistic: statistic.view,
          status: status.view,
          ...(mode === 'tiled' ? {includedCount: count.view} : {count: count.view})
        }
      })
    );
    const compiled = rig.graph.compile();
    try {
      for (const affine of [false, true]) {
        rig.bufferOf(values).write(makeValues(affine));
        submitGraph(device, compiled, undefined);
        compare(
          await readFloat32(statistic.buffer, 10),
          MULTIVARIATE_REFERENCE,
          'multivariate Geary'
        );
        expect(await readUint32(status.buffer, 10)).toEqual([0, 0, 1, 0, 0, 0, 0, 1, 2, 1]);
        expect(await readUint32(count.buffer, 1)).toEqual([7]);
      }
      for (const flags of [
        [9, 0],
        [0, 7],
        [0, 0]
      ]) {
        rig.bufferOf(inputOverflow).write(Uint32Array.of(flags[0]));
        rig.bufferOf(inputCandidateOverflow).write(Uint32Array.of(flags[1]));
        submitGraph(device, compiled, undefined);
        const incomplete = flags.some(Boolean);
        compare(
          await readFloat32(statistic.buffer, 10),
          incomplete ? new Array(10).fill(NaN) : MULTIVARIATE_REFERENCE,
          'Geary loss'
        );
        expect(await readUint32(status.buffer, 10)).toEqual(
          incomplete ? new Array(10).fill(3) : [0, 0, 1, 0, 0, 0, 0, 1, 2, 1]
        );
        expect(await readUint32(count.buffer, 1)).toEqual([incomplete ? 0 : 7]);
      }
      rig.bufferOf(values).write(makeValues(false, true));
      submitGraph(device, compiled, undefined);
      compare(await readFloat32(statistic.buffer, 10), new Array(10).fill(NaN), 'constant column');
      expect(await readUint32(status.buffer, 10)).toEqual([2, 2, 1, 2, 2, 2, 2, 1, 2, 1]);
      expect(await readUint32(count.buffer, 1)).toEqual([7]);
    } finally {
      compiled.destroy();
      rig.destroy();
    }
  });
}
