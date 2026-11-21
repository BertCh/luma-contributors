// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUGlobalSpatialStatistics} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {
  GLOBAL_AUDIT_VALUES,
  GLOBAL_AUDIT_SECOND,
  GLOBAL_AUDIT_MASK,
  GLOBAL_AUDIT_WEIGHTS,
  GLOBAL_AUDIT_REFERENCE
} from './global-mathematical-audit-fixture';

const VARIANCE_FIELDS = new Set([10, 13, 18, 21, 26, 29, 37, 45, 49]);
function compare(actual: readonly number[], expected: readonly number[]) {
  expect(actual.length).toBe(expected.length);
  for (const [index, value] of expected.entries()) {
    if (Number.isNaN(value)) expect(actual[index], `field ${index}`).toBeNaN();
    else {
      expect(Number.isFinite(actual[index]), `field ${index}`).toBe(true);
      // Compensated f32 reductions plus fourth-order moments: stated 1e-4 relative contract.
      // Variance absolute error stays below 1e-10 so writing zero cannot pass a small variance.
      const tolerance = (VARIANCE_FIELDS.has(index) ? 1e-10 : 2e-6) + 1e-4 * Math.abs(value);
      expect(
        Math.abs(actual[index] - value),
        `field ${index}: ${actual[index]} vs ${value}`
      ).toBeLessThanOrEqual(tolerance);
    }
  }
}

function createAudit(
  device: Device,
  mode: 'reference' | 'tiled',
  route: 'whole' | 'pairwise' | 'transpose',
  reference: (typeof GLOBAL_AUDIT_REFERENCE)[number],
  alternative: 'two-sided' | 'greater' | 'lesser' = 'two-sided'
) {
  const rig = new WeightsRig(device);
  const whole = rig.uploadWeights(GLOBAL_AUDIT_WEIGHTS, 3);
  const chunks = [0, 3].map((rowStart, index) => {
    const rowEnd = index === 0 ? 3 : 10;
    const start = GLOBAL_AUDIT_WEIGHTS.offsets[rowStart],
      end = GLOBAL_AUDIT_WEIGHTS.offsets[rowEnd];
    return {
      rowStart,
      rowCount: rowEnd - rowStart,
      weights: rig.uploadWeights(
        {
          offsets: GLOBAL_AUDIT_WEIGHTS.offsets
            .slice(rowStart, rowEnd + 1)
            .map(offset => offset - start),
          neighbors: GLOBAL_AUDIT_WEIGHTS.neighbors.slice(start, end),
          weights: GLOBAL_AUDIT_WEIGHTS.weights.slice(start, end),
          distances: []
        },
        2
      )
    };
  });
  const finalLoss = rig.input(Uint32Array.of(0), 'uint32', 1),
    candidateLoss = rig.input(Uint32Array.of(0), 'uint32', 1);
  const weights = {
    ...(route === 'whole' ? whole : {rowCount: 10, chunks}),
    information: reference.standardized
      ? {rowCount: 10, transform: 'row' as const, symmetric: 'unknown' as const}
      : undefined,
    status: {overflow: finalLoss, candidateOverflow: candidateLoss}
  };
  const values = rig.input(Float32Array.from(GLOBAL_AUDIT_VALUES), 'float32', 10),
    second = rig.input(Float32Array.from(GLOBAL_AUDIT_SECOND), 'float32', 10);
  const summary = rig.output('float32', 52),
    counts = rig.output('uint32', 3),
    joins = rig.output('uint32', 3),
    overflow = rig.output('uint32', 1),
    candidate = rig.output('uint32', 1);
  rig.graph.add(
    new GPUGlobalSpatialStatistics({
      mode,
      analyticAlternative: alternative,
      weights,
      values,
      secondValues: second,
      islands: reference.islands,
      symmetryStrategy: route === 'whole' ? 'auto' : route,
      mask: rig.input(Uint32Array.from(GLOBAL_AUDIT_MASK), 'uint32', 10),
      statistics: ['moran', 'geary', 'getisOrdG', 'bivariateMoran', 'joinCount'],
      output: {
        summary: summary.view,
        counts: counts.view,
        joinCounts: joins.view,
        status: {overflow: overflow.view, candidateOverflow: candidate.view}
      }
    })
  );
  const compiled = rig.graph.compile();
  return {
    rig,
    compiled,
    values,
    second,
    finalLoss,
    candidateLoss,
    summary,
    counts,
    joins,
    overflow,
    candidate,
    whole,
    chunks
  };
}

for (const mode of ['reference', 'tiled'] as const) {
  it(`${mode} global statistics match every exhaustive/normality oracle field in every CSR route`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const route of ['whole', 'pairwise', 'transpose'] as const)
      for (const reference of GLOBAL_AUDIT_REFERENCE) {
        const context = createAudit(device, mode, route, reference);
        const {rig, compiled} = context;
        try {
          for (const scale of [1, 2 ** 20]) {
            rig
              .bufferOf(context.values)
              .write(Float32Array.from(GLOBAL_AUDIT_VALUES, value => value * scale));
            rig
              .bufferOf(context.second)
              .write(Float32Array.from(GLOBAL_AUDIT_SECOND, value => value * scale));
            submitGraph(device, compiled, undefined);
            compare(
              await readFloat32(context.summary.buffer, 52),
              reference.summary.map((value, index) =>
                index === 4 ? value * scale : index === 5 ? value * scale * scale : value
              )
            );
            expect(await readUint32(context.counts.buffer, 3)).toEqual(reference.counts);
            expect(await readUint32(context.joins.buffer, 3)).toEqual(reference.joinCounts);
            expect(await readUint32(context.overflow.buffer, 1)).toEqual([0]);
            expect(await readUint32(context.candidate.buffer, 1)).toEqual([0]);
          }
          rig.bufferOf(context.values).write(Float32Array.from(GLOBAL_AUDIT_VALUES));
          rig.bufferOf(context.second).write(new Float32Array(10).fill(5));
          submitGraph(device, compiled, undefined);
          const constant = Array.from(reference.summary);
          for (const field of [32, 37, 38, 39]) constant[field] = NaN;
          compare(await readFloat32(context.summary.buffer, 52), constant);
          rig.bufferOf(context.second).write(Float32Array.from(GLOBAL_AUDIT_SECOND));
          const padded = Float32Array.from(
            route === 'whole'
              ? GLOBAL_AUDIT_WEIGHTS.weights
              : GLOBAL_AUDIT_WEIGHTS.weights.slice(0, 11)
          );
          padded[10] = Infinity;
          rig
            .bufferOf(route === 'whole' ? context.whole.weights : context.chunks[0].weights.weights)
            .write(padded);
          submitGraph(device, compiled, undefined);
          compare(await readFloat32(context.summary.buffer, 52), reference.summary);
          for (const flags of [
            [7, 0],
            [0, 9],
            [0, 0]
          ]) {
            rig.bufferOf(context.finalLoss).write(Uint32Array.of(flags[0]));
            rig.bufferOf(context.candidateLoss).write(Uint32Array.of(flags[1]));
            submitGraph(device, compiled, undefined);
            const expected = reference.summary.map((value, index) =>
              flags.some(Boolean) && ![0, 6, 7].includes(index) ? NaN : value
            );
            compare(await readFloat32(context.summary.buffer, 52), expected);
            expect(await readUint32(context.counts.buffer, 3)).toEqual(reference.counts);
            expect(await readUint32(context.joins.buffer, 3)).toEqual(
              flags.some(Boolean) ? [0, 0, 0] : reference.joinCounts
            );
            expect(await readUint32(context.overflow.buffer, 1)).toEqual([Number(flags[0] !== 0)]);
            expect(await readUint32(context.candidate.buffer, 1)).toEqual([Number(flags[1] !== 0)]);
          }
        } finally {
          compiled.destroy();
          rig.destroy();
        }
      }
  }, 180000);
}

for (const alternative of ['greater', 'lesser'] as const) {
  it(`global ${alternative} normal tails match independently enumerated z-scores`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const mode of ['reference', 'tiled'] as const) {
      const reference = GLOBAL_AUDIT_REFERENCE[0];
      const context = createAudit(device, mode, 'whole', reference, alternative);
      try {
        submitGraph(device, context.compiled, undefined);
        const expected = Array.from(reference.summary);
        // Exact normal symmetry from independent two-sided erfc literals, not production p.
        for (const field of [12, 15, 20, 23, 28, 31, 39, 47, 51]) {
          const positive = expected[field - 1] > 0;
          const upper = positive ? expected[field] / 2 : 1 - expected[field] / 2;
          expected[field] = alternative === 'greater' ? upper : 1 - upper;
        }
        compare(await readFloat32(context.summary.buffer, 52), expected);
      } finally {
        context.compiled.destroy();
        context.rig.destroy();
      }
    }
  });
}
