// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPULOSHResidualPermutationTest} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  GPULocalPermutationTest,
  GPUGlobalPermutationTest,
  getGPUPermutationParameterValues
} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  GLOBAL_AUDIT_VALUES,
  GLOBAL_AUDIT_SECOND,
  GLOBAL_AUDIT_MASK,
  GLOBAL_AUDIT_WEIGHTS
} from '../global-spatial-statistics/global-mathematical-audit-fixture';
import {
  HANDOFF_REFERENCE,
  HANDOFF_BINARY_SECOND,
  HANDOFF_GLOBAL_LEE,
  LOSH_HANDOFF_REFERENCE,
  LOSH_HANDOFF_TAILS
} from './permutation-handoff-audit-fixture';

function compare(actual: readonly number[], expected: readonly number[]) {
  expect(actual.length).toBe(expected.length);
  for (const [index, value] of expected.entries())
    if (Number.isNaN(value)) expect(actual[index]).toBeNaN();
    else {
      expect(Number.isFinite(actual[index])).toBe(true);
      expect(Math.abs(actual[index] - value)).toBeLessThanOrEqual(2e-5 + 1e-5 * Math.abs(value));
    }
}

it('adapter observed values match independent local and global definitions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const weights = rig.uploadWeights(GLOBAL_AUDIT_WEIGHTS);
  const parameters = rig.input(
    getGPUPermutationParameterValues({seed: 43, permutationCount: 19, significanceLevel: 0.6}),
    'uint32',
    4
  );
  const mask = rig.input(Uint32Array.from(GLOBAL_AUDIT_MASK), 'uint32', 10);
  const outputs = [];
  for (const name of [
    'localBivariateMoran',
    'localLeeL',
    'localJoinCount',
    'BJC',
    'CLC'
  ] as const) {
    const binary = ['localJoinCount', 'BJC', 'CLC'].includes(name);
    const observed = rig.output('float32', 10);
    outputs.push({name, observed});
    rig.graph.add(
      new GPULocalPermutationTest({
        id: `handoff-${name}`,
        weights,
        parameters,
        mask,
        maximumPermutations: 19,
        maximumNeighbors: 4,
        values: rig.input(
          Float32Array.from(GLOBAL_AUDIT_VALUES, value =>
            binary && Number.isFinite(value) ? Number(value !== 0) : value
          ),
          'float32',
          10
        ),
        secondValues:
          name === 'localJoinCount'
            ? undefined
            : rig.input(
                Float32Array.from(binary ? HANDOFF_BINARY_SECOND : GLOBAL_AUDIT_SECOND),
                'float32',
                10
              ),
        statistic: name === 'BJC' || name === 'CLC' ? 'localBivariateJoinCount' : name,
        joinCase: name === 'BJC' || name === 'CLC' ? name : undefined,
        output: {
          observed: observed.view,
          exceedances: rig.output('uint32', 10).view,
          pseudoPValues: rig.output('float32', 10).view,
          status: {overflow: rig.output('uint32', 1).view}
        }
      })
    );
  }
  const global = rig.output('float32', 12);
  rig.graph.add(
    new GPUGlobalPermutationTest({
      id: 'handoff-global-Lee',
      weights,
      parameters,
      mask,
      maximumPermutations: 19,
      statistic: 'leesL',
      values: rig.input(Float32Array.from(GLOBAL_AUDIT_VALUES), 'float32', 10),
      secondValues: rig.input(Float32Array.from(GLOBAL_AUDIT_SECOND), 'float32', 10),
      output: {summary: global.view}
    })
  );
  const compiled = rig.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    for (const output of outputs)
      compare(await readFloat32(output.observed.buffer, 10), HANDOFF_REFERENCE[output.name]);
    compare((await readFloat32(global.buffer, 12)).slice(0, 1), [HANDOFF_GLOBAL_LEE]);
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});

for (const tail of LOSH_HANDOFF_TAILS) {
  it(`LOSH ${tail.alternative} passes the fixed residual field to inference with every output checked`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const reference of LOSH_HANDOFF_REFERENCE) {
      const rig = new WeightsRig(device);
      const values = rig.input(Float32Array.of(0, 1, 4, 10, 22, 4, 99, NaN), 'float32', 8);
      const candidate = rig.input(Uint32Array.of(0), 'uint32', 1);
      const statistic = rig.output('float32', 8),
        normalized = rig.output('float32', 8),
        status = rig.output('uint32', 8),
        residualCount = rig.output('uint32', 1);
      const exceedances = rig.output('uint32', 8),
        probabilities = rig.output('float32', 8),
        adjusted = rig.output('float32', 8),
        significant = rig.output('uint32', 8),
        overflow = rig.output('uint32', 1),
        invalid = rig.output('uint32', 1);
      const weightsCause = rig.output('uint32', 1);
      const neighborCause = rig.output('uint32', 1);
      const invalidCause = rig.output('uint32', 1);
      const candidateCause = rig.output('uint32', 1);
      rig.graph.add(
        new GPULOSHResidualPermutationTest({
          values,
          residualPower: reference.power,
          alternative: tail.alternative,
          weights: {
            ...rig.uploadWeights({
              offsets: [0, 2, 4, 6, 8, 10, 11, 12, 13],
              neighbors: [1, 4, 0, 2, 1, 3, 2, 4, 0, 3, 0, 6, 0],
              weights: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 99, 1],
              distances: []
            }),
            status: {candidateOverflow: candidate}
          },
          mask: rig.input(Uint32Array.of(1, 1, 1, 1, 1, 0, 1, 1), 'uint32', 8),
          maximumPermutations: 19,
          maximumNeighbors: 2,
          parameters: rig.input(
            getGPUPermutationParameterValues({
              seed: 43,
              permutationCount: 19,
              significanceLevel: 0.6
            }),
            'uint32',
            4
          ),
          output: {
            statistic: statistic.view,
            status: status.view,
            normalizedResiduals: normalized.view,
            residualCount: residualCount.view,
            inference: {
              exceedances: exceedances.view,
              pseudoPValues: probabilities.view,
              adjustedPValues: adjusted.view,
              significant: significant.view,
              status: {
                overflow: overflow.view,
                invalidCount: invalid.view,
                weightsOverflow: weightsCause.view,
                neighborOverflow: neighborCause.view,
                invalidInput: invalidCause.view,
                candidateOverflow: candidateCause.view
              }
            }
          }
        })
      );
      const compiled = rig.graph.compile();
      try {
        for (const scale of [1, 2]) {
          rig
            .bufferOf(values)
            .write(
              Float32Array.from(
                [0, 1, 4, 10, 22, 4, 99, NaN],
                value => scale * value + (scale === 2 ? 32 : 0)
              )
            );
          for (const loss of [0, 9, 0]) {
            rig.bufferOf(candidate).write(Uint32Array.of(loss));
            submitGraph(device, compiled, undefined);
            compare(
              await readFloat32(statistic.buffer, 8),
              loss ? new Array(8).fill(NaN) : [...reference.statistic, NaN, NaN, NaN]
            );
            compare(
              await readFloat32(normalized.buffer, 8),
              loss ? new Array(8).fill(NaN) : [...reference.normalized, NaN, NaN, NaN]
            );
            expect(await readUint32(status.buffer, 8)).toEqual(
              loss ? new Array(8).fill(3) : [0, 0, 0, 0, 0, 1, 1, 1]
            );
            expect(await readUint32(residualCount.buffer, 1)).toEqual([loss ? 0 : 5]);
            expect(await readUint32(exceedances.buffer, 8)).toEqual(
              loss
                ? new Array(8).fill(0xffffffff)
                : [...tail.exceedances, 0xffffffff, 0xffffffff, 0xffffffff]
            );
            compare(
              await readFloat32(probabilities.buffer, 8),
              loss ? new Array(8).fill(NaN) : [...tail.probabilities, NaN, NaN, NaN]
            );
            compare(
              await readFloat32(adjusted.buffer, 8),
              loss ? new Array(8).fill(NaN) : [...tail.probabilities, NaN, NaN, NaN]
            );
            expect(await readUint32(significant.buffer, 8)).toEqual(
              loss ? new Array(8).fill(0) : [...tail.significant, 0, 0, 0]
            );
            expect(await readUint32(overflow.buffer, 1)).toEqual([Number(loss !== 0)]);
            expect(await readUint32(invalid.buffer, 1)).toEqual([0]);
            expect(await readUint32(weightsCause.buffer, 1)).toEqual([0]);
            expect(await readUint32(neighborCause.buffer, 1)).toEqual([0]);
            expect(await readUint32(invalidCause.buffer, 1)).toEqual([0]);
            expect(await readUint32(candidateCause.buffer, 1)).toEqual([Number(loss !== 0)]);
          }
        }
      } finally {
        compiled.destroy();
        rig.destroy();
      }
    }
  });
}
