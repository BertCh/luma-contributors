// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPU_UPPER_INCOMPLETE_GAMMA_WGSL} from '../../../src/analysis-core/kernels/incomplete-gamma-wgsl';
import {GET_NAN_WGSL, IS_NAN_BITS_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

function getLogGamma(z: number): number {
  // Lanczos approximation in double precision.
  const coefficients = [
    676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
  ];
  const shifted = z - 1;
  let sum = 0.99999999999980993;
  coefficients.forEach((coefficient, index) => {
    sum += coefficient / (shifted + index + 1);
  });
  const t = shifted + coefficients.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (shifted + 0.5) * Math.log(t) - t + Math.log(sum);
}

/** Double precision regularized upper incomplete gamma Q(a, x) (series or continued fraction). */
function getChiSquareSurvival(value: number, degreesOfFreedom: number): number {
  const a = degreesOfFreedom / 2;
  const x = value / 2;
  const logPrefactor = -x + a * Math.log(x) - getLogGamma(a);
  if (x < a + 1) {
    let term = 1 / a;
    let sum = term;
    for (let iteration = 1; iteration < 100000 && term > sum * 1e-15; iteration++) {
      term = (term * x) / (a + iteration);
      sum += term;
    }
    return 1 - sum * Math.exp(logPrefactor);
  }
  const tiny = 1e-300;
  let b = x + 1 - a;
  let c = 1 / tiny;
  let d = 1 / b;
  let h = d;
  for (let iteration = 1; iteration < 100000; iteration++) {
    const an = -iteration * (iteration - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < tiny) d = tiny;
    c = b + an / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < 1e-15) break;
  }
  return Math.exp(logPrefactor) * h;
}

it('chi-square survival stays within 5e-4 of double precision for very large degrees of freedom', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const cases: [number, number][] = [];
  for (const degreesOfFreedom of [500, 1000, 1999, 2001, 4000, 8000]) {
    for (const offset of [-3, -1, 0, 1, 3]) {
      const value = degreesOfFreedom - 0.67 + offset * Math.sqrt(2 * degreesOfFreedom);
      cases.push([Math.fround(value), degreesOfFreedom]);
    }
  }
  const graph = new GPUCommandGraph(device);
  const valuesBuffer = createInputBuffer(device, new Float32Array(cases.map(([value]) => value)));
  const dofBuffer = createInputBuffer(device, new Float32Array(cases.map(([, dof]) => dof)));
  const outputBuffer = createOutputBuffer(device, cases.length);
  graph.add({
    getCommandNodes: (target: GPUCommandGraph) => [
      createWGSLKernelNode(target, {
        id: 'chi-square-survival',
        operation: 'ChiSquareSurvivalTest',
        invocationCount: cases.length,
        bindings: [
          {
            name: 'values',
            view: importGraphBuffer(target, 'values', valuesBuffer, 'float32', cases.length),
            type: 'f32',
            access: 'read'
          },
          {
            name: 'dofs',
            view: importGraphBuffer(target, 'dofs', dofBuffer, 'float32', cases.length),
            type: 'f32',
            access: 'read'
          },
          {
            name: 'result',
            view: importGraphBuffer(target, 'result', outputBuffer, 'float32', cases.length),
            type: 'f32',
            access: 'read_write'
          }
        ],
        declarations: `${GET_NAN_WGSL}${IS_NAN_BITS_WGSL}${GPU_UPPER_INCOMPLETE_GAMMA_WGSL}`,
        body: 'result[resultOffset + index] = getChiSquareSurvivalValue(values[valuesOffset + index], dofs[dofsOffset + index]);'
      })
    ]
  } as never);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const actual = await readFloat32(outputBuffer, cases.length);
    cases.forEach(([value, degreesOfFreedom], index) => {
      const expected = getChiSquareSurvival(value, degreesOfFreedom);
      expect(
        Math.abs(actual[index] - expected),
        `dof ${degreesOfFreedom} value ${value}`
      ).toBeLessThan(5e-4);
    });
  } finally {
    compiled.destroy();
    [valuesBuffer, dofBuffer, outputBuffer].forEach(buffer => buffer.destroy());
  }
});
