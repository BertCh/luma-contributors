// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createWaveHarness} from '../geometry-validity/w5-test-harness';
it.each([false, true])('executor normalizes fractional weights with frexp=%s', withFrexp => {
  const harness = createWaveHarness();
  const input = harness.allocate('weights', 'float32', 2, [0.5, 0.25]);
  const output = harness.allocate('normalized', 'float32x2', 2);
  const node = createWGSLKernelNode(harness.graph, {
    id: 'exponent-only',
    operation: 'ExponentTest',
    invocationCount: 2,
    bindings: [
      {name: 'input', view: input, type: 'f32', access: 'read'},
      {name: 'output', view: output, type: 'f32', access: 'read_write'}
    ],
    body: `let weight=input[inputOffset+index];
 ${withFrexp ? 'let split=frexp(weight);let exponent=split.exp;let fraction=split.fract;' : 'let exponent=i32((bitcast<u32>(weight)>>23u)&255u)-126;let fraction=ldexp(weight,-exponent);'}
 output[outputOffset+2u*index]=fraction;
 output[outputOffset+2u*index+1u]=ldexp(fraction,exponent);`
  });
  try {
    // Binary fractions normalize to .5, with exponents 0 and -1; reconstruction is exact.
    return harness
      .run([node])
      .then(async () => expect(await harness.read(output)).toEqual([0.5, 0.5, 0.5, 0.25]))
      .finally(() => harness.destroy());
  } catch (error) {
    harness.destroy();
    throw error;
  }
});
