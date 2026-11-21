// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {POISSON_OBSERVED_TERM_WGSL} from '../../src/analysis-core/kernels/wgsl-poisson-observed-term';
import {createWGSLKernelNode} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {createNullWebGPUDevice} from './gpu-contributor-test-utils';

// CORE-104: getObservedTerm evaluates e ((1 + d) ln(1 + d) - d) with d = (o - e) / e. For a large
// count over a tiny (normal float32) baseline d, or (1 + d) ln(1 + d), overflows and the term was
// NaN or +inf although o ln(o / e) - o + e is finite. The shared WGSL runs in the wgsl_reflect
// interpreter (float32 arithmetic). References are float64 o (ln o - ln e) - o + e of the float32
// inputs (/usr/bin/python3 with NumPy float32 rounding of the inputs):
//   (1e9, f32(1e-30)) 88800818623.5967; (1e8, f32(1e-30)) 8649823353.060265;
//   (1e6, f32(1e-36)) 95708573.86981079; (f32(5e37), f32(1e-3)) 4.6356e39, beyond float32.
// The headless twin is wgsl-poisson-observed-term.spec.ts.

function runObservedTerm(pairs: readonly (readonly [number, number])[]): number[] {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const inputs = createTransientView(graph, 'inputs', 'float32', pairs.length * 2);
    const output = createTransientView(graph, 'output', 'float32', pairs.length);
    const memory = createCPUGraphMemory();
    memory.write(inputs, Float32Array.from(pairs.flat()));
    const node = createWGSLKernelNode(graph, {
      id: 'observed-term',
      operation: 'PoissonObservedTermTest',
      invocationCount: pairs.length,
      bindings: [
        {name: 'inputs', view: inputs, type: 'f32', access: 'read'},
        {name: 'output', view: output, type: 'f32', access: 'read_write'}
      ],
      declarations: POISSON_OBSERVED_TERM_WGSL,
      body: 'output[outputOffset + index] = getObservedTerm(inputs[inputsOffset + 2u * index], inputs[inputsOffset + 2u * index + 1u]);'
    });
    executeCommandNodesOnCPU([node], memory);
    return Array.from(memory.readFloat32(output));
  } finally {
    device.destroy();
  }
}

it('CORE-104 the Poisson observed term stays finite when its ratio overflows', () => {
  const values = runObservedTerm([
    [1e9, 1e-30],
    [1e8, 1e-30],
    [1e6, 1e-36],
    [5e37, 1e-3],
    // Ordinary inputs keep the previous branch: o = 2e gives e (2 ln 2 - 1).
    [20, 10]
  ]);
  const references = [88800818623.5967, 8649823353.060265, 95708573.86981079];
  references.forEach((reference, index) => {
    expect(Number.isFinite(values[index]), `pair ${index}`).toBe(true);
    expect(Math.abs(values[index] - reference)).toBeLessThanOrEqual(1e-6 * reference);
  });
  // The term itself exceeds float32: +inf, not NaN.
  expect(values[3]).toBe(Infinity);
  expect(Math.abs(values[4] - 10 * (2 * Math.LN2 - 1))).toBeLessThan(1e-5);
});

it('CORE-104 the fallback runs only when the ratio form is not finite', () => {
  const source = POISSON_OBSERVED_TERM_WGSL.replace(/\s+/g, ' ');
  expect(source).toContain(
    'if ((bitcast<u32>(term) & 0x7f800000u) != 0x7f800000u) { return term; }'
  );
  expect(source).toContain(
    'return observed * (log(observed) - log(expected)) - observed + expected;'
  );
});
