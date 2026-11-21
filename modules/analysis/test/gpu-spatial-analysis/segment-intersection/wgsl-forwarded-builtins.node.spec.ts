// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {EXACT_ORIENTATION_WGSL} from '../../../src/analysis-core/kernels/exact-orientation-wgsl';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createCPUGraphMemory, executeCommandNodesOnCPU} from './wgsl-graph-executor';
import {getUndeclaredCalls} from './wgsl-call-sites';

it('shared WGSL audit recognizes workgroupUniformLoad while retaining missing-call detection', () => {
  expect(
    getUndeclaredCalls('fn main() { let value = workgroupUniformLoad(&word); missing(value); }')
  ).toEqual(['missing']);
});

it('shared interpreter preserves integer vector constructors and exact product limbs', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const output = createTransientView(graph, 'output', 'uint32', 9);
    const memory = createCPUGraphMemory();
    const node = createWGSLKernelNode(graph, {
      id: 'integer-constructors',
      operation: 'test',
      invocationCount: 1,
      bindings: [{name: 'output', view: output, type: 'u32', access: 'read_write'}],
      declarations: EXACT_ORIENTATION_WGSL,
      body: `let explicit = vec2<u32>(0xc0000032u, 0x800001u);
  let shorthand = vec2u(0xc0000032u, 0x800001u);
  let zero = vec2<u32>();
  let splat = vec2<u32>(0xc0000032u);
  output[0] = explicit.x; output[1] = explicit.y;
  output[2] = shorthand.x; output[3] = shorthand.y;
  output[4] = zero.x; output[5] = zero.y;
  output[6] = splat.x; output[7] = splat.y;
  output[8] = multiply24(0x800001u, 0xc65d40u).x;`
    });
    executeCommandNodesOnCPU([node], memory);
    // Exact integer literals; the product's low limb is (0x800001 * 0xc65d40) mod 2^32.
    expect(Array.from(memory.readUint32(output))).toEqual([
      3221225522, 8388609, 3221225522, 8388609, 0, 0, 3221225522, 3221225522, 2697354560
    ]);
  } finally {
    device.destroy();
  }
});

it('shared interpreter supports ldexp without any frexp call', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const output = createTransientView(graph, 'output', 'float32', 1);
    const memory = createCPUGraphMemory();
    const node = createWGSLKernelNode(graph, {
      id: 'ldexp-only',
      operation: 'test',
      invocationCount: 1,
      bindings: [{name: 'output', view: output, type: 'f32', access: 'read_write'}],
      body: 'output[0] = ldexp(0.875, -3i);'
    });
    executeCommandNodesOnCPU([node], memory);
    expect(Array.from(memory.readFloat32(output))).toEqual([0.109375]);
  } finally {
    device.destroy();
  }
});
