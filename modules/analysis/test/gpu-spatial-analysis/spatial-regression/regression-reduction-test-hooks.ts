// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUReduction, GraphVectorView, type GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterEach, beforeEach, vi} from 'vitest';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

let nativeNodeDelta = 0;
/** Difference between native cooperative declaration count and the CPU max stand-in. */
export function getRegressionReductionNativeNodeDelta(): number {
  return nativeNodeDelta;
}

/** CPU-interpreter stand-in for the trusted upstream magnitude max (its workgroup has barriers). */
export function useRegressionReductionTestHooks(): void {
  let restore: (() => void) | undefined;
  beforeEach(() => {
    nativeNodeDelta = 0;
    const original = GPUReduction.prototype.getCommandNodes;
    const spy = vi.spyOn(GPUReduction.prototype, 'getCommandNodes').mockImplementation(function <
      Parameters
    >(this: GPUReduction, graph: GPUCommandGraph<Parameters>) {
      const nativeNodes = (original<Parameters>).call(this, graph);
      if (
        this.operation !== 'max' ||
        this.input instanceof GraphVectorView ||
        !['float32', 'uint32'].includes(this.input.format) ||
        this.mask
      ) {
        return nativeNodes;
      }
      // Count the real upstream declaration for tests of contributor command estimates.
      nativeNodeDelta += nativeNodes.length - 1;
      const unsigned = this.input.format === 'uint32';
      const scalarType = unsigned ? ('u32' as const) : ('f32' as const);
      const zero = unsigned ? '0u' : '0.0';
      return [
        createWGSLKernelNode(graph, {
          id: `${this.id}-cpu-max`,
          operation: 'RegressionCPUPrimitive',
          invocationCount: 1,
          bindings: [
            {name: 'values', view: this.input, type: scalarType, access: 'read'},
            {name: 'maximum', view: this.output, type: scalarType, access: 'read_write'}
          ],
          body: `var value = ${unsigned ? '0u' : '-3.402823466e38'};
  for (var row = 0u; row < ${this.input.length}u; row++) { value = max(value, values[valuesOffset + row]); }
  maximum[maximumOffset] = select(value, ${zero}, ${this.input.length}u == 0u);`
        })
      ];
    });
    restore = () => spy.mockRestore();
  });
  afterEach(() => {
    restore?.();
    restore = undefined;
  });
}
