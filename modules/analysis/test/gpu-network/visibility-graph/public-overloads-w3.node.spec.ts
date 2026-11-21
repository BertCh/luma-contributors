// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it} from 'vitest';
import {
  addVisibilityGraphRecipe,
  getGPUEdgeBundlingParameterValues,
  getNetworkVisibilityGraphExecutionPlan,
  type GPUNetworkVisibilityGraphCSRProps
} from '@luma-contributors/analysis/gpu-network';
import type {GPUVisibilityGraphProps} from '@luma-contributors/analysis/gpu-spatial-analysis';

it('TYP-10 keeps literal packer returns precise and accepts a supported format union', () => {
  expectTypeOf(getGPUEdgeBundlingParameterValues({}, 'float32')).toEqualTypeOf<Float32Array>();
  expectTypeOf(getGPUEdgeBundlingParameterValues({}, 'uint32')).toEqualTypeOf<Uint32Array>();
  for (const format of ['float32', 'uint32'] as const) {
    const values = getGPUEdgeBundlingParameterValues({}, format);
    expectTypeOf(values).toEqualTypeOf<Float32Array | Uint32Array>();
    expect(values).toBeInstanceOf(format === 'float32' ? Float32Array : Uint32Array);
  }
  expect(() => getGPUEdgeBundlingParameterValues({}, 'other' as 'uint32')).toThrow(
    /format.*uint32.*float32.*other/
  );
});

// These calls are checked by the lead's test typecheck; no GPU allocation is needed for types.
function checkRecipeOverloads(
  graph: GPUCommandGraph,
  reference: GPUVisibilityGraphProps,
  csr: GPUNetworkVisibilityGraphCSRProps,
  selected: GPUVisibilityGraphProps | GPUNetworkVisibilityGraphCSRProps
) {
  const preciseCSR = addVisibilityGraphRecipe(graph, csr);
  const preciseReference = addVisibilityGraphRecipe(graph, reference);
  const union = addVisibilityGraphRecipe(graph, selected);
  expectTypeOf(preciseCSR.outputs.betweenness).toEqualTypeOf<GraphDataView<'float32'>>();
  expectTypeOf(preciseReference.outputs.adjacency).toEqualTypeOf<GraphDataView<'uint32'>>();
  expectTypeOf(union).toEqualTypeOf<typeof preciseCSR | typeof preciseReference>();
}
void checkRecipeOverloads;

it('ENT-14 public visibility forwarding exposes preflight beside qualified contract types', () => {
  expect(getNetworkVisibilityGraphExecutionPlan).toBe(addVisibilityGraphRecipe.getExecutionPlan);
});
