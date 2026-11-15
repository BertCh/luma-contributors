// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getViewBindingRange,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateGPUStorageBindingByteLength,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateRequirement,
  validateSinglePackedView,
  validateViewLength
} from './gpu-contributor-utils';

/** Default WebGPU storage binding limit; fitting inputs are deliberately minibatch sized. */
export const GPU_FITTING_MAXIMUM_BINDING_BYTES = 128 * 1024 * 1024;
/** Tier-zero fitted model budget from the new-contributors research (400 kB of weights). */
export const GPU_FITTING_MAXIMUM_PARAMETER_COUNT = 100_000;
/** Counts remain exactly representable in float32 when normalizing gradients. */
export const GPU_FITTING_MAXIMUM_ROW_COUNT = 2 ** 24;
/** Planning work budget at 0.1 trillion multiply-accumulates per second. */
export const GPU_FITTING_MAXIMUM_WORK = 5_000_000_000;

/** Validate a scalar packed fitting port at construction. @internal */
export function validateFittingView(
  id: string,
  name: string,
  view: GraphDataView,
  format: 'float32' | 'uint32',
  length: number
): void {
  validateSinglePackedView(view, [format], `${id} ${name}`);
  validateViewLength(id, name, view, length);
  validateMaximumCount(
    id,
    `${name} byteLength`,
    getViewBindingRange(view).size,
    GPU_FITTING_MAXIMUM_BINDING_BYTES
  );
}

/** Fitting writable ports use distinct buffers, including persistent optimizer state. @internal */
export function validateFittingWritableViews(
  id: string,
  writable: readonly GraphDataView[],
  inputs: readonly GraphDataView[]
): void {
  validateRequirement(
    id,
    new Set(writable.map(view => view.buffer)).size === writable.length,
    'writable views must use separate buffers'
  );
  validateRequirement(
    id,
    writable.every(output => inputs.every(input => output.buffer !== input.buffer)),
    'writable views must use separate buffers from read-only inputs'
  );
}

/** Check device limits before allocating any transient view or command node. @internal */
export function validateFittingGraph<Parameters>(
  id: string,
  graph: GPUCommandGraph<Parameters>,
  views: readonly GraphDataView[]
): void {
  validateGraphViewsBelongToGraph(id, graph, views);
  for (const [index, view] of views.entries()) {
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      `view ${index}`,
      getViewBindingRange(view).size,
      'reduce the minibatch or feature count'
    );
  }
}
