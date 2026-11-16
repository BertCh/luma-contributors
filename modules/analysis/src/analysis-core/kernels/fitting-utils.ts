// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getViewBindingRange,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateMaximumCount,
  validateGraphOutputsDisjointFromInputs,
  validateSinglePackedView,
  validateViewLength
} from './gpu-contributor-utils';

import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from './graph-chunk-dispatch';

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

/** Co-bound writable ports need disjoint aligned ranges and separate read-only buffers. @internal */
export function validateFittingWritableViews(
  id: string,
  writable: readonly GraphDataView[],
  inputs: readonly GraphDataView[]
): void {
  validateDisjointWritableViews(
    id,
    Object.fromEntries(writable.map((view, index) => [`output ${index}`, view]))
  );
  validateGraphOutputsDisjointFromInputs(id, writable, inputs);
}

/** Check device limits before allocating any transient view or command node. @internal */
export function validateFittingGraph<Parameters>(
  id: string,
  graph: GPUCommandGraph<Parameters>,
  views: readonly GraphDataView[]
): void {
  validateGraphChunkBindingRanges(
    graph,
    id,
    Object.fromEntries(views.map((view, index) => [`view ${index}`, view]))
  );
}
