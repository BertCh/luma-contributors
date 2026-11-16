// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GraphDataView, GraphVectorView, type GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  validateGraphChunkBindingRanges,
  validateDisjointWritableViews,
  type GraphChunkDispatchViews
} from '../analysis-core/kernels/graph-chunk-dispatch';
import {getCoreDefaultBuffer} from '../analysis-core/network-topology';
import {
  getGraphViewChunks,
  validateGPUStorageBindingByteLength
} from '../analysis-core/kernels/gpu-contributor-utils';

/**
 * Validates output ranges without traversing graph handles or textures. With dispatch groups,
 * compare logical ranges globally, aligned writable ranges per dispatch, and physical read/write
 * separation per dispatch. Omit optional views that a dispatch does not bind. @internal
 */
export function validateNetworkWritableViews(
  id: string,
  output: object,
  dispatches?: readonly {outputs: object; inputs?: object}[]
): void {
  const fields: Record<string, GraphDataView | GraphVectorView> = {};
  const collect = (name: string, value: unknown): void => {
    if (value instanceof GraphDataView || value instanceof GraphVectorView) fields[name] = value;
    else if (
      value &&
      typeof value === 'object' &&
      (Object.getPrototypeOf(value) === Object.prototype || Array.isArray(value))
    ) {
      for (const [field, child] of Object.entries(value)) collect(`${name}.${field}`, child);
    }
  };
  collect('output', output);
  validateDisjointWritableViews(id, fields as GraphChunkDispatchViews, {
    separateDispatches: Boolean(dispatches)
  });
  for (const [index, dispatch] of dispatches?.entries() ?? []) {
    validateNetworkWritableViews(`${id} dispatch ${index}`, dispatch.outputs);
    validateNetworkOutputsDisjointFromInputs(
      id,
      getNetworkViews(dispatch.outputs),
      getNetworkViews(dispatch.inputs)
    );
  }
}

/** Checks caller STORAGE usage, ownership, aligned ranges and derived scratch before declaration. @internal */
export function validateNetworkViewBindings<Parameters>(
  id: string,
  graph: GPUCommandGraph<Parameters>,
  views: readonly (GraphDataView | GraphVectorView | undefined)[],
  scratchByteLengths: Readonly<Record<string, number>> = {}
): void {
  const physicalHandles = new Map<unknown, GraphDataView['buffer']>();
  for (const view of views) {
    for (const chunk of view ? getGraphViewChunks(view) : []) {
      const physical = getNetworkPhysicalBuffer(chunk);
      const previous = physicalHandles.get(physical);
      if (previous && previous !== chunk.buffer) {
        throw new Error(`${id} a physical buffer must be imported once`);
      }
      physicalHandles.set(physical, chunk.buffer);
    }
  }
  validateGraphChunkBindingRanges(
    graph,
    id,
    Object.fromEntries(views.map((view, index) => [`view ${index}`, view]))
  );
  for (const [name, byteLength] of Object.entries(scratchByteLengths)) {
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      Math.max(4, byteLength),
      'reduce batch size or output capacity'
    );
  }
}

/** Element alignment that keeps batch blocks on 256-byte storage boundaries. @internal */
export const NETWORK_ALIGNMENT_ELEMENTS = 64;

/** Returns a packed range of `length` rows starting at row `firstRow` of a packed view. @internal */
export function getSubView<Format extends 'uint32' | 'float32', Parameters>(
  graph: GPUCommandGraph<Parameters>,
  view: GraphDataView<Format>,
  firstRow: number,
  length: number
): GraphDataView<Format> {
  return graph.createDataView(view.buffer, {
    format: view.format as Format,
    length,
    byteOffset: view.byteOffset + firstRow * view.byteStride
  });
}

/** Rounds an element count up to a multiple of {@link NETWORK_ALIGNMENT_ELEMENTS}. @internal */
export function alignElementCount(count: number): number {
  return Math.ceil(count / NETWORK_ALIGNMENT_ELEMENTS) * NETWORK_ALIGNMENT_ELEMENTS;
}

/** Collects graph views from plain contributor property/output bags. @internal */
export function getNetworkViews(value: unknown): (GraphDataView | GraphVectorView)[] {
  if (value instanceof GraphDataView || value instanceof GraphVectorView) return [value];
  if (
    value &&
    typeof value === 'object' &&
    (Object.getPrototypeOf(value) === Object.prototype || Array.isArray(value))
  ) {
    return Object.values(value).flatMap(getNetworkViews);
  }
  return [];
}

/** Uses the physical allocation identity for imported views, graph identity for scratch. @internal */
export function getNetworkPhysicalBuffer(view: GraphDataView): unknown {
  return view.buffer.defaultBuffer ? getCoreDefaultBuffer(view.buffer.defaultBuffer) : view.buffer;
}

/** A WebGPU dispatch cannot bind one physical buffer both read-only and writable. @internal */
export function validateNetworkOutputsDisjointFromInputs(
  id: string,
  outputs: readonly (GraphDataView | GraphVectorView | undefined)[],
  inputs: readonly (GraphDataView | GraphVectorView | undefined)[]
): void {
  const physicalInputs = new Set(
    inputs.flatMap(view => (view ? getGraphViewChunks(view).map(getNetworkPhysicalBuffer) : []))
  );
  for (const view of outputs) {
    if (
      view &&
      getGraphViewChunks(view).some(chunk => physicalInputs.has(getNetworkPhysicalBuffer(chunk)))
    ) {
      throw new Error(`${id} outputs must not share buffers with inputs (physical storage)`);
    }
  }
}
