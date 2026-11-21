// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  isGraphDataView,
  isGraphVectorView,
  validateCapacity,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {type GraphDataView, GraphVectorView, type GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges,
  type GraphChunkDispatchViews
} from '../../analysis-core/kernels/graph-chunk-dispatch';

/** Collects trajectory-family view metadata; never visits rows or graph/resource objects. */
export function getTrajectoryBindingViews(value: object): GraphChunkDispatchViews {
  const views: Record<string, GraphDataView | GraphVectorView> = {};
  const visit = (item: unknown, path: string): void => {
    if (isGraphDataView(item) || isGraphVectorView(item)) {
      views[path] = item;
    } else if (
      item &&
      typeof item === 'object' &&
      (Array.isArray(item) || Object.getPrototypeOf(item) === Object.prototype)
    ) {
      for (const [name, child] of Object.entries(item))
        visit(child, path ? `${path}.${name}` : name);
    }
  };
  visit(value, '');
  return views;
}

/** Preflights every caller view, including storage usage and aligned containment, before declaration. */
export function validateTrajectoryBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: object
): void {
  validateGraphChunkBindingRanges(graph, id, getTrajectoryBindingViews(props));
}

/**
 * Checks aligned writable ranges within each dispatch, and logical writes across dispatches.
 * The aggregate vector is metadata only: each packed output is a separate logical chunk, so
 * the shared validator compares logical ranges across it without forbidding adjacent writes.
 */
export function validateTrajectoryWritableDispatches(
  id: string,
  groups: readonly GraphChunkDispatchViews[]
): void {
  const fields = [...new Map(groups.flatMap(group => Object.entries(group))).values()].filter(
    (view): view is GraphDataView => isGraphDataView(view) && view.length > 0
  );
  const data = fields.map(view =>
    (view.buffer.graph as GPUCommandGraph).createDataView(view.buffer, {
      format: 'uint32',
      byteOffset: view.byteOffset,
      length: (view.length * view.rowByteLength) / 4
    })
  );
  const length = data.reduce((total, view) => total + view.length, 0);
  validateDisjointWritableViews(id, {
    outputs: new GraphVectorView({
      id: `${id}-writable-ranges`,
      name: 'outputs',
      format: 'uint32',
      length,
      valueLength: length,
      stride: 1,
      byteStride: 4,
      rowByteLength: 4,
      data
    })
  });
  for (const group of groups) validateDisjointWritableViews(id, group);
}

/** Rejects logical output aliasing even when fields are published by different kernels. */
export function validateTrajectoryOutputRanges(id: string, output: object): void {
  const fields = getTrajectoryBindingViews(output);
  validateTrajectoryWritableDispatches(
    id,
    Object.entries(fields).map(([name, view]) => ({[name]: view}))
  );
}

/** Checks derived trajectory records before creating any scratch buffer. */
export function validateTrajectoryScratch<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  sizes: Readonly<Record<string, number>>
): void {
  for (const [name, requestedBytes] of Object.entries(sizes)) {
    const byteLength = Math.max(4, requestedBytes);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      byteLength,
      'reduce the input or output capacity'
    );
    validateCapacity(
      id,
      `${name} allocation bytes`,
      byteLength,
      4,
      graph.device.limits.maxBufferSize
    );
  }
}
