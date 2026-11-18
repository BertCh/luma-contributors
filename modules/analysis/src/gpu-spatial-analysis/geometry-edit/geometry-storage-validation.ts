// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Checks derived geometry tables (and nested primitive scratch) before graph mutation. */
export function validateGeometryScratchByteLengths<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  byteLengths: Record<string, number>
): void {
  for (const [name, bytes] of Object.entries(byteLengths)) {
    const byteLength = Math.max(4, bytes);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      byteLength,
      'split the input or reduce capacity'
    );
    validateCapacity(id, `${name} byteLength`, byteLength, 4, graph.device.limits.maxBufferSize);
  }
}

/** Metal forbids a physical buffer with both read-only and writable storage in one dispatch. */
export function validateGeometryReadWriteViews(
  id: string,
  reads: readonly (GraphDataView | undefined)[],
  writes: readonly (GraphDataView | undefined)[]
): void {
  const physical = (view: GraphDataView) =>
    view.buffer.defaultBuffer ? getCoreDefaultBuffer(view.buffer.defaultBuffer) : view.buffer;
  const readBuffers = new Set(
    reads.filter((view): view is GraphDataView => Boolean(view?.length)).map(physical)
  );
  for (const view of writes)
    if (view?.length)
      validateRequirement(
        id,
        !readBuffers.has(physical(view)),
        'read-only and writable storage views in one dispatch must use separate physical buffers'
      );
}

/** Largest radix-sort column or digit histogram; scan summaries are smaller than these. */
export function getGeometrySortScratchByteLengths(rows: number): Record<string, number> {
  return {
    sortColumns: Math.max(1, rows) * 4,
    sortDigitHistogram: Math.max(1, Math.ceil(rows / 256)) * 16 * 4
  };
}

/** Retained inputs are read by later encodings/consumers; separate dispatches protect logical words. */
export function validateGeometryRetainedViews(
  id: string,
  retained: readonly GraphDataView[],
  writes: readonly (GraphDataView | undefined)[]
): void {
  const physical = (view: GraphDataView) =>
    view.buffer.defaultBuffer ? getCoreDefaultBuffer(view.buffer.defaultBuffer) : view.buffer;
  for (const input of retained)
    for (const output of writes)
      if (output?.length && input.length) {
        validateRequirement(
          id,
          physical(input) !== physical(output) ||
            input.byteOffset + input.length * input.rowByteLength <= output.byteOffset ||
            output.byteOffset + output.length * output.rowByteLength <= input.byteOffset,
          'writable outputs must not overlap retained input logical byte ranges'
        );
      }
}

/** Resolves a canonical status port and its deprecated alias without binding/writing it twice. */
export function resolveGeometryUint32StatusView(
  id: string,
  name: string,
  canonical: GraphDataView<'uint32'> | undefined,
  legacy: GraphDataView<'uint32'> | undefined
): GraphDataView<'uint32'> | undefined {
  if (canonical && legacy) {
    const physical = (view: GraphDataView) =>
      view.buffer.defaultBuffer ? getCoreDefaultBuffer(view.buffer.defaultBuffer) : view.buffer;
    validateRequirement(
      id,
      canonical.format === legacy.format &&
        physical(canonical) === physical(legacy) &&
        canonical.byteOffset === legacy.byteOffset &&
        canonical.length === legacy.length &&
        canonical.rowByteLength === legacy.rowByteLength,
      `${name} and its deprecated alias must identify the same uint32 view`
    );
  }
  return canonical ?? legacy;
}
