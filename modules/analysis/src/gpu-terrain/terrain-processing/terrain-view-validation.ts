// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getViewBindingRange,
  GraphVectorView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';

/** Actual simultaneous storage bindings, excluding unused optional/empty views. @internal */
export type TerrainViewDispatch = {
  reads?: readonly GraphDataView[];
  writes: readonly GraphDataView[];
};

function getTerrainPhysicalBuffer(view: GraphDataView) {
  return view.buffer.defaultBuffer ? getCoreDefaultBuffer(view.buffer.defaultBuffer) : view.buffer;
}

/** Physical storage-usage separation within a dispatch; logical separation across dispatches. @internal */
export function validateTerrainReadWriteRanges(
  id: string,
  inputs: readonly GraphDataView[],
  outputs: readonly GraphDataView[],
  aligned: boolean
): void {
  const groups = new Map<
    ReturnType<typeof getTerrainPhysicalBuffer>,
    {reads: {start: number; end: number}[]; writes: {start: number; end: number}[]}
  >();
  for (const [views, writable] of [
    [inputs, false],
    [outputs, true]
  ] as const) {
    for (const view of views) {
      if (!aligned && view.length === 0) continue;
      const binding = getViewBindingRange(view);
      const start = aligned ? binding.offset : view.byteOffset;
      const end = aligned
        ? start + binding.size
        : start + (view.length - 1) * view.byteStride + view.rowByteLength;
      const buffer = getTerrainPhysicalBuffer(view);
      const group = groups.get(buffer) ?? {reads: [], writes: []};
      (writable ? group.writes : group.reads).push({start, end});
      groups.set(buffer, group);
    }
  }
  for (const group of groups.values()) {
    if (aligned) {
      // A WebGPU usage scope cannot bind one physical buffer as both read-only and writable
      // storage, including disjoint storage-aligned views. Multiple writable bindings are
      // allowed when their binding ranges are disjoint (checked separately below).
      validateRequirement(
        id,
        group.reads.length === 0 || group.writes.length === 0,
        'input and output aligned binding usage scopes must not overlap one physical buffer: read-only and writable storage require separate buffers in the same dispatch'
      );
      continue;
    }
    const reads = group.reads.sort((first, second) => first.start - second.start);
    const writes = group.writes.sort((first, second) => first.start - second.start);
    let readIndex = 0;
    for (const write of writes) {
      while (readIndex < reads.length && reads[readIndex].end <= write.start) readIndex++;
      validateRequirement(
        id,
        readIndex === reads.length || write.end <= reads[readIndex].start,
        'input and output logical byte ranges must not overlap, including separately imported physical aliases'
      );
    }
  }
}

/** Logical separation globally; aligned separation only among bindings in the same dispatch. @internal */
export function validateTerrainOutputDispatches(
  id: string,
  outputs: readonly GraphDataView[],
  dispatches: readonly TerrainViewDispatch[]
): void {
  // Give every output a distinct chunk index. Empty padding is metadata only, and is skipped by
  // the shared chunk validator. This requests logical cross-dispatch checks across all formats.
  const fields: Record<string, GraphVectorView> = {};
  for (const format of new Set(outputs.map(view => view.format))) {
    const template = outputs.find(view => view.format === format)!;
    const empty = (template.buffer.graph as GPUCommandGraph).createDataView(template.buffer, {
      format,
      length: 0
    });
    const data = outputs.map(view => (view.format === format ? view : empty));
    const length = data.reduce((total, view) => total + view.length, 0);
    fields[format] = new GraphVectorView({
      id: format,
      name: format,
      format,
      length,
      valueLength: length,
      stride: 1,
      byteStride: template.byteStride,
      rowByteLength: template.rowByteLength,
      data
    });
  }
  validateDisjointWritableViews(id, fields);
  for (const dispatch of dispatches) {
    validateDisjointWritableViews(
      id,
      Object.fromEntries(dispatch.writes.map((view, index) => [`output[${index}]`, view])),
      {strictAlignedBindings: true}
    );
    validateTerrainReadWriteRanges(id, dispatch.reads ?? [], dispatch.writes, true);
  }
}
