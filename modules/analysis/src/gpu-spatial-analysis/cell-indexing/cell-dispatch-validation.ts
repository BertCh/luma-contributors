// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import {
  isGraphDataView,
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

type CellFields = Readonly<Record<string, GraphDataView | undefined>>;

/** Physical storage identity also covers separately imported views of the same default buffer. @internal */
export function getCellDispatchBuffer(view: GraphDataView): unknown {
  return view.buffer.defaultBuffer ? getCoreDefaultBuffer(view.buffer.defaultBuffer) : view.buffer;
}

/** Metal usage scopes forbid read-only and writable storage bindings of one physical buffer. @internal */
export function validateCellReadWriteDispatch(
  id: string,
  reads: CellFields,
  writes: CellFields
): void {
  for (const [readName, readView] of Object.entries(reads)) {
    if (!readView) continue;
    for (const [writeName, writeView] of Object.entries(writes))
      if (writeView)
        validateRequirement(
          id,
          getCellDispatchBuffer(readView) !== getCellDispatchBuffer(writeView),
          `${readName} and ${writeName} must not share a physical buffer as read-only and writable storage in one dispatch`
        );
  }
}

/** Logical output ranges must be disjoint; only co-bound fields need separate aligned bindings. @internal */
export function validateCellDispatchOutputs(
  id: string,
  outputs: CellFields,
  dispatches: readonly CellFields[]
): void {
  const ranges = new Map<unknown, {name: string; begin: number; end: number}[]>();
  for (const [name, view] of Object.entries(outputs)) {
    if (!view || view.length === 0) continue;
    const buffer = getCellDispatchBuffer(view);
    const fields = ranges.get(buffer) ?? [];
    fields.push({
      name,
      begin: view.byteOffset,
      end: view.byteOffset + (view.length - 1) * view.byteStride + view.rowByteLength
    });
    ranges.set(buffer, fields);
  }
  for (const fields of ranges.values()) {
    fields.sort((first, second) => first.begin - second.begin);
    for (let index = 1; index < fields.length; index++) {
      validateRequirement(
        id,
        fields[index - 1].end <= fields[index].begin,
        `${fields[index - 1].name} and ${fields[index].name} logical output byte ranges must not overlap`
      );
    }
  }
  for (const fields of dispatches) validateDisjointWritableViews(id, fields);
}

/** Checks actual storage bindings and every derived scratch plane before graph declaration. @internal */
export function validateCellDispatchBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  fields: CellFields,
  scratch: readonly (readonly [string, number])[] = []
): void {
  validateGraphChunkBindingRanges(graph, id, fields);
  for (const [name, view] of Object.entries(fields))
    if (view) {
      validateCapacity(
        id,
        `${name} buffer byteLength`,
        view.buffer.byteLength,
        0,
        graph.device.limits.maxBufferSize
      );
    }
  for (const [name, byteLength] of scratch) {
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      Math.max(4, byteLength),
      'reduce input rows or output capacity'
    );
    validateCapacity(
      id,
      `${name} byteLength`,
      Math.max(4, byteLength),
      0,
      graph.device.limits.maxBufferSize
    );
  }
}

/** Flattens packed geometry/status records without adding graph resources. @internal */
export function getCellPackedFields(
  record: object,
  prefix = ''
): Record<string, GraphDataView | undefined> {
  const fields: Record<string, GraphDataView | undefined> = {};
  for (const [name, value] of Object.entries(record)) {
    const path = prefix ? `${prefix}.${name}` : name;
    if (isGraphDataView(value)) fields[path] = value;
    else if (value && typeof value === 'object')
      Object.assign(fields, getCellPackedFields(value, path));
  }
  return fields;
}
