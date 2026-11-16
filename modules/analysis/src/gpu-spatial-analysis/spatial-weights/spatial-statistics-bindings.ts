// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GraphDataView, GraphVectorView, type GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  getGraphViewChunks,
  validateGPUStorageBindingByteLength,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCoreDefaultBuffer} from '../../analysis-core/network-topology';

/**
 * Preflights the metadata of every caller view in a statistics configuration, including nested
 * CSR chunks and status ports. Typed arrays are never traversed: this visits props/chunks, not
 * dataset rows. Scratch sizes are explicit analysis-layout byte lengths supplied by the caller.
 * @internal
 */
export function validateSpatialStatisticsBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: object,
  scratch: Readonly<Record<string, number>> = {}
): void {
  const fields: Record<string, GraphDataView | GraphVectorView> = {};
  const seen = new Set<object>();
  const visit = (value: unknown, name: string): void => {
    if (value instanceof GraphDataView || value instanceof GraphVectorView) {
      fields[name] = value;
      return;
    }
    if (!value || typeof value !== 'object' || ArrayBuffer.isView(value) || seen.has(value)) return;
    seen.add(value);
    for (const [key, child] of Object.entries(value)) visit(child, name ? `${name}.${key}` : key);
  };
  visit(props, '');
  validateGraphChunkBindingRanges(graph, id, fields);
  const imports = new Map<object, {name: string; buffer: GraphDataView['buffer']}>();
  for (const [name, field] of Object.entries(fields)) {
    for (const view of getGraphViewChunks(field)) {
      if (!view.buffer.defaultBuffer) continue;
      const physical = getCoreDefaultBuffer(view.buffer.defaultBuffer);
      const previous = imports.get(physical);
      validateRequirement(
        id,
        !previous || previous.buffer === view.buffer,
        `${name} and ${previous?.name} share a physical buffer: import it once and use offset views`
      );
      imports.set(physical, {name, buffer: view.buffer});
    }
  }
  for (const [name, byteLength] of Object.entries(scratch))
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      byteLength,
      'reduce the statistics input or capacity, or request larger device bindings'
    );
}
