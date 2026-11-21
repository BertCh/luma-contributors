// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUData} from '@luma.gl/graph/gpu-data';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/**
 * Reimports persistent input identities on a fresh graph after failed declaration. Transients
 * have no resident identity to move: recreate their producer and preparation handle instead.
 * Supply retryViews when a fresh producer already has a different compatible logical import
 * (for example a core Buffer versus DynamicBuffer, or a narrower usage descriptor). Physical
 * equivalence alone does not establish shared graph hazards; the supplied fresh handle is used.
 * @internal
 */
export function createSpatialPreparationRetryViews<Views extends object, Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  views: Views,
  retryViews?: Partial<Views>
): Views {
  return Object.fromEntries(
    Object.entries(views).map(([name, value]) => {
      const view = value as GraphDataView | undefined;
      if (!view || typeof view !== 'object' || !('buffer' in view)) return [name, value];
      const replacement = retryViews?.[name as keyof Views] as GraphDataView | undefined;
      if (replacement) {
        validateSpatialPreparationRetryView(graph, id, name, view, replacement);
        return [name, replacement];
      }
      if (view.buffer.graph === graph) return [name, value];
      validateRequirement(
        id,
        Boolean(view.buffer.defaultBuffer),
        `${name} retry on a fresh graph requires an imported persistent input; recreate transient producers and this handle`
      );
      return [
        name,
        graph.importGPUData(
          `${id}-retry-${name}`,
          new GPUData({
            buffer: view.buffer.defaultBuffer!,
            format: view.format,
            length: view.length,
            byteOffset: view.byteOffset,
            byteStride: view.byteStride,
            rowByteLength: view.rowByteLength,
            ownsBuffer: false
          })
        )
      ];
    })
  ) as Views;
}

/** Validate physical equivalence separately from the logical handle that tracks graph writes. @internal */
export function validateSpatialPreparationRetryView<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  name: string,
  source: GraphDataView,
  replacement: GraphDataView
): void {
  validateRequirement(
    id,
    Boolean(source.buffer.defaultBuffer),
    `${name} retry requires an imported persistent input; recreate transient producers and this handle`
  );
  validateRequirement(
    id,
    replacement.buffer.graph === graph,
    `${name} retry view must belong to the target graph`
  );
  const sourceBuffer = source.buffer.defaultBuffer;
  const replacementBuffer = replacement.buffer.defaultBuffer;
  validateRequirement(
    id,
    Boolean(
      sourceBuffer &&
        replacementBuffer &&
        getCoreDefaultBuffer(sourceBuffer) === getCoreDefaultBuffer(replacementBuffer) &&
        source.format === replacement.format &&
        source.length === replacement.length &&
        source.byteOffset === replacement.byteOffset &&
        source.byteStride === replacement.byteStride &&
        source.rowByteLength === replacement.rowByteLength
    ),
    `${name} retry view must describe the same physical range, format, length, byteOffset, byteStride and rowByteLength`
  );
}
