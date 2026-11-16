// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getCoreDefaultBuffer} from '../../analysis-core/network-topology';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/**
 * Logical separation across dispatches. Aligned separation for fields bound together is checked
 * separately with validateDisjointWritableViews. Optional geometry is deliberately not included
 * in the read-only set: centrality never binds or reads it. @internal
 */
export function validateCentralityLogicalRanges(
  id: string,
  outputs: Readonly<Record<string, GraphDataView | undefined>>,
  inputs: Readonly<Record<string, GraphDataView | undefined>>
): void {
  const ranges = (views: typeof outputs) =>
    Object.entries(views).flatMap(([name, view]) =>
      view && view.length > 0
        ? [
            {
              name,
              buffer: view.buffer.defaultBuffer
                ? getCoreDefaultBuffer(view.buffer.defaultBuffer)
                : view.buffer,
              start: view.byteOffset,
              end: view.byteOffset + (view.length - 1) * view.byteStride + view.rowByteLength
            }
          ]
        : []
    );
  const writable = ranges(outputs);
  const readable = ranges(inputs);
  const overlaps = (first: (typeof writable)[number], second: (typeof writable)[number]) =>
    first.buffer === second.buffer && first.start < second.end && second.start < first.end;
  for (const [index, first] of writable.entries()) {
    for (const second of writable.slice(index + 1))
      validateRequirement(
        id,
        !overlaps(first, second),
        `outputs must use disjoint logical byte ranges for ${first.name} and ${second.name} (separate buffers or disjoint packed fields)`
      );
    for (const second of readable)
      validateRequirement(
        id,
        !overlaps(first, second),
        `outputs must not overlap input logical byte ranges for ${first.name} and ${second.name} (separate buffers or disjoint packed fields)`
      );
  }
}
