// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getCoreDefaultBuffer} from '../../analysis-core/network-topology';
import {
  getGraphViewChunks,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateDisjointWritableViews,
  type GraphChunkDispatchViews
} from '../../analysis-core/kernels/graph-chunk-dispatch';

/** Logical separation across passes; aligned separation only among fields bound together. */
export function validateSpatialJoinOutputGroups(
  id: string,
  outputs: GraphChunkDispatchViews,
  dispatchGroups: readonly GraphChunkDispatchViews[]
): void {
  const rangesByBuffer = new Map<object, {name: string; first: number; end: number}[]>();
  for (const [name, output] of Object.entries(outputs)) {
    for (const view of output ? getGraphViewChunks(output) : []) {
      if (view.length === 0) continue;
      const buffer = view.buffer.defaultBuffer
        ? getCoreDefaultBuffer(view.buffer.defaultBuffer)
        : view.buffer;
      const ranges = rangesByBuffer.get(buffer) ?? [];
      ranges.push({
        name,
        first: view.byteOffset,
        end: view.byteOffset + (view.length - 1) * view.byteStride + view.rowByteLength
      });
      rangesByBuffer.set(buffer, ranges);
    }
  }
  for (const ranges of rangesByBuffer.values()) {
    ranges.sort((first, second) => first.first - second.first);
    for (let index = 1; index < ranges.length; index++) {
      validateRequirement(
        id,
        ranges[index - 1].end <= ranges[index].first,
        `${ranges[index - 1].name} and ${ranges[index].name} logical output byte ranges must not overlap`
      );
    }
  }
  // These output passes skip empty row views; an unused empty view has no binding conflict.
  for (const group of dispatchGroups)
    validateDisjointWritableViews(
      id,
      Object.fromEntries(Object.entries(group).filter(([, view]) => view?.length !== 0))
    );
}
