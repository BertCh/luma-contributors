// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import {
  getGraphViewChunks,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {SpaceTimeKrigingRowViews} from './gpu-space-time-kriging';

/**
 * All output fields and chunks publish in separate dispatches, each with one writable caller
 * binding. Their aligned binding prefixes may overlap; their logical writable bytes may not.
 * Empty chunks never dispatch. Resolve imported aliases to the underlying buffer too.
 * @internal
 */
export function validateSpaceTimeKrigingOutputs(
  id: string,
  output: SpaceTimeKrigingRowViews['output']
): void {
  const rangesByBuffer = new Map<
    Buffer | GraphDataView['buffer'],
    {name: string; offset: number; end: number}[]
  >();
  for (const [field, view] of Object.entries(output)) {
    for (const [chunkIndex, chunk] of getGraphViewChunks(view).entries()) {
      if (chunk.length === 0) continue;
      const buffer = chunk.buffer.defaultBuffer
        ? getCoreDefaultBuffer(chunk.buffer.defaultBuffer)
        : chunk.buffer;
      const ranges = rangesByBuffer.get(buffer) ?? [];
      ranges.push({
        name: `output.${field} chunk ${chunkIndex}`,
        offset: chunk.byteOffset,
        end: chunk.byteOffset + chunk.length * chunk.rowByteLength
      });
      rangesByBuffer.set(buffer, ranges);
    }
  }
  for (const ranges of rangesByBuffer.values()) {
    ranges.sort((first, second) => first.offset - second.offset);
    for (let rangeIndex = 1; rangeIndex < ranges.length; rangeIndex++) {
      const previous = ranges[rangeIndex - 1];
      const current = ranges[rangeIndex];
      validateRequirement(
        id,
        previous.end <= current.offset,
        `${previous.name} and ${current.name} byte ranges must not overlap`
      );
    }
  }
}
