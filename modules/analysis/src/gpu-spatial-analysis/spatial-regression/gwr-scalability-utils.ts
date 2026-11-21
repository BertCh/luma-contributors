// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GWR_GRID_EXTENT_COMPONENT_COUNT} from './gwr-grid-layout';
import {Buffer, type Device} from '@luma.gl/core';
import {getViewBindingRange, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  validateFiniteNumber,
  validateGPUStorageBindingByteLength,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUApproximationStatusPort} from '../../analysis-core/contracts/status';

import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';

/** Largest finite f32 literal; JavaScript's finite range is much larger than WGSL's. */
export const GWR_MAXIMUM_SHADER_FLOAT = 3.4028234663852886e38;
/** Smallest positive f32 literal; smaller positive props round to zero in the shader. */
export const GWR_MINIMUM_POSITIVE_SHADER_FLOAT = 2 ** -149;

/** Validate compile-time positive distances before any resource or shader declaration. @internal */
export function validateGWRPositiveShaderFloat(id: string, name: string, value: number): void {
  validateFiniteNumber(
    id,
    name,
    value,
    GWR_MINIMUM_POSITIVE_SHADER_FLOAT,
    GWR_MAXIMUM_SHADER_FLOAT
  );
}

/** Flatten the named status for internal binding and range validation. @internal */
export function getGWRBindingOutputs<Output extends {status?: GPUApproximationStatusPort}>(
  output: Output
): Omit<Output, 'status'> & GPUApproximationStatusPort {
  const {status, ...fields} = output;
  return {...fields, approximation: status?.approximation};
}

/** Reject logical overlaps globally and aligned overlaps only for outputs bound together. @internal */
export function validateGWROutputs(
  id: string,
  outputs: readonly (GraphDataView | undefined)[],
  dispatches: readonly {
    writes: readonly (GraphDataView | undefined)[];
    reads?: readonly (GraphDataView | undefined)[];
  }[] = [{writes: outputs}]
): void {
  for (const dispatch of dispatches) {
    validateDisjointWritableViews(
      id,
      Object.fromEntries(dispatch.writes.map((view, index) => [`write[${index}]`, view]))
    );
    for (const read of dispatch.reads ?? []) {
      if (!read) continue;
      const readBuffer = read.buffer.defaultBuffer
        ? getCoreDefaultBuffer(read.buffer.defaultBuffer)
        : read.buffer;
      for (const write of dispatch.writes) {
        if (!write) continue;
        const writeBuffer = write.buffer.defaultBuffer
          ? getCoreDefaultBuffer(write.buffer.defaultBuffer)
          : write.buffer;
        validateRequirement(
          id,
          readBuffer !== writeBuffer,
          'outputs must not bind one physical buffer read-only and writable in the same dispatch'
        );
      }
    }
  }
  // There are at most eighteen GWR output fields. Their logical ranges must remain disjoint
  // even across dispatches, including when separate imported handles share a core buffer.
  const ranges = outputs.flatMap((view, index) =>
    view && view.length > 0
      ? [
          {
            index,
            buffer: view.buffer.defaultBuffer
              ? getCoreDefaultBuffer(view.buffer.defaultBuffer)
              : view.buffer,
            start: view.byteOffset,
            end: view.byteOffset + (view.length - 1) * view.byteStride + view.rowByteLength
          }
        ]
      : []
  );
  for (let first = 0; first < ranges.length; first++) {
    for (let second = first + 1; second < ranges.length; second++) {
      const left = ranges[first];
      const right = ranges[second];
      validateRequirement(
        id,
        left.buffer !== right.buffer || left.start >= right.end || right.start >= left.end,
        `output[${left.index}] and output[${right.index}] logical writable byte ranges overlap`
      );
    }
  }
}

/** Preflight every caller binding and scratch shape before graph declaration. @internal */
export function validateGWRBindings(
  device: Device,
  id: string,
  views: readonly (GraphDataView | undefined)[],
  scratch: readonly (readonly [string, number])[]
): void {
  for (const [index, view] of views.entries()) {
    if (view) {
      validateRequirement(
        id,
        (view.buffer.usage & Buffer.STORAGE) !== 0,
        `view[${index}] buffer usage must include Buffer.STORAGE`
      );
      validateGPUStorageBindingByteLength(
        device,
        id,
        `view[${index}] aligned binding`,
        getViewBindingRange(view).size,
        'reduce the packed view length'
      );
      validateRequirement(
        id,
        view.buffer.byteLength <= device.limits.maxBufferSize,
        `view[${index}] backing buffer needs ${view.buffer.byteLength} bytes (maxBufferSize ${device.limits.maxBufferSize})`
      );
    }
  }
  for (const [name, byteLength] of scratch) {
    validateGPUStorageBindingByteLength(
      device,
      id,
      name,
      byteLength,
      'reduce the row count or capacities'
    );
    validateRequirement(
      id,
      byteLength <= device.limits.maxBufferSize,
      `${name} scratch needs ${byteLength} bytes (maxBufferSize ${device.limits.maxBufferSize})`
    );
  }
}

/**
 * Global grid scratch, including the published stable radix sort and scan upper bounds. Radix
 * histograms have at most 16 bins per 256-row workgroup, and scan levels only shrink. @internal
 */
export function getGWRGridScratchSizes(
  rowCount: number,
  gridSize: readonly [number, number] | null
): readonly (readonly [string, number])[] {
  if (!gridSize) return [];
  return [
    ['grid positions and normalized positions', rowCount * 8],
    [
      'grid tile bounds',
      Math.ceil(Math.ceil(rowCount / 256) / 64) *
        64 *
        GWR_GRID_EXTENT_COMPONENT_COUNT *
        Float32Array.BYTES_PER_ELEMENT
    ],
    ['grid bounds', GWR_GRID_EXTENT_COMPONENT_COUNT * Float32Array.BYTES_PER_ELEMENT],
    ['grid cell counts, offsets and scan scratch', (gridSize[0] * gridSize[1] + 1) * 4],
    ['grid object IDs, cell keys, row IDs and sort ping-pong', rowCount * 4],
    [
      'grid sorted IDs and extent',
      (rowCount + GWR_GRID_EXTENT_COMPONENT_COUNT) * Uint32Array.BYTES_PER_ELEMENT
    ],
    ['grid radix histograms, offsets and scan scratch', Math.ceil(rowCount / 256) * 16 * 4]
  ];
}
