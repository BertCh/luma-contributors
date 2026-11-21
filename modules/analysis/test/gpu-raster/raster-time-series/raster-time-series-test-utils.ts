// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

type ViewFormat = 'float32' | 'uint32' | 'uint32x2' | 'sint32';

export function createRasterTimeSeriesHarness(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const outputs = new Map<string, {buffer: Buffer; length: number; integer: boolean}>();
  let serial = 0;
  const importView = <Format extends ViewFormat>(
    buffer: Buffer,
    format: Format,
    length: number
  ) => {
    buffers.push(buffer);
    const handle = graph.importBuffer(
      {id: `rts-${serial++}`, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    return graph.createDataView(handle, {format, length});
  };
  const destroy = () => {
    outputs.clear();
    for (const buffer of buffers.splice(0)) {
      buffer.destroy();
    }
  };
  return {
    graph,
    destroy,
    floats(values: ArrayLike<number>) {
      return importView(
        createInputBuffer(device, Float32Array.from(values)),
        'float32',
        values.length
      );
    },
    integers(values: ArrayLike<number>) {
      return importView(
        createInputBuffer(device, Uint32Array.from(values)),
        'uint32',
        values.length
      );
    },
    /** `times` as `uint32x2` Int64 `(low, high)` rows. */
    words(times: BigInt64Array) {
      const words = new Uint32Array(times.length * 2);
      times.forEach((time, index) => {
        const unsigned = BigInt.asUintN(64, time);
        words[index * 2] = Number(unsigned & 0xffffffffn);
        words[index * 2 + 1] = Number(unsigned >> 32n);
      });
      return importView(createInputBuffer(device, words), 'uint32x2', times.length);
    },
    output<Format extends 'float32' | 'uint32' | 'sint32'>(
      name: string,
      format: Format,
      length: number
    ) {
      const buffer = createOutputBuffer(device, length);
      outputs.set(name, {buffer, length, integer: format !== 'float32'});
      return importView(buffer, format, length);
    },
    async run(contributors: GPUCommandNodeProducer | GPUCommandNodeProducer[]) {
      let compiled: ReturnType<typeof graph.compile> | undefined;
      try {
        for (const contributor of Array.isArray(contributors) ? contributors : [contributors]) {
          graph.add(contributor);
        }
        compiled = graph.compile();
        submitGraph(device, compiled, undefined);
        const result: Record<string, number[]> = {};
        for (const [name, entry] of outputs) {
          result[name] = await (entry.integer
            ? readUint32(entry.buffer, entry.length)
            : readFloat32(entry.buffer, entry.length));
        }
        return result;
      } finally {
        try {
          compiled?.destroy();
        } finally {
          destroy();
        }
      }
    }
  };
}

/**
 * Largest absolute difference between two equally long series, treating NaN as equal to NaN and
 * equal infinities as equal. Throws when the lengths differ, so an empty or short readback cannot
 * report zero error. A NaN on one side only returns `Infinity`, which fails every tolerance.
 */
export function getMaximumAbsoluteDifference(
  left: ArrayLike<number>,
  right: ArrayLike<number>
): number {
  if (left.length !== right.length) {
    throw new Error(
      `getMaximumAbsoluteDifference: lengths differ (${left.length} and ${right.length})`
    );
  }
  let largest = 0;
  for (let index = 0; index < left.length; index++) {
    const a = left[index];
    const b = right[index];
    if (a === b || (Number.isNaN(a) && Number.isNaN(b))) continue;
    const difference = Math.abs(a - b);
    if (Number.isNaN(difference)) return Infinity;
    largest = Math.max(largest, difference);
  }
  return largest;
}
