// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {getInt64TimeWords} from '../../../src/dataframe-analysis/time-window-filter/time-words';
import {
  createInputBuffer,
  createOutputBuffer,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

/** Trajectory port inputs as CPU columns. */
export type TrajectoryColumns = {
  positions: number[][] | Float32Array;
  /** Integer times in the timestamps' unit (Int64 words) or `Float32Array` relative times. */
  times: bigint[] | Float32Array;
  trackOffsets: number[];
};

/** A graph harness that tracks every buffer and destroys them after readback. */
export class GraphHarness<Parameters = undefined> {
  readonly graph: GPUCommandGraph<Parameters>;
  private readonly buffers: Buffer[] = [];

  constructor(
    readonly device: Device,
    id: string
  ) {
    this.graph = new GPUCommandGraph(device, {id}) as GPUCommandGraph<Parameters>;
  }

  /** Imports a float32 or uint32 column as a graph view. */
  input<Format extends 'float32' | 'uint32' | 'float32x2' | 'uint32x2'>(
    id: string,
    values: Float32Array | Uint32Array,
    format: Format,
    length: number
  ): GraphDataView<Format> {
    const buffer = createInputBuffer(this.device, values);
    this.buffers.push(buffer);
    return importGraphBuffer(this.graph, id, buffer, format, length) as GraphDataView<Format>;
  }

  /** Imports a fresh writable column and returns its buffer for readback. */
  output<Format extends 'float32' | 'uint32' | 'float32x2'>(
    id: string,
    format: Format,
    length: number
  ): {view: GraphDataView<Format>; buffer: Buffer} {
    const elements = format === 'float32x2' ? length * 2 : length;
    const buffer = createOutputBuffer(this.device, elements);
    this.buffers.push(buffer);
    return {
      view: importGraphBuffer(this.graph, id, buffer, format, length) as GraphDataView<Format>,
      buffer
    };
  }

  /** Compiles and submits; call the reader while the buffers are alive. */
  async run<Result>(read: () => Promise<Result>): Promise<Result> {
    const compiled = this.graph.compile();
    try {
      submitGraph(this.device, compiled, undefined);
      return await read();
    } finally {
      compiled.destroy();
      for (const buffer of this.buffers) {
        buffer.destroy();
      }
    }
  }
}

/** Imports a trajectory port's columns. */
export function importTrajectory(harness: GraphHarness, columns: TrajectoryColumns) {
  const rowCount =
    columns.positions instanceof Float32Array
      ? columns.positions.length / 2
      : columns.positions.length;
  const positionValues =
    columns.positions instanceof Float32Array
      ? columns.positions
      : Float32Array.from(columns.positions.flat());
  const positions = harness.input('positions', positionValues, 'float32x2', rowCount);
  const timestamps =
    columns.times instanceof Float32Array
      ? harness.input('timestamps', columns.times, 'float32', rowCount)
      : harness.input(
          'timestamps',
          getInt64TimeWords(BigInt64Array.from(columns.times)),
          'uint32x2',
          rowCount
        );
  const trackOffsets = harness.input(
    'track-offsets',
    Uint32Array.from(columns.trackOffsets),
    'uint32',
    columns.trackOffsets.length
  );
  return {positions, timestamps, trackOffsets, rowCount};
}
