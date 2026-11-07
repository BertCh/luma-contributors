// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import type {GPUCommandGraph, GraphDataView} from '@luma-contributors/graph/gpu-core';
import type {GPUHotSpotAnalysisRecipeProps} from '@luma-contributors/graph/gpu-spatial-analysis';
import type {KeplerComputeSession, KeplerMemoryReservation} from '@luma-contributors/kepler';

/** Lazy caller-owned outputs that survive graph replacement and asynchronous inspection. */
export class AnalysisHotspotOutputs {
  readonly pValues: GPUVector<'float32'>;
  readonly bins: GPUVector<'sint32'>;
  readonly neighborCounts: GPUVector<'uint32'>;
  readonly diagnostics: readonly Buffer[];
  private readonly buffers: Buffer[] = [];
  private readonly reservation: KeplerMemoryReservation;

  constructor(session: KeplerComputeSession, chunkLengths: readonly number[]) {
    const rowCount = chunkLengths.reduce((total, length) => total + length, 0);
    this.reservation = session.registry.reserveMemory(rowCount * 12 + 12, 'gpu', 'output');
    const allocate = (id: string, length: number): Buffer => {
      const buffer = session.device.createBuffer({
        id,
        byteLength: Math.max(1, length) * 4,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      this.buffers.push(buffer);
      return buffer;
    };
    try {
      this.pValues = makeAnalysisOutputVector(
        'hotspot-p-values',
        'float32',
        allocate('hotspot-p-values', rowCount),
        chunkLengths
      );
      this.bins = makeAnalysisOutputVector(
        'hotspot-bins',
        'sint32',
        allocate('hotspot-bins', rowCount),
        chunkLengths
      );
      this.neighborCounts = makeAnalysisOutputVector(
        'hotspot-neighbors',
        'uint32',
        allocate('hotspot-neighbors', rowCount),
        chunkLengths
      );
      this.diagnostics = ['weights-overflow', 'weights-required-count', 'invalid-count'].map(name =>
        allocate(`hotspot-${name}`, 1)
      );
    } catch (error) {
      for (const buffer of this.buffers) buffer.destroy();
      this.reservation.release();
      throw error;
    }
  }

  importToGraph(
    graph: GPUCommandGraph,
    zScores: GraphDataView<'float32'>
  ): GPUHotSpotAnalysisRecipeProps['outputs'] {
    const importOutput = <Format extends 'float32' | 'sint32' | 'uint32'>(
      id: string,
      vector: GPUVector<Format>
    ): GraphDataView<Format> => {
      const buffer = vector.data[0].buffer;
      return graph.createDataView(
        graph.importBuffer({id, byteLength: buffer.byteLength, usage: buffer.usage}, buffer),
        {format: vector.format!, length: vector.length}
      );
    };
    const status = this.diagnostics.map((buffer, index) =>
      graph.createDataView(
        graph.importBuffer(
          {id: `hotspot-diagnostic-${index}`, byteLength: buffer.byteLength, usage: buffer.usage},
          buffer
        ),
        {format: 'uint32', length: 1}
      )
    );
    return {
      zScores,
      pValues: importOutput('hotspot-p-values', this.pValues),
      bins: importOutput('hotspot-bins', this.bins),
      neighborCounts: importOutput('hotspot-neighbors', this.neighborCounts),
      status: {
        weightsOverflow: status[0],
        weightsRequiredCount: status[1],
        invalidCount: status[2]
      }
    };
  }

  destroy(): void {
    this.pValues.destroy();
    this.bins.destroy();
    this.neighborCounts.destroy();
    for (const buffer of this.buffers) buffer.destroy();
    this.reservation.release();
  }
}

/** Publish source-aligned chunks over one analysis-owned output buffer. */
export function makeAnalysisOutputVector<
  Format extends 'float32' | 'uint32' | 'sint32' | 'unorm8x4'
>(
  name: string,
  format: Format,
  buffer: Buffer,
  chunkLengths: readonly number[]
): GPUVector<Format> {
  let rowOffset = 0;
  const data = chunkLengths.map(length => {
    const chunk = new GPUData({
      buffer,
      format,
      byteOffset: rowOffset * 4,
      length,
      ownsBuffer: false
    });
    rowOffset += length;
    return chunk;
  });
  return new GPUVector({name, type: 'data', format, data, ownsData: true});
}
