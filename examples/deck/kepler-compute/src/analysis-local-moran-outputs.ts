// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {getAnalysisMoranColors} from './analysis-result-colors';
import {packGPUColor} from '@luma-contributors/analysis/dataframe-analysis';
import type {GPULocalMoranOutput} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {
  KeplerComputeSession,
  KeplerMemoryReservation,
  KeplerPointLocalMoranOutputs
} from '@luma-contributors/kepler';
import {makeAnalysisOutputVector} from './analysis-hotspot-outputs';

/** Source-aligned scalar results; the runtime supplies its existing local-I result buffer. */
export class AnalysisLocalMoranOutputs {
  readonly zScores: GPUVector<'float32'>;
  readonly spatialLag: GPUVector<'float32'>;
  readonly pValues: GPUVector<'float32'>;
  readonly adjustedPValues: GPUVector<'float32'>;
  readonly quadrants: GPUVector<'uint32'>;
  readonly neighborCounts: GPUVector<'uint32'>;
  readonly globalStatistics: Buffer;
  readonly invalidCount: Buffer;
  readonly inferenceOverflow: Buffer;
  readonly palette: Buffer;
  private readonly buffers: Buffer[] = [];
  private readonly vectorBuffers = new Map<string, Buffer>();
  private readonly reservation: KeplerMemoryReservation;

  constructor(session: KeplerComputeSession, chunkLengths: readonly number[]) {
    const rowCount = chunkLengths.reduce((total, length) => total + length, 0);
    this.reservation = session.registry.reserveMemory(
      Math.max(1, rowCount) * 24 + 52,
      'gpu',
      'output'
    );
    const allocate = (name: string, length: number): Buffer => {
      const buffer = session.device.createBuffer({
        id: `local-moran-${name}`,
        byteLength: Math.max(1, length) * 4,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      this.buffers.push(buffer);
      return buffer;
    };
    const vector = <Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format
    ): GPUVector<Format> => {
      const buffer = allocate(name, rowCount);
      this.vectorBuffers.set(name, buffer);
      return makeAnalysisOutputVector(`local-moran-${name}`, format, buffer, chunkLengths);
    };
    try {
      this.zScores = vector('z-scores', 'float32');
      this.spatialLag = vector('spatial-lag', 'float32');
      this.pValues = vector('p-values', 'float32');
      this.adjustedPValues = vector('adjusted-p-values', 'float32');
      this.quadrants = vector('quadrants', 'uint32');
      this.neighborCounts = vector('neighbor-counts', 'uint32');
      this.globalStatistics = allocate('global-statistics', 5);
      this.invalidCount = allocate('undefined-count', 1);
      this.inferenceOverflow = allocate('inference-overflow', 1);
      this.palette = allocate('quadrant-palette', 6);
      this.palette.write(
        new Uint32Array(getAnalysisMoranColors().map(color => packGPUColor(...color)))
      );
    } catch (error) {
      for (const buffer of this.buffers) buffer.destroy();
      this.reservation.release();
      throw error;
    }
  }

  importToGraph(
    graph: GPUCommandGraph,
    localI: GraphDataView<'float32'>
  ): GPULocalMoranOutput &
    KeplerPointLocalMoranOutputs & {
      status: {invalidCount: GraphDataView<'uint32'>; inferenceOverflow: GraphDataView<'uint32'>};
    } {
    const importBuffer = <Format extends 'float32' | 'uint32'>(
      name: string,
      buffer: Buffer,
      format: Format,
      length: number
    ): GraphDataView<Format> =>
      graph.createDataView(
        graph.importBuffer(
          {id: `local-moran-${name}`, byteLength: buffer.byteLength, usage: buffer.usage},
          buffer
        ),
        {format, length}
      );
    const importVector = <Format extends 'float32' | 'uint32'>(
      name: string,
      vector: GPUVector<Format>
    ): GraphDataView<Format> =>
      // Import the complete owned allocation; vectors retain the source chunk boundaries.
      importBuffer(name, this.vectorBuffers.get(name)!, vector.format!, vector.length);
    return {
      localI,
      zScores: importVector('z-scores', this.zScores),
      spatialLag: importVector('spatial-lag', this.spatialLag),
      pValues: importVector('p-values', this.pValues),
      adjustedPValues: importVector('adjusted-p-values', this.adjustedPValues),
      quadrants: importVector('quadrants', this.quadrants),
      neighborCounts: importVector('neighbor-counts', this.neighborCounts),
      globalStatistics: importBuffer('global-statistics', this.globalStatistics, 'float32', 5),
      status: {
        invalidCount: importBuffer('undefined-count', this.invalidCount, 'uint32', 1),
        inferenceOverflow: importBuffer('inference-overflow', this.inferenceOverflow, 'uint32', 1)
      }
    };
  }

  destroy(): void {
    for (const vector of [
      this.zScores,
      this.spatialLag,
      this.pValues,
      this.adjustedPValues,
      this.quadrants,
      this.neighborCounts
    ])
      vector.destroy();
    for (const buffer of this.buffers) buffer.destroy();
    this.reservation.release();
  }
}
