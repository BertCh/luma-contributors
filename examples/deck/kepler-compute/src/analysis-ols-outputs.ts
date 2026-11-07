// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {
  KeplerComputeSession,
  KeplerMemoryReservation,
  KeplerOLSOutputs
} from '@luma-contributors/kepler';
import {makeAnalysisOutputVector} from './analysis-hotspot-outputs';

/** A source-aligned fitted column and bounded model diagnostics, owned across graph replacement. */
export class AnalysisOLSOutputs {
  readonly fitted: GPUVector<'float32'>;
  readonly coefficients: Buffer;
  readonly standardErrors: Buffer;
  readonly tStatistics: Buffer;
  readonly summary: Buffer;
  readonly status: Buffer;
  /** Read coefficient arrays only through predictorCount + 1 entries. */
  readonly diagnostics: readonly [Buffer, Buffer, Buffer, Buffer, Buffer];
  private readonly buffers = new Map<string, Buffer>();
  private readonly reservation: KeplerMemoryReservation;

  constructor(session: KeplerComputeSession, chunkLengths: readonly number[]) {
    const rowCount = chunkLengths.reduce((total, length) => total + length, 0);
    this.reservation = session.registry.reserveMemory(
      Math.max(1, rowCount) * 4 + 3 * 16 * 4 + GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH * 4 + 4,
      'gpu',
      'output'
    );
    const allocate = (name: string, length: number): Buffer => {
      const buffer = session.device.createBuffer({
        id: `ols-${name}`,
        byteLength: Math.max(1, length) * 4,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      this.buffers.set(name, buffer);
      return buffer;
    };
    try {
      this.fitted = makeAnalysisOutputVector(
        'ols-fitted',
        'float32',
        allocate('fitted', rowCount),
        chunkLengths
      );
      this.coefficients = allocate('coefficients', 16);
      this.standardErrors = allocate('standard-errors', 16);
      this.tStatistics = allocate('t-statistics', 16);
      this.summary = allocate('summary', GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH);
      this.status = allocate('status', 1);
      this.diagnostics = [
        this.coefficients,
        this.standardErrors,
        this.tStatistics,
        this.summary,
        this.status
      ];
    } catch (error) {
      for (const buffer of this.buffers.values()) buffer.destroy();
      this.reservation.release();
      throw error;
    }
  }

  importToGraph(
    graph: GPUCommandGraph,
    residuals: GraphDataView<'float32'>,
    predictorCount: number
  ): KeplerOLSOutputs {
    const view = <Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      length: number
    ): GraphDataView<Format> => {
      const buffer = this.buffers.get(name)!;
      return graph.createDataView(
        graph.importBuffer(
          {id: `ols-output-${name}`, byteLength: buffer.byteLength, usage: buffer.usage},
          buffer
        ),
        {format, length}
      );
    };
    return {
      residuals,
      fitted: view('fitted', 'float32', this.fitted.length),
      coefficients: view('coefficients', 'float32', predictorCount + 1),
      standardErrors: view('standard-errors', 'float32', predictorCount + 1),
      tStatistics: view('t-statistics', 'float32', predictorCount + 1),
      summary: view('summary', 'float32', GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH),
      status: {code: view('status', 'uint32', 1)}
    };
  }

  destroy(): void {
    this.fitted.destroy();
    for (const buffer of this.buffers.values()) buffer.destroy();
    this.reservation.release();
  }
}
