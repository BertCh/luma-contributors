// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {describeOutputs} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
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
    const layout = describeOutputs(
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUOrdinaryLeastSquares'
      },
      'default',
      {rowCount: Math.max(1, rowCount), predictorCount: 15}
    );
    const capacity = (path: string) => {
      const declaration = layout.outputs.find(output => output.path === `output.${path}`);
      if (!declaration) throw new Error(`OLS output layout missing ${path}`);
      return declaration.capacity;
    };
    this.reservation = session.registry.reserveMemory(
      ['fitted', 'coefficients', 'standardErrors', 'tStatistics', 'summary', 'status.code'].reduce(
        (bytes, path) => bytes + capacity(path) * 4,
        0
      ),
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
      this.coefficients = allocate('coefficients', capacity('coefficients'));
      this.standardErrors = allocate('standard-errors', capacity('standardErrors'));
      this.tStatistics = allocate('t-statistics', capacity('tStatistics'));
      this.summary = allocate('summary', capacity('summary'));
      this.status = allocate('status', capacity('status.code'));
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
    const layout = describeOutputs(
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUOrdinaryLeastSquares'
      },
      'default',
      {rowCount: Math.max(1, this.fitted.length), predictorCount}
    );
    const view = <Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      path = name
    ): GraphDataView<Format> => {
      const declaration = layout.outputs.find(output => output.path === `output.${path}`);
      if (!declaration || declaration.format !== format || declaration.activeLength === undefined)
        throw new Error(`OLS output layout missing active ${format} declaration for ${path}`);
      const length = path === 'fitted' ? this.fitted.length : declaration.activeLength;
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
      fitted: view('fitted', 'float32'),
      coefficients: view('coefficients', 'float32'),
      standardErrors: view('standard-errors', 'float32', 'standardErrors'),
      tStatistics: view('t-statistics', 'float32', 'tStatistics'),
      summary: view('summary', 'float32'),
      status: {code: view('status', 'uint32', 'status.code')}
    };
  }

  destroy(): void {
    this.fitted.destroy();
    for (const buffer of this.buffers.values()) buffer.destroy();
    this.reservation.release();
  }
}
