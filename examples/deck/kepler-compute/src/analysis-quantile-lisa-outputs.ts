// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {packGPUColor} from '@luma-contributors/analysis/dataframe-analysis';
import type {GPUQuantileLISAOutput} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {KeplerComputeSession, KeplerMemoryReservation} from '@luma-contributors/kepler';
import {makeAnalysisOutputVector} from './analysis-hotspot-outputs';

/** Explicitly owned packed outputs exposed as original source chunks for table and export. */
export class AnalysisQuantileLisaOutputs {
  readonly observed: GPUVector<'float32'>;
  readonly membership: GPUVector<'uint32'>;
  readonly classes: GPUVector<'uint32'>;
  readonly exceedances: GPUVector<'uint32'>;
  readonly significant: GPUVector<'uint32'>;
  readonly palette: Buffer;
  readonly diagnostics: readonly Buffer[];
  private readonly buffers = new Map<string, Buffer>();
  private readonly reservation: KeplerMemoryReservation;

  constructor(session: KeplerComputeSession, chunkLengths: readonly number[]) {
    const rowCount = chunkLengths.reduce((total, length) => total + length, 0);
    this.reservation = session.registry.reserveMemory(
      Math.max(1, rowCount) * 20 + 40,
      'gpu',
      'output'
    );
    const allocate = (name: string, length: number): Buffer => {
      const buffer = session.device.createBuffer({
        id: `quantile-lisa-${name}`,
        byteLength: Math.max(1, length) * 4,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      this.buffers.set(name, buffer);
      return buffer;
    };
    const vector = <Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format
    ): GPUVector<Format> =>
      makeAnalysisOutputVector(
        `quantile-lisa-${name}`,
        format,
        allocate(name, rowCount),
        chunkLengths
      );
    try {
      this.observed = vector('observed', 'float32');
      this.membership = vector('membership', 'uint32');
      this.classes = vector('classes', 'uint32');
      this.exceedances = vector('exceedances', 'uint32');
      this.significant = vector('significant', 'uint32');
      this.diagnostics = [
        allocate('boundaries', 2),
        allocate('valid-count', 1),
        allocate('island-count', 1),
        allocate('invalid-quantile', 1),
        allocate('overflow', 1)
      ];
      this.palette = allocate('palette', 4);
      this.palette.write(
        new Uint32Array([
          packGPUColor(150, 150, 150), // 0: not significant or not tested
          packGPUColor(215, 25, 28), // 1: significant under the recipe's explicit p-value alternative
          packGPUColor(65, 65, 65), // 2: undefined
          packGPUColor(253, 174, 97) // 3: isolated
        ])
      );
    } catch (error) {
      for (const buffer of this.buffers.values()) buffer.destroy();
      this.reservation.release();
      throw error;
    }
  }

  importToGraph(
    graph: GPUCommandGraph,
    pseudoPValues: GraphDataView<'float32'>
  ): GPUQuantileLISAOutput {
    const view = <Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      length: number
    ): GraphDataView<Format> => {
      const buffer = this.buffers.get(name)!;
      return graph.createDataView(
        graph.importBuffer(
          {id: `quantile-lisa-output-${name}`, byteLength: buffer.byteLength, usage: buffer.usage},
          buffer
        ),
        {format, length}
      );
    };
    const length = this.observed.length;
    return {
      pseudoPValues,
      observed: view('observed', 'float32', length),
      membership: view('membership', 'uint32', length),
      classes: view('classes', 'uint32', length),
      exceedances: view('exceedances', 'uint32', length),
      significant: view('significant', 'uint32', length),
      boundaries: view('boundaries', 'float32', 2),
      validCount: view('valid-count', 'uint32', 1),
      islandCount: view('island-count', 'uint32', 1),
      invalidQuantile: view('invalid-quantile', 'uint32', 1),
      status: {overflow: view('overflow', 'uint32', 1)}
    };
  }

  destroy(): void {
    for (const vector of [
      this.observed,
      this.membership,
      this.classes,
      this.exceedances,
      this.significant
    ])
      vector.destroy();
    for (const buffer of this.buffers.values()) buffer.destroy();
    this.reservation.release();
  }
}
