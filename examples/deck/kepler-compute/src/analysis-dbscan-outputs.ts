// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {Kernel} from '@luma-contributors/graph';
import type {GPUVector} from '@luma-contributors/graph/gpu-data';
import {
  createTransientView,
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {packGPUColor} from '@luma-contributors/graph/gpu-dataframe';
import type {
  KeplerComputeSession,
  KeplerMemoryReservation,
  KeplerPointDBSCANOutputs
} from '@luma-contributors/kepler';
import {makeAnalysisOutputVector} from './analysis-hotspot-outputs';

export const DBSCAN_PALETTE_LENGTH = 16;

/** Owned packed outputs with borrowed source-aligned chunks for inspection and export. */
export class AnalysisDBSCANOutputs {
  readonly labels: GPUVector<'uint32'>;
  readonly coreDistances: GPUVector<'float32'>;
  readonly reachability: GPUVector<'float32'>;
  readonly predecessors: GPUVector<'uint32'>;
  readonly palette: Buffer;
  /** [OPTICS status, cluster summary]; each buffer contains three uint32 values. */
  readonly diagnostics: readonly Buffer[];
  private readonly buffers = new Map<string, Buffer>();
  private readonly reservation: KeplerMemoryReservation;

  constructor(session: KeplerComputeSession, chunkLengths: readonly number[]) {
    const rowCount = chunkLengths.reduce((total, length) => total + length, 0);
    this.reservation = session.registry.reserveMemory(
      Math.max(1, rowCount) * 20 + 24 + DBSCAN_PALETTE_LENGTH * 4,
      'gpu',
      'output'
    );
    const allocate = (name: string, length: number): Buffer => {
      const buffer = session.device.createBuffer({
        id: `dbscan-${name}`,
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
      makeAnalysisOutputVector(`dbscan-${name}`, format, allocate(name, rowCount), chunkLengths);
    try {
      this.labels = vector('labels', 'uint32');
      this.coreDistances = vector('core-distances', 'float32');
      this.reachability = vector('reachability', 'float32');
      this.predecessors = vector('predecessors', 'uint32');
      // Visit rank -> original row is not a source-aligned table column.
      allocate('ordering', rowCount);
      this.diagnostics = [allocate('optics-status', 3), allocate('cluster-summary', 3)];
      this.palette = allocate('palette', DBSCAN_PALETTE_LENGTH);
      this.palette.write(
        new Uint32Array([
          packGPUColor(150, 150, 150), // Noise; excluded rows use the validity mask.
          packGPUColor(31, 119, 180),
          packGPUColor(255, 127, 14),
          packGPUColor(44, 160, 44),
          packGPUColor(214, 39, 40),
          packGPUColor(148, 103, 189),
          packGPUColor(140, 86, 75),
          packGPUColor(227, 119, 194),
          packGPUColor(188, 189, 34),
          packGPUColor(23, 190, 207),
          packGPUColor(57, 59, 121),
          packGPUColor(82, 84, 163),
          packGPUColor(99, 121, 57),
          packGPUColor(140, 109, 49),
          packGPUColor(132, 60, 57),
          packGPUColor(123, 65, 115)
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
    values: GraphDataView<'float32'>
  ): KeplerPointDBSCANOutputs {
    const view = <Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      length: number
    ): GraphDataView<Format> => {
      const buffer = this.buffers.get(name)!;
      return graph.createDataView(
        graph.importBuffer(
          {id: `dbscan-output-${name}`, byteLength: buffer.byteLength, usage: buffer.usage},
          buffer
        ),
        {format, length}
      );
    };
    const length = this.labels.length;
    return {
      values,
      labels: view('labels', 'uint32', length),
      coreDistances: view('core-distances', 'float32', length),
      reachability: view('reachability', 'float32', length),
      predecessors: view('predecessors', 'uint32', length),
      ordering: view('ordering', 'uint32', length),
      status: {
        optics: view('optics-status', 'uint32', 3),
        summary: view('cluster-summary', 'uint32', 3)
      }
    };
  }

  /** Palette zero is noise; positive entries repeat without replacing the original cluster IDs. */
  addColorIndices(
    graph: GPUCommandGraph,
    labels: GraphDataView<'uint32'>
  ): GraphDataView<'uint32'> {
    const indices = createTransientView(graph, 'dbscan-color-indices', 'uint32', labels.length);
    if (!labels.length) return indices;
    graph.addComputePass({
      id: 'dbscan-color-indices',
      resources: [
        {buffer: labels, usage: 'storage-read'},
        {buffer: indices, usage: 'storage-write'}
      ],
      compile: ({device}) => {
        const kernel = new Kernel(device, {
          id: 'dbscan-color-indices',
          source: `@group(0) @binding(0) var<storage, read> labels: array<u32>;
@group(0) @binding(1) var<storage, read_write> indices: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let row = invocation.x;
  if (row >= ${labels.length}u) { return; }
  let label = labels[${getViewElementOffset(labels)}u + row];
  indices[${getViewElementOffset(indices)}u + row] = select(1u + label % ${DBSCAN_PALETTE_LENGTH - 1}u, 0u, label == 0xffffffffu);
}`,
          shaderLayout: {
            bindings: [
              {name: 'labels', type: 'read-only-storage', group: 0, location: 0},
              {name: 'indices', type: 'storage', group: 0, location: 1}
            ]
          }
        });
        return {
          encode: ({computePass, getBuffer}) =>
            kernel.dispatch(computePass, {
              bindings: {
                labels: getViewBinding(labels, getBuffer),
                indices: getViewBinding(indices, getBuffer)
              },
              x: Math.ceil(labels.length / 64),
              y: 1,
              z: 1
            }),
          destroy: () => kernel.destroy()
        };
      }
    });
    return indices;
  }

  destroy(): void {
    for (const vector of [this.labels, this.coreDistances, this.reachability, this.predecessors])
      vector.destroy();
    for (const buffer of this.buffers.values()) buffer.destroy();
    this.reservation.release();
  }
}
