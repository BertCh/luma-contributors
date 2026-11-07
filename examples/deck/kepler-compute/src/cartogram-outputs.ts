// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {createTransientView, GPUReduction, type GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUDorlingCartogramOutput} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {KeplerComputeSession, KeplerMemoryReservation} from '@luma-contributors/kepler';

/** Caller-owned result storage; every vector preserves the original source chunks. */
export class CartogramOutputs {
  readonly positions: GPUVector<'float32x2'>;
  readonly centerX: GPUVector<'float32'>;
  readonly centerY: GPUVector<'float32'>;
  readonly radii: GPUVector<'float32'>;
  readonly sourceIds: GPUVector<'uint32'>;
  readonly validity: GPUVector<'uint32'>;
  readonly metrics: Buffer;
  readonly status: Buffer;
  readonly bounds: Buffer;
  readonly rowCount: number;
  private readonly buffers = new Map<string, Buffer>();
  private readonly reservation: KeplerMemoryReservation;

  constructor(session: KeplerComputeSession, chunkLengths: readonly number[]) {
    this.rowCount = chunkLengths.reduce((total, length) => total + length, 0);
    this.reservation = session.registry.reserveMemory(
      Math.max(1, this.rowCount) * 28 + 40,
      'gpu',
      'output'
    );
    const allocate = (name: string, byteLength: number, usage = 0): Buffer => {
      const buffer = session.device.createBuffer({
        id: `cartogram-${name}`,
        byteLength,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST | usage
      });
      this.buffers.set(name, buffer);
      return buffer;
    };
    const vector = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      name: string,
      format: Format
    ): GPUVector<Format> => {
      const stride = format === 'float32x2' ? 8 : 4;
      const buffer = allocate(name, Math.max(1, this.rowCount) * stride, Buffer.VERTEX);
      let rowOffset = 0;
      const data = chunkLengths.map(length => {
        const chunk = new GPUData({
          buffer,
          format,
          byteOffset: rowOffset * stride,
          length,
          ownsBuffer: false
        });
        rowOffset += length;
        return chunk;
      });
      return new GPUVector({name: `cartogram-${name}`, type: 'data', format, data, ownsData: true});
    };
    try {
      this.positions = vector('positions', 'float32x2');
      this.centerX = vector('center-x', 'float32');
      this.centerY = vector('center-y', 'float32');
      this.radii = vector('radii', 'float32');
      this.sourceIds = vector('source-ids', 'uint32');
      this.validity = vector('validity', 'uint32');
      this.metrics = allocate('metrics', 8);
      this.status = allocate('status', 16);
      this.bounds = allocate('bounds', 16);
    } catch (error) {
      for (const buffer of this.buffers.values()) buffer.destroy();
      this.reservation.release();
      throw error;
    }
  }

  importToGraph(graph: GPUCommandGraph): GPUDorlingCartogramOutput {
    return {
      positions: this.importView(graph, 'positions', 'float32x2', this.rowCount),
      radii: this.importView(graph, 'radii', 'float32', this.rowCount),
      sourceIds: this.importView(graph, 'source-ids', 'uint32', this.rowCount),
      metrics: this.importView(graph, 'metrics', 'float32x2', 1),
      status: this.importView(graph, 'status', 'uint32', 4)
    };
  }

  /** Separate scalar fields support the resident table without reading result rows on the CPU. */
  addScalarOutputs(
    graph: GPUCommandGraph,
    output: GPUDorlingCartogramOutput
  ): GraphDataView<'uint32'> {
    const centerX = this.importView(graph, 'center-x', 'float32', this.rowCount);
    const centerY = this.importView(graph, 'center-y', 'float32', this.rowCount);
    const validity = this.importView(graph, 'validity', 'uint32', this.rowCount);
    const rowCount = this.rowCount;
    const extents = createTransientView(graph, 'cartogram-circle-extents', 'float32', rowCount * 4);
    const views = [output.positions, output.radii, centerX, centerY, validity, extents];
    graph.addComputePass({
      id: 'cartogram-scalar-outputs',
      resources: views.map((buffer, index) => ({
        buffer,
        usage: index < 2 ? 'storage-read' : 'storage-write'
      })),
      compile: ({device}) => {
        const kernel = new Kernel(device, {
          id: 'cartogram-scalar-outputs',
          source: `@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> radii: array<f32>;
@group(0) @binding(2) var<storage, read_write> centerX: array<f32>;
@group(0) @binding(3) var<storage, read_write> centerY: array<f32>;
@group(0) @binding(4) var<storage, read_write> validity: array<u32>;
@group(0) @binding(5) var<storage, read_write> extents: array<f32>;
fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let row = invocation.x;
  if (row >= ${rowCount}u) { return; }
  let x = positions[row * 2u];
  let y = positions[row * 2u + 1u];
  let radius = radii[row];
  centerX[row] = x;
  centerY[row] = y;
  validity[row] = select(0u, 1u, finite(x) && finite(y) && finite(radius) && radius >= 0.0);
  extents[row] = x - radius;
  extents[${rowCount}u + row] = y - radius;
  extents[${rowCount * 2}u + row] = x + radius;
  extents[${rowCount * 3}u + row] = y + radius;
}`,
          shaderLayout: {
            bindings: ['positions', 'radii', 'centerX', 'centerY', 'validity', 'extents'].map(
              (name, index) => ({
                name,
                type: index < 2 ? 'read-only-storage' : 'storage',
                group: 0,
                location: index
              })
            )
          }
        });
        return {
          encode: ({computePass, getBuffer}) =>
            kernel.dispatch(computePass, {
              bindings: {
                positions: getBuffer(output.positions),
                radii: getBuffer(output.radii),
                centerX: getBuffer(centerX),
                centerY: getBuffer(centerY),
                validity: getBuffer(validity),
                extents: getBuffer(extents)
              },
              x: Math.ceil(rowCount / 64),
              y: 1,
              z: 1
            }),
          destroy: () => kernel.destroy()
        };
      }
    });
    const bounds = this.importView(graph, 'bounds', 'float32', 4);
    for (let index = 0; index < 4; index++)
      graph.add(
        new GPUReduction({
          id: `cartogram-circle-bound-${index}`,
          input: graph.createDataView(extents.buffer, {
            format: 'float32',
            byteOffset: extents.byteOffset + index * rowCount * 4,
            length: rowCount
          }),
          mask: validity,
          output: graph.createDataView(bounds.buffer, {
            format: 'float32',
            byteOffset: bounds.byteOffset + index * 4,
            length: 1
          }),
          operation: index < 2 ? 'min' : 'max'
        })
      );
    return validity;
  }

  destroy(): void {
    for (const vector of [
      this.positions,
      this.centerX,
      this.centerY,
      this.radii,
      this.sourceIds,
      this.validity
    ])
      vector.destroy();
    for (const buffer of this.buffers.values()) buffer.destroy();
    this.reservation.release();
  }

  private importView<Format extends 'float32' | 'float32x2' | 'uint32'>(
    graph: GPUCommandGraph,
    name: string,
    format: Format,
    length: number
  ): GraphDataView<Format> {
    const buffer = this.buffers.get(name)!;
    return graph.createDataView(
      graph.importBuffer(
        {id: `cartogram-output-${name}`, byteLength: buffer.byteLength, usage: buffer.usage},
        buffer
      ),
      {format, length}
    );
  }
}
