// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '../runtime/dataset-registry';
import {Kernel} from '@luma.gl/graph';
import {
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';

/**
 * Rows (or raster cells) per compute dispatch for the tiled analysis kernels. Tiles keep global
 * indices, so results never depend on this value; it only bounds the work of one dispatch.
 */
export const KEPLER_ANALYSIS_ROW_TILE_SIZE = 262144;

/** Source rows at or below which the legacy 64 by 32 interpolation index is kept bit for bit. */
const LEGACY_INDEX_ROW_COUNT = 4096;

/** Largest scaled interpolation index (cells). Offsets cost four bytes per cell. */
const MAXIMUM_INDEX_CELL_COUNT = 2 ** 22;

/**
 * Chooses the `GPUGridIndex` shape over the whole world for `rowCount` sources, aiming at about
 * one source per index cell on average so that neighbor search cost does not grow with the
 * source count. Two columns per row, matching the 360 by 180 degree domain.
 */
export function getKeplerIndexGridSize(rowCount: number): [number, number] {
  if (rowCount <= LEGACY_INDEX_ROW_COUNT) return [64, 32];
  const cellCount = Math.min(rowCount, MAXIMUM_INDEX_CELL_COUNT);
  const height = Math.max(32, Math.ceil(Math.sqrt(cellCount / 2)));
  return [height * 2, height];
}

/**
 * Neighbor-search lattice: the legacy 32 by 16 up to 16384 rows, then about eight rows per cell.
 * Without `bounds` the lattice is 2:1 (the world in degrees). With the search extent of projected
 * coordinates it takes the extent's aspect ratio, so cells stay near-square and the cell budget is
 * not wasted on a stretched axis; the lattice never exceeds the same cell budget either way.
 */
export function getKeplerNeighborGridSize(
  rowCount: number,
  bounds?: readonly [number, number, number, number]
): [number, number] {
  if (rowCount <= 16384) return [32, 16];
  const cellCount = Math.min(rowCount / 8, 2 ** 22);
  const width = bounds ? bounds[2] - bounds[0] : 2;
  const height = bounds ? bounds[3] - bounds[1] : 1;
  if (!(width > 0) || !(height > 0) || Math.max(width / height, height / width) > 1024) {
    const rows = Math.max(16, Math.ceil(Math.sqrt(cellCount / 2)));
    return [rows * 2, rows];
  }
  const aspect = width / height;
  const rows = Math.max(16, Math.ceil(Math.sqrt(cellCount / aspect)));
  const columns = Math.max(16, Math.ceil(rows * aspect));
  // Rounding up can overshoot the budget by a row or column; trim to the budget.
  const scale = Math.min(1, Math.sqrt(2 ** 22 / (rows * columns)));
  return [Math.max(1, Math.floor(columns * scale)), Math.max(1, Math.floor(rows * scale))];
}

export type KeplerLinearKernelBinding = {
  name: string;
  view: GraphDataView;
  write?: boolean;
  /** Declares the binding as `array<atomic<u32>>` (uint32 views only). */
  atomic?: boolean;
};

const LINEAR_WORKGROUP_SIZE = 64;

/**
 * Adds a one-dimensional compute pass of `length` invocations with a bounded dispatch layout
 * derived from the device `maxComputeWorkgroupsPerDimension` limit, so lengths above 65535
 * workgroups (about 4.2 million rows) are supported. `body` sees `index` in `[0, length)`; atomic
 * bindings give exact, order-independent integer sums and are always bound read-write.
 */
export function addKeplerLinearKernel<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  length: number,
  bindings: KeplerLinearKernelBinding[],
  body: string
): void {
  graph.addComputePass({
    id,
    resources: bindings.map(({view, write, atomic}) => ({
      buffer: view,
      usage: write || atomic ? ('storage-read-write' as const) : ('storage-read' as const)
    })),
    compile: ({device}) => {
      const layout = getBoundedDispatchLayout(
        id,
        length,
        LINEAR_WORKGROUP_SIZE,
        device.limits.maxComputeWorkgroupsPerDimension
      );
      const kernel = new Kernel(device, {
        id,
        source: `${bindings.map(({name, view, write, atomic}, index) => `@group(0) @binding(${index}) var<storage, ${write || atomic ? 'read_write' : 'read'}> ${name}: array<${atomic ? 'atomic<u32>' : view.format === 'uint32' ? 'u32' : 'f32'}>;\nconst ${name}Offset: u32 = ${getViewElementOffset(view)}u;`).join('\n')}
fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
@compute @workgroup_size(${LINEAR_WORKGROUP_SIZE}) fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
${getBoundedInvocationIndexSource(layout, LINEAR_WORKGROUP_SIZE)}
if (index >= ${length}u) { return; }
${body}
}`,
        shaderLayout: {
          bindings: bindings.map(({name}, location) => ({
            name,
            type: 'storage' as const,
            group: 0,
            location
          }))
        }
      });
      return {
        encode: ({computePass, getBuffer}) =>
          kernel.dispatch(computePass, {
            bindings: Object.fromEntries(
              bindings.map(({name, view}) => [name, getViewBinding(view, getBuffer)])
            ),
            x: layout.x,
            y: layout.y,
            z: layout.z
          }),
        destroy: () => kernel.destroy()
      };
    }
  });
}

/**
 * Throws when `elementCount` elements of `bytesPerElement` cannot be bound as one storage buffer.
 * The default WebGPU binding limit is 128 MiB, so large neighbor graphs and result columns must
 * be reduced or tiled by the caller instead of failing inside pipeline creation.
 */
export function assertKeplerStorageBindingFits(
  limits: {maxStorageBufferBindingSize: number; maxBufferSize: number},
  elementCount: number,
  bytesPerElement: number,
  label: string
): void {
  const byteLength = elementCount * bytesPerElement;
  // The message names the label so a device-limit failure is attributable without a long string.
  if (byteLength > limits.maxStorageBufferBindingSize || byteLength > limits.maxBufferSize)
    throw new KeplerAllocationError('capacity', {
      message: `${label} exceed the device storage binding limit`
    });
}

/** Largest element count a single storage binding of `bytesPerElement` can hold on this device. */
export function getKeplerMaximumBindingElements(
  limits: {maxStorageBufferBindingSize: number; maxBufferSize: number},
  bytesPerElement: number
): number {
  return Math.floor(
    Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize) / bytesPerElement
  );
}
