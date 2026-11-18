// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphVectorView,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {Kernel} from '@luma.gl/graph';
import {
  createGraphChunkDispatchParameters,
  planGraphChunkDispatch,
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  getGraphViewChunks,
  validateCommandNodeBudget,
  validateMaximumCount,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  type WGSLKernelProps,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

import {dggs} from '@luma.gl/shaders';

import {CELL_KEY_WGSL} from '../cell-aggregation/cell-keys';
import {
  validateCellIndexResolution,
  type GPUCellTopologyFamily
} from '../cell-indexing/cell-index-families';
import type {GPUCellWordOrder} from '../cell-aggregation/gpu-cell-aggregation';
import {H3_NEIGHBOR_WGSL} from './h3-neighbor-wgsl';
import {
  H3_TOPOLOGY_WGSL,
  QUADBIN_TOPOLOGY_WGSL,
  CELL_TOPOLOGY_NEIGHBORHOOD_WGSL
} from './cell-topology-wgsl';
import {S2_NEIGHBORS_WGSL} from './s2-neighbor-wgsl';
import {S2_TOPOLOGY_WGSL} from './s2-topology-wgsl';

/** Fixed workgroup size makes chunk lengths runtime dispatch metadata. */
const CHUNK_WORKGROUP_SIZE = 256;
/** Storage component indices of whole fixed-stride records cannot wrap uint32. */
const MAXIMUM_CHUNK_COMPONENT_COUNT = 0xffffffff;
/** Guard padded workgroups before multiplying their index into a uint32 invocation index. */
const MAXIMUM_LINEAR_WORKGROUP_COUNT = 16777216;

const OPERATION = 'GPUCellTopology';

/** Largest bounded neighborhood radius, shared by H3, Quadbin and S2. */
export const GPU_CELL_TOPOLOGY_MAXIMUM_RADIUS = 8;

/** Largest per-row output stride of a `children` operation. */
export const GPU_CELL_TOPOLOGY_MAXIMUM_CHILDREN_STRIDE = 4096;

/**
 * One per-cell topology operation.
 *
 * - `disk`: cells within grid distance `k` (`gridDisk` / `gridDiskDistances`), sorted by
 *   (distance, key) ascending. Stride `3k(k+1)+1` (H3), `(2k+1)^2` (Quadbin), `2k(k+1)+1` (S2).
 * - `ring`: cells at grid distance exactly `k`, sorted by key. Stride `6k` (H3), `8k` (Quadbin), `4k` (S2), 1 for `k = 0`.
 * - `parent`: the ancestor at `resolution`; stride 1. Input cells coarser than `resolution` give zero.
 * - `children`: all descendants at `resolution` of cells at exactly `inputResolution`, ascending
 *   key order (`cellToChildren`). Stride `4^d` (Quadbin/S2) or `7^d` (H3) for `d = resolution - inputResolution`,
 *   at most {@link GPU_CELL_TOPOLOGY_MAXIMUM_CHILDREN_STRIDE}. Pentagons have fewer children
 *   (`(5 * 7^d + 1) / 6`), padded with zeros. Input cells at another resolution give zero rows.
 *
 * Quadbin distance is the Chebyshev distance of tile columns and rows. Columns wrap across the
 * antimeridian, rows do not (tiles outside `[0, 2^z)` are dropped), so a neighborhood near a pole
 * is clipped. When the wrap makes the square overlap itself at tiny zoom, every tile appears once
 * at its smallest circular distance. H3 distance is the grid distance of `gridDiskDistances`,
 * exact across faces and pentagons. S2 distance counts steps over its four edge neighbors,
 * including cube-face transitions; diagonal contacts are not steps. Its disk and ring strides
 * bound the flat square-grid sizes: cube corners reduce the number of reachable cells.
 */
export type GPUCellTopologyOperation =
  | {type: 'disk'; k: number}
  | {type: 'ring'; k: number}
  | {type: 'parent'; resolution: number}
  | {type: 'children'; resolution: number; inputResolution: number};

/** Output columns of {@link GPUCellTopology}. */
export type GPUCellTopologyOutput = {
  /** `uint32x2` little-endian `(low, high)` cells, `rows * stride` entries, row-major, zero padded. */
  cells: GraphDataView<'uint32x2'> | GraphVectorView<'uint32x2'>;
  /** Optional `uint32` grid distances aligned with `cells` (`disk` only), zero padded. */
  distances?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Optional `uint32` number of valid entries per input row (0 for invalid or masked rows). */
  counts?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
};

/**
 * Properties for {@link GPUCellTopology}.
 *
 * Per-frame (no recompile): the contents of `cells` and `mask`. Topology: family, operation, the
 * view layouts.
 */
export type GPUCellTopologyProps = {
  /** Prefix for generated node IDs. Defaults to `'cell-topology'`. */
  id?: string;
  /** Cell family. */
  family: GPUCellTopologyFamily;
  /** Operation, fixing the output stride. */
  operation: GPUCellTopologyOperation;
  /** Input `uint32x2` cell keys. Zero or invalid keys produce zero rows. */
  cells: GraphDataView<'uint32x2'> | GraphVectorView<'uint32x2'>;
  /** Word order of `cells`. Defaults to `'little-endian'`. */
  wordOrder?: GPUCellWordOrder;
  /** Optional per-row mask; zero produces a zero row with count 0. */
  mask?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUCellTopologyOutput;
};

/** Number of H3 cells within grid distance `k` of a hexagon. */
function getH3DiskSize(k: number): number {
  return 3 * k * (k + 1) + 1;
}

/**
 * Returns the fixed output stride (entries per input row) of an operation.
 *
 * @throws if `k`, resolutions or the stride are out of range.
 */
export function getCellTopologyStride(
  family: GPUCellTopologyFamily,
  operation: GPUCellTopologyOperation
): number {
  validateRequirement(
    'GPUCellTopology',
    ['quadbin', 'h3', 's2'].includes(family),
    "family must be 'quadbin', 'h3' or 's2'"
  );
  if (operation.type === 'disk' || operation.type === 'ring') {
    const {k} = operation;
    if (!Number.isInteger(k) || k < 0 || k > GPU_CELL_TOPOLOGY_MAXIMUM_RADIUS) {
      throw new Error(
        `${operation.type} k must be an integer in [0, ${GPU_CELL_TOPOLOGY_MAXIMUM_RADIUS}]`
      );
    }
    if (operation.type === 'disk') {
      return family === 'h3'
        ? getH3DiskSize(k)
        : family === 's2'
          ? 2 * k * (k + 1) + 1
          : (2 * k + 1) ** 2;
    }
    return k === 0 ? 1 : (family === 'h3' ? 6 : family === 's2' ? 4 : 8) * k;
  }
  validateCellIndexResolution('GPUCellTopology', family, operation.resolution);
  if (operation.type === 'parent') {
    return 1;
  }
  validateCellIndexResolution('GPUCellTopology', family, operation.inputResolution);
  const depth = operation.resolution - operation.inputResolution;
  if (depth < 0) {
    throw new Error('children resolution must not be below inputResolution');
  }
  const stride = (family === 'h3' ? 7 : 4) ** depth;
  if (stride > GPU_CELL_TOPOLOGY_MAXIMUM_CHILDREN_STRIDE) {
    throw new Error(
      `children stride ${stride} exceeds ${GPU_CELL_TOPOLOGY_MAXIMUM_CHILDREN_STRIDE}; reduce the resolution depth`
    );
  }
  return stride;
}

/**
 * Per-cell fixed fan-out topology of Quadbin, H3 and S2 cells: `gridDisk`, `gridDiskDistances`,
 * `gridRing`, `cellToParent` and `cellToChildren`.
 *
 * One kernel; each output row `i` occupies entries `[i * stride, (i + 1) * stride)` of the output
 * columns, filled in ascending (distance, key) order and zero padded. Invalid, zero and masked
 * input cells produce all-zero rows with count 0. H3 disks and rings run a bounded breadth-first
 * search over {@link H3_NEIGHBOR_WGSL} in private arrays, so they are exact across icosahedron
 * faces and pentagons; use `k <= 8`. Everything is integer arithmetic, so results are identical
 * on every device and equal the h3-js and BigInt Quadbin references. S2 uses the same bounded
 * breadth-first search over four exact integer edge steps; parents and multi-level children
 * are exact at levels 0–30, with children in ascending Hilbert-key order.
 *
 * Consumes cell keys only, so it takes no `spatialContext`.
 *
 * Accepts packed or chunked row inputs; output chunks mirror inputs and keep whole fixed-stride records.
 */
export class GPUCellTopology implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCellTopologyProps;
  /** Output entries per input row. */
  readonly stride: number;

  /** Upload plus one dispatch per nonempty chunk. */
  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;
  private readonly recordChunks: Readonly<Record<string, readonly GraphDataView[]>>;

  constructor(props: GPUCellTopologyProps) {
    this.id = props.id ?? 'cell-topology';
    this.props = props;
    const id = this.id;
    validateRequirement(
      id,
      ['quadbin', 'h3', 's2'].includes(props.family),
      "family must be 'quadbin', 'h3' or 's2'"
    );
    validateRequirement(
      id,
      props.wordOrder === undefined || ['little-endian', 'high-low'].includes(props.wordOrder),
      "wordOrder must be 'little-endian' or 'high-low'"
    );
    try {
      this.stride = getCellTopologyStride(props.family, props.operation);
    } catch (error) {
      throw new Error(`${id} ${(error as Error).message}`);
    }
    const {output} = props;
    validateRequirement(
      id,
      !output.distances || props.operation.type === 'disk',
      'output.distances is only available for the disk operation'
    );
    this.chunkPlan = planGraphChunkDispatch(id, {
      cells: props.cells,
      mask: props.mask,
      counts: output.counts
    });
    validateCapacity(id, 'cells row count', this.chunkPlan.totalLength);
    for (const chunk of this.chunkPlan.chunks) {
      if (chunk.views['cells'])
        validateSinglePackedView(
          chunk.views['cells'],
          ['uint32x2'],
          `${id} cells chunk ${chunk.chunkIndex}`
        );
      validateMaximumCount(
        id,
        'cells chunk components',
        chunk.length * 2,
        MAXIMUM_CHUNK_COMPONENT_COUNT
      );
      if (chunk.views['mask'])
        validateSinglePackedView(
          chunk.views['mask'],
          ['uint32'],
          `${id} mask chunk ${chunk.chunkIndex}`
        );
      validateMaximumCount(
        id,
        'mask chunk components',
        chunk.length,
        MAXIMUM_CHUNK_COMPONENT_COUNT
      );
      if (chunk.views['counts'])
        validateSinglePackedView(
          chunk.views['counts'],
          ['uint32'],
          `${id} counts chunk ${chunk.chunkIndex}`
        );
      validateMaximumCount(
        id,
        'counts chunk components',
        chunk.length,
        MAXIMUM_CHUNK_COMPONENT_COUNT
      );
    }
    const recordChunks: Record<string, readonly GraphDataView[]> = {};
    recordChunks['outputCells'] = output.cells ? getGraphViewChunks(output.cells) : [];
    if (output.cells) {
      validateRequirement(
        id,
        output.cells.length === this.chunkPlan.totalLength * this.stride,
        'output.cells length must equal input rows * record stride'
      );
      validateRequirement(
        id,
        recordChunks['outputCells'].length === this.chunkPlan.chunks.length,
        'output.cells must match input chunk topology'
      );
      for (const chunk of this.chunkPlan.chunks) {
        const view = recordChunks['outputCells'][chunk.chunkIndex];
        validateSinglePackedView(view, ['uint32x2'], `${id} outputCells chunk ${chunk.chunkIndex}`);
        validateRequirement(
          id,
          view.length === chunk.length * this.stride,
          `output.cells chunk ${chunk.chunkIndex} length must equal input chunk rows * record stride`
        );
        validateMaximumCount(
          id,
          'outputCells chunk components',
          view.length * 2,
          MAXIMUM_CHUNK_COMPONENT_COUNT
        );
      }
    }
    recordChunks['distances'] = output.distances ? getGraphViewChunks(output.distances) : [];
    if (output.distances) {
      validateRequirement(
        id,
        output.distances.length === this.chunkPlan.totalLength * this.stride,
        'output.distances length must equal input rows * record stride'
      );
      validateRequirement(
        id,
        recordChunks['distances'].length === this.chunkPlan.chunks.length,
        'output.distances must match input chunk topology'
      );
      for (const chunk of this.chunkPlan.chunks) {
        const view = recordChunks['distances'][chunk.chunkIndex];
        validateSinglePackedView(view, ['uint32'], `${id} distances chunk ${chunk.chunkIndex}`);
        validateRequirement(
          id,
          view.length === chunk.length * this.stride,
          `output.distances chunk ${chunk.chunkIndex} length must equal input chunk rows * record stride`
        );
        validateMaximumCount(
          id,
          'distances chunk components',
          view.length * 1,
          MAXIMUM_CHUNK_COMPONENT_COUNT
        );
      }
    }
    this.recordChunks = recordChunks;
    validateDisjointWritableViews(id, props.output);
    this.estimatedCommandNodeCount =
      (this.chunkPlan.chunks.length > 0 ? 1 : 0) + this.chunkPlan.nonemptyChunkCount;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.cells, output.distances, output.counts],
      [props.cells, props.mask]
    );
  }

  /** Uploads chunk parameters and dispatches each nonempty cell chunk. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, stride} = this;
    const {family, operation, output} = props;
    validateMaximumCount(
      id,
      'storage bindings',
      2 + Number(Boolean(props.mask)) + Object.values(props.output).filter(Boolean).length,
      graph.device.limits.maxStorageBuffersPerShaderStage
    );
    validateGraphChunkDispatchBindings(graph, id, this.chunkPlan);
    validateGraphChunkBindingRanges(graph, id, {
      'output.cells': output.cells,
      'output.distances': output.distances
    });
    const invocationsPerRow = props.operation.type === 'children' ? this.stride : 1;
    const layouts = this.chunkPlan.chunks.map(chunk =>
      getBoundedDispatchLayout(
        id,
        Math.max(chunk.length * invocationsPerRow, 1),
        CHUNK_WORKGROUP_SIZE,
        graph.device.limits.maxComputeWorkgroupsPerDimension
      )
    );
    const {slots, nodes} = createGraphChunkDispatchParameters(graph, {
      id: `${id}-chunks`,
      plan: this.chunkPlan
    });
    // Every chunk closes over the same source string, including assembled fp64 modules.
    let sharedShaderSource: string | undefined;
    const chunkNodes = this.chunkPlan.chunks
      .filter(chunk => chunk.length > 0)
      .map(chunk => {
        const isH3 = family === 'h3';
        const isS2 = family === 's2';
        const bindings: WGSLKernelBinding[] = [
          {name: 'chunkParameters', view: slots[chunk.chunkIndex], type: 'u32', access: 'read'},
          {name: 'cells', view: chunk.views['cells'], type: 'u32', access: 'read'},
          ...(props.mask
            ? [{name: 'rowMask', view: chunk.views['mask'], type: 'u32', access: 'read'} as const]
            : []),
          {
            name: 'outCells',
            view: this.recordChunks['outputCells'][chunk.chunkIndex],
            type: 'u32',
            access: 'read_write'
          },
          ...(output.distances
            ? [
                {
                  name: 'outDistances',
                  view: this.recordChunks['distances'][chunk.chunkIndex],
                  type: 'u32',
                  access: 'read_write'
                } as const
              ]
            : []),
          ...(output.counts
            ? [
                {
                  name: 'outCounts',
                  view: chunk.views['counts'],
                  type: 'u32',
                  access: 'read_write'
                } as const
              ]
            : [])
        ];

        const isNeighborhood = operation.type === 'disk' || operation.type === 'ring';
        const radius = isNeighborhood ? operation.k : 0;
        const capacity = isNeighborhood
          ? isH3
            ? getH3DiskSize(radius)
            : getCellTopologyStride(family, {type: 'disk', k: radius})
          : 1;
        // The source resolution a row must have (children), or the minimum (parent).
        const targetResolution =
          operation.type === 'parent' || operation.type === 'children' ? operation.resolution : 0;
        const inputResolution = operation.type === 'children' ? operation.inputResolution : 0;
        const depth = operation.type === 'children' ? operation.resolution - inputResolution : 0;

        const resolutionOf = isH3
          ? 'dggs_h3_get_resolution(key)'
          : isS2
            ? 'dggs_s2_get_level(key)'
            : '((key.x >> 20u) & 0x1fu)';
        const validity = isH3
          ? 'dggs_h3_is_valid_cell_id(key)'
          : isS2
            ? 'dggs_s2_is_valid_cell_id(key)'
            : 'cellIsValidQuadbin(key)';
        let resolutionCheck = 'true';
        if (operation.type === 'parent') {
          resolutionCheck = `${resolutionOf} >= TARGET_RESOLUTION`;
        } else if (operation.type === 'children') {
          resolutionCheck = `${resolutionOf} == INPUT_RESOLUTION`;
        }
        const readRow = (rowExpression: string) => `
  let words = vec2u(cells[cellsOffset + 2u * ${rowExpression}], cells[cellsOffset + 2u * ${rowExpression} + 1u]);
  let key = ${props.wordOrder === 'high-low' ? 'words' : 'words.yx'};
  var isValid = ${validity} && ${resolutionCheck};
  ${props.mask ? `isValid = isValid && rowMask[rowMaskOffset + ${rowExpression}] != 0u;` : ''}`;
        const writeCell = (slot: string, cell: string, distance?: string) => `
  outCells[outCellsOffset + 2u * ${slot}] = ${cell}.y;
  outCells[outCellsOffset + 2u * ${slot} + 1u] = ${cell}.x;
  ${output.distances ? `outDistances[outDistancesOffset + ${slot}] = ${distance ?? '0u'};` : ''}`;

        let body: string;
        if (isNeighborhood) {
          body = `${readRow('index')}
  var written = 0u;
  if (isValid) {
    written = ${isH3 || isS2 ? 'cellTopologyNeighborhood' : 'cellTopologyQuadbinNeighborhood'}(key, ${operation.type === 'ring'});
  }
  for (var j = 0u; j < STRIDE; j++) {
    let slot = index * STRIDE + j;
    if (j < written) {${writeCell('slot', 'cellTopologyCells[j]', 'cellTopologyDistances[j]')}
    } else {${writeCell('slot', 'vec2u(0u)')}
    }
  }
  ${output.counts ? 'outCounts[outCountsOffset + index] = written;' : ''}`;
        } else if (operation.type === 'parent') {
          const parent = isH3
            ? 'dggs_h3_get_parent(key, TARGET_RESOLUTION)'
            : isS2
              ? 'cellTopologyS2GetParent(key, TARGET_RESOLUTION)'
              : 'cellTopologyQuadbinGetParent(key, TARGET_RESOLUTION)';
          body = `${readRow('index')}
  var parent = vec2u(0u);
  if (isValid) {
    parent = ${parent};
  }${writeCell('index', 'parent')}
  ${output.counts ? 'outCounts[outCountsOffset + index] = select(0u, 1u, isValid);' : ''}`;
        } else {
          const child = isH3
            ? 'cellTopologyH3GetChild(key, TARGET_RESOLUTION, DEPTH, slotInRow)'
            : isS2
              ? 'cellTopologyS2GetChild(key, TARGET_RESOLUTION, DEPTH, slotInRow)'
              : 'cellTopologyQuadbinGetChild(key, TARGET_RESOLUTION, DEPTH, slotInRow)';
          const count = isH3 ? 'cellTopologyH3GetChildCount(key, DEPTH)' : 'STRIDE';
          body = `let row = index / STRIDE;
  let slotInRow = index % STRIDE;${readRow('row')}
  var child = vec2u(0u);
  if (isValid) {
    child = ${child};
  }${writeCell('index', 'child')}
  ${output.counts ? `if (slotInRow == 0u) {\n    outCounts[outCountsOffset + row] = select(0u, ${count}, isValid);\n  }` : ''}`;
        }

        const kernelProps: WGSLKernelProps<Parameters> = {
          id: `${id}-topology`,
          operation: OPERATION,
          variant: `${family}-${operation.type}`,
          bindings,
          invocationCount: chunk.length * invocationsPerRow,
          declarations: `const STRIDE: u32 = ${stride}u;
const CELL_TOPOLOGY_K: u32 = ${radius}u;
const CELL_TOPOLOGY_CAPACITY: u32 = ${capacity}u;
const TARGET_RESOLUTION: u32 = ${targetResolution}u;
const INPUT_RESOLUTION: u32 = ${inputResolution}u;
const DEPTH: u32 = ${depth}u;
${isH3 || isS2 ? dggs.source : ''}
${CELL_KEY_WGSL}
${isH3 ? H3_NEIGHBOR_WGSL : ''}
${QUADBIN_TOPOLOGY_WGSL}
${isH3 ? H3_TOPOLOGY_WGSL : ''}
${isS2 ? `${S2_NEIGHBORS_WGSL}\n${S2_TOPOLOGY_WGSL}` : ''}
${
  isH3 || isS2
    ? `
const CELL_TOPOLOGY_NEIGHBOR_COUNT: u32 = ${isH3 ? 6 : 4}u;
fn cellTopologyNeighbor(cell: vec2u, direction: u32) -> vec2u {
  return ${isH3 ? 'cellTopologyH3Neighbor(cell, direction + 1u)' : 's2Neighbor(cell, direction)'};
}
${CELL_TOPOLOGY_NEIGHBORHOOD_WGSL}`
    : ''
}`,
          body
        };
        const storageDeclarations = kernelProps.bindings
          .map(
            (binding, location) =>
              `@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
          )
          .join('\n');
        const bindingComponentCounts: Record<string, number> = {
          cells: 2,
          rowMask: 1,
          outCells: stride * 2,
          outDistances: stride * 1,
          outCounts: 1
        };
        const bindingOffsets = kernelProps.bindings
          .filter(binding => binding.name !== 'chunkParameters')
          .map(
            binding =>
              `let ${binding.name}Offset = arrayLength(&${binding.name}) - rowCount * ${bindingComponentCounts[binding.name]}u;`
          )
          .join('\n  ');
        const source =
          sharedShaderSource ??
          `${storageDeclarations}
${kernelProps.declarations ?? ''}
@compute @workgroup_size(${CHUNK_WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
  @builtin(num_workgroups) workgroupCount: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32) {
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${MAXIMUM_LINEAR_WORKGROUP_COUNT}u) { return; }
  let index = workgroupIndex * ${CHUNK_WORKGROUP_SIZE}u + localInvocationIndex;
  let rowCount = chunkParameters[1];
  if (index >= rowCount * ${invocationsPerRow}u) { return; }
  ${bindingOffsets}
  ${kernelProps.body}
}`;
        sharedShaderSource = source;
        const kernelBindings = kernelProps.bindings;
        const operationName = kernelProps.operation;
        const layout = layouts[chunk.chunkIndex];
        return createGPUComputeCommandNode<Parameters>({
          id: `${kernelProps.id}-${chunk.chunkIndex}`,
          resources: kernelProps.bindings.map(binding => ({
            buffer: binding.view,
            usage: binding.access === 'read' ? 'storage-read' : 'storage-read-write'
          })),
          workload: {
            operation: kernelProps.operation,
            ...(kernelProps.variant ? {variant: kernelProps.variant} : {}),
            commandCount: 1,
            maximumWorkgroupCount: layout.x * layout.y * layout.z,
            maximumInvocationCount: chunk.length * invocationsPerRow
          },
          compile: ({device}) => {
            const kernel = new Kernel(device, {
              id: operationName,
              source,
              shaderLayout: {
                bindings: kernelBindings.map((binding, location) => ({
                  name: binding.name,
                  group: 0,
                  location,
                  type: binding.access === 'read' ? 'read-only-storage' : 'storage'
                }))
              }
            });
            return {
              encode: ({computePass, getBuffer}) =>
                kernel.dispatch(computePass, {
                  bindings: Object.fromEntries(
                    kernelBindings.map(binding => [
                      binding.name,
                      getViewBinding(binding.view, getBuffer)
                    ])
                  ),
                  ...layout
                }),
              destroy: () => kernel.destroy()
            };
          }
        });
      });
    return [...nodes, ...chunkNodes];
  }
}
