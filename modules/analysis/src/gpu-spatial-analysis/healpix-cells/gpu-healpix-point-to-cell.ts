// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {Kernel} from '@luma.gl/graph';
import {
  createGraphChunkDispatchParameters,
  planGraphChunkDispatch,
  validateDisjointWritableViews,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  validateCommandNodeBudget,
  validateMaximumCount,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {WGSLKernelProps} from '../../analysis-core/kernels/wgsl-kernel-nodes';

import {IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {HEALPIX_INDEX_WGSL, HEALPIX_MAXIMUM_ORDER} from './healpix-index-wgsl';

/** Fixed workgroup size makes chunk lengths runtime dispatch metadata. */
const CHUNK_WORKGROUP_SIZE = 256;
/** Storage component indices of whole fixed-stride records cannot wrap uint32. */
const MAXIMUM_CHUNK_COMPONENT_COUNT = 0xffffffff;
/** Guard padded workgroups before multiplying their index into a uint32 invocation index. */
const MAXIMUM_LINEAR_WORKGROUP_COUNT = 16777216;

const OPERATION = 'GPUHealpixPointToCell';

/**
 * Index scheme of the {@link GPUHealpixPointToCell} keys. `'nested'` is the plain nested pixel
 * number (what healpix-geo `nested` returns), in which cell 0 of face 0 has key 0, the "no cell"
 * sentinel of the other families, so read `output.validity`. `'uniq'` adds `4^(order + 1)` (the
 * IVOA UNIQ scheme): never zero, and the order is part of the key.
 */
export type GPUHealpixIndexScheme = 'nested' | 'uniq';

/** Caller-owned outputs of {@link GPUHealpixPointToCell}. */
export type GPUHealpixPointToCellOutput = {
  /** One 64-bit key per row as little-endian `(low, high)` words (the Arrow `Uint64` layout). Zero for masked or non-finite rows. */
  cells: GraphDataView<'uint32x2'> | GraphVectorView<'uint32x2'>;
  /** Optional `1` for every row that received a key and `0` for masked or non-finite rows. */
  validity?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
};

/**
 * Properties for {@link GPUHealpixPointToCell}.
 *
 * Per-frame (no recompile): the contents of `positions` and `mask`. Compile-time: `order`,
 * `scheme`, the row count and which optional views exist.
 */
export type GPUHealpixPointToCellProps = {
  /** Prefix for generated node IDs. Defaults to `'healpix-point-to-cell'`. */
  id?: string;
  /** HEALPix order (depth), 0 to 29: `nside = 2^order`, `12 * 4^order` cells. */
  order: number;
  /** Key scheme. Defaults to `'nested'`. */
  scheme?: GPUHealpixIndexScheme;
  /** Coordinate contract of `positions`: longitude/latitude without a metric (the default). */
  spatialContext?: GPUSpatialContext;
  /** Longitude/latitude degrees per row. */
  positions: GraphDataView<'float32x2'> | GraphVectorView<'float32x2'>;
  /** Optional per-row mask; zero produces a zero key. */
  mask?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUHealpixPointToCellOutput;
};

/**
 * Converts longitude/latitude points to HEALPix nested cell keys (RTS-08), one row per row, with
 * the same output layout as `GPUPointToCell` (little-endian `(low, high)` words and an optional
 * validity flag; masked and non-finite rows get the zero key).
 *
 * The projection is the closed-form `ang2pix_nest` of Gorski et al. (2005) on the sphere; the
 * cells have equal area. There is no ellipsoid: healpix-geo's `ellipsoid='sphere'` is the
 * reference. Geographic latitude on a spheroid needs the authalic conversion first, which this
 * contributor does not do.
 *
 * **float32 accuracy.** Positions are rounded to float32 and the projection is evaluated in f32,
 * so a point can fall in a neighbouring cell when it lies within the rounding error of a cell
 * edge. The error is a few units in `2^-23` of `4 * nside` cell widths, so the share of wrong
 * cells grows linearly with `nside`; the spec measures it against healpix-geo on float32 inputs
 * and the documentation lists the rates by order. Orders above about 14 (cells under 400 m) are
 * dominated by the 1.5e-5 degree quantization of float32 longitude near 180 degrees: the key is
 * the cell of the rounded position. A mismatch is always an adjacent cell.
 *
 * Packed and chunked inputs are supported. Outputs mirror input chunks and must not share buffers with the inputs.
 */
export class GPUHealpixPointToCell implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUHealpixPointToCellProps;
  /** Resolved spatial context (longitude/latitude, no metric). */
  readonly spatialContext: GPUSpatialContext;

  /** Upload plus one dispatch per nonempty chunk. */
  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;

  constructor(props: GPUHealpixPointToCellProps) {
    this.id = props.id ?? 'healpix-point-to-cell';
    this.props = props;
    const id = this.id;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext ?? GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
      GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT_SUPPORT
    );
    validateCapacity(id, 'order', props.order, 0, HEALPIX_MAXIMUM_ORDER);
    validateRequirement(
      id,
      props.scheme === undefined || ['nested', 'uniq'].includes(props.scheme),
      "scheme must be 'nested' or 'uniq'"
    );
    this.chunkPlan = planGraphChunkDispatch(id, {
      positions: props.positions,
      mask: props.mask,
      outputCells: props.output.cells,
      validity: props.output.validity
    });
    validateCapacity(id, 'positions row count', this.chunkPlan.totalLength);
    for (const chunk of this.chunkPlan.chunks) {
      if (chunk.views['positions'])
        validateSinglePackedView(
          chunk.views['positions'],
          ['float32x2'],
          `${id} positions chunk ${chunk.chunkIndex}`
        );
      validateMaximumCount(
        id,
        'positions chunk components',
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
      if (chunk.views['outputCells'])
        validateSinglePackedView(
          chunk.views['outputCells'],
          ['uint32x2'],
          `${id} outputCells chunk ${chunk.chunkIndex}`
        );
      validateMaximumCount(
        id,
        'outputCells chunk components',
        chunk.length * 2,
        MAXIMUM_CHUNK_COMPONENT_COUNT
      );
      if (chunk.views['validity'])
        validateSinglePackedView(
          chunk.views['validity'],
          ['uint32'],
          `${id} validity chunk ${chunk.chunkIndex}`
        );
      validateMaximumCount(
        id,
        'validity chunk components',
        chunk.length,
        MAXIMUM_CHUNK_COMPONENT_COUNT
      );
    }
    validateDisjointWritableViews(id, props.output);
    this.estimatedCommandNodeCount =
      (this.chunkPlan.chunks.length > 0 ? 1 : 0) + this.chunkPlan.nonemptyChunkCount;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.cells, props.output.validity],
      [props.positions, props.mask]
    );
  }

  /** Uploads chunk parameters and dispatches each nonempty position chunk. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    const isUniq = props.scheme === 'uniq';
    validateMaximumCount(
      id,
      'storage bindings',
      2 + Number(Boolean(props.mask)) + Object.values(props.output).filter(Boolean).length,
      graph.device.limits.maxStorageBuffersPerShaderStage
    );
    validateGraphChunkDispatchBindings(graph, id, this.chunkPlan);
    const invocationsPerRow = 1;
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
        const kernelProps: WGSLKernelProps<Parameters> = {
          id: `${id}-keys`,
          operation: OPERATION,
          variant: `healpix-${isUniq ? 'uniq' : 'nested'}`,
          bindings: [
            {name: 'chunkParameters', view: slots[chunk.chunkIndex], type: 'u32', access: 'read'},
            {name: 'positions', view: chunk.views['positions'], type: 'f32', access: 'read'},
            ...(props.mask
              ? [{name: 'rowMask', view: chunk.views['mask'], type: 'u32', access: 'read'} as const]
              : []),
            {name: 'cellsOut', view: chunk.views['outputCells'], type: 'u32', access: 'read_write'},
            ...(output.validity
              ? [
                  {
                    name: 'validityOut',
                    view: chunk.views['validity'],
                    type: 'u32',
                    access: 'read_write'
                  } as const
                ]
              : [])
          ],
          invocationCount: chunk.length * invocationsPerRow,
          declarations: `const ORDER: u32 = ${props.order}u;
${HEALPIX_INDEX_WGSL}
${IS_FINITE_BITS_WGSL}`,
          body: `let longitude = positions[positionsOffset + 2u * index];
  let latitude = positions[positionsOffset + 2u * index + 1u];
  var isValid = isFiniteBits(longitude) && isFiniteBits(latitude);
  ${props.mask ? 'isValid = isValid && rowMask[rowMaskOffset + index] != 0u;' : ''}
  var key = vec2u(0u);
  if (isValid) {
    key = cellIndexHealpixNestedFromLngLat(longitude, latitude, ORDER);
    ${isUniq ? 'key = cellIndexHealpixUniq(key, ORDER);' : ''}
  }
  cellsOut[cellsOutOffset + 2u * index] = key.y;
  cellsOut[cellsOutOffset + 2u * index + 1u] = key.x;
  ${output.validity ? 'validityOut[validityOutOffset + index] = select(0u, 1u, isValid);' : ''}`
        };
        const storageDeclarations = kernelProps.bindings
          .map(
            (binding, location) =>
              `@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
          )
          .join('\n');
        const bindingComponentCounts: Record<string, number> = {
          positions: 2,
          rowMask: 1,
          cellsOut: 2,
          validityOut: 1
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
