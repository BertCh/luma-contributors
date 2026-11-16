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
import {Kernel, getPlatformInfo} from '@luma.gl/graph';
import {WGSLShaderAssembler, dggs, fp64} from '@luma.gl/shaders';
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
  validateSinglePackedView,
  validateRequirement,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {WGSLKernelProps} from '../../analysis-core/kernels/wgsl-kernel-nodes';

import {S2_ANALYSIS_WGSL} from './s2-analysis-wgsl';
import {CELL_MEASURES_FP64_WGSL} from '../cell-indexing/cell-measures-fp64-wgsl';
import {S2_CELL_MEASURES_FP64_WGSL} from '../cell-indexing/s2-cell-measures-fp64-wgsl';

/** Fixed workgroup size makes chunk lengths runtime dispatch metadata. */
const CHUNK_WORKGROUP_SIZE = 256;
/** Storage component indices of whole fixed-stride records cannot wrap uint32. */
const MAXIMUM_CHUNK_COMPONENT_COUNT = 0xffffffff;
/** Guard padded workgroups before multiplying their index into a uint32 invocation index. */
const MAXIMUM_LINEAR_WORKGROUP_COUNT = 16777216;

/** Status of one cell: the measures are valid. */
export const GPU_S2_CELL_MEASURES_STATUS_VALID = 0;
/** Status of one cell: the input is not a valid S2 cell ID, so its measures are NaN. */
export const GPU_S2_CELL_MEASURES_STATUS_INVALID = 1;
export type GPUS2CellMeasuresProps = {
  id?: string;
  cells: GraphDataView<'uint32x2'> | GraphVectorView<'uint32x2'>;
  /** Compile-time arithmetic choice; defaults to 'float32'. Outputs stay float32. */
  precision?: 'float32' | 'fp64';
  output: {
    /** Spherical area (steradians), perimeter (radians). Multiply by radius squared/radius externally. */
    measures: GraphDataView<'float32x2'> | GraphVectorView<'float32x2'>;
    /**
     * Per cell: {@link GPU_S2_CELL_MEASURES_STATUS_VALID} (0) or
     * {@link GPU_S2_CELL_MEASURES_STATUS_INVALID} (1, invalid cell ID).
     */
    status: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  };
};
/**
 * S2 spherical cell measurements at levels 0–30 using cancellation-free local increments.
 * Per-frame: cells contents. Compile-time: precision, row count and view layouts.
 * With precision 'fp64', uses the published double-single module for projection and local increments.
 *
 * Accepts packed or chunked row inputs; output chunks mirror inputs and keep whole fixed-stride records.
 */
export class GPUS2CellMeasures implements GPUCommandNodeProducer {
  readonly id: string;
  /** Upload plus one dispatch per nonempty chunk. */
  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;

  constructor(readonly props: GPUS2CellMeasuresProps) {
    this.id = props.id ?? 's2-cell-measures';
    validateRequirement(
      this.id,
      props.precision === undefined || ['float32', 'fp64'].includes(props.precision),
      "precision must be 'float32' or 'fp64'"
    );
    const id = this.id;
    this.chunkPlan = planGraphChunkDispatch(id, {
      cells: props.cells,
      measures: props.output.measures,
      status: props.output.status
    });
    validateCapacity(id, 'cells row count', this.chunkPlan.totalLength, 0);
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
      if (chunk.views['measures'])
        validateSinglePackedView(
          chunk.views['measures'],
          ['float32x2'],
          `${id} measures chunk ${chunk.chunkIndex}`
        );
      validateMaximumCount(
        id,
        'measures chunk components',
        chunk.length * 2,
        MAXIMUM_CHUNK_COMPONENT_COUNT
      );
      if (chunk.views['status'])
        validateSinglePackedView(
          chunk.views['status'],
          ['uint32'],
          `${id} status chunk ${chunk.chunkIndex}`
        );
      validateMaximumCount(
        id,
        'status chunk components',
        chunk.length,
        MAXIMUM_CHUNK_COMPONENT_COUNT
      );
    }
    validateDisjointWritableViews(id, props.output);
    this.estimatedCommandNodeCount =
      (this.chunkPlan.chunks.length > 0 ? 1 : 0) + this.chunkPlan.nonemptyChunkCount;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [props.cells]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateMaximumCount(
      id,
      'storage bindings',
      4,
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
    let sharedKernelSource: string | undefined;
    const chunkNodes = this.chunkPlan.chunks
      .filter(chunk => chunk.length > 0)
      .map(chunk => {
        const usesFP64 = props.precision === 'fp64';
        const kernelProps: WGSLKernelProps<Parameters> = {
          id,
          operation: 'GPUS2CellMeasures',
          variant: usesFP64 ? 'unit-sphere-fp64' : 'unit-sphere',
          modules: usesFP64 ? [fp64] : undefined,
          invocationCount: chunk.length * invocationsPerRow,
          bindings: [
            {name: 'chunkParameters', view: slots[chunk.chunkIndex], type: 'u32', access: 'read'},
            {name: 'cells', view: chunk.views['cells'], type: 'u32', access: 'read'},
            {name: 'measures', view: chunk.views['measures'], type: 'f32', access: 'read_write'},
            {name: 'status', view: chunk.views['status'], type: 'u32', access: 'read_write'}
          ],
          declarations: `${dggs.source}\n${usesFP64 ? `${CELL_MEASURES_FP64_WGSL}\n${S2_CELL_MEASURES_FP64_WGSL}` : S2_ANALYSIS_WGSL}`,
          body: `let cell = vec2u(cells[cellsOffset + index * 2u + 1u], cells[cellsOffset + index * 2u]);
var failure = select(1u, 0u, dggs_s2_is_valid_cell_id(cell));
var area = bitcast<f32>(0x7fc00000u | (index & 0u)); var perimeter = area;
if (failure == 0u) {
  let measured = ${usesFP64 ? 'cellMeasuresGetS2' : 's2AnalysisStableMeasures'}(cell);
  area = measured.${usesFP64 ? 'area' : 'x'}; perimeter = measured.${usesFP64 ? 'perimeter' : 'y'};
}
measures[measuresOffset + index * 2u] = area; measures[measuresOffset + index * 2u + 1u] = perimeter; status[statusOffset + index] = failure;`
        };
        const storageDeclarations = kernelProps.bindings
          .map(
            (binding, location) =>
              `@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
          )
          .join('\n');
        const bindingComponentCounts: Record<string, number> = {
          cells: 2,
          measures: 2,
          status: 1
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
        const kernelSource =
          sharedKernelSource ??
          (usesFP64
            ? new WGSLShaderAssembler().assembleWGSLShader({
                platformInfo: getPlatformInfo(graph.device),
                source,
                modules: [fp64],
                defines: {LUMA_FP64_INTEGER_ARITHMETIC: true},
                shaderStage: 'compute',
                computeEntryPoint: 'main',
                scanVertexAttributes: false
              }).source
            : source);
        sharedKernelSource = kernelSource;
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
              source: kernelSource,
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
