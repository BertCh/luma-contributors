// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A one-node GPU step that copies an operation output into a float32 buffer, so counts can be drawn
// as chart heights (the kit draws uint32 as categories), one column of an interleaved table can be
// drawn as a line, and a grid can be flipped to the raster's top-down row order. It runs in the same
// command graph, after the operation that wrote the source.

import type {Binding} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {RunBuffer, RunBuilder, RunFormat} from '../../../define-run';

const WORKGROUP_SIZE = 64;

/** Which rows of the source are copied: element `index` reads `offset + index * stride`. */
export type CopyLayout = {
  /** Source elements between consecutive output rows. Default 1. */
  stride?: number;
  /** First source element. Default 0. */
  offset?: number;
  /** Treat the output as a grid of this many columns and flip its rows (bottom-up to top-down). */
  flipColumns?: number;
  /** Components per output row (1 to 4): output row r reads `componentStart + c` of source row r. */
  outputComponents?: 1 | 2 | 3 | 4;
  /** First source component of each row to copy. Default 0. */
  componentStart?: number;
};

const OUTPUT_FORMATS = {1: 'float32', 2: 'float32x2', 3: 'float32x3', 4: 'float32x4'} as const;

class IntegerToFloat implements GPUCommandNodeProducer {
  constructor(
    private readonly id: string,
    private readonly source: GraphDataView<RunFormat>,
    private readonly target: GraphDataView<RunFormat>,
    private readonly layout: CopyLayout
  ) {}

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, source, target} = this;
    const {outputComponents = 1, componentStart = 0} = this.layout;
    const outputCount = target.length * outputComponents;
    const sourceComponents = source.rowByteLength / 4;
    const dispatch = getBoundedDispatchLayout(
      id,
      outputCount,
      WORKGROUP_SIZE,
      graph.device.limits.maxComputeWorkgroupsPerDimension
    );
    const sourceType = source.format.startsWith('sint')
      ? 'i32'
      : source.format.startsWith('uint')
        ? 'u32'
        : 'f32';
    const {stride = sourceComponents, offset = 0, flipColumns} = this.layout;
    const rowCount = flipColumns ? Math.floor(target.length / flipColumns) : 0;
    const sourceIndex = flipColumns
      ? `(${rowCount - 1}u - row / ${flipColumns}u) * ${flipColumns}u + row % ${flipColumns}u`
      : 'row';
    const shaderSource = /* wgsl */ `
const inputOffset: u32 = ${getViewElementOffset(source)}u;
const outputOffset: u32 = ${getViewElementOffset(target)}u;
@group(0) @binding(0) var<storage, read> inputValues: array<${sourceType}>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<f32>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${getBoundedInvocationIndexSource(dispatch, WORKGROUP_SIZE)}
  if (index >= ${outputCount}u) { return; }
  let row = index / ${outputComponents}u;
  let component = index % ${outputComponents}u;
  outputValues[outputOffset + index] = f32(inputValues[inputOffset + ${offset}u + (${sourceIndex}) * ${stride}u + ${componentStart}u + component]);
}`;
    return [
      createGPUComputeCommandNode<Parameters>({
        id,
        resources: [
          {buffer: source, usage: 'storage-read'},
          {buffer: target, usage: 'storage-read-write'}
        ],
        workload: {
          operation: 'IntegerToFloat',
          variant: source.format,
          commandCount: 1,
          maximumWorkgroupCount: dispatch.x * dispatch.y * dispatch.z,
          maximumInvocationCount: dispatch.x * dispatch.y * dispatch.z * WORKGROUP_SIZE,
          readByteLength: target.length * 4,
          writeByteLength: target.length * 4
        },
        compile: ({device}) => {
          const kernel = new Kernel(device, {
            id,
            source: shaderSource,
            shaderLayout: {
              bindings: [
                {name: 'inputValues', type: 'storage' as const, group: 0, location: 0},
                {name: 'outputValues', type: 'storage' as const, group: 0, location: 1}
              ]
            }
          });
          return {
            encode: ({computePass, getBuffer}) => {
              const bindings: Record<string, Binding> = {
                inputValues: getViewBinding(source, getBuffer),
                outputValues: getViewBinding(target, getBuffer)
              };
              kernel.dispatch(computePass, {bindings, ...dispatch});
            },
            destroy: () => kernel.destroy()
          };
        }
      })
    ];
  }
}

/** Adds a float32 copy of an integer output (a bar height or a scatter y); returns the copy. */
export function addIntegerToFloat(
  builder: RunBuilder,
  id: string,
  source: RunBuffer<'uint32'> | RunBuffer<'sint32'>
): RunBuffer {
  return addFloatCopy(builder, id, source, source.length);
}

/** Adds a float32 copy of `length` rows of an output, picked by `layout`; returns the copy. */
export function addFloatCopy(
  builder: RunBuilder,
  id: string,
  source: RunBuffer,
  length: number,
  layout: CopyLayout = {}
): RunBuffer {
  const copy = builder.allocate(id, OUTPUT_FORMATS[layout.outputComponents ?? 1], length);
  builder.graph.add(new IntegerToFloat(`${id}-copy`, source.view, copy.view, layout));
  return copy;
}
