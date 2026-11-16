// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateMaximumCount,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';

const OPERATION = 'GPUCubeLayoutTranspose';
/** Element indices are `uint32`. */
const MAXIMUM_ELEMENT_COUNT = 0xffffffff;

/**
 * Dense cube layouts, one scalar per (cell, period):
 * - `'cell-major'`: `cell * periodCount + period`, the layout of the hot spot cube and of
 *   `GPUEmergingHotSpots`.
 * - `'period-major'`: `period * cellCount + cell`, the layout of Markov chains, spatial panels and
 *   per-period maps.
 */
export type GPUCubeLayout = 'cell-major' | 'period-major';

/** Properties for {@link GPUCubeLayoutTranspose}. */
export type GPUCubeLayoutTransposeProps = {
  /** Prefix for generated node IDs. Defaults to `'cube-layout-transpose'`. */
  id?: string;
  /** Packed dense cube, `cellCount * periodCount` values in the `from` layout. */
  values: GraphDataView<'uint32'> | GraphDataView<'float32'>;
  /** Layout of `values`. The output holds the other one. */
  from: GPUCubeLayout;
  /** Number of cells (compile-time). */
  cellCount: number;
  /** Number of periods (compile-time). */
  periodCount: number;
  /** Caller-owned cube of the same format and length as `values`, in the other layout. */
  output: GraphDataView<'uint32'> | GraphDataView<'float32'>;
};

/**
 * Moves a dense (cell, period) cube between the cell-major layout of the space-time hot spot cube
 * and the period-major layout of dynamics and panel operators, for `uint32` or `float32` values.
 * Every value is copied bit for bit, so the transpose is exact (NaN payloads and counts alike) and
 * transposing twice returns the original cube.
 *
 * The cube is one packed view, so `cellCount * periodCount` values must fit one storage binding
 * (33,554,432 values on a default device) and a `uint32` index. The transpose is a scatter with
 * one thread per element; the writes of neighboring threads are `periodCount` or `cellCount`
 * elements apart, which is acceptable for a layout change done once per cube.
 * The published `GPUTranspose` pads tiles past the uint32 invocation limit for large degenerate
 * cubes and does not check aligned storage binding sizes, so this kernel handles the copy.
 *
 * Per-frame (no recompile): the contents of `values` and `output`. Topology (needs a new graph):
 * `from`, `cellCount`, `periodCount` and the value format.
 */
export class GPUCubeLayoutTranspose implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCubeLayoutTransposeProps;
  /** Number of values in the cube. */
  readonly elementCount: number;

  constructor(props: GPUCubeLayoutTransposeProps) {
    this.id = props.id ?? 'cube-layout-transpose';
    this.props = props;
    const id = this.id;
    const {values, output, cellCount, periodCount} = props;
    validateRequirement(
      id,
      props.from === 'cell-major' || props.from === 'period-major',
      'from must be cell-major or period-major'
    );
    validateCapacity(id, 'cellCount', cellCount, 1, MAXIMUM_ELEMENT_COUNT);
    validateCapacity(id, 'periodCount', periodCount, 1, MAXIMUM_ELEMENT_COUNT);
    this.elementCount = cellCount * periodCount;
    validateMaximumCount(id, 'cellCount * periodCount', this.elementCount, MAXIMUM_ELEMENT_COUNT);
    validateSinglePackedView(values, ['uint32', 'float32'], `${id} values`);
    validateSinglePackedView(output, [values.format], `${id} output`);
    validateViewLength(id, 'values', values, this.elementCount);
    validateViewLength(id, 'output', output, this.elementCount);
    validateGraphOutputsDisjointFromInputs(id, [output], [values]);
  }

  /** Returns the one transpose kernel. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, elementCount} = this;
    const {values, output, cellCount, periodCount, from} = props;
    validateGraphChunkBindingRanges(graph, id, {values, output});
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'cube',
      elementCount * 4,
      'reduce cellCount * periodCount'
    );
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-transpose`,
        operation: OPERATION,
        variant: from === 'cell-major' ? 'to-period-major' : 'to-cell-major',
        bindings: [
          {name: 'cube', view: values, type: 'u32', access: 'read'},
          {name: 'transposed', view: output, type: 'u32', access: 'read_write'}
        ],
        invocationCount: elementCount,
        // Sizes are overrides so that cubes of different shapes share one shader module.
        declarations: `override CELL_COUNT: u32;
override PERIOD_COUNT: u32;`,
        constants: {CELL_COUNT: cellCount, PERIOD_COUNT: periodCount},
        body:
          from === 'cell-major'
            ? `let cell = index / PERIOD_COUNT;
  let period = index % PERIOD_COUNT;
  transposed[transposedOffset + period * CELL_COUNT + cell] = cube[cubeOffset + index];`
            : `let period = index / CELL_COUNT;
  let cell = index % CELL_COUNT;
  transposed[transposedOffset + cell * PERIOD_COUNT + period] = cube[cubeOffset + index];`
      })
    ];
  }
}
