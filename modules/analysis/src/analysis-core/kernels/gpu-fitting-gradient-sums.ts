// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength
} from './gpu-contributor-utils';
import {
  GPU_FITTING_MAXIMUM_PARAMETER_COUNT,
  GPU_FITTING_MAXIMUM_ROW_COUNT,
  validateFittingGraph,
  validateFittingView,
  validateFittingWritableViews
} from './fitting-utils';
import {createWGSLKernelNode} from './wgsl-kernel-nodes';

/** A fixed tile bounds every serial walk; subsequent levels merge pairs in separate dispatches. */
export const GPU_FITTING_GRADIENT_TILE_ROWS = 256;

export type GPUFittingGradientSumsOutput = {
  /** Sum per parameter (or loss term); skipped rows contribute zero. */
  sums: GraphDataView<'float32'>;
};

export type GPUFittingGradientSumsProps = {
  id?: string;
  /** Compile-time number of rows and terms. Zero rows produces zero sums. */
  rowCount: number;
  parameterCount: number;
  /** Per-frame row-major gradients, rowCount * parameterCount scalar floats. */
  gradients: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  output: GPUFittingGradientSumsOutput;
};

/**
 * Fixed-order minibatch gradient sums, also usable for row losses. Compile-time: shapes and mask
 * presence. Per-frame: gradient and mask contents. Each invocation walks at most 256 rows for one
 * term; tile partials merge in a fixed binary tree. No floating atomics or row-sized serial merge.
 * Repeatable on one device/compiler; cross-device bit identity is not promised. Non-finite retained
 * terms propagate; callers should produce finite derivatives. Divide by the retained batch size
 * in the derivative kernel when a mean gradient is wanted.
 */
export class GPUFittingGradientSums implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUFittingGradientSumsProps;
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUFittingGradientSumsProps) {
    this.id = props.id ?? 'fitting-gradient-sums';
    this.props = props;
    validateCapacity(this.id, 'rowCount', props.rowCount, 0, GPU_FITTING_MAXIMUM_ROW_COUNT);
    validateCapacity(
      this.id,
      'parameterCount',
      props.parameterCount,
      1,
      GPU_FITTING_MAXIMUM_PARAMETER_COUNT
    );
    validateFittingView(
      this.id,
      'gradients',
      props.gradients,
      'float32',
      props.rowCount * props.parameterCount
    );
    validateFittingView(this.id, 'output.sums', props.output.sums, 'float32', props.parameterCount);
    if (props.mask) validateFittingView(this.id, 'mask', props.mask, 'uint32', props.rowCount);
    validateFittingWritableViews(
      this.id,
      [props.output.sums],
      [props.gradients, ...(props.mask ? [props.mask] : [])]
    );
    const tileCount = Math.max(1, Math.ceil(props.rowCount / GPU_FITTING_GRADIENT_TILE_ROWS));
    this.estimatedCommandNodeCount = 1 + Math.ceil(Math.log2(tileCount));
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {rowCount, parameterCount, gradients, mask, output} = props;
    let tileCount = Math.max(1, Math.ceil(rowCount / GPU_FITTING_GRADIENT_TILE_ROWS));
    validateFittingGraph(id, graph, [gradients, output.sums, ...(mask ? [mask] : [])]);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'gradient partials',
      tileCount * parameterCount * 4,
      'reduce the minibatch or parameter count'
    );
    let partials =
      tileCount === 1
        ? output.sums
        : createTransientView(graph, `${id}-tiles`, 'float32', tileCount * parameterCount);
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-tiles`,
        operation: 'GPUFittingGradientSums',
        variant: 'row-tiles',
        invocationCount: tileCount * parameterCount,
        bindings: [
          {name: 'gradients', view: gradients, type: 'f32', access: 'read'},
          ...(mask
            ? [{name: 'mask', view: mask, type: 'u32' as const, access: 'read' as const}]
            : []),
          {name: 'partials', view: partials, type: 'f32', access: 'read_write'}
        ],
        body: `let parameter = index % ${parameterCount}u;
  let firstRow = (index / ${parameterCount}u) * ${GPU_FITTING_GRADIENT_TILE_ROWS}u;
  var total = 0.0;
  for (var offset = 0u; offset < ${GPU_FITTING_GRADIENT_TILE_ROWS}u; offset++) {
    let row = firstRow + offset;
    if (row < ${rowCount}u) {
      ${mask ? 'if (mask[maskOffset + row] == 0u) { continue; }' : ''}
      total += gradients[gradientsOffset + row * ${parameterCount}u + parameter];
    }
  }
  partials[partialsOffset + index] = total;`
      })
    ];
    let level = 0;
    while (tileCount > 1) {
      const nextCount = Math.ceil(tileCount / 2);
      const next =
        nextCount === 1
          ? output.sums
          : createTransientView(
              graph,
              `${id}-level-${level}`,
              'float32',
              nextCount * parameterCount
            );
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-merge-${level++}`,
          operation: 'GPUFittingGradientSums',
          variant: 'pairwise-merge',
          invocationCount: nextCount * parameterCount,
          bindings: [
            {name: 'source', view: partials, type: 'f32', access: 'read'},
            {name: 'destination', view: next, type: 'f32', access: 'read_write'}
          ],
          body: `let parameter = index % ${parameterCount}u;
  let firstTile = 2u * (index / ${parameterCount}u);
  var total = source[sourceOffset + firstTile * ${parameterCount}u + parameter];
  if (firstTile + 1u < ${tileCount}u) {
    total += source[sourceOffset + (firstTile + 1u) * ${parameterCount}u + parameter];
  }
  destination[destinationOffset + index] = total;`
        })
      );
      partials = next;
      tileCount = nextCount;
    }
    return nodes;
  }
}
