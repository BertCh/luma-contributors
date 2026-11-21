// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {validateCapacity, validateCommandNodeBudget} from './gpu-contributor-utils';
import {
  GPU_FITTING_MAXIMUM_ROW_COUNT,
  validateFittingGraph,
  validateFittingView,
  validateFittingWritableViews
} from './fitting-utils';
import {PHILOX_PURPOSES, PHILOX_WGSL} from './philox-random';
import {createWGSLKernelNode} from './wgsl-kernel-nodes';
import {snapshotConstructionFields} from './construction-snapshot';

/** Upper bound for the published 32-bit radix sort's passes plus key generation. */
export const GPU_FITTING_MINIBATCH_ORDER_SORT_NODE_ALLOWANCE = 128;

export type GPUFittingMinibatchOrderOutput = {
  /** Zero-based source row indices sorted by Philox key, ties in original row order. */
  order: GraphDataView<'uint32'>;
};
export type GPUFittingMinibatchOrderProps = {
  id?: string;
  /** Compile-time number of rows; divide the permutation into consecutive batches on the GPU. */
  rowCount: number;
  /** Per-frame [seedLow, seedHigh, epoch], three uint32 scalars. */
  parameters: GraphDataView<'uint32'>;
  output: GPUFittingMinibatchOrderOutput;
};

/** Packs a seed and epoch without signed truncation or loss of uint32 precision. */
export function getGPUFittingMinibatchOrderParameterValues(
  seed: readonly [number, number],
  epoch: number
): Uint32Array {
  for (const [name, value] of [
    ['seedLow', seed[0]],
    ['seedHigh', seed[1]],
    ['epoch', epoch]
  ] as const) {
    validateCapacity('fitting-minibatch-order', name, value, 0, 0xffffffff);
  }
  return new Uint32Array([seed[0], seed[1], epoch]);
}

/**
 * Seeded row permutation for model fitting using Philox keys and published stable GPUSort.
 * Total source rowCount is at most 2^24 (16,777,216). Each packed binding is at most 128 MiB
 * and must also fit the device binding limit. output.order contains zero-based source rows.
 * Compile-time: row count. Per-frame: both seed words and epoch. Equal keys preserve ascending
 * source row order; there is no sampling with replacement and no lost row on a key collision.
 * A random-key sort has a small finite-key tie bias; it is not claimed to be a uniform draw from
 * all n! permutations. No CPU row-sized work or readback.
 */
export class GPUFittingMinibatchOrder implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUFittingMinibatchOrderProps;
  readonly estimatedCommandNodeCount = GPU_FITTING_MINIBATCH_ORDER_SORT_NODE_ALLOWANCE;

  constructor(props: GPUFittingMinibatchOrderProps) {
    props = Object.freeze({
      ...snapshotConstructionFields(props, ['id', 'rowCount', 'parameters', 'output']),
      output: snapshotConstructionFields(props.output, ['order'])
    });
    this.id = props.id ?? 'fitting-minibatch-order';
    this.props = props;
    validateCapacity(this.id, 'rowCount', props.rowCount, 0, GPU_FITTING_MAXIMUM_ROW_COUNT);
    validateFittingView(this.id, 'parameters', props.parameters, 'uint32', 3);
    validateFittingView(this.id, 'output.order', props.output.order, 'uint32', props.rowCount);
    validateFittingWritableViews(this.id, [props.output.order], [props.parameters]);
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateFittingGraph(id, graph, [props.parameters, props.output.order]);
    if (props.rowCount === 0) return [];
    const keys = createTransientView(graph, `${id}-keys`, 'uint32', props.rowCount);
    const indices = createTransientView(graph, `${id}-indices`, 'uint32', props.rowCount);
    const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', props.rowCount);
    return [
      createWGSLKernelNode(graph, {
        id: `${id}-keys`,
        operation: 'GPUFittingMinibatchOrder',
        variant: 'philox-epoch-keys',
        invocationCount: props.rowCount,
        bindings: [
          {name: 'parameters', view: props.parameters, type: 'u32', access: 'read'},
          {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
          {name: 'indices', view: indices, type: 'u32', access: 'read_write'}
        ],
        declarations: PHILOX_WGSL,
        body: `let key = vec2<u32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
  keys[keysOffset + index] = philox4x32(
    vec4<u32>(index, parameters[parametersOffset + 2u], 0u, ${PHILOX_PURPOSES.fittingMinibatchOrder.id}u), key).x;
  indices[indicesOffset + index] = index;`
      }),
      ...new GPUSort({
        id: `${id}-sort`,
        keys,
        values: indices,
        outputKeys: sortedKeys,
        outputValues: props.output.order,
        keyBits: 32
      }).getCommandNodes(graph)
    ];
  }
}
