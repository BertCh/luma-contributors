// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUSpatialWeights, GPUSpatialWeightsOutput} from './spatial-weights';
import {
  createCSRStages,
  getMergeVisitWGSL,
  type CSRStageAttribute,
  type CSRStageProps,
  type SpatialWeightsMergeOperation
} from './spatial-weights-algebra-wgsl';

/**
 * Stage settings that differ between whole and chunked weights: everything {@link createCSRStages}
 * takes except what the row-local operations below supply.
 *
 * @internal
 */
export type RowLocalStageBase = Omit<CSRStageProps, 'bindings' | 'visit' | 'attributes'>;

/**
 * WGSL expression of the chunk-local row of the visited global row `index`: `index` for whole
 * weights, `(index - WEIGHTS_ROW_START)` for a chunk.
 *
 * @internal
 */
export function getLocalRowWGSL(base: RowLocalStageBase): string {
  return base.chunk ? '(index - WEIGHTS_ROW_START)' : 'index';
}

/**
 * `union`, `intersection`, `difference` and `symmetricDifference` of two CSRs as CSR stages. The
 * one implementation behind {@link GPUSpatialWeightsAlgebra} and its chunked wrapper.
 *
 * @param combineExpression WGSL expression over `leftValue` and `rightValue` for a shared neighbor.
 * @internal
 */
export function getBinaryMergeStages<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  base: RowLocalStageBase,
  props: {
    operation: SpatialWeightsMergeOperation;
    left: GPUSpatialWeights;
    right: GPUSpatialWeights;
    output: GPUSpatialWeightsOutput;
    isBinaryRule: boolean;
    combineExpression: string;
  }
): GPUCommandNode<Parameters>[] {
  const {operation, left, right, output} = props;
  const attributes: CSRStageAttribute[] = [
    {name: 'neighbors', output: output.neighbors, type: 'u32', value: 'id'},
    {
      name: 'weights',
      output: output.weights,
      type: 'f32',
      value: 'combineWeight(leftSlot, rightSlot)',
      bindings: [
        {name: 'leftWeights', view: left.weights, type: 'f32', access: 'read'},
        {name: 'rightWeights', view: right.weights, type: 'f32', access: 'read'}
      ],
      declarations: `fn combineWeight(leftSlot: u32, rightSlot: u32) -> f32 {
  ${props.isBinaryRule ? 'return 1.0;' : ''}
  if (rightSlot == NO_SLOT) {
    return leftWeights[leftWeightsOffset + leftSlot];
  }
  if (leftSlot == NO_SLOT) {
    return rightWeights[rightWeightsOffset + rightSlot];
  }
  let leftValue = leftWeights[leftWeightsOffset + leftSlot];
  let rightValue = rightWeights[rightWeightsOffset + rightSlot];
  return ${props.combineExpression};
}`
    }
  ];
  if (output.distances) {
    const rightDistances =
      right.distances && operation !== 'intersection' && operation !== 'difference';
    attributes.push({
      name: 'distances',
      output: output.distances,
      type: 'f32',
      value: 'pickDistance(leftSlot, rightSlot)',
      bindings: [
        {name: 'leftDistances', view: left.distances!, type: 'f32', access: 'read'},
        ...(rightDistances
          ? [
              {
                name: 'rightDistances',
                view: right.distances!,
                type: 'f32' as const,
                access: 'read' as const
              }
            ]
          : [])
      ],
      declarations: `fn pickDistance(leftSlot: u32, rightSlot: u32) -> f32 {
  if (leftSlot != NO_SLOT) {
    return leftDistances[leftDistancesOffset + leftSlot];
  }
  ${rightDistances ? 'return rightDistances[rightDistancesOffset + rightSlot];' : 'return 0.0;'}
}`
    });
  }
  const localRow = getLocalRowWGSL(base);
  return createCSRStages(graph, {
    ...base,
    bindings: [
      {name: 'leftOffsets', view: left.offsets, type: 'u32', access: 'read'},
      {name: 'leftNeighbors', view: left.neighbors, type: 'u32', access: 'read'},
      {name: 'rightOffsets', view: right.offsets, type: 'u32', access: 'read'},
      {name: 'rightNeighbors', view: right.neighbors, type: 'u32', access: 'read'}
    ],
    visit: onItem => getMergeVisitWGSL(operation, onItem, localRow),
    attributes
  });
}

/** `selfWeight` of one CSR as CSR stages; shared by whole and chunked weights. @internal */
export function getSelfWeightStages<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  base: RowLocalStageBase,
  props: {
    weights: GPUSpatialWeights;
    output: GPUSpatialWeightsOutput;
    selfWeight: number | GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {weights, output, selfWeight} = props;
  const selfValue =
    typeof selfWeight === 'number'
      ? getWGSLFloatLiteral(selfWeight)
      : 'selfWeights[selfWeightsOffset + index]';
  const attributes: CSRStageAttribute[] = [
    {name: 'neighbors', output: output.neighbors, type: 'u32', value: 'id'},
    {
      name: 'weights',
      output: output.weights,
      type: 'f32',
      value: `select(sourceWeights[sourceWeightsOffset + sourceSlot], ${selfValue}, id == index)`,
      bindings: [
        {name: 'sourceWeights', view: weights.weights, type: 'f32', access: 'read'},
        ...(typeof selfWeight === 'number'
          ? []
          : [
              {
                name: 'selfWeights',
                view: selfWeight,
                type: 'f32' as const,
                access: 'read' as const
              }
            ])
      ]
    }
  ];
  if (output.distances) {
    attributes.push({
      name: 'distances',
      output: output.distances,
      type: 'f32',
      value: 'select(sourceDistances[sourceDistancesOffset + sourceSlot], 0.0, id == index)',
      bindings: [{name: 'sourceDistances', view: weights.distances!, type: 'f32', access: 'read'}]
    });
  }
  const localRow = getLocalRowWGSL(base);
  return createCSRStages(graph, {
    ...base,
    bindings: [
      {name: 'sourceOffsets', view: weights.offsets, type: 'u32', access: 'read'},
      {name: 'sourceNeighbors', view: weights.neighbors, type: 'u32', access: 'read'}
    ],
    // The source slot of the inserted or replaced self entry is clamped to a valid slot by
    // `select`; the value is only read for `id != index`.
    visit: onItem => `var inserted = false;
  for (var slot = sourceOffsets[sourceOffsetsOffset + ${localRow}]; slot < sourceOffsets[sourceOffsetsOffset + ${localRow} + 1u]; slot++) {
    let neighbor = sourceNeighbors[sourceNeighborsOffset + slot];
    if (!inserted && neighbor >= index) {
      inserted = true;
      {
        let id = index;
        let sourceSlot = slot;
        ${onItem}
      }
      if (neighbor == index) {
        continue;
      }
    }
    {
      let id = neighbor;
      let sourceSlot = slot;
      ${onItem}
    }
  }
  if (!inserted) {
    let id = index;
    let sourceSlot = 0u;
    ${onItem}
  }`,
    attributes
  });
}

/** `subgraph` of one CSR as CSR stages; shared by whole and chunked weights. @internal */
export function getSubgraphStages<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  base: RowLocalStageBase,
  props: {
    weights: GPUSpatialWeights;
    output: GPUSpatialWeightsOutput;
    mask: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {weights, output, mask} = props;
  const attributes: CSRStageAttribute[] = [
    {name: 'neighbors', output: output.neighbors, type: 'u32', value: 'id'},
    {
      name: 'weights',
      output: output.weights,
      type: 'f32',
      value: 'sourceWeights[sourceWeightsOffset + slot]',
      bindings: [{name: 'sourceWeights', view: weights.weights, type: 'f32', access: 'read'}]
    }
  ];
  if (output.distances) {
    attributes.push({
      name: 'distances',
      output: output.distances,
      type: 'f32',
      value: 'sourceDistances[sourceDistancesOffset + slot]',
      bindings: [{name: 'sourceDistances', view: weights.distances!, type: 'f32', access: 'read'}]
    });
  }
  const localRow = getLocalRowWGSL(base);
  return createCSRStages(graph, {
    ...base,
    bindings: [
      {name: 'sourceOffsets', view: weights.offsets, type: 'u32', access: 'read'},
      {name: 'sourceNeighbors', view: weights.neighbors, type: 'u32', access: 'read'},
      {name: 'mask', view: mask, type: 'u32', access: 'read'}
    ],
    visit: onItem => `if (mask[maskOffset + index] != 0u) {
    for (var slot = sourceOffsets[sourceOffsetsOffset + ${localRow}]; slot < sourceOffsets[sourceOffsetsOffset + ${localRow} + 1u]; slot++) {
      let id = sourceNeighbors[sourceNeighborsOffset + slot];
      if (id < ROWS && mask[maskOffset + id] != 0u) {
        ${onItem}
      }
    }
  }`,
    attributes
  });
}
