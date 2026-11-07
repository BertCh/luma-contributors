// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {HASH_UINT32_WGSL} from '../shared/hash-uint32';
import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {createWGSLKernelNode, getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {
  validateCapacity,
  validatePackedView,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {validatePredictionViews} from '../../utils/gpu-prediction-utils';
import {
  createSpatialValidationDesignNodes,
  type GPUSpatialValidationDesign,
  validateSpatialValidationDesign
} from './spatial-validation-design';
import {IS_FINITE_BITS_WGSL} from '../../utils/wgsl-common';

export type GPUSpatialCrossValidationProps = {
  id?: string;
  /** Planar positions in the same units as bufferDistance. */
  positions: GraphDataView<'float32x2'>;
  /** Explicit block IDs. All members of a block receive the same fold. */
  blocks: GraphDataView<'uint32'>;
  mask?: GraphDataView<'uint32'>;
  /** Optional explicit fold design; omitted preserves seeded block hashing. */
  design?: GPUSpatialValidationDesign;
  foldCount: number;
  /** Stable uint32 seed; assignment is deterministic, not balanced. */
  seed?: number;
  /** Training points at distance <= bufferDistance of any test point are excluded. */
  bufferDistance?: number;
  output: {
    /** Per-row fold ID, or 0xffffffff for invalid/masked positions. */
    folds: GraphDataView<'uint32'>;
    /** Fold-major row masks. These are the ONLY rows a fold's preprocessing/fit may use. */
    training: GraphDataView<'uint32'>;
    testing: GraphDataView<'uint32'>;
    /** Per fold: training count, test count, buffered-out count. */
    counts: GraphDataView<'uint32'>;
    /** Per fold bitmask: 1 empty training set, 2 empty test set. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Seeded spatial block holdouts with exact all-pairs planar buffer exclusion. Every valid row
 * appears in exactly one test fold. Invalid rows appear in neither set. This bounded baseline
 * costs O(folds * rows²); it makes no claim to blockCV's balancing or polygon generation.
 * Fit preprocessing, feature selection and bandwidth/variogram selection separately using each
 * training mask. This producer never fits a model or silently reuses a whole-sample fit.
 */
export class GPUSpatialCrossValidation implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUSpatialCrossValidationProps) {
    this.id = props.id ?? 'spatial-cross-validation';
    const rows = props.positions.length;
    validateSpatialValidationDesign(this.id, props);
    validateCapacity(this.id, 'rows', rows, 1, 65536);
    validateCapacity(this.id, 'foldCount', props.foldCount, 2, 64);
    if (
      !Number.isInteger(props.seed ?? 0) ||
      (props.seed ?? 0) < 0 ||
      (props.seed ?? 0) > 0xffffffff ||
      !Number.isFinite(props.bufferDistance ?? 0) ||
      (props.bufferDistance ?? 0) < 0 ||
      (props.bufferDistance ?? 0) > 1e18
    )
      throw new Error(`${this.id}: invalid seed or distance`);
    for (const [view, format, length] of [
      [props.positions, 'float32x2', rows],
      [props.blocks, 'uint32', rows],
      [props.mask, 'uint32', rows],
      [props.output.folds, 'uint32', rows],
      [props.output.training, 'uint32', rows * props.foldCount],
      [props.output.testing, 'uint32', rows * props.foldCount],
      [props.output.counts, 'uint32', 3 * props.foldCount],
      [props.output.status, 'uint32', props.foldCount]
    ] as const)
      if (view) {
        validatePackedView(view, [format], this.id);
        if (view.length !== length) throw new Error(`${this.id}: shape mismatch`);
      }
    validatePredictionViews(
      this.id,
      [props.positions, props.blocks, ...(props.mask ? [props.mask] : [])],
      Object.values(props.output)
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const rows = props.positions.length;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.blocks,
      props.mask,
      ...Object.values(props.output)
    ]);
    const design = props.design ? createSpatialValidationDesignNodes(graph, props, id) : undefined;
    return [
      ...(design?.nodes ?? [
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-assign`,
          operation: 'GPUSpatialCrossValidation',
          variant: 'assign',
          invocationCount: rows,
          bindings: [
            {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
            {name: 'blocks', view: props.blocks, type: 'u32', access: 'read'},
            ...(props.mask
              ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
              : []),
            {name: 'folds', view: props.output.folds, type: 'u32', access: 'read_write'}
          ],
          declarations: `${IS_FINITE_BITS_WGSL}${HASH_UINT32_WGSL}`,
          body: `let position = vec2<f32>(positions[positionsOffset + index * 2u], positions[positionsOffset + index * 2u + 1u]);
var fold = 0xffffffffu;
if (isFiniteBits(position.x) && isFiniteBits(position.y) ${props.mask ? '&& mask[maskOffset + index] != 0u' : ''}) {
  let key = hashUint32(blocks[blocksOffset + index] ^ ${props.seed ?? 0}u);
  fold = key % ${props.foldCount}u;
}
folds[foldsOffset + index] = fold;`
        })
      ]),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-masks`,
        operation: 'GPUSpatialCrossValidation',
        variant: 'buffered-masks',
        invocationCount: rows * props.foldCount,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'folds', view: props.output.folds, type: 'u32', access: 'read'},
          ...(design
            ? [
                {
                  name: 'designState',
                  view: design.state,
                  type: 'f32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'training', view: props.output.training, type: 'u32', access: 'read_write'},
          {name: 'testing', view: props.output.testing, type: 'u32', access: 'read_write'}
        ],
        body: `let fold = index / ${rows}u;
let row = index % ${rows}u;
let assigned = folds[foldsOffset + row];
let test = assigned == fold;
var train = assigned < ${props.foldCount}u && !test;
${design ? 'train = train && designState[designStateOffset] == 0.0;' : ''}
if (train) {
  let position = vec2<f32>(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
  for (var other = 0u; other < ${rows}u; other++) {
    if (folds[foldsOffset + other] == fold) {
      let difference = position - vec2<f32>(positions[positionsOffset + other * 2u], positions[positionsOffset + other * 2u + 1u]);
      if (${design && props.design?.mode === 'nndm' ? 'length(difference) <= designState[designStateOffset + fold + 4u]' : `dot(difference, difference) <= ${getWGSLFloatLiteral((props.bufferDistance ?? 0) ** 2)}`}) { train = false; break; }
    }
  }
}
training[trainingOffset + index] = select(0u, 1u, train);
testing[testingOffset + index] = select(0u, 1u, test);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-counts`,
        operation: 'GPUSpatialCrossValidation',
        variant: 'counts',
        invocationCount: props.foldCount,
        bindings: [
          {name: 'folds', view: props.output.folds, type: 'u32', access: 'read'},
          {name: 'training', view: props.output.training, type: 'u32', access: 'read'},
          {name: 'testing', view: props.output.testing, type: 'u32', access: 'read'},
          {name: 'counts', view: props.output.counts, type: 'u32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        body: `var train = 0u; var test = 0u; var valid = 0u;
for (var row = 0u; row < ${rows}u; row++) {
 train += training[trainingOffset + index * ${rows}u + row]; test += testing[testingOffset + index * ${rows}u + row]; valid += select(0u, 1u, folds[foldsOffset + row] < ${props.foldCount}u);
}
counts[countsOffset + 3u * index] = train;
counts[countsOffset + 3u * index + 1u] = test;
counts[countsOffset + 3u * index + 2u] = valid - train - test;
status[statusOffset + index] = select(0u, 1u, train == 0u) | select(0u, 2u, test == 0u);`
      })
    ];
  }
}
