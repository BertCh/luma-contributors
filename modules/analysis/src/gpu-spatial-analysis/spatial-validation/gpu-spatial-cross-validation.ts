// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateSpatialCrossValidationBindings} from './validation-binding-plans';

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateSinglePackedView,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {validateRegressionOutputDispatches} from '../spatial-regression/regression-output-dispatches';
import {validateGraphOutputsDisjointFromInputs} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createSpatialValidationDesignNodes,
  type GPUSpatialValidationDesign,
  validateSpatialValidationDesign
} from './spatial-validation-design';
import {IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {PHILOX_PURPOSES, PHILOX_WGSL} from '../../analysis-core/kernels/philox-random';

/** Most rows: each buffer-mask invocation scans all rows, so work grows as folds times rows squared. */
const MAXIMUM_ROW_COUNT = 65536;
/** Most folds: masks are fold-major, rows times folds entries each. */
const MAXIMUM_FOLD_COUNT = 64;

export type GPUSpatialCrossValidationProps = {
  id?: string;
  /** Planar only; longitude-latitude is rejected. Default planar. */
  spatialContext?: GPUSpatialContext;
  /** Planar positions in the same units as bufferDistance. */
  positions: GraphDataView<'float32x2'>;
  /** Explicit block IDs. All members of a block receive the same fold. */
  blocks: GraphDataView<'uint32'>;
  mask?: GraphDataView<'uint32'>;
  /** Optional explicit fold design; omitted deals the blocks to folds in a seeded random order. */
  design?: GPUSpatialValidationDesign;
  foldCount: number;
  /**
   * Stable uint32 seed of the block order. The same seed, blocks and usable rows always give the
   * same folds; a different seed gives a different order. Default 0.
   */
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
 *
 * Block folds (no `design`). Every block that holds a usable row (finite position, nonzero
 * `mask`) draws one Philox word keyed by `(seed, block id)`. The blocks are ranked by that word,
 * ties by block id, and block of rank `r` goes to fold `r % foldCount`. So fold sizes differ by
 * at most one BLOCK (not one row: blocks of unequal size give folds of unequal size), no fold is
 * empty when there are at least `foldCount` such blocks, and a new seed is a new random order of
 * the blocks, not a relabelling of the old one. Block ids are arbitrary uint32 labels.
 *
 * The fold of a block is a function of the seed and of the SET of blocks that hold a usable row.
 * A mask that removes rows but leaves every block one usable row changes no fold. A mask that
 * removes whole blocks re-deals the remaining ones, so the same block can get another fold. That
 * is deliberate: ranking every block id present in the input whatever its mask would make the
 * fold depend on the seed and block ids only, but would leave folds empty whenever the masked
 * blocks happen to hold the ranks of one fold (8 blocks, 4 usable, 4 folds: an empty fold for 54
 * of the 70 equally likely cases). Nested validation therefore deals its inner folds anew for
 * each outer fold, among the blocks of that fold's training rows, as nested resampling does.
 *
 * Reference tier and limits. Block folds are two all-pairs passes over the rows, O(rows²). Fold designs
 * (`design`) are one serial invocation each, so they accept at most 256 rows (64 for NNDM, 256
 * prediction locations); buffered masks run one invocation per (fold, row) but scan every row,
 * so rows go up to 65,536 and folds to 64 with O(folds * rows²) work. Planar positions only.
 * Parallel designs (not implemented): block folds from a stable GPUSort by block key plus a scan
 * over block boundaries to deal whole blocks to folds; environmental and kNNDM clustering on
 * GPUSpatialKMeans (which has k-means++); buffer masks from a GPUNeighborSearch or grid-index radius
 * query of each test row, marking only the rows within bufferDistance, for O(rows * neighbors).
 * Fit preprocessing, feature selection and bandwidth/variogram selection separately using each
 * training mask. This producer never fits a model or silently reuses a whole-sample fit.
 */
export class GPUSpatialCrossValidation implements GPUCommandNodeProducer {
  readonly id: string;
  /** Resolved spatial context (planar). */
  readonly spatialContext: GPUSpatialContext;
  constructor(readonly props: GPUSpatialCrossValidationProps) {
    this.id = props.id ?? 'spatial-cross-validation';
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    const rows = props.positions.length;
    validateSpatialValidationDesign(this.id, props);
    validateCapacity(this.id, 'rows', rows, 1, MAXIMUM_ROW_COUNT);
    validateCapacity(this.id, 'foldCount', props.foldCount, 2, MAXIMUM_FOLD_COUNT);
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
        validateSinglePackedView(view, [format], this.id);
        if (view.length !== length) throw new Error(`${this.id}: shape mismatch`);
      }
    validateRegressionOutputDispatches(
      this.id,
      {...props.output, diagnostics: props.design?.diagnostics},
      [
        ['folds', 'diagnostics'],
        {read: ['folds', 'diagnostics'], write: ['training', 'testing']},
        {read: ['folds', 'training', 'testing'], write: ['counts', 'status']}
      ]
    );
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.positions,
      ...(!props.design || props.design.mode === 'groups' ? [props.blocks] : []),
      ...(props.mask ? [props.mask] : [])
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateSpatialCrossValidationBindings(graph, id, props);
    const rows = props.positions.length;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      !props.design || props.design.mode === 'groups' ? props.blocks : undefined,
      props.mask,
      ...Object.values(props.output)
    ]);
    const design = props.design ? createSpatialValidationDesignNodes(graph, props, id) : undefined;
    // Block folds. Pass one gives every usable row the random key of its block and marks the
    // first usable row of each block; pass two ranks a block among the marked rows and deals the
    // ranks round-robin. Both scan every row, as the buffered masks below already do.
    const blockOrder = design
      ? undefined
      : createTransientView(graph, `${id}-block-order`, 'uint32', 2 * rows);
    const usableRowWGSL = /* wgsl */ `${IS_FINITE_BITS_WGSL}
fn isUsableRow(row: u32) -> bool {
  return isFiniteBits(positions[positionsOffset + row * 2u]) &&
    isFiniteBits(positions[positionsOffset + row * 2u + 1u])${props.mask ? ' && mask[maskOffset + row] != 0u' : ''};
}`;
    const usableRowBindings = [
      {name: 'positions', view: props.positions, type: 'f32' as const, access: 'read' as const},
      ...(props.mask
        ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
        : [])
    ];
    return [
      ...(design?.nodes ?? [
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-block-order`,
          operation: 'GPUSpatialCrossValidation',
          variant: 'block-order',
          invocationCount: rows,
          bindings: [
            ...usableRowBindings,
            {name: 'blocks', view: props.blocks, type: 'u32', access: 'read'},
            {name: 'blockOrder', view: blockOrder!, type: 'u32', access: 'read_write'}
          ],
          declarations: `${usableRowWGSL}${PHILOX_WGSL}`,
          body: `var key = 0u;
var representative = 0u;
if (isUsableRow(index)) {
  let blockId = blocks[blocksOffset + index];
  let random = philox4x32(vec4<u32>(blockId, 0u, 0u, 0u), vec2<u32>(${props.seed ?? 0}u, ${PHILOX_PURPOSES.spatialCrossValidationBlockOrder.id}u));
  key = random.x;
  representative = 1u;
  for (var other = 0u; other < index; other++) {
    if (blocks[blocksOffset + other] == blockId && isUsableRow(other)) { representative = 0u; break; }
  }
}
blockOrder[blockOrderOffset + 2u * index] = key;
blockOrder[blockOrderOffset + 2u * index + 1u] = representative;`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-assign`,
          operation: 'GPUSpatialCrossValidation',
          variant: 'assign',
          invocationCount: rows,
          bindings: [
            ...usableRowBindings,
            {name: 'blocks', view: props.blocks, type: 'u32', access: 'read'},
            {name: 'blockOrder', view: blockOrder!, type: 'u32', access: 'read'},
            {name: 'folds', view: props.output.folds, type: 'u32', access: 'read_write'}
          ],
          declarations: usableRowWGSL,
          body: `var fold = 0xffffffffu;
if (isUsableRow(index)) {
  let blockId = blocks[blocksOffset + index];
  let key = blockOrder[blockOrderOffset + 2u * index];
  // Rank of this block among the blocks that hold a usable row, by (key, block id).
  var rank = 0u;
  for (var other = 0u; other < ${rows}u; other++) {
    if (blockOrder[blockOrderOffset + 2u * other + 1u] != 0u) {
      let otherKey = blockOrder[blockOrderOffset + 2u * other];
      if (otherKey < key || (otherKey == key && blocks[blocksOffset + other] < blockId)) { rank++; }
    }
  }
  fold = rank % ${props.foldCount}u;
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
