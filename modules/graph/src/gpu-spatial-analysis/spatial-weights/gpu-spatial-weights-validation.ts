// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {IS_FINITE_FLOAT_WGSL} from '../../utils/wgsl-common';
import {describeGPUSpatialWeights, type GPUSpatialWeights} from './spatial-weights';
import {
  getGPUSpatialWeightsChunkDispatches,
  getGPUSpatialWeightsOrChunksViews,
  isGPUChunkedSpatialWeights,
  validateGPUSpatialWeightsOrChunks,
  type GPUChunkedSpatialWeights
} from './gpu-chunked-spatial-weights';

const OPERATION = 'GPUSpatialWeightsValidation';
const CHUNKED_OPERATION = 'GPUChunkedSpatialWeightsValidation';

/**
 * Bits of the per-row flags written by {@link GPUSpatialWeightsValidation}. A row's flags are the
 * bitwise OR of the violations found in it; 0 means the row satisfies every invariant.
 */
export const GPU_SPATIAL_WEIGHTS_VIOLATION = {
  /**
   * Bad offsets: `offsets[0] != 0` (reported on row 0), an offset that decreases (`offsets[i] >
   * offsets[i + 1]`, reported on row `i`), or `offsets[rows]` above the slot capacity (reported on
   * the last row).
   */
  offsets: 1,
  /** A neighbor ID `>= rows`. */
  neighborRange: 2,
  /** Neighbor IDs of the row are not strictly ascending (a duplicate or a descending pair). */
  order: 4,
  /** The row lists itself as a neighbor. */
  self: 8,
  /** A weight is NaN, infinite or negative. */
  weight: 16
} as const;

/** Properties for {@link GPUSpatialWeightsValidation}. */
export type GPUSpatialWeightsValidationProps<
  Weights extends GPUSpatialWeights | GPUChunkedSpatialWeights = GPUSpatialWeights
> = {
  /** Prefix for generated node and transient IDs. Defaults to `'spatial-weights-validation'`. */
  id?: string;
  /**
   * Square weights to check. Read only. Chunked weights run one rows dispatch per chunk (chunk `c`
   * checks its own offsets and capacity, and neighbor IDs against the total row count), with the
   * per-row flags and the invalid count in global row order; the result does not depend on the
   * chunking.
   */
  weights: Weights;
  /** Caller-owned outputs. */
  output: GPUSpatialWeightsValidationOutput;
};

/** Caller-owned outputs of {@link GPUSpatialWeightsValidation}. */
export type GPUSpatialWeightsValidationOutput = {
  /**
   * Optional uint32 per row (length equal to the row count) receiving the violation bits of
   * {@link GPU_SPATIAL_WEIGHTS_VIOLATION}.
   */
  flags?: GraphDataView<'uint32'>;
  status: {
    /**
     * One-row uint32 receiving the number of rows with at least one violation. Zero means the
     * weights satisfy every {@link GPUSpatialWeights} invariant.
     */
    invalidCount: GraphDataView<'uint32'>;
  };
};

/**
 * Checks the {@link GPUSpatialWeights} invariants on the GPU, for CSR that was built elsewhere
 * and uploaded (producers in this package already guarantee them).
 *
 * Uploaded CSR must satisfy the invariants, because consumers rely on them without checking:
 * `findWeight` binary-searches the neighbor IDs of a row, so unsorted or duplicate IDs give wrong
 * `S1`, `S2` and symmetry counts. Run this node once after upload (the CSR rarely changes) and
 * read `output.status.invalidCount` before using the weights. There is no GPU sort or dedupe: fix the CSR on
 * the CPU (sort each row by neighbor ID, merge duplicates by summing, drop self slots) and upload
 * it again.
 *
 * Checked per row `i`: `offsets[0] = 0`, offsets non-decreasing and `offsets[rows] <=
 * neighbors.length`; every neighbor ID `< rows`; neighbor IDs strictly ascending within the row
 * (sorted, no duplicates); no self slot; every weight finite and `>= 0`. Slots past
 * `offsets[rows]` are not inspected.
 *
 * {@link GPUSpatialWeightsValidation.outputWeights} is `weights` with `invalidCount` attached to
 * its `status`.
 */
export class GPUSpatialWeightsValidation<
  Weights extends GPUSpatialWeights | GPUChunkedSpatialWeights = GPUSpatialWeights
> implements GPUCommandNodeProducer
{
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialWeightsValidationProps<Weights>;
  /** The checked weights with `status.invalidCount` attached. */
  readonly outputWeights: Weights;

  constructor(props: GPUSpatialWeightsValidationProps<Weights>) {
    const id =
      props.id ??
      (isGPUChunkedSpatialWeights(props.weights)
        ? 'chunked-spatial-weights-validation'
        : 'spatial-weights-validation');
    this.id = id;
    this.props = props;
    const rows = validateGPUSpatialWeightsOrChunks(id, props.weights);
    validatePackedView(props.output.status.invalidCount, ['uint32'], `${id} invalidCount`);
    if (props.output.status.invalidCount.length < 1) {
      throw new Error(`${id} invalidCount must hold one uint32`);
    }
    if (props.output.flags) {
      validatePackedView(props.output.flags, ['uint32'], `${id} flags`);
      if (props.output.flags.length !== rows) {
        throw new Error(`${id} flags length must equal the weights row count`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.status.invalidCount, props.output.flags],
      getGPUSpatialWeightsOrChunksViews(props.weights)
    );
    const information = props.weights.information ?? {
      rowCount: rows,
      transform: 'unknown' as const,
      symmetric: 'unknown' as const
    };
    // The cast is the generic parameter: whole weights stay whole, chunked weights stay chunked.
    this.outputWeights = (
      isGPUChunkedSpatialWeights(props.weights)
        ? {
            ...props.weights,
            information,
            status: {...props.weights.status, invalidCount: props.output.status.invalidCount}
          }
        : describeGPUSpatialWeights(props.weights, {
            information,
            status: {invalidCount: props.output.status.invalidCount}
          })
    ) as Weights;
  }

  /** Returns the validation nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {weights} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getGPUSpatialWeightsOrChunksViews(weights),
      props.output.status.invalidCount,
      props.output.flags
    ]);
    const dispatches = getGPUSpatialWeightsChunkDispatches(weights);
    const rows = dispatches.reduce((total, dispatch) => total + dispatch.rowCount, 0);
    const violated = createTransientView(graph, `${id}-violated`, 'uint32', rows);
    const flags = props.output.flags ?? createTransientView(graph, `${id}-flags`, 'uint32', rows);
    const bits = GPU_SPATIAL_WEIGHTS_VIOLATION;
    const operation = isGPUChunkedSpatialWeights(weights) ? CHUNKED_OPERATION : OPERATION;
    const nodes: GPUCommandNode<Parameters>[] = dispatches.map(dispatch =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-rows${dispatch.isWhole ? '' : dispatch.idSuffix}`,
        operation,
        variant: 'rows',
        bindings: [
          {name: 'offsets', view: dispatch.weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: dispatch.weights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: dispatch.weights.weights, type: 'f32', access: 'read'},
          {name: 'flags', view: flags, type: 'u32', access: 'read_write'},
          {name: 'violated', view: violated, type: 'u32', access: 'read_write'}
        ],
        invocationCount: dispatch.rowCount,
        declarations: `${IS_FINITE_FLOAT_WGSL}
const ROWS: u32 = ${rows}u;
const CHUNK_ROWS: u32 = ${dispatch.rowCount}u;
const CAPACITY: u32 = ${dispatch.weights.neighbors.length}u;
${dispatch.constantsWGSL}`,
        // `index` is the row inside the dispatch's chunk; `row` is the global row.
        body: `let row = index${dispatch.isWhole ? '' : ' + WEIGHTS_ROW_START'};
  let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  var rowFlags = 0u;
  if ((index == 0u && begin != 0u) || begin > end || (index + 1u == CHUNK_ROWS && end > CAPACITY)) {
    rowFlags |= ${bits.offsets}u;
  }
  var previous = 0u;
  for (var slot = begin; slot < min(end, CAPACITY); slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    let weight = weights[weightsOffset + slot];
    if (neighbor >= ROWS) {
      rowFlags |= ${bits.neighborRange}u;
    }
    if (neighbor == row) {
      rowFlags |= ${bits.self}u;
    }
    if (slot > begin && neighbor <= previous) {
      rowFlags |= ${bits.order}u;
    }
    if (!isFiniteFloat(weight) || weight < 0.0) {
      rowFlags |= ${bits.weight}u;
    }
    previous = neighbor;
  }
  flags[flagsOffset + row] = rowFlags;
  violated[violatedOffset + row] = select(0u, 1u, rowFlags != 0u);`
      })
    );
    nodes.push(
      ...new GPUReduction({
        id: `${id}-count`,
        input: violated,
        output: props.output.status.invalidCount,
        operation: 'sum'
      }).getCommandNodes(graph)
    );
    return nodes;
  }
}
