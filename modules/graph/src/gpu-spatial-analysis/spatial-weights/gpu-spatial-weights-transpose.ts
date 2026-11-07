// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer
} from '@luma-contributors/graph/gpu-core';
import type {GPUSpatialWeights} from './spatial-weights';
import {
  getSpatialWeightsTransposeNodes,
  planSpatialWeightsTranspose,
  type GPUSpatialWeightsTransposeOutput
} from './spatial-weights-transpose-pipeline';

export type {GPUSpatialWeightsTransposeOutput} from './spatial-weights-transpose-pipeline';

const OPERATION = 'GPUSpatialWeightsTranspose';

/** Properties for {@link GPUSpatialWeightsTranspose}. */
export type GPUSpatialWeightsTransposeProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'spatial-weights-transpose'`. */
  id?: string;
  /**
   * Weights `W` to transpose. Rows are the `offsets.length - 1` focus rows; neighbor IDs are
   * column indices in `[0, columnCount)`. Slots with a neighbor ID `>= columnCount` are dropped.
   */
  weights: GPUSpatialWeights;
  /**
   * Number of columns of `W` (rows of the transpose). Defaults to the row count of `weights`
   * (square weights). Set it for rectangular cross weights, whose neighbor IDs index the target
   * rows. Rows of the transpose that no column of `W` reaches are empty.
   */
  columnCount?: number;
  /**
   * Caller-owned CSR of `W'`: `offsets` holds `columnCount + 1` values, and `neighbors`,
   * `weights` (and `distances` when `weights.distances` is set) hold at least as many slots as the
   * input. Row `j` lists the source rows `i` with `w_ij` present, ascending.
   */
  output: GPUSpatialWeightsTransposeOutput;
  /** Absolute weight tolerance of the symmetry check. Defaults to 0 (exact). */
  symmetryTolerance?: number;
};

/**
 * CSR transpose of a {@link GPUSpatialWeights} on the GPU (for square or rectangular weights),
 * the sparse `W.T` that libpysal `Graph` gets from swapping focal and neighbor.
 *
 * The result is deterministic and independent of any atomics ordering: slots are keyed by their
 * column and stable-sorted (so within a column they stay in source row order, which makes every
 * output row ascending), per-column counts of the sorted keys give the output offsets, and a
 * scatter writes the transposed slots. This is the same algorithm as
 * {@link GPUChunkedSpatialWeightsTranspose} with a single chunk (see
 * `spatial-weights-transpose-pipeline.ts`). Capacity is the input capacity (`nnz` cannot grow);
 * slots of the output past `offsets[columnCount]` are written as zero. `distances` are carried
 * along when present.
 *
 * Optionally reports whether `W` is symmetric (`output.asymmetricSlots`).
 *
 * {@link GPUSpatialWeightsTranspose.outputWeights} is `output` with `information` and `status` attached.
 * The transpose of `W` is symmetric exactly when `W` is, so `symmetric` is carried over. A
 * `'row'` or `'variance'` transform state becomes `'unknown'` (the transpose is column, not row,
 * standardized); other states are unchanged. The input `status` (for example `overflow` of a
 * truncated `W`) is carried over, because a truncated `W` gives a truncated `W'`.
 */
export class GPUSpatialWeightsTranspose implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialWeightsTransposeProps;
  /** Rows of `W` (columns of the transpose). */
  readonly rowCount: number;
  /** Columns of `W` (rows of the transpose). */
  readonly columnCount: number;
  /** The `output` CSR with `information` and `status` attached, for consumers. */
  readonly outputWeights: GPUSpatialWeights;

  constructor(props: GPUSpatialWeightsTransposeProps) {
    this.id = props.id ?? 'spatial-weights-transpose';
    this.props = props;
    const plan = planSpatialWeightsTranspose(this.id, OPERATION, props);
    this.rowCount = plan.rowCount;
    this.columnCount = plan.columnCount;
    this.outputWeights = plan.outputWeights;
  }

  /** Returns the transpose nodes (and the symmetry check when requested) in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    return getSpatialWeightsTransposeNodes(graph, id, OPERATION, props, {
      rowCount: this.rowCount,
      columnCount: this.columnCount,
      outputWeights: this.outputWeights
    });
  }
}
