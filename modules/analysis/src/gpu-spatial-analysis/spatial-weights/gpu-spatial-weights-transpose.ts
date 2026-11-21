// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateFiniteNumber,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUSpatialWeights} from './spatial-weights';
import {snapshotWeightsFields, snapshotWeightsPort} from './weights-construction-snapshot';
import {
  getSpatialWeightsTransposeNodes,
  planSpatialWeightsTranspose,
  type GPUSpatialWeightsTransposeOutput
} from './spatial-weights-transpose-pipeline';

import {
  getGPUSpatialWeightsTransposeParameterValues,
  GPU_SPATIAL_WEIGHTS_TRANSPOSE_PARAMETER_LENGTH
} from './spatial-weights-parameters';

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
  /** Per-frame absolute symmetry tolerance: numeric default or one-float parameter view. Defaults to 0. */
  symmetryTolerance?: number | GraphDataView<'float32'>;
};

/**
 * Per-frame: CSR contents and symmetryTolerance. Compile-time: view shapes, columnCount and
 * which optional outputs are present. Numeric tolerances use the same GPUParameterBuffer layout
 * as getGPUSpatialWeightsTransposeParameterValues.
 *
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
 * standardized); `'double'` also becomes `'unknown'` because the source column domain is
 * undeclared, so even square output may filter out-of-range columns and remove mass. The one
 * exception is square output of weights declared `symmetric: true`: their neighbors index
 * `[0, rows)`, every slot is kept and `'double'` is carried over.
 * Other states are unchanged. Input loss status (for example `overflow` of a
 * truncated `W`) is carried over, because a truncated `W` gives a truncated `W'`. Input `count`
 * and `requiredCount` are omitted because columns may be dropped; read the terminal output offset.
 * Destination status ports are refused, as are distances without source distances.
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
  private readonly toleranceValues: Float32Array | GraphDataView<'float32'>;
  /** The `output` CSR with `information` and `status` attached, for consumers. */
  readonly outputWeights: GPUSpatialWeights;

  constructor(props: GPUSpatialWeightsTransposeProps) {
    props = snapshotWeightsFields(props, [
      'id',
      'weights',
      'columnCount',
      'output',
      'symmetryTolerance'
    ]);
    props = Object.freeze({
      ...props,
      weights: snapshotWeightsPort(props.weights),
      output: snapshotWeightsPort(props.output)
    });
    this.id = props.id ?? 'spatial-weights-transpose';
    this.props = props;
    const tolerance = props.symmetryTolerance ?? 0;
    if (typeof tolerance === 'number') {
      validateFiniteNumber(this.id, 'symmetryTolerance', tolerance, 0);
      this.toleranceValues = getGPUSpatialWeightsTransposeParameterValues({
        symmetryTolerance: tolerance
      });
    } else {
      validateSinglePackedView(tolerance, ['float32'], `${this.id} symmetryTolerance`);
      validateCapacity(
        this.id,
        'symmetryTolerance.length',
        tolerance.length,
        GPU_SPATIAL_WEIGHTS_TRANSPOSE_PARAMETER_LENGTH
      );
      this.toleranceValues = tolerance;
    }
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
    return getSpatialWeightsTransposeNodes(
      graph,
      id,
      OPERATION,
      props,
      {
        rowCount: this.rowCount,
        columnCount: this.columnCount,
        outputWeights: this.outputWeights
      },
      this.toleranceValues
    );
  }
}
