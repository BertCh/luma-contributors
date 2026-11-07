// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import type {GPUChunkedSpatialWeights} from './gpu-chunked-spatial-weights';
import type {GPUSpatialWeights} from './spatial-weights';
import {
  getSpatialWeightsTransposeNodes,
  planSpatialWeightsTranspose,
  type GPUSpatialWeightsTransposeOutput
} from './spatial-weights-transpose-pipeline';

const OPERATION = 'GPUChunkedSpatialWeightsTranspose';

/** Properties for {@link GPUChunkedSpatialWeightsTranspose}. */
export type GPUChunkedSpatialWeightsTransposeProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'chunked-spatial-weights-transpose'`. */
  id?: string;
  /** Chunked weights `W`. Neighbor IDs index columns in `[0, columnCount)`. */
  weights: GPUChunkedSpatialWeights;
  /** Number of columns of `W`. Defaults to the total row count (square weights). */
  columnCount?: number;
  /**
   * Caller-owned CSR of `W'` as one whole CSR (packing is the caller's explicit choice): `columnCount + 1`
   * offsets and at least as many slots as the summed capacity of the chunks.
   */
  output: GPUSpatialWeightsTransposeOutput;
  /** Absolute weight tolerance of the symmetry check. Defaults to 0 (exact). */
  symmetryTolerance?: number;
};

/**
 * {@link GPUSpatialWeightsTranspose} of chunked weights, written as one whole CSR.
 *
 * Same algorithm as {@link GPUSpatialWeightsTranspose} (one shared pipeline; whole weights are one chunk). Linear in the chunk count: per chunk one stable sort of its slots by column and one pass that
 * adds the chunk's column counts; one scan over the summed counts gives the output offsets; then per
 * chunk one destination pass (`offsets[column] + slots of earlier chunks in that column + rank in
 * this chunk`) and one scatter, and a cursor update. Chunks run in source order and the sort is
 * stable, so every output row lists source rows ascending exactly as the whole transpose does, and
 * the result does not depend on the chunking. Tail slots past the used count are written as zero.
 *
 * The output stays one binding: a transpose whose output does not fit one storage binding needs a
 * chunked output (not implemented; WEIGHTS7 backlog).
 */
export class GPUChunkedSpatialWeightsTranspose implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUChunkedSpatialWeightsTransposeProps;
  readonly rowCount: number;
  readonly columnCount: number;
  /** The `output` CSR with `information` and `status` attached. */
  readonly outputWeights: GPUSpatialWeights;

  constructor(props: GPUChunkedSpatialWeightsTransposeProps) {
    this.id = props.id ?? 'chunked-spatial-weights-transpose';
    this.props = props;
    const plan = planSpatialWeightsTranspose(this.id, OPERATION, props);
    this.rowCount = plan.rowCount;
    this.columnCount = plan.columnCount;
    this.outputWeights = plan.outputWeights;
  }

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
