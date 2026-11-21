// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateSpatialStatisticsBindings} from './spatial-statistics-bindings';
import {createSpatialWeightsSuppressionNodes} from './spatial-weights-suppression';
import {createWeightsLossNodes, getWeightsLossSources} from './spatial-weights-loss';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  createLoadBalancedSegmentSumNodes,
  getLoadBalancedTileCount
} from '../../analysis-core/kernels/load-balanced-for-each';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createFloat32ValuesNodes,
  type GPUFloat32OrUint32Values,
  validateFloat32OrUint32Values
} from './float32-values';
import type {GPUSpatialWeights} from './spatial-weights';
import {
  getGPUSpatialWeightsChunkDispatches,
  getGPUSpatialWeightsChunks,
  getGPUSpatialWeightsOrChunksViews,
  isGPUChunkedSpatialWeights,
  validateGPUSpatialWeightsOrChunks,
  type GPUChunkedSpatialWeights
} from './gpu-chunked-spatial-weights';

const OPERATION = 'GPUSpatialLag';
/** A load-balanced dispatch is encoded with uint32 invocation indices and 256 lanes. */
const SPATIAL_LAG_MAXIMUM_WORKGROUP_COUNT = Math.floor(0xffffffff / 256);
/** Merge-path diagonal and interleaved output addressing use uint32 indices. */
const SPATIAL_LAG_MAXIMUM_MERGE_STEPS = 0xffffffff;

/**
 * Properties for {@link GPUSpatialLag}.
 *
 * Compile-time: `normalize`, row count, view lengths and which optional views exist. Per-frame: the
 * contents of `values`, `mask` and `weights`.
 */
export type GPUSpatialLagProps = {
  /** Prefix for generated node IDs. Defaults to `'spatial-lag'`. */
  id?: string;
  /**
   * Per-source values `y`, row-major with `columnCount` columns, `float32` or `uint32` (counts,
   * exact below 2^24). Its row count (`values.length / columnCount`) is the neighbor-ID space: the
   * weights row count for square weights, or the source-value rows of cross weights (for example
   * the area-share weights of `GPUArealInterpolation`, whose CSR rows are targets and neighbors
   * are sources). Neighbors at or above that row count are skipped.
   * `mask` requires square weights.
   */
  values: GPUFloat32OrUint32Values;
  /**
   * Number of value columns lagged at once. Defaults to 1. Column `c` of row `i` is
   * `output[i * columnCount + c]`; `normalize` divides each row by the same weight sum.
   */
  columnCount?: number;
  /**
   * Weights `W` whose neighbor IDs index `values`. Chunked weights run one lag per row chunk,
   * each writing its rows of `output`, with the same per-row result as the whole matrix.
   */
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  /** Optional selection (`rows` entries): zero rows output 0 and zero neighbors are skipped. */
  mask?: GraphDataView<'uint32'>;
  /**
   * When true, divides by the sum of the weights of the included neighbors, which gives the lag
   * under row-standardized weights without writing a transformed copy. Rows whose included
   * weight sum is zero output 0. Defaults to false.
   */
  normalize?: boolean;
  /** Caller-owned lag output with `rows * columnCount` entries; NaN on either supplied neighborhood loss flag. */
  output: GraphDataView<'float32'>;
  /**
   * Optional caller-owned scalar uint32 loss causes. Each supplied word is rewritten on every
   * submission: `overflow` ORs final-edge loss, `candidateOverflow` ORs candidate loss, across
   * the outer port and all chunks. Either cause still suppresses the complete `output` to NaN.
   * These words do not certify uploaded CSR validity. Run weights validation with the declared
   * source-row bound and diagonal policy, then inspect its `invalidCount` on the host before
   * submitting this lag.
   * Construction rejects supplied `weights.status.complete` at outer or chunk scope, even with
   * failure-polarity flags: this lag's loss gate is not completeness-aware. Do not strip the
   * certificate to bypass that restriction; establish complete input upstream or on the host
   * and retain its diagnostics alongside the lag result.
   */
  status?: {
    overflow?: GraphDataView<'uint32'>;
    candidateOverflow?: GraphDataView<'uint32'>;
  };
};

/**
 * Spatial lag `lag_i = sum_j w_ij y_j` (PySAL `lag_spatial`) for a {@link GPUSpatialWeights}.
 *
 * Neighbors `j` with a zero mask, or outside the value rows, are skipped. With `normalize` the
 * result is `sum_j w_ij y_j / sum_j w_ij` over the included neighbors. Rows are summed with the
 * shared load-balanced CSR loop, so hub rows (one row with 1e5 neighbors) spread over many
 * workgroups. The association depends only on the offsets, so the result is bitwise reproducible
 * and matches a sequential CPU sum within f32 rounding. This is a single-purpose row loop.
 *
 * Uploaded CSR can be checked with `GPUSpatialWeightsValidation` (or its chunked form): declare
 * `columnCount` as the source-row count (square by default) and `diagonal` as `forbid` (default),
 * `allow` or `require-unit`. The last policy requires one exactly unit-weight diagonal per
 * target row. Read `output.status.invalidCount` on the host after validation and before use.
 * For two square rows, offsets `[0, 1, 1]`, neighbor `[2]` and weight `[1]` contain one invalid
 * row under the default square policy, not a genuine island.
 * Cross-weight neighbor IDs instead index `values.length / columnCount` source rows: a one-row
 * CSR with offsets `[0, 1]` and neighbor `[2]` validly selects source row 2 from three values.
 * Validate that cross CSR with `columnCount: 3`, checking its own target-row offsets and slot
 * capacity. An attached `invalidCount` is metadata, not an automatic GPU gate. Validation failures
 * and deliberately excluded coordinates/values are distinct from final-edge and candidate capacity loss; neither sets
 * these optional loss words merely because a row was rejected or excluded.
 */
export class GPUSpatialLag implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialLagProps;

  private readonly rowStart?: number;

  constructor(props: GPUSpatialLagProps);
  constructor(props: GPUSpatialLagProps, rowStart?: number) {
    this.rowStart = rowStart;
    const id = props.id ?? 'spatial-lag';
    this.id = id;
    this.props = props;
    const rows = validateGPUSpatialWeightsOrChunks(id, props.weights);
    const valueCount = validateFloat32OrUint32Values(props.values, `${id} values`);
    validateSinglePackedView(props.output, ['float32'], `${id} output`);
    const status = getSpatialLagStatusOutputs(props);
    for (const [name, view] of Object.entries(status)) {
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} status.${name}`);
        validateCapacity(id, `status.${name} length`, view.length, 1, 1);
      }
    }
    const columnCount = props.columnCount ?? 1;
    validateCapacity(id, 'columnCount', columnCount);
    for (const dispatch of getGPUSpatialWeightsChunkDispatches(props.weights)) {
      validateCapacity(
        id,
        'columnCount * merge steps (SPATIAL_LAG_MAXIMUM_MERGE_STEPS)',
        columnCount * (dispatch.rowCount + dispatch.weights.neighbors.length),
        1,
        SPATIAL_LAG_MAXIMUM_MERGE_STEPS
      );
      validateCapacity(
        id,
        'load-balanced workgroups (SPATIAL_LAG_MAXIMUM_WORKGROUP_COUNT)',
        columnCount *
          getLoadBalancedTileCount(dispatch.rowCount, dispatch.weights.neighbors.length),
        1,
        SPATIAL_LAG_MAXIMUM_WORKGROUP_COUNT
      );
    }
    if (valueCount % columnCount !== 0 || valueCount === 0) {
      throw new Error(`${id} values length must be a multiple of columnCount`);
    }
    if (props.output.length !== rows * columnCount) {
      throw new Error(`${id} output length must equal rows * columnCount`);
    }
    if (props.mask) {
      const maskRows = valueCount / columnCount;
      if (rowStart === undefined ? maskRows !== rows : maskRows < rows) {
        throw new Error(`${id} mask requires square weights`);
      }
      validateSinglePackedView(props.mask, ['uint32'], `${id} mask`);
      if (props.mask.length !== maskRows) {
        throw new Error(`${id} mask length must equal the weights row count`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output, ...Object.values(status)],
      [
        props.values,
        props.mask,
        ...getGPUSpatialWeightsOrChunksViews(props.weights),
        ...[
          props.weights,
          ...getGPUSpatialWeightsChunks(props.weights).map(chunk => chunk.weights)
        ].flatMap(part => Object.values(part.status ?? {}))
      ]
    );
    validateDisjointWritableViews(
      id,
      {output: props.output, ...status},
      {separateDispatches: true}
    );
  }

  /** Number of value rows, the neighbor-ID space of `weights`. */
  getValueRowCount(): number {
    return this.props.values.length / (this.props.columnCount ?? 1);
  }

  /** Returns the lag nodes: value conversion, load-balanced sum and fixup, optional normalize. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const status = getSpatialLagStatusOutputs(this.props);
    validateSpatialStatisticsBindings(
      graph,
      this.id,
      {...this.props, status},
      {
        'row scratch': this.props.output.length * 4,
        'load-balanced carry scratch': Math.max(
          ...getGPUSpatialWeightsChunkDispatches(this.props.weights).map(
            dispatch =>
              getLoadBalancedTileCount(dispatch.rowCount, dispatch.weights.neighbors.length) *
              (this.props.columnCount ?? 1) *
              2 *
              4
          )
        )
      }
    );
    const {id, props} = this;
    const {mask, output} = props;
    const columnCount = props.columnCount ?? 1;
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      mask,
      output,
      ...Object.values(status),
      ...getGPUSpatialWeightsOrChunksViews(props.weights)
    ]);
    // Publish only at the parent level; recursive row chunks never share writable status ports.
    const sources = getWeightsLossSources([props.weights]);
    const statusNodes = (['overflow', 'candidateOverflow'] as const).flatMap(name =>
      status[name]
        ? createWeightsLossNodes(graph, `${id}-status-${name}`, sources[name], status[name])
        : []
    );
    if (isGPUChunkedSpatialWeights(props.weights)) {
      // One lag per row chunk; the chunk's output is its slice of the caller's output.
      const nodes = props.weights.chunks.flatMap((chunk, chunkIndex) =>
        (
          Reflect.construct(GPUSpatialLag, [
            {
              ...props,
              id: `${id}-chunk${chunkIndex}`,
              weights: chunk.weights,
              status: undefined,
              output: graph.createDataView(output.buffer, {
                format: 'float32',
                length: chunk.rowCount * columnCount,
                byteOffset: output.byteOffset + chunk.rowStart * columnCount * 4
              })
            },
            chunk.rowStart
          ]) as GPUSpatialLag
        ).getCommandNodes(graph)
      );
      return [
        ...nodes,
        ...createSpatialWeightsSuppressionNodes(graph, id, OPERATION, props.weights, {output}),
        ...statusNodes
      ];
    }
    const weights = props.weights;
    const rowStart = this.rowStart;
    const rows = weights.offsets.length - 1;
    const valueRowCount = this.getValueRowCount();
    const values = createFloat32ValuesNodes(graph, `${id}-values`, props.values);
    const rowIncluded = mask
      ? `mask[maskOffset + ${rowStart === undefined ? '' : `${rowStart}u + `}row] != 0u`
      : 'true';
    const neighborIncluded = mask ? 'mask[maskOffset + neighbor] != 0u' : 'true';
    // Load-balanced CSR sum (algorithms roadmap decision 3): a hub row spreads over many
    // workgroups instead of one thread. One lane per value column.
    // Normalize with one weight sum per row; interleaved (lag, weight) scratch would double
    // the caller's binding size and exceed the device limit for otherwise valid outputs.
    const accumulatorType = 'f32';
    const nodes: GPUCommandNode<Parameters>[] = [
      ...values.nodes,
      ...createLoadBalancedSegmentSumNodes<Parameters>(graph, {
        id: `${id}-lag`,
        operation: OPERATION,
        variant: props.normalize ? 'normalized' : 'sum',
        offsets: weights.offsets,
        itemCapacity: weights.neighbors.length,
        laneCount: columnCount,
        accumulatorType,
        output,
        bindings: [
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
          {name: 'values', view: values.view, type: 'f32', access: 'read'},
          ...(mask
            ? [{name: 'mask', view: mask, type: 'u32' as const, access: 'read' as const}]
            : [])
        ],
        declarations: `const ROWS: u32 = ${valueRowCount}u;
const COLUMNS: u32 = ${columnCount}u;
fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> ${accumulatorType} {
  let neighbor = neighbors[neighborsOffset + item];
  if (!(${rowIncluded}) || neighbor >= ROWS || !(${neighborIncluded})) {
    return ${accumulatorType}();
  }
  let weight = weights[weightsOffset + item];
  let lagTerm = weight * values[valuesOffset + neighbor * COLUMNS + lane];
  return lagTerm;
}`
      })
    ];
    if (props.normalize) {
      const weightSums = createTransientView(graph, `${id}-weight-sums`, 'float32', rows);
      nodes.push(
        ...createLoadBalancedSegmentSumNodes<Parameters>(graph, {
          id: `${id}-weights`,
          operation: OPERATION,
          offsets: weights.offsets,
          itemCapacity: weights.neighbors.length,
          accumulatorType: 'f32',
          output: weightSums,
          bindings: [
            {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
            ...(mask
              ? [{name: 'mask', view: mask, type: 'u32' as const, access: 'read' as const}]
              : [])
          ],
          declarations: `fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> f32 {
  let neighbor = neighbors[neighborsOffset + item];
  if (!(${rowIncluded}) || neighbor >= ${valueRowCount}u || !(${neighborIncluded})) {
    return 0.0;
  }
  return weights[weightsOffset + item];
}`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-normalize`,
          operation: OPERATION,
          variant: 'normalize',
          bindings: [
            {name: 'weightSums', view: weightSums, type: 'f32', access: 'read'},
            {name: 'output', view: output, type: 'f32', access: 'read_write'}
          ],
          invocationCount: rows * columnCount,
          body: `let lag = output[outputOffset + index];
  let weightSum = weightSums[weightSumsOffset + index / ${columnCount}u];
  output[outputOffset + index] = select(0.0, lag / weightSum, weightSum > 0.0);`
        })
      );
    }
    return [
      ...nodes,
      ...createSpatialWeightsSuppressionNodes(graph, id, OPERATION, props.weights, {output}),
      ...statusNodes
    ];
  }
}

/** Select supported output ports so a structurally wider status object remains read-only. */
function getSpatialLagStatusOutputs(props: GPUSpatialLagProps) {
  return {overflow: props.status?.overflow, candidateOverflow: props.status?.candidateOverflow};
}
