// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateSpatialOutputViews} from '../spatial-clustering/spatial-output-validation';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {getGraphOutputViews} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUReduction,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createLoadBalancedSegmentSumNodes} from '../../analysis-core/kernels/load-balanced-for-each';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateSinglePackedView,
  validateGPUStorageBindingByteLength,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';
import {
  getGPUSpatialWeightsChunkDispatches,
  getGPUSpatialWeightsOrChunksViews,
  isGPUChunkedSpatialWeights,
  validateGPUSpatialWeightsOrChunks,
  type GPUChunkedSpatialWeights
} from '../spatial-weights/gpu-chunked-spatial-weights';
import {GET_NAN_WGSL, IS_FINITE_FLOAT_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  createRegionalizationLossSuppressionNodes,
  validateRegionalizationWeightsCompleteness
} from './adjacency-loss-suppression';
import {snapshotWeightsPort} from '../spatial-weights/weights-construction-snapshot';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {validateGPUStatusPort} from '../../analysis-core/contracts/index';
import {ROUNDED_DIFFERENCE_WGSL} from '../spatial-regression/rounded-difference-wgsl';
import {REGIONAL_MOMENT_WGSL} from './regional-moment-wgsl';
import {createRegionalMomentSumNodes} from './regional-moment-sums';

const OPERATION = 'GPURegionPartitionEvaluation';
/** Rows or labels reduced by one thread of a summary tile pass. */
const SUMMARY_TILE = 256;

/** Largest supported attribute column count. */
export const GPU_REGION_PARTITION_EVALUATION_MAXIMUM_COLUMNS = 32;

/**
 * Largest `labels` row count of {@link GPURegionPartitionEvaluation} on `device`: the
 * contiguity-link pass writes two uint32 link counts (8 bytes) per row in one storage binding, so
 * 16777216 on a default device (128 MiB bindings).
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPURegionPartitionEvaluationMaximumRowCount(device: Device): number {
  return getStorageBindingElementLimit(device, 8);
}

/**
 * Largest `labelCapacity` of {@link GPURegionPartitionEvaluation} on `device`: the label offsets
 * hold `labelCapacity + 1` uint32 values in one storage binding, so 33554431 on a default device
 * (128 MiB bindings). The per-region column sums also bound `labelCapacity * columnCount` by
 * {@link getGPURegionPartitionEvaluationMaximumRegionValueCount}.
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPURegionPartitionEvaluationMaximumLabelCapacity(device: Device): number {
  return getStorageBindingElementLimit(device, 4, 1);
}

/**
 * Largest `labelCapacity * columnCount` of {@link GPURegionPartitionEvaluation} on `device`: one
 * float32 region sum per label and column in one storage binding, so 33554432 on a default device
 * (128 MiB bindings).
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPURegionPartitionEvaluationMaximumRegionValueCount(device: Device): number {
  return getStorageBindingElementLimit(device, 4);
}

/** Layout of the `output.summary` words written by {@link GPURegionPartitionEvaluation}. */
export const GPU_REGION_PARTITION_EVALUATION_LAYOUT = {
  /** Number of non-empty regions. */
  regionCount: 0,
  /** Sum over regions of the within-region sum of squared deviations from the region mean. */
  withinSumOfSquares: 1,
  /**
   * The sum of squares explained by the partition, `sum over regions of n_r |mean_r - mean|^2`
   * with `mean` the mean of the used rows. It equals `totalSumOfSquares - withinSumOfSquares` up
   * to float32 rounding and is computed directly, so it keeps its digits when the partition
   * explains a small share of the total. Its accuracy is that of `mean_r - mean`: center the
   * table when the overall mean is large against the differences between region means.
   */
  betweenSumOfSquares: 2,
  /** Sum of squared deviations from the global mean over the labelled rows. */
  totalSumOfSquares: 3,
  /** Smallest non-empty region size in rows. */
  minimumSize: 4,
  /** Largest region size in rows. */
  maximumSize: 5,
  /**
   * Fraction of weights links (between two labelled rows) whose endpoints lie in different
   * regions: 0 for a partition with no boundary, higher for ragged regions. 0 without `weights`.
   */
  crossLinkFraction: 6,
  /**
   * Rows left out of every statistic: those whose label is `0xffffffff` or at least
   * `labelCapacity`, and labelled rows with a NaN or infinite value in any column (the latter are
   * also counted by the optional `output.invalidRowCount`).
   */
  ignoredRows: 7,
  /**
   * Pseudo-F (Calinski-Harabasz) statistic
   * `(betweenSumOfSquares / (k - 1)) / (withinSumOfSquares / (n - k))` with `k` the region count
   * and `n` the labelled rows, the `sklearn.metrics.calinski_harabasz_score` convention (1 when
   * `withinSumOfSquares` is 0). NaN when `k < 2` or `n <= k`, where it is undefined.
   * Larger is a better separated partition; it compares partitions of the same table.
   */
  pseudoF: 8,
  /** Total word count. */
  length: 9
} as const;

/**
 * Properties for {@link GPURegionPartitionEvaluation}.
 *
 * Compile-time: row count, `columnCount`, `labelCapacity` and which outputs exist. Per-frame: the
 * contents of `values`, `labels`, `labelsComplete` and `weights`.
 */
export type GPURegionPartitionEvaluationProps = {
  /** Prefix for generated node IDs. Defaults to `'region-partition-evaluation'`. */
  id?: string;
  /**
   * Attribute table, row-major with `columnCount` columns (`rows * columnCount` entries). A row
   * with a NaN or infinite value in any column is left out of every statistic, as an unlabelled
   * row is, and counted in `summary.ignoredRows` and `output.invalidRowCount`.
   */
  values: GraphDataView<'float32'>;
  /** Number of attribute columns. Defaults to 1; at most 32. */
  columnCount?: number;
  /**
   * Region label per row. Any label column works: `GPUSkaterRegions`, a clustering, a
   * classification or an administrative code. Labels at or above `labelCapacity` and
   * `0xffffffff` (unlabelled) are ignored.
   */
  labels: GraphDataView<'uint32'>;
  /**
   * Exclusive bound on label values `[0, labelCapacity)`, not the number of regions. Use `compactRegionLabels` when passing a region count. Defaults to the row count. At most
   * {@link getGPURegionPartitionEvaluationMaximumLabelCapacity}, and `labelCapacity * columnCount`
   * at most {@link getGPURegionPartitionEvaluationMaximumRegionValueCount}.
   */
  labelCapacity?: number;
  /** Optional per-frame uint32 scalar: nonzero when solver labels are complete. Requires output.labelsComplete when supplied. Omission means ordinary complete labels. Scores remain available for provisional labels. */
  labelsComplete?: GraphDataView<'uint32'>;
  /** Optional weights for `crossLinkFraction` (a compactness proxy on the contiguity graph). */
  weights?: GPUSpatialWeights | GPUChunkedSpatialWeights;
  /** Caller-owned outputs. */
  output: GPURegionPartitionEvaluationOutput;
};

/** Caller-owned outputs of {@link GPURegionPartitionEvaluation}. */
export type GPURegionPartitionEvaluationOutput = {
  /**
   * Summary words, see {@link GPU_REGION_PARTITION_EVALUATION_LAYOUT} (`length` float32
   * entries).
   */
  summary: GraphDataView<'float32'>;
  /** Optional row count per label (`labelCapacity` entries). */
  regionSizes?: GraphDataView<'uint32'>;
  /** Optional within-region sum of squared deviations per label (`labelCapacity` entries). */
  regionWithinSumOfSquares?: GraphDataView<'float32'>;
  /**
   * Optional mean per label and column (`labelCapacity * columnCount` entries, label-major), NaN
   * for an empty label. The geographical detector reads these for its non-centrality.
   */
  regionMeans?: GraphDataView<'float32'>;
  /**
   * Optional one-row count of rows that carry a label in `[0, labelCapacity)` but were left out
   * because one of their values is NaN or infinite. They are part of `summary.ignoredRows`.
   */
  /** Optional scalar provenance, independent of adjacency loss; defaults to 1 without an input word. */
  labelsComplete?: GraphDataView<'uint32'>;
  invalidRowCount?: GraphDataView<'uint32'>;
};

/**
 * Evaluates a region partition on any label column: within-region and between-region sum of
 * squares, the pseudo-F (Calinski-Harabasz) statistic, region sizes and a contiguity-graph
 * boundary measure. `GPUMaxPRegions`, `GPUAZPRegions` and `GPUWardRegions` report the same
 * within-region sum of squares as their `objective`, so pass the table they used (the standardized
 * one unless they ran with `standardize: false`) to compare.
 *
 * Rows are grouped by label with a stable radix sort (ascending row inside a label), so every
 * label is a segment of the sorted rows. Its split column sums use a staged parallel segment
 * scan; squared deviations use load-balanced segment sums (merge-path partitioning over label
 * segments). The contiguity link counts are load-balanced over the weights rows, so a huge region or a hub row
 * spreads over many workgroups instead of serialising one thread. The summary is a tiled
 * reduction of 256 rows or labels per thread, followed by published parallel scalar reductions.
 * Results match a sequential CPU evaluation within f32 rounding; reduction association can depend
 * on the device's published reduction strategy. Split regional sums use `O(rows * log(rows))`
 * parallel work per column, independent of the region size distribution; summary work is
 * `O(rows + labels + links)`. Each regional sum stage combines at most two split moments.
 * Regional and global coordinates use member origins for same-sign columns, preserving
 * constants exactly; mixed-sign columns use zero. Subtraction and summation residuals survive
 * cancellation and the final regional/global mean contrast. Pseudo-F uses
 * normal mantissas and exponent differences, avoiding overflowing products or underflowing mean
 * squares when the SSDs and final ratio are representable.
 *
 * **Missing values.** A labelled row with a NaN or infinite value in any column is treated as
 * unlabelled: it enters no sum, size, link count or mean, it is counted in `ignoredRows`, and
 * `output.invalidRowCount` says how many such rows there were. One missing value therefore
 * removes one row instead of turning every statistic into NaN.
 * Nonzero weights.status overflow or candidateOverflow (including chunk-local flags) makes
 * crossLinkFraction unavailable (NaN). All statistics depending only on labels/values remain usable.
 * Optional labelsComplete provenance reports whether the supplied labels are final; provisional
 * scores remain available, independently of adjacency loss.
 * Accepts 1–32 attribute columns. At a 128 MiB storage binding limit, the enforced bounds
 * are 16,777,216 label rows, a label capacity of 33,554,431, and 33,554,432 entries in
 * `labelCapacity * columnCount`.
 */
export class GPURegionPartitionEvaluation implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURegionPartitionEvaluationProps;

  constructor(props: GPURegionPartitionEvaluationProps) {
    const captured = snapshotConstructionFields(props, [
      'id',
      'values',
      'columnCount',
      'labels',
      'labelCapacity',
      'labelsComplete',
      'weights',
      'output'
    ]);
    props = Object.freeze({
      ...captured,
      weights: captured.weights ? snapshotWeightsPort(captured.weights) : undefined,
      output: snapshotConstructionFields(captured.output, [
        'summary',
        'regionSizes',
        'regionWithinSumOfSquares',
        'regionMeans',
        'labelsComplete',
        'invalidRowCount'
      ])
    });
    const id = props.id ?? 'region-partition-evaluation';
    this.id = id;
    this.props = props;
    const columnCount = props.columnCount ?? 1;
    const {output} = props;
    validateCapacity(
      id,
      'columnCount',
      columnCount,
      1,
      GPU_REGION_PARTITION_EVALUATION_MAXIMUM_COLUMNS
    );
    validateSinglePackedView(props.values, ['float32'], `${id} values`);
    validateSinglePackedView(props.labels, ['uint32'], `${id} labels`);
    validateRequirement(
      id,
      props.labelsComplete === undefined || output.labelsComplete !== undefined,
      'output.labelsComplete is required when labelsComplete is supplied',
      {
        code: 'MISSING_PROP',
        path: 'output.labelsComplete',
        received: output.labelsComplete,
        expected: 'a uint32 provenance output'
      }
    );
    for (const [name, view] of [
      ['labelsComplete', props.labelsComplete],
      ['output.labelsComplete', output.labelsComplete]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
        validateCapacity(id, `${name}.length`, view.length, 1);
      }
    }
    const rows = props.labels.length;
    if (rows < 1 || props.values.length !== rows * columnCount) {
      validateRequirement(
        id,
        false,
        `values length must equal labels.length * columnCount; received props.values.length=${props.values.length}`
      );
    }
    const labelCapacity = props.labelCapacity ?? rows;
    validateCapacity(id, 'labelCapacity', labelCapacity);
    if (props.weights && isGPUChunkedSpatialWeights(props.weights) && props.weights.status)
      validateGPUStatusPort(`${id} weights.status`, props.weights.status);
    if (props.weights) {
      validateRegionalizationWeightsCompleteness(id, props.weights);
      const statusPorts = [
        {name: 'weights.status', status: props.weights.status},
        ...(isGPUChunkedSpatialWeights(props.weights)
          ? props.weights.chunks.map((chunk, index) => ({
              name: `weights.chunks[${index}].weights.status`,
              status: chunk.weights.status
            }))
          : [])
      ];
      for (const port of statusPorts)
        for (const [name, scalar] of Object.entries(port.status ?? {}))
          if (scalar) validateSinglePackedView(scalar, ['uint32'], `${id} ${port.name}.${name}`);
    }
    if (props.weights && validateGPUSpatialWeightsOrChunks(id, props.weights) !== rows) {
      validateRequirement(
        id,
        false,
        `weights row count must equal labels.length; received rows=${rows}`
      );
    }
    validateSinglePackedView(output.summary, ['float32'], `${id} output.summary`);
    if (output.summary.length < GPU_REGION_PARTITION_EVALUATION_LAYOUT.length) {
      validateRequirement(
        id,
        false,
        `output.summary must hold ${GPU_REGION_PARTITION_EVALUATION_LAYOUT.length} words; received output.summary.length=${output.summary.length}, GPU_REGION_PARTITION_EVALUATION_LAYOUT.length=${GPU_REGION_PARTITION_EVALUATION_LAYOUT.length}`
      );
    }
    if (output.regionSizes) {
      validateSinglePackedView(output.regionSizes, ['uint32'], `${id} output.regionSizes`);
      if (output.regionSizes.length < labelCapacity) {
        validateRequirement(
          id,
          false,
          `output.regionSizes must hold labelCapacity entries; received output.regionSizes.length=${output.regionSizes.length}`
        );
      }
    }
    if (output.regionWithinSumOfSquares) {
      validateSinglePackedView(
        output.regionWithinSumOfSquares,
        ['float32'],
        `${id} output.regionWithinSumOfSquares`
      );
      if (output.regionWithinSumOfSquares.length < labelCapacity) {
        validateRequirement(
          id,
          false,
          `output.regionWithinSumOfSquares must hold labelCapacity entries; received output.regionWithinSumOfSquares.length=${output.regionWithinSumOfSquares.length}`
        );
      }
    }
    if (output.regionMeans) {
      validateSinglePackedView(output.regionMeans, ['float32'], `${id} output.regionMeans`);
      if (output.regionMeans.length < labelCapacity * columnCount) {
        validateRequirement(
          id,
          false,
          `output.regionMeans must hold labelCapacity * columnCount entries; received output.regionMeans.length=${output.regionMeans.length}`
        );
      }
    }
    if (output.invalidRowCount) {
      validateSinglePackedView(output.invalidRowCount, ['uint32'], `${id} output.invalidRowCount`);
      validateRequirement(
        id,
        output.invalidRowCount.length >= 1,
        'output.invalidRowCount must hold one row'
      );
    }
    validateSpatialOutputViews(
      this.id,
      Object.fromEntries(
        getGraphOutputViews(props.output).map((view, index) => [`output[${index}]`, view])
      ),
      [
        [output.regionSizes, output.regionMeans],
        [output.summary, output.invalidRowCount]
      ]
    );
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.summary,
        output.regionSizes,
        output.regionWithinSumOfSquares,
        output.regionMeans,
        output.invalidRowCount,
        output.labelsComplete
      ],
      [
        props.values,
        props.labels,
        props.labelsComplete,
        ...(props.weights ? getGPUSpatialWeightsOrChunksViews(props.weights) : [])
      ]
    );
  }

  /** Returns the per-label, per-link and summary nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {values, labels, weights, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      values,
      labels,
      output.summary,
      output.regionSizes,
      output.regionWithinSumOfSquares,
      output.regionMeans,
      output.invalidRowCount,
      output.labelsComplete,
      props.labelsComplete,
      ...(weights ? getGPUSpatialWeightsOrChunksViews(weights) : [])
    ]);
    validateGraphChunkBindingRanges(graph, id, {
      values,
      labels,
      labelsComplete: props.labelsComplete,
      ...Object.fromEntries(
        getGraphOutputViews(output).map((view, index) => [`output[${index}]`, view])
      ),
      ...Object.fromEntries(
        (weights ? getGPUSpatialWeightsOrChunksViews(weights) : []).map((view, index) => [
          `weights[${index}]`,
          view
        ])
      )
    });
    const rows = labels.length;
    const columnCount = props.columnCount ?? 1;
    const labelCapacity = props.labelCapacity ?? rows;
    const reductionStride = Math.max(
      1,
      Math.ceil(graph.device.limits.minStorageBufferOffsetAlignment / 4)
    );
    validateMaximumCount(
      id,
      'labels rows',
      rows,
      getGPURegionPartitionEvaluationMaximumRowCount(graph.device)
    );
    validateCapacity(
      id,
      'labelCapacity',
      labelCapacity,
      1,
      getGPURegionPartitionEvaluationMaximumLabelCapacity(graph.device)
    );
    validateMaximumCount(
      id,
      'labelCapacity x columnCount',
      labelCapacity * columnCount,
      getGPURegionPartitionEvaluationMaximumRegionValueCount(graph.device)
    );
    for (const [name, byteLength] of [
      ['region table scratch', labelCapacity * columnCount * 4],
      ['region origin modes', labelCapacity * columnCount * 4],
      ['region centroid scratch', labelCapacity * columnCount * 4],
      ['global coordinate scratch', columnCount * 4],
      ['label offsets', (labelCapacity + 1) * 4],
      ['row scratch', rows * 4],
      ['link counts', weights ? rows * 8 : 0],
      ['row tile counts', Math.ceil(rows / SUMMARY_TILE) * 20],
      ['row tile sums', Math.ceil(rows / SUMMARY_TILE) * columnCount * 8],
      ['label tile counts', Math.ceil(labelCapacity / SUMMARY_TILE) * 12],
      ['summary integer result slots', 7 * reductionStride * 4],
      ['summary square result slots', 3 * reductionStride * 4]
    ] as const)
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        byteLength,
        'reduce rows, labelCapacity or columnCount'
      );
    const sizes =
      output.regionSizes ?? createTransientView(graph, `${id}-sizes`, 'uint32', labelCapacity);
    const within =
      output.regionWithinSumOfSquares ??
      createTransientView(graph, `${id}-within`, 'float32', labelCapacity);
    // Interleaved (total, cross) link counts per row.
    const links = weights
      ? createTransientView(graph, `${id}-link-counts`, 'uint32', 2 * rows)
      : undefined;
    const labelOffsets = createTransientView(
      graph,
      `${id}-label-offsets`,
      'uint32',
      labelCapacity + 1
    );
    const regionSums = createTransientView(
      graph,
      `${id}-region-column-sums`,
      'float32',
      labelCapacity * columnCount
    );
    const regionResiduals = createTransientView(
      graph,
      `${id}-region-residuals`,
      'float32',
      labelCapacity * columnCount
    );
    const centroidResiduals = createTransientView(
      graph,
      `${id}-centroid-residuals`,
      'float32',
      labelCapacity * columnCount
    );
    const globalOrigins = createTransientView(
      graph,
      `${id}-global-origins`,
      'float32',
      columnCount
    );
    const regionOrigins = createTransientView(
      graph,
      `${id}-region-origins`,
      'float32',
      labelCapacity * columnCount
    );
    const regionOriginModes = createTransientView(
      graph,
      `${id}-region-origin-modes`,
      'uint32',
      labelCapacity * columnCount
    );
    const globalOriginModes = createTransientView(
      graph,
      `${id}-global-origin-modes`,
      'uint32',
      columnCount
    );
    const regionCentroids = createTransientView(
      graph,
      `${id}-region-centroids`,
      'float32',
      labelCapacity * columnCount
    );
    const regionDeviations = createTransientView(
      graph,
      `${id}-region-column-deviations`,
      'float32',
      labelCapacity * columnCount
    );
    const declarations = `const ROWS: u32 = ${rows}u;
const COLUMNS: u32 = ${columnCount}u;
const LABELS: u32 = ${labelCapacity}u;`;
    // Rows grouped by label with a stable sort (ascending row inside a label), so each label's
    // thread reads only its own rows, in the same order as a scan of all rows would, instead of
    // every label scanning every row (`labelCapacity * rows`, quadratic with the default capacity).
    const sortKeys = createTransientView(graph, `${id}-sort-keys`, 'uint32', rows);
    const rowIds = createTransientView(graph, `${id}-row-ids`, 'uint32', rows);
    const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', rows);
    const sortedRows = createTransientView(graph, `${id}-sorted-rows`, 'uint32', rows);
    // The label every later kernel reads: the row's label when the row is used, 0xffffffff when
    // it is unlabelled, out of range, or has a non-finite value in some column. A NaN that reached
    // a sum would turn every statistic into NaN.
    const usedLabels = createTransientView(graph, `${id}-used-labels`, 'uint32', rows);
    // Key `labelCapacity` collects the ignored rows (out of range, unlabelled or non-finite).
    const keyBits = Math.max(1, Math.ceil(Math.log2(labelCapacity + 1)));
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-sort-keys`,
        operation: OPERATION,
        variant: 'sort-keys',
        bindings: [
          {name: 'labels', view: labels, type: 'u32', access: 'read'},
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'sortKeys', view: sortKeys, type: 'u32', access: 'read_write'},
          {name: 'usedLabels', view: usedLabels, type: 'u32', access: 'read_write'},
          {name: 'rowIds', view: rowIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `${declarations}
${IS_FINITE_FLOAT_WGSL}`,
        body: `let label = labels[labelsOffset + index];
  var isUsed = label < LABELS;
  for (var column = 0u; column < COLUMNS; column++) {
    isUsed = isUsed && isFiniteFloat(values[valuesOffset + index * COLUMNS + column]);
  }
  sortKeys[sortKeysOffset + index] = select(LABELS, label, isUsed);
  usedLabels[usedLabelsOffset + index] = select(0xffffffffu, label, isUsed);
  rowIds[rowIdsOffset + index] = index;`
      }),
      ...new GPUSort({
        id: `${id}-sort`,
        keys: sortKeys,
        values: rowIds,
        outputKeys: sortedKeys,
        outputValues: sortedRows,
        keyBits
      }).getCommandNodes(graph),
      // Label segments of the sorted rows: offsets[label] is the first position with key >= label,
      // offsets[LABELS] the first ignored row, so the labels form CSR rows over the sorted rows.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-label-offsets`,
        operation: OPERATION,
        variant: 'label-offsets',
        bindings: [
          {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
          {name: 'labelOffsets', view: labelOffsets, type: 'u32', access: 'read_write'},
          {name: 'sizes', view: sizes, type: 'u32', access: 'read_write'}
        ],
        invocationCount: labelCapacity,
        declarations: `${declarations}
fn getLowerBound(key: u32) -> u32 {
  var low = 0u;
  var high = ROWS;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (sortedKeys[sortedKeysOffset + middle] < key) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`,
        body: `let first = getLowerBound(index);
  let end = getLowerBound(index + 1u);
  labelOffsets[labelOffsetsOffset + index] = first;
  sizes[sizesOffset + index] = end - first;
  if (index + 1u == LABELS) {
    labelOffsets[labelOffsetsOffset + LABELS] = end;
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-region-origins`,
        operation: OPERATION,
        variant: 'region-origins',
        invocationCount: labelCapacity * columnCount,
        declarations,
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
          {name: 'labelOffsets', view: labelOffsets, type: 'u32', access: 'read'},
          {name: 'origins', view: regionOrigins, type: 'f32', access: 'read_write'},
          {name: 'regionModes', view: regionOriginModes, type: 'u32', access: 'read_write'},
          {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
          {name: 'globalOrigins', view: globalOrigins, type: 'f32', access: 'read_write'},
          {name: 'globalModes', view: globalOriginModes, type: 'u32', access: 'read_write'}
        ],
        body: `let label = index / COLUMNS;
  let begin = labelOffsets[labelOffsetsOffset + label];
  var origin = 0.0;
  if (begin < labelOffsets[labelOffsetsOffset + label + 1u]) {
    origin = values[valuesOffset + sortedRows[sortedRowsOffset + begin] * COLUMNS + index % COLUMNS];
  }
  origins[originsOffset + index] = origin;
  regionModes[regionModesOffset + index] = 0u;
  if (index < COLUMNS) {
    var globalOrigin = 0.0;
    if (sortedKeys[sortedKeysOffset] < LABELS) {
      globalOrigin = values[valuesOffset + sortedRows[sortedRowsOffset] * COLUMNS + index];
    }
    globalOrigins[globalOriginsOffset + index] = globalOrigin;
    globalModes[globalModesOffset + index] = labelOffsets[labelOffsetsOffset];
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-origin-signs`,
        operation: OPERATION,
        invocationCount: rows * columnCount,
        declarations,
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'labels', view: usedLabels, type: 'u32', access: 'read'},
          {name: 'origins', view: regionOrigins, type: 'f32', access: 'read'},
          {name: 'regionModes', view: regionOriginModes, type: 'atomic<u32>', access: 'read_write'},
          {name: 'globalOrigins', view: globalOrigins, type: 'f32', access: 'read'},
          {name: 'globalModes', view: globalOriginModes, type: 'atomic<u32>', access: 'read_write'}
        ],
        body: `let label = labels[labelsOffset + index / COLUMNS];
  if (label >= LABELS) { return; }
  let column = index % COLUMNS;
  let value = values[valuesOffset + index];
  let origin = origins[originsOffset + label * COLUMNS + column];
  if ((value < 0.0 && origin > 0.0) || (value > 0.0 && origin < 0.0)) {
    atomicOr(&regionModes[regionModesOffset + label * COLUMNS + column], 1u);
  }
  let globalOrigin = globalOrigins[globalOriginsOffset + column];
  if ((value < 0.0 && globalOrigin > 0.0) || (value > 0.0 && globalOrigin < 0.0)) {
    atomicOr(&globalModes[globalModesOffset + column], 1u);
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-select-origins`,
        operation: OPERATION,
        invocationCount: labelCapacity * columnCount,
        declarations,
        bindings: [
          {name: 'origins', view: regionOrigins, type: 'f32', access: 'read_write'},
          {name: 'regionModes', view: regionOriginModes, type: 'u32', access: 'read'},
          {name: 'globalOrigins', view: globalOrigins, type: 'f32', access: 'read_write'},
          {name: 'globalModes', view: globalOriginModes, type: 'u32', access: 'read'}
        ],
        // Centering same-sign columns preserves constants; zero-origin signed sums preserve
        // cancellation when subtracting a large first member would lose low bits.
        body: `if (regionModes[regionModesOffset + index] != 0u) { origins[originsOffset + index] = 0.0; }
  if (index < COLUMNS && globalModes[globalModesOffset + index] != 0u) {
    globalOrigins[globalOriginsOffset + index] = 0.0;
  }`
      }),
      // Per-label centered column sums, then deviations in the same coordinates. A constant
      // region has exact zero deviations even when its value times its size rounds in float32.
      // Staged parallel segment sums retain both rounding lanes, including errors in the input
      // subtraction. One huge region spreads over many workgroups rather than one serial thread.
      ...createRegionalMomentSumNodes<Parameters>(graph, {
        id: `${id}-region-sums`,
        operation: OPERATION,
        values,
        sortedRows,
        sortedKeys,
        offsets: labelOffsets,
        origins: regionOrigins,
        columnCount,
        high: regionSums,
        low: regionResiduals
      }),
      ...createLoadBalancedSegmentSumNodes<Parameters>(graph, {
        id: `${id}-region-deviations`,
        operation: OPERATION,
        offsets: labelOffsets,
        itemCapacity: rows,
        laneCount: columnCount,
        accumulatorType: 'f32',
        output: regionDeviations,
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
          {name: 'regionSums', view: regionSums, type: 'f32', access: 'read'},
          {name: 'residuals', view: regionResiduals, type: 'f32', access: 'read'},
          {name: 'origins', view: regionOrigins, type: 'f32', access: 'read'}
        ],
        declarations: `${declarations}
${REGIONAL_MOMENT_WGSL}
fn getLoadBalancedContribution(label: u32, item: u32, column: u32) -> f32 {
  // A visited label has at least one row, so the count is positive.
  let size = loadBalancedOffsets[loadBalancedOffsetsOffset + label + 1u] - loadBalancedOffsets[loadBalancedOffsetsOffset + label];
  let zero = loadBalancedOffsets[loadBalancedOffsetsOffset];
  let cell = label * COLUMNS + column;
  let mean = divideRegionalMoment(vec2<f32>(regionSums[regionSumsOffset + cell], residuals[residualsOffset + cell]), f32(size), zero);
  let centered = subtractRegionalValue(values[valuesOffset + sortedRows[sortedRowsOffset + item] * COLUMNS + column], origins[originsOffset + cell], zero);
  let difference = addRegionalMoment(centered, -mean, zero);
  let delta = difference.x + difference.y;
  return delta * delta;
}`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-region-centroids`,
        operation: OPERATION,
        variant: 'region-means',
        bindings: [
          {name: 'regionSums', view: regionSums, type: 'f32', access: 'read'},
          {name: 'residuals', view: regionResiduals, type: 'f32', access: 'read'},
          {name: 'origins', view: regionOrigins, type: 'f32', access: 'read'},
          {name: 'globalOrigins', view: globalOrigins, type: 'f32', access: 'read'},
          {name: 'labelOffsets', view: labelOffsets, type: 'u32', access: 'read'},
          {name: 'centroids', view: regionCentroids, type: 'f32', access: 'read_write'},
          {name: 'centroidResiduals', view: centroidResiduals, type: 'f32', access: 'read_write'},
          ...(output.regionMeans
            ? [
                {
                  name: 'regionMeans',
                  view: output.regionMeans,
                  type: 'f32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: labelCapacity * columnCount,
        declarations: `${declarations}
${GET_NAN_WGSL}
${REGIONAL_MOMENT_WGSL}`,
        body: `let label = index / COLUMNS;
  let size = labelOffsets[labelOffsetsOffset + label + 1u] - labelOffsets[labelOffsetsOffset + label];
  var globalFrameSum = 0.0;
  var globalFrameResidual = 0.0;
  var absoluteMean = getNaN();
  if (size > 0u) {
    let zero = labelOffsets[labelOffsetsOffset];
    let mean = divideRegionalMoment(vec2<f32>(regionSums[regionSumsOffset + index], residuals[residualsOffset + index]), f32(size), zero);
    // The regional sum in the global frame: local sum plus size times the origin offset, kept
    // as a split sum so the between-region contrast can be formed before any division.
    let relativeOrigin = subtractRegionalValue(origins[originsOffset + index], globalOrigins[globalOriginsOffset + index % COLUMNS], zero);
    let relative = addRegionalMoment(vec2<f32>(regionSums[regionSumsOffset + index], residuals[residualsOffset + index]),
      multiplyRegionalMoment(relativeOrigin, f32(size), zero), zero);
    globalFrameSum = relative.x;
    globalFrameResidual = relative.y;
    let absolute = addRegionalMoment(vec2<f32>(origins[originsOffset + index], 0.0), mean, zero);
    absoluteMean = absolute.x + absolute.y;
  }
  centroids[centroidsOffset + index] = globalFrameSum;
  centroidResiduals[centroidResidualsOffset + index] = globalFrameResidual;
  ${output.regionMeans ? 'regionMeans[regionMeansOffset + index] = absoluteMean;' : ''}`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-region-within`,
        operation: OPERATION,
        variant: 'region-within',
        bindings: [
          {name: 'regionDeviations', view: regionDeviations, type: 'f32', access: 'read'},
          {name: 'within', view: within, type: 'f32', access: 'read_write'}
        ],
        invocationCount: labelCapacity,
        declarations,
        body: `var ssd = 0.0;
  for (var column = 0u; column < COLUMNS; column++) {
    ssd += regionDeviations[regionDeviationsOffset + index * COLUMNS + column];
  }
  within[withinOffset + index] = ssd;`
      })
    ];
    if (weights && links) {
      // Per-row (total, cross) link counts, load balanced over the weights rows (hub rows). Each
      // weights chunk is summed on its own; a chunk's counts are copied to its rows of `links`.
      for (const dispatch of getGPUSpatialWeightsChunkDispatches(weights)) {
        const chunkLinks = dispatch.isWhole
          ? links
          : createTransientView(
              graph,
              `${id}-link-counts${dispatch.idSuffix}`,
              'uint32',
              2 * dispatch.rowCount
            );
        nodes.push(
          ...createLoadBalancedSegmentSumNodes<Parameters>(graph, {
            id: `${id}-links${dispatch.idSuffix}`,
            operation: OPERATION,
            offsets: dispatch.weights.offsets,
            itemCapacity: dispatch.weights.neighbors.length,
            accumulatorType: 'vec2<u32>',
            output: chunkLinks,
            bindings: [
              {name: 'neighbors', view: dispatch.weights.neighbors, type: 'u32', access: 'read'},
              // The used labels: a link to or from a left-out row is not counted.
              {name: 'labels', view: usedLabels, type: 'u32', access: 'read'}
            ],
            declarations: `${declarations}${dispatch.constantsWGSL ? `\n${dispatch.constantsWGSL}` : ''}
fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> vec2<u32> {
  let own = labels[labelsOffset + ${dispatch.isWhole ? 'row' : 'WEIGHTS_ROW_START + row'}];
  let other = labels[labelsOffset + neighbors[neighborsOffset + item]];
  if (own >= LABELS || other >= LABELS) {
    return vec2<u32>();
  }
  return vec2<u32>(1u, select(0u, 1u, other != own));
}`
          })
        );
        if (!dispatch.isWhole) {
          nodes.push(
            createWGSLKernelNode<Parameters>(graph, {
              id: `${id}-link-counts-copy${dispatch.idSuffix}`,
              operation: OPERATION,
              variant: 'link-counts-copy',
              bindings: [
                {name: 'chunkLinks', view: chunkLinks, type: 'u32', access: 'read'},
                {name: 'links', view: links, type: 'u32', access: 'read_write'}
              ],
              invocationCount: 2 * dispatch.rowCount,
              declarations: dispatch.constantsWGSL,
              body: 'links[linksOffset + 2u * WEIGHTS_ROW_START + index] = chunkLinks[chunkLinksOffset + index];'
            })
          );
        }
      }
    }
    // The summary is a tiled reduction (fixed tile order, no atomics) rather than one thread
    // walking every row and label: tiles of rows and of labels in parallel, the column means,
    // the deviations about those means, and parallel scalar reductions of the tiles.
    const rowTileCount = Math.ceil(rows / SUMMARY_TILE);
    const labelTileCount = Math.ceil(labelCapacity / SUMMARY_TILE);
    // Per row tile: used rows, ignored rows, cross links, links, and ignored rows that had a label.
    const rowCounts = createTransientView(graph, `${id}-row-counts`, 'uint32', rowTileCount * 5);
    const rowSums = createTransientView(
      graph,
      `${id}-row-sums`,
      'float32',
      rowTileCount * columnCount * 2
    );
    const labelCounts = createTransientView(
      graph,
      `${id}-label-counts`,
      'uint32',
      labelTileCount * 3
    );
    const labelWithin = createTransientView(graph, `${id}-label-within`, 'float32', labelTileCount);
    const labelBetween = createTransientView(
      graph,
      `${id}-label-between`,
      'float32',
      labelTileCount
    );
    const means = createTransientView(graph, `${id}-means`, 'float32', columnCount);
    const meanResiduals = createTransientView(
      graph,
      `${id}-mean-residuals`,
      'float32',
      columnCount
    );
    const deviations = createTransientView(
      graph,
      `${id}-deviations`,
      'float32',
      rowTileCount * columnCount
    );
    const rowChannels = Array.from({length: 4}, (_, channel) =>
      createTransientView(graph, `${id}-summary-row-channel-${channel}`, 'uint32', rowTileCount)
    );
    const labelChannels = Array.from({length: 3}, (_, channel) =>
      createTransientView(graph, `${id}-summary-label-channel-${channel}`, 'uint32', labelTileCount)
    );
    const totalSquares = createTransientView(
      graph,
      `${id}-summary-total-squares`,
      'float32',
      rowTileCount
    );
    // Aligned scalar slots let published reductions write into shared result buffers while
    // the loop-free publisher stays below the eight-storage-binding limit.
    const countTotals = createTransientView(
      graph,
      `${id}-summary-count-totals`,
      'uint32',
      7 * reductionStride
    );
    const squareTotals = createTransientView(
      graph,
      `${id}-summary-square-totals`,
      'float32',
      3 * reductionStride
    );
    const tileDeclarations = `${declarations}
const SUMMARY_TILE: u32 = ${SUMMARY_TILE}u;
const ROW_TILES: u32 = ${rowTileCount}u;
const LABEL_TILES: u32 = ${labelTileCount}u;`;
    let meanRows = rowCounts;
    let meanSums = rowSums;
    const meanReductionNodes: GPUCommandNode<Parameters>[] = [];
    let meanTileCount = rowTileCount;
    let meanStage = 0;
    while (meanTileCount > 1) {
      const nextCount = Math.ceil(meanTileCount / SUMMARY_TILE);
      const nextRows = createTransientView(
        graph,
        `${id}-mean-rows-${meanStage}`,
        'uint32',
        nextCount * 5
      );
      const nextSums = createTransientView(
        graph,
        `${id}-mean-sums-${meanStage}`,
        'float32',
        nextCount * columnCount * 2
      );
      meanReductionNodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-mean-stage-${meanStage++}`,
          operation: OPERATION,
          invocationCount: nextCount * columnCount,
          bindings: [
            {name: 'rows', view: meanRows, type: 'u32', access: 'read'},
            {name: 'sums', view: meanSums, type: 'f32', access: 'read'},
            {name: 'labelOffsets', view: labelOffsets, type: 'u32', access: 'read'},
            {name: 'nextRows', view: nextRows, type: 'u32', access: 'read_write'},
            {name: 'nextSums', view: nextSums, type: 'f32', access: 'read_write'}
          ],
          declarations: `${declarations}\n${REGIONAL_MOMENT_WGSL}`,
          body: `let group = index / COLUMNS;
  let column = index % COLUMNS;
  var count = 0u;
  var sum = vec2<f32>(0.0);
  for (var tile = group * ${SUMMARY_TILE}u; tile < min((group + 1u) * ${SUMMARY_TILE}u, ${meanTileCount}u); tile++) {
    count += rows[rowsOffset + 5u * tile];
    let source = 2u * (tile * COLUMNS + column);
    sum = addRegionalMoment(sum, vec2<f32>(sums[sumsOffset + source], sums[sumsOffset + source + 1u]), labelOffsets[labelOffsetsOffset]);
  }
  if (column == 0u) { nextRows[nextRowsOffset + 5u * group] = count; }
  nextSums[nextSumsOffset + 2u * index] = sum.x;
  nextSums[nextSumsOffset + 2u * index + 1u] = sum.y;`
        })
      );
      meanRows = nextRows;
      meanSums = nextSums;
      meanTileCount = nextCount;
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-row-tiles`,
        operation: OPERATION,
        variant: 'row-tiles',
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'labels', view: usedLabels, type: 'u32', access: 'read'},
          {name: 'sourceLabels', view: labels, type: 'u32', access: 'read'},
          {name: 'globalOrigins', view: globalOrigins, type: 'f32', access: 'read'},
          {name: 'labelOffsets', view: labelOffsets, type: 'u32', access: 'read'},
          ...(links
            ? [{name: 'links', view: links, type: 'u32' as const, access: 'read' as const}]
            : []),
          {name: 'rowCounts', view: rowCounts, type: 'u32', access: 'read_write'},
          {name: 'rowSums', view: rowSums, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rowTileCount,
        declarations: `${tileDeclarations}\n${REGIONAL_MOMENT_WGSL}`,
        body: `var labelled = 0u;
  var ignored = 0u;
  var invalid = 0u;
  var cross = 0u;
  var total = 0u;
  var sums: array<vec2<f32>, ${columnCount}>;
  let firstRow = index * SUMMARY_TILE;
  for (var row = firstRow; row < min(firstRow + SUMMARY_TILE, ROWS); row++) {
    if (labels[labelsOffset + row] < LABELS) {
      labelled += 1u;
      ${links ? 'total += links[linksOffset + 2u * row]; cross += links[linksOffset + 2u * row + 1u];' : ''}
      for (var column = 0u; column < COLUMNS; column++) {
        let zero = labelOffsets[labelOffsetsOffset];
        sums[column] = addRegionalMoment(sums[column], subtractRegionalValue(values[valuesOffset + row * COLUMNS + column], globalOrigins[globalOriginsOffset + column], zero), zero);
      }
    } else {
      ignored += 1u;
      // Labelled in the input, so it was left out for a non-finite value.
      invalid += select(0u, 1u, sourceLabels[sourceLabelsOffset + row] < LABELS);
    }
  }
  rowCounts[rowCountsOffset + 5u * index] = labelled;
  rowCounts[rowCountsOffset + 5u * index + 1u] = ignored;
  rowCounts[rowCountsOffset + 5u * index + 2u] = cross;
  rowCounts[rowCountsOffset + 5u * index + 3u] = total;
  rowCounts[rowCountsOffset + 5u * index + 4u] = invalid;
  for (var column = 0u; column < COLUMNS; column++) {
    rowSums[rowSumsOffset + 2u * (index * COLUMNS + column)] = sums[column].x;
    rowSums[rowSumsOffset + 2u * (index * COLUMNS + column) + 1u] = sums[column].y;
  }`
      }),
      ...meanReductionNodes,
      // The overall column means, for the total deviations. The between-region tiles read the
      // grand split sums and count directly.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-means`,
        operation: OPERATION,
        variant: 'means',
        bindings: [
          {name: 'rowCounts', view: meanRows, type: 'u32', access: 'read'},
          {name: 'rowSums', view: meanSums, type: 'f32', access: 'read'},
          {name: 'labelOffsets', view: labelOffsets, type: 'u32', access: 'read'},
          {name: 'means', view: means, type: 'f32', access: 'read_write'},
          {name: 'meanResiduals', view: meanResiduals, type: 'f32', access: 'read_write'}
        ],
        invocationCount: columnCount,
        declarations: `${tileDeclarations}\n${REGIONAL_MOMENT_WGSL}`,
        body: `let labelled = rowCounts[rowCountsOffset];
  let mean = divideRegionalMoment(vec2<f32>(rowSums[rowSumsOffset + 2u * index], rowSums[rowSumsOffset + 2u * index + 1u]), max(f32(labelled), 1.0), labelOffsets[labelOffsetsOffset]);
  means[meansOffset + index] = mean.x;
  meanResiduals[meanResidualsOffset + index] = mean.y;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-label-tiles`,
        operation: OPERATION,
        variant: 'label-tiles',
        bindings: [
          {name: 'sizes', view: sizes, type: 'u32', access: 'read'},
          {name: 'within', view: within, type: 'f32', access: 'read'},
          {name: 'labelCounts', view: labelCounts, type: 'u32', access: 'read_write'},
          {name: 'labelWithin', view: labelWithin, type: 'f32', access: 'read_write'}
        ],
        invocationCount: labelTileCount,
        declarations: tileDeclarations,
        body: `var regionCount = 0u;
  var withinSsd = 0.0;
  var minimumSize = 0xffffffffu;
  var maximumSize = 0u;
  let firstLabel = index * SUMMARY_TILE;
  for (var label = firstLabel; label < min(firstLabel + SUMMARY_TILE, LABELS); label++) {
    let size = sizes[sizesOffset + label];
    if (size > 0u) {
      regionCount += 1u;
      withinSsd += within[withinOffset + label];
      minimumSize = min(minimumSize, size);
      maximumSize = max(maximumSize, size);
    }
  }
  labelCounts[labelCountsOffset + 3u * index] = regionCount;
  labelCounts[labelCountsOffset + 3u * index + 1u] = minimumSize;
  labelCounts[labelCountsOffset + 3u * index + 2u] = maximumSize;
  labelWithin[labelWithinOffset + index] = withinSsd;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-between-tiles`,
        operation: OPERATION,
        invocationCount: labelTileCount,
        declarations: `${tileDeclarations}\n${REGIONAL_MOMENT_WGSL}`,
        bindings: [
          {name: 'sizes', view: sizes, type: 'u32', access: 'read'},
          {name: 'centroids', view: regionCentroids, type: 'f32', access: 'read'},
          {name: 'centroidResiduals', view: centroidResiduals, type: 'f32', access: 'read'},
          {name: 'grandCounts', view: meanRows, type: 'u32', access: 'read'},
          {name: 'grandSums', view: meanSums, type: 'f32', access: 'read'},
          {name: 'labelOffsets', view: labelOffsets, type: 'u32', access: 'read'},
          {name: 'labelBetween', view: labelBetween, type: 'f32', access: 'read_write'}
        ],
        // The contrast mean_r - mean = (N S_r - n_r S) / (n_r N) is formed from split sums and
        // exact products before the one division, so equal means give exactly zero rather than
        // two differently rounded quotients. Both counts are scaled by one power of two at or
        // above N, which is exact and keeps the products no larger than the sums.
        body: `var betweenSsd = 0.0;
  let zero = labelOffsets[labelOffsetsOffset];
  let labelled = max(grandCounts[grandCountsOffset], 1u);
  let countScale = ldexp(1.0, -i32(firstLeadingBit(labelled)) - 1);
  let scaledTotal = f32(labelled) * countScale;
  for (var label = index * SUMMARY_TILE; label < min((index + 1u) * SUMMARY_TILE, LABELS); label++) {
    let size = sizes[sizesOffset + label];
    if (size > 0u) {
      var squaredDistance = 0.0;
      let scaledSize = f32(size) * countScale;
      for (var column = 0u; column < COLUMNS; column++) {
        let cell = label * COLUMNS + column;
        let regional = vec2<f32>(centroids[centroidsOffset + cell], centroidResiduals[centroidResidualsOffset + cell]);
        let global = vec2<f32>(grandSums[grandSumsOffset + 2u * column], grandSums[grandSumsOffset + 2u * column + 1u]);
        let difference = addRegionalMoment(multiplyRegionalMoment(regional, scaledTotal, zero),
          -multiplyRegionalMoment(global, scaledSize, zero), zero);
        let delta = (difference.x + difference.y) / (f32(size) * scaledTotal);
        squaredDistance += delta * delta;
      }
      betweenSsd += f32(size) * squaredDistance;
    }
  }
  labelBetween[labelBetweenOffset + index] = betweenSsd;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-deviation-tiles`,
        operation: OPERATION,
        variant: 'deviation-tiles',
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'labels', view: usedLabels, type: 'u32', access: 'read'},
          {name: 'means', view: means, type: 'f32', access: 'read'},
          {name: 'meanResiduals', view: meanResiduals, type: 'f32', access: 'read'},
          {name: 'deviations', view: deviations, type: 'f32', access: 'read_write'},
          {name: 'globalOrigins', view: globalOrigins, type: 'f32', access: 'read'},
          {name: 'labelOffsets', view: labelOffsets, type: 'u32', access: 'read'}
        ],
        invocationCount: rowTileCount,
        declarations: `${tileDeclarations}\n${REGIONAL_MOMENT_WGSL}`,
        body: `var squares: array<f32, ${columnCount}>;
  let firstRow = index * SUMMARY_TILE;
  for (var row = firstRow; row < min(firstRow + SUMMARY_TILE, ROWS); row++) {
    if (labels[labelsOffset + row] < LABELS) {
      for (var column = 0u; column < COLUMNS; column++) {
        let zero = labelOffsets[labelOffsetsOffset];
        let centered = subtractRegionalValue(values[valuesOffset + row * COLUMNS + column], globalOrigins[globalOriginsOffset + column], zero);
        let difference = addRegionalMoment(centered, -vec2<f32>(means[meansOffset + column], meanResiduals[meanResidualsOffset + column]), zero);
        let delta = difference.x + difference.y;
        squares[column] += delta * delta;
      }
    }
  }
  for (var column = 0u; column < COLUMNS; column++) {
    deviations[deviationsOffset + index * COLUMNS + column] = squares[column];
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summary-row-channels`,
        operation: OPERATION,
        invocationCount: rowTileCount,
        declarations: tileDeclarations,
        bindings: [
          {name: 'rowCounts', view: rowCounts, type: 'u32', access: 'read'},
          {name: 'deviations', view: deviations, type: 'f32', access: 'read'},
          ...rowChannels.map((view, channel) => ({
            name: `channel${channel}`,
            view,
            type: 'u32' as const,
            access: 'read_write' as const
          })),
          {name: 'totalSquares', view: totalSquares, type: 'f32', access: 'read_write'}
        ],
        body: `channel0[channel0Offset + index] = rowCounts[rowCountsOffset + 5u * index + 1u];
  channel1[channel1Offset + index] = rowCounts[rowCountsOffset + 5u * index + 2u];
  channel2[channel2Offset + index] = rowCounts[rowCountsOffset + 5u * index + 3u];
  channel3[channel3Offset + index] = rowCounts[rowCountsOffset + 5u * index + 4u];
  var sum = 0.0;
  // Only the construction-validated attribute count (at most 32), never rows or tiles.
  for (var column = 0u; column < COLUMNS; column++) {
    sum += deviations[deviationsOffset + index * COLUMNS + column];
  }
  totalSquares[totalSquaresOffset + index] = sum;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summary-label-channels`,
        operation: OPERATION,
        invocationCount: labelTileCount,
        bindings: [
          {name: 'labelCounts', view: labelCounts, type: 'u32', access: 'read'},
          ...labelChannels.map((view, channel) => ({
            name: `channel${channel}`,
            view,
            type: 'u32' as const,
            access: 'read_write' as const
          }))
        ],
        body: `channel0[channel0Offset + index] = labelCounts[labelCountsOffset + 3u * index];
  channel1[channel1Offset + index] = labelCounts[labelCountsOffset + 3u * index + 1u];
  channel2[channel2Offset + index] = labelCounts[labelCountsOffset + 3u * index + 2u];`
      }),
      ...[...labelChannels, ...rowChannels].flatMap((input, channel) =>
        new GPUReduction({
          id: `${id}-summary-count-${channel}`,
          input,
          operation: channel === 1 ? 'min' : channel === 2 ? 'max' : 'sum',
          output: graph.createDataView(countTotals.buffer, {
            format: 'uint32',
            length: 1,
            byteOffset: channel * reductionStride * 4
          })
        }).getCommandNodes(graph)
      ),
      ...[labelWithin, labelBetween, totalSquares].flatMap((input, channel) =>
        new GPUReduction({
          id: `${id}-summary-squares-${channel}`,
          input,
          operation: 'sum',
          output: graph.createDataView(squareTotals.buffer, {
            format: 'float32',
            length: 1,
            byteOffset: channel * reductionStride * 4
          })
        }).getCommandNodes(graph)
      ),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summary`,
        operation: OPERATION,
        variant: 'summary',
        bindings: [
          {name: 'countTotals', view: countTotals, type: 'u32', access: 'read'},
          {name: 'squareTotals', view: squareTotals, type: 'f32', access: 'read'},
          {name: 'summary', view: output.summary, type: 'f32', access: 'read_write'},
          {name: 'labelOffsets', view: labelOffsets, type: 'u32', access: 'read'},
          ...(output.invalidRowCount
            ? [
                {
                  name: 'invalidRowCount',
                  view: output.invalidRowCount,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: 1,
        declarations: `${declarations}
const RESULT_STRIDE: u32 = ${reductionStride}u;
${GET_NAN_WGSL}
${ROUNDED_DIFFERENCE_WGSL}`,
        body: `let regionCount = countTotals[countTotalsOffset];
  let minimumSize = countTotals[countTotalsOffset + RESULT_STRIDE];
  let maximumSize = countTotals[countTotalsOffset + 2u * RESULT_STRIDE];
  let ignored = countTotals[countTotalsOffset + 3u * RESULT_STRIDE];
  let cross = countTotals[countTotalsOffset + 4u * RESULT_STRIDE];
  let total = countTotals[countTotalsOffset + 5u * RESULT_STRIDE];
  let invalid = countTotals[countTotalsOffset + 6u * RESULT_STRIDE];
  let withinSsd = squareTotals[squareTotalsOffset];
  let between = squareTotals[squareTotalsOffset + RESULT_STRIDE];
  let totalSsd = squareTotals[squareTotalsOffset + 2u * RESULT_STRIDE];
  ${output.invalidRowCount ? 'invalidRowCount[invalidRowCountOffset] = invalid;' : ''}
  summary[summaryOffset + 0u] = f32(regionCount);
  summary[summaryOffset + 1u] = withinSsd;
  summary[summaryOffset + 2u] = between;
  summary[summaryOffset + 3u] = totalSsd;
  summary[summaryOffset + 4u] = f32(select(minimumSize, 0u, regionCount == 0u));
  summary[summaryOffset + 5u] = f32(maximumSize);
  summary[summaryOffset + 6u] = select(0.0, f32(cross) / f32(total), total > 0u);
  summary[summaryOffset + 7u] = f32(ignored);
  // Calinski-Harabasz: undefined below two regions or without more rows than regions.
  let labelledRows = f32(ROWS - ignored);
  let regions = f32(regionCount);
  var pseudoF = getNaN();
  if (regionCount >= 2u && labelledRows > regions) {
    if (withinSsd == 0.0) {
      pseudoF = 1.0;
    } else if (between == 0.0) {
      pseudoF = 0.0;
    } else {
      // Form the SSD ratio from normal mantissas. Neither between*(n-k) nor within/(n-k)
      // is formed, avoiding the overflowing product and the underflowing mean square.
      let betweenParts = frexp(between);
      let withinParts = frexp(withinSsd);
      let zero = labelOffsets[labelOffsetsOffset];
      let ratio = getRoundedDifference(betweenParts.fract / withinParts.fract, 0.0, zero);
      let degreesRatio = getRoundedDifference((labelledRows - regions) / (regions - 1.0), 0.0, zero);
      let mantissa = getRoundedDifference(ratio * degreesRatio, 0.0, zero);
      let exponentDifference = betweenParts.exp - withinParts.exp;
      // Each ldexp exponent stays in its WGSL-defined range. All powers have the same sign,
      // so an intermediate cannot overflow/underflow when the final result is representable.
      let firstPower = clamp(exponentDifference, -126i, 126i);
      let remainingPower = exponentDifference - firstPower;
      let secondPower = clamp(remainingPower, -126i, 126i);
      let scaled = ldexp(ldexp(mantissa, firstPower), secondPower);
      pseudoF = ldexp(scaled, remainingPower - secondPower);
    }
  }
  summary[summaryOffset + 8u] = pseudoF;`
      })
    );
    if (weights)
      nodes.push(
        ...createRegionalizationLossSuppressionNodes(graph, id, OPERATION, weights, [
          {
            view: output.summary,
            length: 1,
            wordOffset: GPU_REGION_PARTITION_EVALUATION_LAYOUT.crossLinkFraction,
            value: 'nan'
          }
        ])
      );
    if (output.labelsComplete)
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-labels-complete`,
          operation: OPERATION,
          invocationCount: 1,
          bindings: [
            ...(props.labelsComplete
              ? [
                  {
                    name: 'input',
                    view: props.labelsComplete,
                    type: 'u32' as const,
                    access: 'read' as const
                  }
                ]
              : []),
            {name: 'complete', view: output.labelsComplete, type: 'u32', access: 'read_write'}
          ],
          body: `complete[completeOffset] = ${props.labelsComplete ? 'select(0u, 1u, input[inputOffset] != 0u)' : '1u'};`
        })
      );
    return nodes;
  }
}
