// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUBoundedResultStatusPort} from '../../analysis-core/contracts/index';
import {
  createFloat32ValuesNodes,
  type GPUFloat32OrUint32Values,
  validateFloat32OrUint32Values
} from '../spatial-weights/float32-values';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights,
  isGPUSpatialWeightsRowStandardized
} from '../spatial-weights/spatial-weights';
import {SPATIAL_AUTOCORRELATION_FLOAT_WGSL} from '../spatial-autocorrelation/spatial-autocorrelation-kernels';

import {validateSpatialStatisticsBindings} from '../spatial-weights/spatial-statistics-bindings';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  createWeightsLossNodes,
  validateWeightsLossSources
} from '../spatial-weights/spatial-weights-loss';
import {ROUNDED_DIFFERENCE_WGSL} from '../spatial-regression/rounded-difference-wgsl';
import {ADDITION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/addition-rounding-error-wgsl';
import {MULTIPLICATION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/multiplication-rounding-error-wgsl';
import {
  NEIGHBORHOOD_STOPPING_FIELDS,
  snapshotNeighborhoodWeightsStatus,
  validateNeighborhoodChunkLossBindings,
  validateNeighborhoodWeightsCertificates,
  type NeighborhoodWeightsChunkMetadata
} from './neighborhood-weights-status';

const OPERATION = 'GPUNeighborhoodSummary';

/** Largest supported compile-time `maximumNeighbors` (members per row for the median). */
export const GPU_NEIGHBORHOOD_SUMMARY_MAXIMUM_NEIGHBORS = 64;

/** Reserved missing-category value; also the `modes` value of a row with no categorical members. */
export const GPU_NEIGHBORHOOD_SUMMARY_NO_MODE = 0xffffffff;

/** A numeric statistic of {@link GPUNeighborhoodSummary}; one `output.table` column each. */
export type GPUNeighborhoodSummaryStatistic =
  | 'count'
  | 'weightSum'
  | 'sum'
  | 'mean'
  | 'min'
  | 'max'
  | 'standardDeviation'
  | 'median';

const STATISTICS: readonly GPUNeighborhoodSummaryStatistic[] = [
  'count',
  'weightSum',
  'sum',
  'mean',
  'min',
  'max',
  'standardDeviation',
  'median'
];

/**
 * Properties for {@link GPUNeighborhoodSummary}.
 *
 * Per-frame (no rebuild or recompile): the contents of `weights`, `values`, `categories` and
 * `mask`. Compile-time: the row count, slot capacity, `statistics`, `includeFocal`, `focalWeight`,
 * `maximumNeighbors` and which optional views exist.
 */
export type GPUNeighborhoodSummaryProps = {
  /** Prefix for generated node IDs. Defaults to `'neighborhood-summary'`. */
  id?: string;
  /** Square spatial weights whose neighbor IDs index the value rows. */
  weights: GPUSpatialWeights;
  /**
   * Numeric values, one per row, `float32` or `uint32` (counts are converted to f32 by one node,
   * exact below 2^24). Required when `statistics` is not empty.
   */
  values?: GPUFloat32OrUint32Values;
  /** Categorical uint32 column, one per row. Required with `output.modes` or `output.entropy`. */
  categories?: GraphDataView<'uint32'>;
  /** Optional row selection: nonzero includes the row as a focus and as a neighbor. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Adds the focal row itself to its own neighborhood (momepy's `include_self`, a self weight on
   * the diagonal). The focal member comes first, with weight `focalWeight`. Compile-time.
   */
  includeFocal?: boolean;
  /** Weight of the focal member when `includeFocal` is set. Finite and non-negative; defaults to 1. */
  focalWeight?: number;
  /** Numeric statistics to write, in `output.table` column order. No duplicates. Compile-time. */
  statistics?: readonly GPUNeighborhoodSummaryStatistic[];
  /**
   * Compile-time capacity of the median buffer, in `[1, 64]`. Members beyond it set
   * `output.status.overflow` and the row's median is quiet NaN. It also sizes the private member
   * cache of the categorical `modes` and `entropy` (larger rows stay exact but take a slower
   * scan). Defaults to 32.
   */
  maximumNeighbors?: number;
  /** Caller-owned outputs; at least one of `table`, `modes` and `entropy`. */
  output: GPUNeighborhoodSummaryOutput;
};

/** Caller-owned outputs of {@link GPUNeighborhoodSummary}. */
export type GPUNeighborhoodSummaryOutput = {
  /**
   * Row-major table of `rows * statistics.length` float32 values; row `i` column `c` holds
   * `statistics[c]` at index `i * statistics.length + c`. Required when `statistics` is not empty.
   */
  table?: GraphDataView<'float32'>;
  /** Weighted mode category per row, `GPU_NEIGHBORHOOD_SUMMARY_NO_MODE` if empty. */
  modes?: GraphDataView<'uint32'>;
  /** Shannon entropy (natural log) of the weighted category shares per row. */
  entropy?: GraphDataView<'float32'>;
  /**
   * `overflow`: one-row flag, 1 when a row with the median requested exceeded
   * `maximumNeighbors`. Required with `'median'`.
   */
  status?: Pick<GPUBoundedResultStatusPort, 'overflow'>;
};

/**
 * Per-row summaries of a neighborhood over a {@link GPUSpatialWeights} CSR: the generalized spatial
 * lag family (momepy `describe`, libpysal `lag_categorical`).
 *
 * Members of row `i` are, in this fixed order: the focal row (with `includeFocal`, weight
 * `focalWeight`), then the active CSR slots of the row in slot order. A slot is active when its
 * weight is positive; a listed self slot, slots outside `[0, rows)`, neighbors with a zero mask
 * and, for the numeric statistics, neighbors with a non-finite value are skipped.
 * Categorical summaries exclude the reserved missing-category value `0xffffffff`, including
 * the focal row; numeric summaries of the same row remain independent.
 * Zero-weight slots are therefore not members: `count`, `min`, `max`, `median` and the categorical outputs
 * ignore them. This deliberately differs from libpysal `lag_spatial` and momepy, which count
 * every listed neighbor including zero-weight ones (identical when no zero weights are listed).
 * When `weights.information.transform === 'row'` the slot weights are re-standardized over the active
 * members (`w_s / sum of active slot weights`, the focal weight is unchanged), so `sum` (the lag)
 * stays a weighted average when neighbors are masked or non-finite instead of shrinking toward 0.
 * When `weights.status.overflow` or `weights.status.candidateOverflow` is nonzero the weights are
 * incomplete (retained slots or upstream candidates were dropped) and every output is NaN
 * (`count` too; `modes` holds `GPU_NEIGHBORHOOD_SUMMARY_NO_MODE`). Nonzero
 * `weights.status.iterationLimitReached` or zero `weights.status.converged` also suppresses included
 * rows. Construction rejects `complete`, `requiredCount`, `unresolvedCount`, `invalidCount` and
 * `approximation` at either scope, plus chunk-local `iterationLimitReached` and `converged`.
 * Pass the producer's `overflow` and `candidateOverflow` for capacity loss, and verify the CSR
 * is complete before omitting other loss evidence.
 * All accepted flags are folded into one GPU word each frame, so restoring completeness
 * restores ordinary results on the same compiled graph. A row with a zero mask is not summarized:
 * `count` is 0 and every other column is quiet NaN (`modes` holds `GPU_NEIGHBORHOOD_SUMMARY_NO_MODE`). A focal row's own value
 * is only used with `includeFocal`. With `k` members, weights `w_s` and values `x_s`:
 * - `count` is `k`; `weightSum` is `W = sum w_s`; `sum` is `sum w_s x_s` (the lag).
 * - `mean` is `sum / W` (NaN when `W = 0`).
 * - `min` and `max` are unweighted over the members (NaN when `k = 0`).
 * - `standardDeviation` is the weighted population deviation `sqrt(sum w_s (x_s - mean)^2 / W)`,
 *   computed about a positive-weight anchor using a weight-scaled relative mean (NaN when `W = 0`).
 * - `median` is the unweighted median (mean of the two middle values for even `k`) for `k` up to
 *   `maximumNeighbors`; larger rows have a NaN median and set `output.status.overflow` while
 *   retaining the other summaries.
 * - `modes` is the category with the largest weighted frequency `sum_{s: c_s = c} w_s`; ties go
 *   to the lowest category. `entropy` is `-sum p_c ln p_c` with `p_c = f_c / W` over the categories
 *   in first-occurrence order (0 for one category, NaN when `W = 0`). Category cost is `O(k^2)` per
 *   row with no member limit.
 *
 * Numeric sums run sequentially in member order in f32. Category frequencies, total mass and
 * row-standardized focal mass retain accumulation residuals to preserve ties between differently
 * partitioned masses and consistent entropy normalization. Results are deterministic.
 *
 * The numeric statistics are one kernel and the categorical outputs another, so each stays under
 * the 8 storage-buffer limit.
 */
export class GPUNeighborhoodSummary implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNeighborhoodSummaryProps;
  /** Validated loss view references are fixed; their GPU-written contents remain per-frame. */
  private readonly weightsLossViews: readonly GraphDataView<'uint32'>[];

  constructor(props: GPUNeighborhoodSummaryProps) {
    const {status, chunks, ...weights} = props.weights as GPUSpatialWeights &
      NeighborhoodWeightsChunkMetadata<GraphDataView<'uint32'>>;
    props = {
      ...props,
      statistics: props.statistics ? [...props.statistics] : undefined,
      weights: {
        ...weights,
        ...(chunks !== undefined ? {chunks} : {}),
        information: props.weights.information ? {...props.weights.information} : undefined,
        status: snapshotNeighborhoodWeightsStatus(status)
      },
      output: {...props.output, status: props.output.status ? {...props.output.status} : undefined}
    };
    const id = props.id ?? 'neighborhood-summary';
    this.id = id;
    this.props = props;
    validateNeighborhoodWeightsCertificates(id, 'weights.status', props.weights.status);
    const rows = validateGPUSpatialWeights(id, props.weights);
    const statistics = props.statistics ?? [];
    for (const statistic of statistics) {
      if (!STATISTICS.includes(statistic)) {
        throw new Error(`${id} unknown statistic ${statistic}`);
      }
    }
    if (new Set(statistics).size !== statistics.length) {
      throw new Error(`${id} statistics must not repeat`);
    }
    const {output} = props;
    const categorical = Boolean(output.modes || output.entropy);
    if (statistics.length === 0 && !categorical) {
      throw new Error(`${id} requests no statistic`);
    }
    if (statistics.length > 0) {
      if (!props.values || !output.table) {
        throw new Error(`${id} statistics require values and output.table`);
      }
      if (validateFloat32OrUint32Values(props.values, `${id} values`) !== rows) {
        throw new Error(`${id} values needs one row per weights row`);
      }
      validateSinglePackedView(output.table, ['float32'], `${id} output.table`);
      if (output.table.length !== rows * statistics.length) {
        throw new Error(`${id} output.table needs rows * statistics.length values`);
      }
    } else if (output.table || props.values) {
      throw new Error(`${id} values and output.table require statistics`);
    }
    const overflow = output.status?.overflow;
    if (statistics.includes('median') && !overflow) {
      throw new Error(`${id} median requires output.status.overflow`);
    }
    if (overflow) {
      validateSinglePackedView(overflow, ['uint32'], `${id} output.status.overflow`);
      if (overflow.length < 1) {
        throw new Error(`${id} output.status.overflow needs one row`);
      }
    }
    validateCapacity(
      id,
      'maximumNeighbors',
      props.maximumNeighbors ?? 32,
      1,
      GPU_NEIGHBORHOOD_SUMMARY_MAXIMUM_NEIGHBORS
    );
    if (categorical) {
      if (!props.categories) {
        throw new Error(`${id} modes and entropy require categories`);
      }
      validateSinglePackedView(props.categories, ['uint32'], `${id} categories`);
      if (props.categories.length !== rows) {
        throw new Error(`${id} categories needs one row per weights row`);
      }
    }
    if (output.modes) {
      validateSinglePackedView(output.modes, ['uint32'], `${id} output.modes`);
      if (output.modes.length !== rows) {
        throw new Error(`${id} output.modes needs one row per weights row`);
      }
    }
    if (output.entropy) {
      validateSinglePackedView(output.entropy, ['float32'], `${id} output.entropy`);
      if (output.entropy.length !== rows) {
        throw new Error(`${id} output.entropy needs one row per weights row`);
      }
    }
    if (props.mask) {
      validateSinglePackedView(props.mask, ['uint32'], `${id} mask`);
      if (props.mask.length !== rows) {
        throw new Error(`${id} mask needs one row per weights row`);
      }
    }
    const chunkLossInputs = chunks?.map((chunk, chunkIndex) => {
      const chunkStatus = snapshotNeighborhoodWeightsStatus(chunk.weights.status);
      const path = `weights.chunks[${chunkIndex}].weights.status`;
      validateNeighborhoodWeightsCertificates(id, path, chunkStatus);
      validateNeighborhoodChunkLossBindings(id, path, chunkStatus);
      return {weights: {status: chunkStatus}};
    });
    const weightsLoss = validateWeightsLossSources(id, [
      {status: props.weights.status, chunks: chunkLossInputs}
    ]);
    this.weightsLossViews = Object.freeze([...weightsLoss]);
    for (const name of NEIGHBORHOOD_STOPPING_FIELDS) {
      const view = props.weights.status?.[name];
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} weights.status.${name}`);
        validateCapacity(id, `weights.status.${name} length`, view.length, 1);
      }
    }
    const focalWeight = props.focalWeight ?? 1;
    if (!Number.isFinite(Math.fround(focalWeight)) || focalWeight < 0) {
      throw new Error(`${id} focalWeight must round to a finite non-negative float32`);
    }
    validateDisjointWritableViews(id, {...output, status: undefined, ...output.status});
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.table, overflow, output.modes, output.entropy],
      [
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights,
        ...weightsLoss,
        ...Object.values(props.weights.status ?? {}),
        props.values,
        props.categories,
        props.mask
      ]
    );
  }

  /** Returns the summary nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {weights, mask, output} = props;
    // Child metadata is caller-owned. Preflight the captured loss references instead of
    // rediscovering views through a chunks array that may have changed since validation.
    validateSpatialStatisticsBindings(graph, id, {
      ...props,
      weights: {...weights, chunks: undefined},
      weightsLossViews: this.weightsLossViews
    });
    validateGraphViewsBelongToGraph(id, graph, [
      weights.offsets,
      weights.neighbors,
      weights.weights,
      ...this.weightsLossViews,
      ...Object.values(weights.status ?? {}),
      props.values,
      props.categories,
      mask,
      output.table,
      output.status?.overflow,
      output.modes,
      output.entropy
    ]);
    const rows = weights.offsets.length - 1;
    const statistics = props.statistics ?? [];
    const includeFocal = Boolean(props.includeFocal);
    // Retained-slot loss and upstream candidate loss (matrix and chunk level) folded into one word.
    const lossView = this.weightsLossViews.length
      ? createTransientView(graph, `${id}-weights-loss`, 'uint32', 1)
      : undefined;
    let truncation = lossView;
    const standardized = isGPUSpatialWeightsRowStandardized(weights);
    const maximumNeighbors = props.maximumNeighbors ?? 32;
    const constants = `const ROWS: u32 = ${rows}u;
const CAPACITY: u32 = ${weights.neighbors.length}u;
const FOCAL_COUNT: u32 = ${includeFocal ? 1 : 0}u;
const FOCAL_WEIGHT: f32 = ${getWGSLFloatLiteral(props.focalWeight ?? 1)};
${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
fn getMemberRange(row: u32) -> vec2<u32> {
  let begin = min(offsets[offsetsOffset + row], CAPACITY);
  let end = min(offsets[offsetsOffset + row + 1u], CAPACITY);
  return vec2<u32>(begin, max(end, begin));
}`;
    const maskedWGSL = (rowExpression: string) =>
      mask ? `mask[maskOffset + ${rowExpression}] != 0u` : 'true';
    // Member `visit` of a row: the focal row first (if requested), then CSR slots in order.
    const getMemberWGSL = (validExpression: (row: string) => string, rawCategories = false) => `
struct Member {
  row: u32,
  weight: ${rawCategories ? 'vec2<f32>' : 'f32'},
  valid: bool
}
fn getMember(row: u32, begin: u32, visit: u32, scale: ${rawCategories ? 'vec2<f32>' : 'f32'}) -> Member {
  var member = Member(row, ${rawCategories ? 'vec2<f32>(FOCAL_WEIGHT, 0.0)' : 'FOCAL_WEIGHT'}, false);
  if (visit >= FOCAL_COUNT) {
    let slot = begin + visit - FOCAL_COUNT;
    member.row = neighbors[neighborsOffset + slot];
    member.weight = ${rawCategories ? 'vec2<f32>(weights[weightsOffset + slot], 0.0)' : 'weights[weightsOffset + slot]'};
    // Only active slots are members: another row with a positive weight.
    if (member.row >= ROWS || member.row == row || !(${rawCategories ? 'member.weight.x' : 'member.weight'} > 0.0)) {
      return member;
    }
    ${rawCategories ? '' : 'member.weight = member.weight * scale;'}
  }
  ${
    rawCategories && standardized
      ? `else if (scale.x > 0.0) {
    // The focal mass and competing category masses retain the same two-lane precision.
    let product = getRoundedDifference(FOCAL_WEIGHT * scale.x, 0.0, offsets[offsetsOffset]);
    let remainder = getMultiplicationRoundingError(FOCAL_WEIGHT, scale.x, product);
    member.weight = addCategoryMass(vec2<f32>(product, remainder),
      vec2<f32>(FOCAL_WEIGHT * scale.y, 0.0), offsets[offsetsOffset]);
  }`
      : ''
  }
  member.valid = ${maskedWGSL('member.row')} && ${validExpression('member.row')};
  return member;
}
// Valid slot-weight mass for categories, its reciprocal for numeric values (1 for raw weights).
fn getRowScale(row: u32, begin: u32, visits: u32) -> ${rawCategories ? 'vec2<f32>' : 'f32'} {
  ${
    standardized
      ? `var sum = ${rawCategories ? 'vec2<f32>(0.0)' : '0.0'};
  for (var visit = FOCAL_COUNT; visit < visits; visit++) {
    let member = getMember(row, begin, visit, ${rawCategories ? 'vec2<f32>(1.0, 0.0)' : '1.0'});
    if (member.valid) {
      ${rawCategories ? 'sum = addCategoryMass(sum, member.weight, offsets[offsetsOffset]);' : 'sum += member.weight;'}
    }
  }
  return ${rawCategories ? 'sum' : 'select(0.0, 1.0 / sum, sum > 0.0)'};`
      : `return ${rawCategories ? 'vec2<f32>(1.0, 0.0)' : '1.0'};`
  }
}`;
    const nodes: GPUCommandNode<Parameters>[] = lossView
      ? createWeightsLossNodes(graph, `${id}-weights-loss`, this.weightsLossViews, lossView)
      : [];
    const flags = (
      [
        ['overflow', lossView],
        ['iterationLimitReached', weights.status?.iterationLimitReached],
        ['converged', weights.status?.converged]
      ] as const
    ).flatMap(([name, view]) => (view ? [[name, view] as const] : []));
    if (flags.some(([name]) => name !== 'overflow')) {
      truncation = createTransientView(graph, `${id}-input-incomplete`, 'uint32', 1);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-input-completeness`,
          operation: OPERATION,
          variant: 'input-completeness',
          invocationCount: 1,
          bindings: [
            ...flags.map(([name, view]) => ({
              name,
              view,
              type: 'u32' as const,
              access: 'read' as const
            })),
            {name: 'incomplete', view: truncation, type: 'u32', access: 'read_write'}
          ],
          body: `incomplete[incompleteOffset] = select(0u, 1u, ${flags.map(([name]) => `${name}[${name}Offset] ${name === 'converged' ? '==' : '!='} 0u`).join(' || ')});`
        })
      );
    }
    if (output.status?.overflow && !statistics.includes('median'))
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-overflow-clear`,
          operation: OPERATION,
          view: output.status.overflow,
          type: 'u32',
          value: '0u',
          componentCount: 1
        })
      );

    if (statistics.length > 0) {
      // uint32 values (counts) are converted to f32 once; float32 values are used as given.
      const conversion = createFloat32ValuesNodes(graph, `${id}-values`, props.values!);
      nodes.push(...conversion.nodes);
      const values = conversion.view;
      const table = output.table!;
      const columns = statistics.length;
      const wants = (statistic: GPUNeighborhoodSummaryStatistic) => statistics.includes(statistic);
      const needsMean = wants('mean') || wants('standardDeviation');
      const bindings: WGSLKernelBinding[] = [
        {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
        {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
        {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
        {name: 'values', view: values, type: 'f32', access: 'read'},
        ...(mask
          ? [{name: 'mask', view: mask, type: 'u32' as const, access: 'read' as const}]
          : []),
        {name: 'output', view: table, type: 'f32', access: 'read_write'},
        ...(truncation
          ? [{name: 'truncation', view: truncation, type: 'u32' as const, access: 'read' as const}]
          : [])
      ];
      if (wants('median')) {
        bindings.push({
          name: 'overflow',
          view: output.status!.overflow!,
          type: 'atomic<u32>',
          access: 'read_write'
        });
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-overflow-clear`,
            operation: OPERATION,
            view: output.status!.overflow!,
            type: 'u32',
            value: '0u',
            componentCount: 1
          })
        );
      }
      const columnWGSL = (statistic: GPUNeighborhoodSummaryStatistic) => {
        switch (statistic) {
          case 'count':
            return 'f32(count)';
          case 'weightSum':
            return 'weightSum';
          case 'sum':
            return 'weightedSum';
          case 'mean':
            return 'select(nan, mean, weightSum > 0.0)';
          case 'min':
            return 'select(nan, minimum, count > 0u)';
          case 'max':
            return 'select(nan, maximum, count > 0u)';
          case 'standardDeviation':
            return 'select(nan, sqrt(squares / relativeWeightSum), weightSum > 0.0)';
          default:
            return 'median';
        }
      };
      // Rows with a zero mask: count 0 and NaN elsewhere.
      const excludedWGSL = statistics
        .map(
          (statistic, column) =>
            `output[outputOffset + index * ${columns}u + ${column}u] = ${statistic === 'count' ? '0.0' : 'nan'};`
        )
        .join('\n    ');
      const truncatedWGSL = statistics
        .map((_, column) => `output[outputOffset + index * ${columns}u + ${column}u] = nan;`)
        .join('\n    ');
      const resultWGSL = statistics
        .map(
          (statistic, column) =>
            `output[outputOffset + index * ${columns}u + ${column}u] = ${columnWGSL(statistic)};`
        )
        .join('\n    ');
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-numeric`,
          operation: OPERATION,
          variant: statistics.join('-'),
          bindings,
          invocationCount: rows,
          declarations: `${constants}
${ROUNDED_DIFFERENCE_WGSL}
const MAXIMUM_NEIGHBORS: u32 = ${maximumNeighbors}u;
${getMemberWGSL(row => `isFiniteFloat(values[valuesOffset + ${row}])`)}`,
          body: `let nan = getQuietNaN(index);
  if (!(${maskedWGSL('index')})) {
    ${excludedWGSL}
    return;
  }
  ${
    truncation
      ? `// Truncated weights: no statistic from them is trustworthy.
  if (truncation[truncationOffset] != 0u) {
    ${truncatedWGSL}
    return;
  }`
      : ''
  }
  let range = getMemberRange(index);
  let visits = range.y - range.x + FOCAL_COUNT;
  let scale = getRowScale(index, range.x, visits);
  var count = 0u;
  var weightSum = 0.0;
  var weightedSum = 0.0;
  var anchor = 0.0;
  var hasAnchor = false;
  var maximumMemberWeight = 0.0;
  var minimum = bitcast<f32>(0x7f7fffffu);
  var maximum = -bitcast<f32>(0x7f7fffffu);
  ${wants('median') ? `var sorted: array<f32, ${maximumNeighbors}>;` : ''}
  for (var visit = 0u; visit < visits; visit++) {
    let member = getMember(index, range.x, visit, scale);
    if (!member.valid) {
      continue;
    }
    let value = values[valuesOffset + member.row];
    if (member.weight > 0.0) {
      if (!hasAnchor) { anchor = value; hasAnchor = true; }
      maximumMemberWeight = max(maximumMemberWeight, member.weight);
    }
    weightSum += member.weight;
    weightedSum += member.weight * value;
    minimum = min(minimum, value);
    maximum = max(maximum, value);
    ${
      wants('median')
        ? `if (count < MAXIMUM_NEIGHBORS) {
      var position = count;
      loop {
        if (position == 0u || sorted[position - 1u] <= value) {
          break;
        }
        sorted[position] = sorted[position - 1u];
        position--;
      }
      sorted[position] = value;
    }`
        : ''
    }
    count++;
  }
  ${needsMean ? 'let mean = weightedSum / weightSum;' : ''}
  var squares = 0.0;
  ${
    wants('standardDeviation')
      ? `// The same power-of-two frame applies to relative sums, central squares and W.
  // ldexp avoids a reciprocal of a huge weight, which Metal may flush to zero.
  let weightExponent = i32((bitcast<u32>(maximumMemberWeight) >> 23u) & 255u) - 126;
  var relativeSum = 0.0;
  var relativeWeightSum = 0.0;
  for (var visit = 0u; visit < visits; visit++) {
    let member = getMember(index, range.x, visit, scale);
    if (member.valid && member.weight > 0.0) {
      let relativeWeight = ldexp(member.weight, -weightExponent);
      relativeWeightSum += relativeWeight;
      relativeSum += relativeWeight * getRoundedDifference(values[valuesOffset + member.row], anchor, offsets[offsetsOffset]);
    }
  }
  let relativeMean = relativeSum / relativeWeightSum;
  for (var visit = 0u; visit < visits; visit++) {
    let member = getMember(index, range.x, visit, scale);
    if (member.valid && member.weight > 0.0) {
      let centered = getRoundedDifference(values[valuesOffset + member.row], anchor, offsets[offsetsOffset]);
      let deviation = getRoundedDifference(centered, relativeMean, offsets[offsetsOffset]);
      squares += ldexp(member.weight, -weightExponent) * deviation * deviation;
    }
  }`
      : ''
  }
  ${
    wants('median')
      ? `var median = nan;
  if (count > MAXIMUM_NEIGHBORS) {
    atomicMax(&overflow[overflowOffset], 1u);
  } else if (count > 0u) {
    let middle = count / 2u;
    if ((count & 1u) == 1u) { median = sorted[middle]; }
    else if (max(abs(sorted[middle - 1u]), abs(sorted[middle])) > 1.5e38) {
      // offsets[0] is the CSR's runtime zero; the XOR prevents Metal reassociating
      // the two half-values back into an overflowing sum before division by two.
      let firstHalf = bitcast<f32>(bitcast<u32>(0.5 * sorted[middle - 1u]) ^ offsets[offsetsOffset]);
      median = firstHalf + 0.5 * sorted[middle];
    } else { median = 0.5 * (sorted[middle - 1u] + sorted[middle]); }
  }`
      : ''
  }
  ${resultWGSL}`
        })
      );
    }

    if (output.modes || output.entropy) {
      const bindings: WGSLKernelBinding[] = [
        {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
        {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
        {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
        {name: 'categories', view: props.categories!, type: 'u32', access: 'read'},
        ...(mask
          ? [{name: 'mask', view: mask, type: 'u32' as const, access: 'read' as const}]
          : []),
        ...(output.modes
          ? [
              {
                name: 'modes',
                view: output.modes,
                type: 'u32' as const,
                access: 'read_write' as const
              }
            ]
          : []),
        ...(output.entropy
          ? [
              {
                name: 'entropy',
                view: output.entropy,
                type: 'f32' as const,
                access: 'read_write' as const
              }
            ]
          : []),
        ...(truncation
          ? [{name: 'truncation', view: truncation, type: 'u32' as const, access: 'read' as const}]
          : [])
      ];
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-categorical`,
          operation: OPERATION,
          variant: `categorical${output.modes ? '-mode' : ''}${output.entropy ? '-entropy' : ''}`,
          bindings,
          invocationCount: rows,
          declarations: `${constants}
${ROUNDED_DIFFERENCE_WGSL}
${ADDITION_ROUNDING_ERROR_WGSL}
${MULTIPLICATION_ROUNDING_ERROR_WGSL}
const NO_MODE: u32 = ${GPU_NEIGHBORHOOD_SUMMARY_NO_MODE}u;
const MAXIMUM_NEIGHBORS: u32 = ${maximumNeighbors}u;
fn addCategoryMass(total: vec2<f32>, value: vec2<f32>, zero: u32) -> vec2<f32> {
  let next = getRoundedDifference(total.x + value.x, 0.0, zero);
  let remainder = total.y + value.y + getAdditionRoundingError(total.x, value.x, next);
  let high = getRoundedDifference(next + remainder, 0.0, zero);
  return vec2<f32>(high, getAdditionRoundingError(next, remainder, high));
}
fn isGreaterMass(first: vec2<f32>, second: vec2<f32>) -> bool {
  return first.x > second.x || (first.x == second.x && first.y > second.y);
}
// Sum raw category masses before a common normalization. Focal mass is converted
// into the same units, preserving exact ties between differently partitioned slots.
${getMemberWGSL(row => `categories[categoriesOffset + ${row}] != NO_MODE`, true)}`,
          body: `let nan = getQuietNaN(index);
  if (!(${maskedWGSL('index')})) {
    ${output.modes ? 'modes[modesOffset + index] = NO_MODE;' : ''}
    ${output.entropy ? 'entropy[entropyOffset + index] = nan;' : ''}
    return;
  }
  ${
    truncation
      ? `// Truncated weights: no statistic from them is trustworthy.
  if (truncation[truncationOffset] != 0u) {
    ${output.modes ? 'modes[modesOffset + index] = NO_MODE;' : ''}
    ${output.entropy ? 'entropy[entropyOffset + index] = nan;' : ''}
    return;
  }`
      : ''
  }
  let range = getMemberRange(index);
  let visits = range.y - range.x + FOCAL_COUNT;
  let scale = getRowScale(index, range.x, visits);
  // Decode the members once. Up to MAXIMUM_NEIGHBORS of them are kept in private arrays so the
  // quadratic frequency scan below reads registers instead of re-decoding every member (four
  // global loads) for every pair; larger rows take the unbounded decoding scan.
  var memberCategories: array<u32, ${maximumNeighbors}>;
  var memberWeights: array<vec2<f32>, ${maximumNeighbors}>;
  var memberCount = 0u;
  var weightSum = vec2<f32>(0.0);
  for (var visit = 0u; visit < visits; visit++) {
    let member = getMember(index, range.x, visit, scale);
    if (member.valid) {
      weightSum = addCategoryMass(weightSum, member.weight, offsets[offsetsOffset]);
      if (memberCount < MAXIMUM_NEIGHBORS) {
        memberCategories[memberCount] = categories[categoriesOffset + member.row];
        memberWeights[memberCount] = member.weight;
      }
      memberCount++;
    }
  }
  var bestCategory = NO_MODE;
  var bestFrequency = vec2<f32>(-1.0, 0.0);
  var entropyTotal = 0.0;
  if (memberCount <= MAXIMUM_NEIGHBORS) {
    for (var member = 0u; member < memberCount; member++) {
      let category = memberCategories[member];
      // Frequency of this category, summed in member order, at its first occurrence only.
      var frequency = vec2<f32>(0.0);
      var firstOccurrence = true;
      for (var other = 0u; other < memberCount; other++) {
        if (memberCategories[other] == category) {
          if (other < member) {
            firstOccurrence = false;
            break;
          }
          frequency = addCategoryMass(frequency, memberWeights[other], offsets[offsetsOffset]);
        }
      }
      if (!firstOccurrence) {
        continue;
      }
      if (isGreaterMass(frequency, bestFrequency) || (all(frequency == bestFrequency) && category < bestCategory)) {
        bestFrequency = frequency;
        bestCategory = category;
      }
      if (frequency.x > 0.0) {
        let share = (frequency.x + frequency.y) / (weightSum.x + weightSum.y);
        entropyTotal -= share * log(share);
      }
    }
  } else {
    for (var visit = 0u; visit < visits; visit++) {
      let member = getMember(index, range.x, visit, scale);
      if (!member.valid) {
        continue;
      }
      let category = categories[categoriesOffset + member.row];
      // Frequency of this category, summed in member order, at its first occurrence only.
      var frequency = vec2<f32>(0.0);
      var firstOccurrence = true;
      for (var other = 0u; other < visits; other++) {
        let candidate = getMember(index, range.x, other, scale);
        if (candidate.valid && categories[categoriesOffset + candidate.row] == category) {
          if (other < visit) {
            firstOccurrence = false;
            break;
          }
          frequency = addCategoryMass(frequency, candidate.weight, offsets[offsetsOffset]);
        }
      }
      if (!firstOccurrence) {
        continue;
      }
      if (isGreaterMass(frequency, bestFrequency) || (all(frequency == bestFrequency) && category < bestCategory)) {
        bestFrequency = frequency;
        bestCategory = category;
      }
      if (frequency.x > 0.0) {
        let share = (frequency.x + frequency.y) / (weightSum.x + weightSum.y);
        entropyTotal -= share * log(share);
      }
    }
  }
  ${output.modes ? 'modes[modesOffset + index] = bestCategory;' : ''}
  ${output.entropy ? 'entropy[entropyOffset + index] = select(nan, entropyTotal, weightSum.x > 0.0);' : ''}`
        })
      );
    }
    return nodes;
  }
}
