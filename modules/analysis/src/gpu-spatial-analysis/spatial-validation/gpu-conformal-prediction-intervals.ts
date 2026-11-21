// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRegressionAllocations} from '../spatial-regression/spatial-likelihood-allocations';

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createStableLexicographicIndexSortNodes} from '../../analysis-core/kernels/stable-lexicographic-index-sort';
import {
  validateCapacity,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateRegressionOutputDispatches} from '../spatial-regression/regression-output-dispatches';
import {validateGraphOutputsDisjointFromInputs} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  createConformalScoreNodes,
  snapshotConformalPredictionProps,
  getConformalInputViews,
  GPU_CONFORMAL_PREDICTION_PARAMETER_LENGTH,
  validateConformalInputs,
  type GPUConformalCalibration,
  type GPUConformalQuery
} from './conformal-prediction-nodes';
import {
  CONFORMAL_PREDICTION_RANKS_WGSL,
  GPU_CONFORMAL_MAXIMUM_CALIBRATION_COUNT
} from './conformal-prediction-ranks';

/** Interval status bits: the lower bound is unbounded (minus infinity). */
export const GPU_CONFORMAL_STATUS_LOWER_UNBOUNDED = 1;
/** Interval status bits: the upper bound is unbounded (plus infinity). */
export const GPU_CONFORMAL_STATUS_UPPER_UNBOUNDED = 2;
/** Interval status bits: a query prediction or scale is not usable; the interval is NaN. */
export const GPU_CONFORMAL_STATUS_INVALID_QUERY = 4;
/** Interval status bits: no calibration row (or neighbour weight) is usable; the interval is NaN. */
export const GPU_CONFORMAL_STATUS_NO_CALIBRATION = 8;
/** Supplied CSR span exceeds the internal 256-entry cap; its tail was dropped. Distinct from upstream input loss. */
export const GPU_CONFORMAL_STATUS_NEIGHBORS_TRUNCATED = 16;
/** Upstream calibration-neighbor loss; the localized interval is suppressed to NaN. */
export const GPU_CONFORMAL_STATUS_INPUT_INCOMPLETE = 32;

/** Decodes independent bits of one row, or the first word of a validated uint32 readback. */
export function decodeConformalStatus(input: number | Uint32Array | Uint8Array | ArrayBuffer) {
  let word: number;
  if (typeof input === 'number') {
    word = input;
  } else {
    let bytes: Uint8Array;
    if (ArrayBuffer.isView(input)) {
      const kind = Object.prototype.toString.call(input);
      if (kind !== '[object Uint32Array]' && kind !== '[object Uint8Array]') {
        throw new Error(
          'decodeConformalStatus: readback must be Uint32Array, Uint8Array or ArrayBuffer'
        );
      }
      bytes = new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    } else {
      // The intrinsic getter checks the ArrayBuffer internal slot across realms.
      try {
        const byteLength = Object.getOwnPropertyDescriptor(
          ArrayBuffer.prototype,
          'byteLength'
        )!.get!.call(input) as number;
        bytes = new Uint8Array(input as ArrayBuffer, 0, byteLength);
      } catch {
        throw new Error(
          'decodeConformalStatus: readback must be Uint32Array, Uint8Array or ArrayBuffer'
        );
      }
    }
    if (bytes.byteLength < 4 || bytes.byteLength % 4 !== 0) {
      throw new Error(
        'decodeConformalStatus: readback must contain whole uint32 words and at least one word'
      );
    }
    // DataView also accepts a byte view whose offset is not uint32 aligned.
    word = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(0, true);
  }
  if (!Number.isInteger(word) || word < 0 || word > 0xffffffff) {
    throw new Error('decodeConformalStatus: word must be a uint32');
  }
  return {
    lowerUnbounded: Boolean(word & GPU_CONFORMAL_STATUS_LOWER_UNBOUNDED),
    upperUnbounded: Boolean(word & GPU_CONFORMAL_STATUS_UPPER_UNBOUNDED),
    invalidQuery: Boolean(word & GPU_CONFORMAL_STATUS_INVALID_QUERY),
    noCalibration: Boolean(word & GPU_CONFORMAL_STATUS_NO_CALIBRATION),
    neighborsTruncated: Boolean(word & GPU_CONFORMAL_STATUS_NEIGHBORS_TRUNCATED),
    inputIncomplete: Boolean(word & GPU_CONFORMAL_STATUS_INPUT_INCOMPLETE)
  };
}

const OPERATION = 'GPUConformalPredictionIntervals';

/** @internal */
export const INFINITY_WGSL = /* wgsl */ `
fn getInfinity() -> f32 { var bits = 0x7f800000u; return bitcast<f32>(bits); }
`;

const ORDERED_KEY_WGSL = /* wgsl */ `
fn getOrderedKey(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  // Numerical ties include both signed zeros. Do not use a float comparison here:
  // Metal may also compare a subnormal value as zero.
  if ((bits & 0x7fffffffu) == 0u) { return 0x80000000u; }
  return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
fn getKeyedValue(key: u32) -> f32 {
  return bitcast<f32>(select(~key, key & 0x7fffffffu, (key & 0x80000000u) != 0u));
}
`;

/** Caller-owned outputs of {@link GPUConformalPredictionIntervals}. */
export type GPUConformalPredictionIntervalsOutput = {
  /** `[lower, upper]` per query row; NaN where the status has the invalid or empty bit. */
  intervals: GraphDataView<'float32'>;
  /** Status bits per query row, see the `GPU_CONFORMAL_STATUS_*` constants. */
  status: GraphDataView<'uint32'>;
  /** Number of usable calibration rows. */
  calibrationCount: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUConformalPredictionIntervals}.
 *
 * Topology: `foldCount`, the optional views and the row counts. Per-frame: `parameters`
 * (miscoverage, see `getGPUConformalPredictionParameterValues`) and every buffer's contents.
 */
export type GPUConformalPredictionIntervalsProps = {
  id?: string;
  /** Calibration rows with held-out predictions. */
  calibration: GPUConformalCalibration;
  /** Query rows. */
  query: GPUConformalQuery;
  /**
   * Number of models. 1 is split (holdout) conformal; more is CV+, with one model per fold, for
   * example the folds of `GPUSpatialCrossValidatedOLS`. Default 1.
   */
  foldCount?: number;
  /** At least 4 float32 values from `getGPUConformalPredictionParameterValues`. */
  parameters: GraphDataView<'float32'>;
  output: GPUConformalPredictionIntervalsOutput;
};

/**
 * Split-conformal and CV+ prediction intervals (Vovk et al. 2005; Barber, Candes, Ramdas and
 * Tibshirani 2021) from held-out predictions. Scores are `R_i = |y_i - prediction_i| / scale_i`.
 * With `n` usable calibration rows the interval for a query with model predictions `p_k` and scale
 * `s` is `[l, u]`, with `l` the `floor(alpha (n + 1))`-th smallest of `p_k(i) - s R_i` and `u` the
 * `ceil((1 - alpha)(n + 1))`-th smallest of `p_k(i) + s R_i`, where `k(i)` is the fold of row `i`.
 * For one fold this is the split interval `p -+ s R_(rank)`. A rank outside `[1, n]` gives an
 * unbounded side and a status bit.
 *
 * Calibration scores are sorted once by (fold, score) with the shared stable radix sort. Each
 * query finds its exact order statistic by bisecting the float32 keys of the candidate values
 * with one binary search per fold, so no loop runs over the calibration rows. Ranks are computed
 * exactly from the packed float32 alpha and integer count, without snapping decimal boundaries
 * (see `getConformalPredictionRanks`). The count plus one must fit uint32.
 *
 * Coverage is conditional on statistical assumptions, which these input views cannot verify.
 * Split conformal requires exchangeable calibration/test scores from a common scoring rule
 * fitted independently of those rows: coverage is at least `1 - alpha`, and at most
 * `1 - alpha + 1/(n + 1)` for almost surely distinct scores. Ties can increase coverage.
 * The CV+ theorem requires balanced folds fixed independently of the observations and
 * permutation-invariant fitting on each training complement; its finite-sample bound depends
 * on calibration size and fold count. Arbitrary group/spatial folds do not inherit a universal
 * `1 - 2 alpha` guarantee from row exchangeability alone. This class computes endpoints only.
 * The global route accepts 1–64 folds and at most 4,294,967,294 calibration rows before
 * the device binding and sort limits are applied.
 */
export class GPUConformalPredictionIntervals implements GPUCommandNodeProducer {
  readonly id: string;
  readonly foldCount: number;
  readonly queryRows: number;
  readonly props: GPUConformalPredictionIntervalsProps;
  constructor(props: GPUConformalPredictionIntervalsProps) {
    props = snapshotConformalPredictionProps(props);
    this.props = props;
    this.id = props.id ?? 'conformal-prediction-intervals';
    const {id} = this;
    this.foldCount = props.foldCount ?? 1;
    validateCapacity(id, 'foldCount', this.foldCount, 1, 64);
    validateSinglePackedView(props.query.predictions, ['float32'], `${id} query.predictions`);
    validateCapacity(id, 'query rows', props.query.predictions.length / this.foldCount, 1);
    this.queryRows = props.query.predictions.length / this.foldCount;
    validateConformalInputs(id, props.calibration, props.query, this.foldCount, this.queryRows);
    validateCapacity(
      id,
      'calibration rows',
      props.calibration.observations.length,
      1,
      GPU_CONFORMAL_MAXIMUM_CALIBRATION_COUNT
    );
    validateConformalOutput(id, props.parameters, props.output, this.queryRows);
    validateSinglePackedView(
      props.output.calibrationCount,
      ['uint32'],
      `${id} output.calibrationCount`
    );
    validateViewLength(id, 'output.calibrationCount', props.output.calibrationCount, 1);
    validateRegressionOutputDispatches(this.id, props.output, [['intervals', 'status']]);
    validateGraphOutputsDisjointFromInputs(id, Object.values(props.output), [
      ...getConformalInputViews(props.calibration, props.query),
      props.parameters
    ]);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, foldCount, queryRows} = this;
    const rows = props.calibration.observations.length;
    validateGraphChunkBindingRanges(graph, id, {
      ...Object.fromEntries(
        getConformalInputViews(props.calibration, props.query).map((view, index) => [
          `input ${index}`,
          view
        ])
      ),
      parameters: props.parameters,
      ...props.output
    });
    validateRegressionAllocations(graph.device, id, {
      'conformal scores, keys, sorted rows and published sort/scan scratch': rows,
      'conformal fold offsets': foldCount + 1
    });
    validateGraphViewsBelongToGraph(id, graph, [
      ...getConformalInputViews(props.calibration, props.query),
      props.parameters,
      ...Object.values(props.output)
    ]);
    const scored = createConformalScoreNodes(graph, id, props.calibration, foldCount);
    const sorted = createStableLexicographicIndexSortNodes(graph, {
      id: `${id}-sort`,
      operation: OPERATION,
      indices: scored.indices,
      keys: [
        {view: scored.views.foldKeys, keyBits: Math.max(1, Math.ceil(Math.log2(foldCount + 1)))},
        {view: scored.scoreKeys, keyBits: 32}
      ]
    });
    const sortedScores = createTransientView<'float32', Parameters>(
      graph,
      `${id}-sorted-scores`,
      'float32',
      rows
    );
    const foldOffsets = createTransientView<'uint32', Parameters>(
      graph,
      `${id}-fold-offsets`,
      'uint32',
      foldCount + 1
    );
    return [
      ...scored.nodes,
      ...sorted.nodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-gather`,
        operation: OPERATION,
        variant: 'sorted-scores',
        invocationCount: rows,
        bindings: [
          {name: 'scoreKeys', view: scored.scoreKeys, type: 'u32', access: 'read'},
          {name: 'sortedIndices', view: sorted.sortedIndices, type: 'u32', access: 'read'},
          {name: 'sortedScores', view: sortedScores, type: 'f32', access: 'read_write'}
        ],
        body: `sortedScores[sortedScoresOffset + index] = bitcast<f32>(scoreKeys[scoreKeysOffset + sortedIndices[sortedIndicesOffset + index]]);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-offsets`,
        operation: OPERATION,
        variant: 'fold-offsets',
        invocationCount: foldCount + 1,
        bindings: [
          {name: 'sortedFolds', view: sorted.sortedPrimaryKeys, type: 'u32', access: 'read'},
          {name: 'foldOffsets', view: foldOffsets, type: 'u32', access: 'read_write'},
          {
            name: 'calibrationCount',
            view: props.output.calibrationCount,
            type: 'u32',
            access: 'read_write'
          }
        ],
        // First position whose fold is at least `index`: the start of that fold's segment.
        body: `var low = 0u;
  var high = ${rows}u;
  for (var step = 0u; step < 32u; step++) {
    if (low >= high) { break; }
    let middle = low + (high - low) / 2u;
    if (sortedFolds[sortedFoldsOffset + middle] < index) { low = middle + 1u; } else { high = middle; }
  }
  foldOffsets[foldOffsetsOffset + index] = low;
  if (index == ${foldCount}u) { calibrationCount[calibrationCountOffset] = low; }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-intervals`,
        operation: OPERATION,
        variant: 'order-statistics',
        invocationCount: queryRows,
        bindings: [
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'sortedScores', view: sortedScores, type: 'f32', access: 'read'},
          {name: 'foldOffsets', view: foldOffsets, type: 'u32', access: 'read'},
          {name: 'queryPredictions', view: props.query.predictions, type: 'f32', access: 'read'},
          ...(props.query.scales
            ? [
                {
                  name: 'queryScales',
                  view: props.query.scales,
                  type: 'f32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'intervals', view: props.output.intervals, type: 'f32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `${IS_FINITE_BITS_WGSL}
${GET_NAN_WGSL}
${INFINITY_WGSL}
${ORDERED_KEY_WGSL}
${CONFORMAL_PREDICTION_RANKS_WGSL}
fn getShifted(prediction: f32, scale: f32, score: f32, sign: f32) -> f32 {
  return prediction + sign * (scale * score);
}
// Values of one fold at or below the threshold. Upper values rise with the sorted score and
// lower values fall, so each count is one binary search over the fold's sorted segment.
fn countAtOrBelow(fold: u32, threshold: u32, prediction: f32, scale: f32, sign: f32) -> u32 {
  let first = foldOffsets[foldOffsetsOffset + fold];
  let last = foldOffsets[foldOffsetsOffset + fold + 1u];
  var low = first;
  var high = last;
  for (var step = 0u; step < 32u; step++) {
    if (low >= high) { break; }
    let middle = low + (high - low) / 2u;
    let value = getOrderedKey(getShifted(prediction, scale, sortedScores[sortedScoresOffset + middle], sign));
    // Advance past the values that are at or below the threshold (rising values) or above it (falling values).
    let goRight = select(value > threshold, value <= threshold, sign > 0.0);
    if (goRight) { low = middle + 1u; } else { high = middle; }
  }
  return select(last - low, low - first, sign > 0.0);
}
fn getTotalCount(threshold: u32, query: u32, scale: f32, sign: f32) -> u32 {
  var total = 0u;
  for (var fold = 0u; fold < ${foldCount}u; fold++) {
    total += countAtOrBelow(fold, threshold, queryPredictions[queryPredictionsOffset + fold * ${queryRows}u + query], scale, sign);
  }
  return total;
}
// Smallest candidate value whose count reaches the rank, by bisection over ordered float32 keys.
// Compare keys throughout: decoding a subnormal threshold for a float comparison can
// flush it to zero on Metal and return a value that is not a candidate.
fn getOrderStatistic(rank: u32, query: u32, scale: f32, sign: f32) -> f32 {
  // Keys of minus and plus infinity: NaN keys outside the range would break the monotone count.
  var low = 0x007fffffu;
  var high = 0xff800000u;
  for (var step = 0u; step < 32u; step++) {
    if (low >= high) { break; }
    let middle = low + (high - low) / 2u;
    if (getTotalCount(middle, query, scale, sign) >= rank) { high = middle; } else { low = middle + 1u; }
  }
  return getKeyedValue(low);
}`,
        body: `let miscoverage = parameters[parametersOffset];
  let count = foldOffsets[foldOffsetsOffset + ${foldCount}u];
  let scale = ${props.query.scales ? 'queryScales[queryScalesOffset + index]' : '1.0'};
  var valid = isFiniteBits(scale) && scale > 0.0;
  for (var fold = 0u; fold < ${foldCount}u; fold++) {
    valid = valid && isFiniteBits(queryPredictions[queryPredictionsOffset + fold * ${queryRows}u + index]);
  }
  let lowerBase = intervalsOffset + index * 2u;
  var code = 0u;
  var lower = getNaN();
  var upper = getNaN();
  if (!valid) {
    code = ${GPU_CONFORMAL_STATUS_INVALID_QUERY}u;
  } else if (count == 0u) {
    code = ${GPU_CONFORMAL_STATUS_NO_CALIBRATION}u;
  } else {
    let ranks = getConformalPredictionRanks(count, miscoverage);
    let lowerRank = ranks.x;
    let upperRank = ranks.y;
    if (upperRank > count) {
      upper = getInfinity();
      code |= ${GPU_CONFORMAL_STATUS_UPPER_UNBOUNDED}u;
    } else {
      upper = getOrderStatistic(upperRank, index, scale, 1.0);
    }
    if (lowerRank < 1u) {
      lower = -getInfinity();
      code |= ${GPU_CONFORMAL_STATUS_LOWER_UNBOUNDED}u;
    } else {
      lower = getOrderStatistic(lowerRank, index, scale, -1.0);
    }
  }
  intervals[lowerBase] = lower;
  intervals[lowerBase + 1u] = upper;
  status[statusOffset + index] = code;`
      })
    ];
  }
}

/** @internal */
export function validateConformalOutput(
  id: string,
  parameters: GraphDataView<'float32'>,
  output: Pick<GPUConformalPredictionIntervalsOutput, 'intervals' | 'status'>,
  queryRows: number
): void {
  validateSinglePackedView(parameters, ['float32'], `${id} parameters`);
  if (parameters.length < GPU_CONFORMAL_PREDICTION_PARAMETER_LENGTH) {
    throw new Error(`${id} parameters must have at least 4 values`);
  }
  validateSinglePackedView(output.intervals, ['float32'], `${id} output.intervals`);
  validateViewLength(id, 'output.intervals', output.intervals, queryRows * 2);
  validateSinglePackedView(output.status, ['uint32'], `${id} output.status`);
  validateViewLength(id, 'output.status', output.status, queryRows);
}
