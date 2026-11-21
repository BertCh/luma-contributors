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
import {
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateRegressionOutputDispatches} from '../spatial-regression/regression-output-dispatches';
import {validateGraphOutputsDisjointFromInputs} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  createConformalScoreNodes,
  snapshotConformalPredictionProps,
  getConformalInputViews,
  validateConformalInputs,
  type GPUConformalCalibration,
  type GPUConformalQuery
} from './conformal-prediction-nodes';
import {CONFORMAL_PREDICTION_RANKS_WGSL} from './conformal-prediction-ranks';
import {validateWeightsLossSources} from '../spatial-weights/spatial-weights-loss';
import {
  GPU_CONFORMAL_STATUS_INVALID_QUERY,
  GPU_CONFORMAL_STATUS_INPUT_INCOMPLETE,
  GPU_CONFORMAL_STATUS_LOWER_UNBOUNDED,
  GPU_CONFORMAL_STATUS_NEIGHBORS_TRUNCATED,
  GPU_CONFORMAL_STATUS_NO_CALIBRATION,
  GPU_CONFORMAL_STATUS_UPPER_UNBOUNDED,
  INFINITY_WGSL,
  validateConformalOutput
} from './gpu-conformal-prediction-intervals';

/** Most neighbours of one query the localized contributor weighs. */
export const GPU_GEO_CONFORMAL_MAXIMUM_NEIGHBORS = 256;

// Relative tolerance for the localized approximate weighted quantile, whose sums use float32.
// Global split/CV+ order-statistic ranks instead use exact integer arithmetic.
const GEO_CONFORMAL_WEIGHT_SLACK = 2 ** -20;

/** Neighbour lists of the localized contributor, in compressed sparse row form. */
export type GPUGeoConformalNeighbors = {
  /** Optional producer evidence. A nonzero loss flag or zero certificate suppresses every interval. */
  status?: {
    overflow?: GraphDataView<'uint32'>;
    candidateOverflow?: GraphDataView<'uint32'>;
    /** One when the entire promised neighbor result is present. */
    complete?: GraphDataView<'uint32'>;
    /** Unsupported dynamic requirement; verify it on the host before supplying the CSR. */
    requiredCount?: GraphDataView<'uint32'>;
  };
  /** `queryRows + 1` offsets into `indices` and `weights`. */
  offsets: GraphDataView<'uint32'>;
  /** Calibration row of each neighbour entry. An entry that is out of range or not usable is skipped. */
  indices: GraphDataView<'uint32'>;
  /** Kernel weight of each entry, non-negative; a non-finite or negative weight skips the entry. */
  weights: GraphDataView<'float32'>;
};

/** Caller-owned outputs of {@link GPUGeoConformalPredictionIntervals}. */
export type GPUGeoConformalPredictionIntervalsOutput = {
  /** `[lower, upper]` per query row; NaN for invalid/empty queries or incomplete upstream neighbors. */
  intervals: GraphDataView<'float32'>;
  /** Status bits per query row, see the `GPU_CONFORMAL_STATUS_*` constants. */
  status: GraphDataView<'uint32'>;
};

/** Properties for {@link GPUGeoConformalPredictionIntervals}. */
export type GPUGeoConformalPredictionIntervalsProps = {
  id?: string;
  /** Calibration rows with held-out predictions of one model (`foldCount` 1). */
  calibration: GPUConformalCalibration;
  /** Query rows. */
  query: GPUConformalQuery;
  /** Spatial neighbours of each query among the calibration rows, with kernel weights. */
  neighbors: GPUGeoConformalNeighbors;
  /** At least 4 float32 values from `getGPUConformalPredictionParameterValues`, which also sets `selfWeight`. */
  parameters: GraphDataView<'float32'>;
  output: GPUGeoConformalPredictionIntervalsOutput;
};

/**
 * Spatially weighted (localized) split-conformal intervals, the GeoConformal prediction of Lou,
 * Luo and Meng (2024) built on weighted conformal prediction (Tibshirani, Barber, Candes and
 * Ramdas 2019). The query's nonconformity quantile is the weighted quantile of the neighbouring
 * calibration scores with the query's own weight, `selfWeight`, as a point at infinity: the
 * smallest score `s` with `sum_{R_j <= s} w_j >= (1 - alpha) (sum_j w_j + selfWeight)`, or an
 * unbounded interval if there is none. The interval is `p -+ scale s`.
 *
 * With positive equal weights `w_j = selfWeight` this uses the exact packed-alpha rank of
 * {@link GPUConformalPredictionIntervals} over the same usable entries, including unbounded
 * sides and their status bits. At most
 * {@link GPU_GEO_CONFORMAL_MAXIMUM_NEIGHBORS} = 256 entries per query are used; supplied spans
 * above this cap set NEIGHBORS_TRUNCATED (bit16) and retain only their first 256 entries. Nonzero
 * producer loss flags (`neighbors.status.overflow` or `candidateOverflow`) or a zero
 * `neighbors.status.complete` certificate suppress the interval
 * to NaN, clear the unbounded-endpoint bits and set INPUT_INCOMPLETE (bit32), independently of
 * the internal span-cap bit. Verify any producer requiredCount against the slot capacity before
 * construction and omit that port. If the producer supplies `converged`,
 * `iterationLimitReached` or `unresolvedCount`, verify the lists are complete and omit those
 * views. Clearing the loss flags and restoring any supplied certificate allows the next
 * submission to compute normally. Positive subnormal
 * weight bits are preserved; uniformly tiny or large masses receive a common power-of-two
 * normalization before summing in float32 list order. Non-uniform weights give a level accurate to float32
 * rounding only. Localized coverage is approximate and requires weights reflecting the
 * exchangeability structure. The global variant's finite-sample claims also require assumptions:
 * split conformal needs exchangeable scores from an independently fitted scoring rule; CV+
 * needs exchangeable observations, balanced, data-independent folds and permutation-invariant fitting.
 * Its finite-sample bound depends on fold count. Arbitrary spatial or group folds do not
 * automatically inherit a universal coverage guarantee.
 */
export class GPUGeoConformalPredictionIntervals implements GPUCommandNodeProducer {
  readonly id: string;
  readonly queryRows: number;
  readonly props: GPUGeoConformalPredictionIntervalsProps;
  constructor(props: GPUGeoConformalPredictionIntervalsProps) {
    props = snapshotConformalPredictionProps(props);
    this.props = props;
    this.id = props.id ?? 'geo-conformal-prediction-intervals';
    const {id} = this;
    validateRequirement(
      id,
      !props.neighbors.status?.requiredCount,
      'neighbors.status.requiredCount cannot be checked by this interval contract; verify requiredCount fits the slot capacity, omit that port, and supply overflow and candidateOverflow loss flags'
    );
    const status = props.neighbors.status;
    for (const field of ['converged', 'iterationLimitReached', 'unresolvedCount'] as const) {
      validateRequirement(
        id,
        !status || Reflect.get(status, field) === undefined,
        `neighbors.status.${field} cannot be checked by this interval contract; verify the neighbor lists are complete and omit this port`
      );
    }
    this.queryRows = props.query.predictions.length;
    validateConformalInputs(id, props.calibration, props.query, 1, this.queryRows);
    validateSinglePackedView(props.neighbors.offsets, ['uint32'], `${id} neighbors.offsets`);
    validateViewLength(id, 'neighbors.offsets', props.neighbors.offsets, this.queryRows + 1);
    validateSinglePackedView(props.neighbors.indices, ['uint32'], `${id} neighbors.indices`);
    validateSinglePackedView(props.neighbors.weights, ['float32'], `${id} neighbors.weights`);
    validateViewLength(
      id,
      'neighbors.weights',
      props.neighbors.weights,
      props.neighbors.indices.length
    );
    validateConformalOutput(id, props.parameters, props.output, this.queryRows);
    const lossViews = validateWeightsLossSources(id, [props.neighbors], true);
    validateRegressionOutputDispatches(this.id, props.output, [['intervals', 'status']]);
    validateGraphOutputsDisjointFromInputs(id, Object.values(props.output), [
      ...getConformalInputViews(props.calibration, props.query),
      props.neighbors.offsets,
      props.neighbors.indices,
      props.neighbors.weights,
      ...lossViews,
      props.parameters
    ]);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, queryRows} = this;
    const {neighbors} = props;
    const lossViews = validateWeightsLossSources(id, [neighbors], true);
    const lossFields = (['overflow', 'candidateOverflow', 'complete'] as const).filter(
      name => neighbors.status?.[name]
    );
    validateGraphChunkBindingRanges(graph, id, {
      ...Object.fromEntries(
        getConformalInputViews(props.calibration, props.query).map((view, index) => [
          `input ${index}`,
          view
        ])
      ),
      neighborOffsets: neighbors.offsets,
      neighborIndices: neighbors.indices,
      neighborWeights: neighbors.weights,
      ...Object.fromEntries(lossViews.map((view, index) => [`loss ${index}`, view])),
      parameters: props.parameters,
      ...props.output
    });
    validateRegressionAllocations(graph.device, id, {
      'conformal calibration scores and keys': props.calibration.observations.length,
      'conformal neighbor scores and weights': neighbors.indices.length
    });
    validateGraphViewsBelongToGraph(id, graph, [
      ...getConformalInputViews(props.calibration, props.query),
      neighbors.offsets,
      neighbors.indices,
      neighbors.weights,
      ...lossViews,
      props.parameters,
      ...Object.values(props.output)
    ]);
    const rows = props.calibration.observations.length;
    const scored = createConformalScoreNodes(graph, id, props.calibration, 1);
    const maximum = GPU_GEO_CONFORMAL_MAXIMUM_NEIGHBORS;
    const entryCount = neighbors.indices.length;
    const entryScores = createTransientView<'float32', Parameters>(
      graph,
      `${id}-entry-scores`,
      'float32',
      entryCount
    );
    const entryWeights = createTransientView<'uint32', Parameters>(
      graph,
      `${id}-entry-weights`,
      'uint32',
      entryCount
    );
    return [
      ...scored.nodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-entries`,
        operation: 'GPUGeoConformalPredictionIntervals',
        variant: 'neighbor-scores',
        invocationCount: entryCount,
        bindings: [
          {name: 'scoreKeys', view: scored.scoreKeys, type: 'u32', access: 'read'},
          {name: 'foldKeys', view: scored.views.foldKeys, type: 'u32', access: 'read'},
          {name: 'neighborIndices', view: neighbors.indices, type: 'u32', access: 'read'},
          {name: 'neighborWeights', view: neighbors.weights, type: 'u32', access: 'read'},
          {name: 'entryScores', view: entryScores, type: 'f32', access: 'read_write'},
          {name: 'entryWeights', view: entryWeights, type: 'u32', access: 'read_write'}
        ],
        declarations: IS_FINITE_BITS_WGSL,
        // An entry that is out of range, excluded, or has an unusable weight gets weight zero,
        // which leaves every sum and quantile below unchanged.
        body: `let row = neighborIndices[neighborIndicesOffset + index];
  let weightBits = neighborWeights[neighborWeightsOffset + index];
  let usable = row < ${rows}u && foldKeys[foldKeysOffset + min(row, ${rows - 1}u)] == 0u && weightBits > 0u && weightBits < 0x7f800000u;
  entryScores[entryScoresOffset + index] = select(0.0, bitcast<f32>(scoreKeys[scoreKeysOffset + min(row, ${rows - 1}u)]), usable);
  entryWeights[entryWeightsOffset + index] = select(0u, weightBits, usable);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-intervals`,
        operation: 'GPUGeoConformalPredictionIntervals',
        variant: 'weighted-quantile',
        invocationCount: queryRows,
        bindings: [
          {name: 'parameters', view: props.parameters, type: 'u32', access: 'read'},
          {name: 'entryScores', view: entryScores, type: 'f32', access: 'read'},
          {name: 'entryWeights', view: entryWeights, type: 'u32', access: 'read'},
          {name: 'neighborOffsets', view: neighbors.offsets, type: 'u32', access: 'read'},
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
${CONFORMAL_PREDICTION_RANKS_WGSL}
fn getMassWeight(bits: u32, exponent: i32) -> f32 {
  if (bits == 0u) { return 0.0; }
  if (exponent == 128i) {
    // A subnormal is mantissa*2^-149; decode it through an exact integer, never
    // through subnormal float arithmetic that may flush on the device.
    if ((bits & 0x7f800000u) == 0u) { return f32(bits) * 4.76837158203125e-7; }
    return bitcast<f32>(bits + 0x40000000u);
  }
  if (exponent == -64i) {
    if (bits >= 0x20800000u) { return bitcast<f32>(bits - 0x20000000u); }
    return bitcast<f32>(bits) * 5.421010862427522e-20;
  }
  return bitcast<f32>(bits);
}`,
        body: `let miscoverage = bitcast<f32>(parameters[parametersOffset]);
  let selfWeightBits = parameters[parametersOffset + 1u] & 0x7fffffffu;
  let scale = ${props.query.scales ? 'queryScales[queryScalesOffset + index]' : '1.0'};
  let prediction = queryPredictions[queryPredictionsOffset + index];
  let first = neighborOffsets[neighborOffsetsOffset + index];
  let available = neighborOffsets[neighborOffsetsOffset + index + 1u] - first;
  let used = min(available, ${maximum}u);
  var code = select(0u, ${GPU_CONFORMAL_STATUS_NEIGHBORS_TRUNCATED}u, available > used);
  var maximumWeightBits = selfWeightBits;
  var usable = 0u;
  var equalWeights = selfWeightBits > 0u && selfWeightBits < 0x7f800000u;
  for (var entry = 0u; entry < used; entry++) {
    let weight = entryWeights[entryWeightsOffset + first + entry];
    maximumWeightBits = max(maximumWeightBits, weight);
    usable += u32(weight > 0u);
    equalWeights = equalWeights && (weight == 0u || weight == selfWeightBits);
  }
  // Common powers of two bound large sums and lift uniformly tiny inputs into
  // normal arithmetic, without a reciprocal or subnormal float comparison.
  var massExponent = 0i;
  if (maximumWeightBits >= 0x5f800000u) { massExponent = -64i; }
  else if (maximumWeightBits > 0u && maximumWeightBits < 0x1f800000u) { massExponent = 128i; }
  var total = getMassWeight(selfWeightBits, massExponent);
  for (var entry = 0u; entry < used; entry++) {
    total += getMassWeight(entryWeights[entryWeightsOffset + first + entry], massExponent);
  }
  var lower = getNaN();
  var upper = getNaN();
  if (!(isFiniteBits(prediction) && isFiniteBits(scale) && scale > 0.0)) {
    code |= ${GPU_CONFORMAL_STATUS_INVALID_QUERY}u;
  } else if (usable == 0u) {
    code |= ${GPU_CONFORMAL_STATUS_NO_CALIBRATION}u;
  } else {
    // The smallest score whose cumulative weight reaches the level; ties count together.
    var level = 0.0;
    var equalWeightRank = 0u;
    if (equalWeights) {
      let ranks = getConformalPredictionRanks(usable, miscoverage);
      equalWeightRank = ranks.y;
    } else {
      level = (1.0 - miscoverage) * total * (1.0 - ${GEO_CONFORMAL_WEIGHT_SLACK});
    }
    var best = getInfinity();
    for (var candidate = 0u; candidate < used; candidate++) {
      if (entryWeights[entryWeightsOffset + first + candidate] == 0u) { continue; }
      let candidateScore = entryScores[entryScoresOffset + first + candidate];
      if (candidateScore >= best) { continue; }
      var cumulative = 0.0;
      var cumulativeCount = 0u;
      for (var other = 0u; other < used; other++) {
        if (entryScores[entryScoresOffset + first + other] <= candidateScore) {
          let weight = entryWeights[entryWeightsOffset + first + other];
          cumulative += getMassWeight(weight, massExponent);
          cumulativeCount += u32(weight > 0u);
        }
      }
      if (select(cumulative >= level, cumulativeCount >= equalWeightRank, equalWeights)) {
        best = candidateScore;
      }
    }
    let width = scale * best;
    lower = prediction - width;
    upper = prediction + width;
    if (best >= getInfinity()) {
      code |= ${GPU_CONFORMAL_STATUS_LOWER_UNBOUNDED | GPU_CONFORMAL_STATUS_UPPER_UNBOUNDED}u;
    }
  }
  intervals[intervalsOffset + index * 2u] = lower;
  intervals[intervalsOffset + index * 2u + 1u] = upper;
  status[statusOffset + index] = code;`
      }),
      ...(lossViews.length
        ? [
            createWGSLKernelNode<Parameters>(graph, {
              id: `${id}-input-completeness`,
              operation: 'GPUGeoConformalPredictionIntervals',
              variant: 'input-completeness',
              invocationCount: queryRows,
              bindings: [
                ...lossFields.map(name => ({
                  name,
                  view: neighbors.status![name]!,
                  type: 'u32' as const,
                  access: 'read' as const
                })),
                {
                  name: 'intervals',
                  view: props.output.intervals,
                  type: 'f32',
                  access: 'read_write'
                },
                {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
              ],
              declarations: GET_NAN_WGSL,
              body: `if (${lossFields
                .map(name => `${name}[${name}Offset] ${name === 'complete' ? '==' : '!='} 0u`)
                .join(' || ')}) {
  intervals[intervalsOffset + index * 2u] = getNaN();
  intervals[intervalsOffset + index * 2u + 1u] = getNaN();
  status[statusOffset + index] = (status[statusOffset + index] & ~${GPU_CONFORMAL_STATUS_LOWER_UNBOUNDED | GPU_CONFORMAL_STATUS_UPPER_UNBOUNDED}u) | ${GPU_CONFORMAL_STATUS_INPUT_INCOMPLETE}u;
}`
            })
          ]
        : [])
    ];
  }
}
