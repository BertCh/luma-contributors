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
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateRegressionOutputDispatches} from '../spatial-regression/regression-output-dispatches';
import {validateGraphOutputsDisjointFromInputs} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  createConformalScoreNodes,
  getConformalInputViews,
  GPU_CONFORMAL_RANK_SLACK,
  validateConformalInputs,
  type GPUConformalCalibration,
  type GPUConformalQuery
} from './conformal-prediction-nodes';
import {
  GPU_CONFORMAL_STATUS_INVALID_QUERY,
  GPU_CONFORMAL_STATUS_LOWER_UNBOUNDED,
  GPU_CONFORMAL_STATUS_NEIGHBORS_TRUNCATED,
  GPU_CONFORMAL_STATUS_NO_CALIBRATION,
  GPU_CONFORMAL_STATUS_UPPER_UNBOUNDED,
  INFINITY_WGSL,
  validateConformalOutput
} from './gpu-conformal-prediction-intervals';

/** Most neighbours of one query the localized contributor weighs. */
export const GPU_GEO_CONFORMAL_MAXIMUM_NEIGHBORS = 256;

/** Neighbour lists of the localized contributor, in compressed sparse row form. */
export type GPUGeoConformalNeighbors = {
  /** `queryRows + 1` offsets into `indices` and `weights`. */
  offsets: GraphDataView<'uint32'>;
  /** Calibration row of each neighbour entry. An entry that is out of range or not usable is skipped. */
  indices: GraphDataView<'uint32'>;
  /** Kernel weight of each entry, non-negative; a non-finite or negative weight skips the entry. */
  weights: GraphDataView<'float32'>;
};

/** Caller-owned outputs of {@link GPUGeoConformalPredictionIntervals}. */
export type GPUGeoConformalPredictionIntervalsOutput = {
  /** `[lower, upper]` per query row; NaN where the status has the invalid or empty bit. */
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
 * With equal weights `w_j = selfWeight` this is exactly the global split interval of
 * {@link GPUConformalPredictionIntervals}, which is its equivalence test. At most
 * {@link GPU_GEO_CONFORMAL_MAXIMUM_NEIGHBORS} entries per query are used. Weights are summed in
 * float32 in list order, so non-uniform weights give a quantile level accurate to float32
 * rounding only. Localized coverage is approximate (it holds when the weights reflect the
 * exchangeability structure), unlike the finite-sample guarantee of the global variant.
 */
export class GPUGeoConformalPredictionIntervals implements GPUCommandNodeProducer {
  readonly id: string;
  readonly queryRows: number;
  constructor(readonly props: GPUGeoConformalPredictionIntervalsProps) {
    this.id = props.id ?? 'geo-conformal-prediction-intervals';
    const {id} = this;
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
    validateRegressionOutputDispatches(this.id, props.output, [['intervals', 'status']]);
    validateGraphOutputsDisjointFromInputs(id, Object.values(props.output), [
      ...getConformalInputViews(props.calibration, props.query),
      props.neighbors.offsets,
      props.neighbors.indices,
      props.neighbors.weights,
      props.parameters
    ]);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, queryRows} = this;
    const {neighbors} = props;
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
    const entryWeights = createTransientView<'float32', Parameters>(
      graph,
      `${id}-entry-weights`,
      'float32',
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
          {name: 'neighborWeights', view: neighbors.weights, type: 'f32', access: 'read'},
          {name: 'entryScores', view: entryScores, type: 'f32', access: 'read_write'},
          {name: 'entryWeights', view: entryWeights, type: 'f32', access: 'read_write'}
        ],
        declarations: IS_FINITE_BITS_WGSL,
        // An entry that is out of range, excluded, or has an unusable weight gets weight zero,
        // which leaves every sum and quantile below unchanged.
        body: `let row = neighborIndices[neighborIndicesOffset + index];
  let weight = neighborWeights[neighborWeightsOffset + index];
  let usable = row < ${rows}u && foldKeys[foldKeysOffset + min(row, ${rows - 1}u)] == 0u && isFiniteBits(weight) && weight > 0.0;
  entryScores[entryScoresOffset + index] = select(0.0, bitcast<f32>(scoreKeys[scoreKeysOffset + min(row, ${rows - 1}u)]), usable);
  entryWeights[entryWeightsOffset + index] = select(0.0, weight, usable);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-intervals`,
        operation: 'GPUGeoConformalPredictionIntervals',
        variant: 'weighted-quantile',
        invocationCount: queryRows,
        bindings: [
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'entryScores', view: entryScores, type: 'f32', access: 'read'},
          {name: 'entryWeights', view: entryWeights, type: 'f32', access: 'read'},
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
${INFINITY_WGSL}`,
        body: `let miscoverage = parameters[parametersOffset];
  let selfWeight = parameters[parametersOffset + 1u];
  let scale = ${props.query.scales ? 'queryScales[queryScalesOffset + index]' : '1.0'};
  let prediction = queryPredictions[queryPredictionsOffset + index];
  let first = neighborOffsets[neighborOffsetsOffset + index];
  let available = neighborOffsets[neighborOffsetsOffset + index + 1u] - first;
  let used = min(available, ${maximum}u);
  var code = select(0u, ${GPU_CONFORMAL_STATUS_NEIGHBORS_TRUNCATED}u, available > used);
  var total = selfWeight;
  var usable = 0u;
  for (var entry = 0u; entry < used; entry++) {
    let weight = entryWeights[entryWeightsOffset + first + entry];
    total += weight;
    usable += u32(weight > 0.0);
  }
  var lower = getNaN();
  var upper = getNaN();
  if (!(isFiniteBits(prediction) && isFiniteBits(scale) && scale > 0.0)) {
    code |= ${GPU_CONFORMAL_STATUS_INVALID_QUERY}u;
  } else if (usable == 0u) {
    code |= ${GPU_CONFORMAL_STATUS_NO_CALIBRATION}u;
  } else {
    // The smallest score whose cumulative weight reaches the level; ties count together.
    let level = (1.0 - miscoverage) * total * (1.0 - ${GPU_CONFORMAL_RANK_SLACK});
    var best = getInfinity();
    for (var candidate = 0u; candidate < used; candidate++) {
      if (entryWeights[entryWeightsOffset + first + candidate] <= 0.0) { continue; }
      let candidateScore = entryScores[entryScoresOffset + first + candidate];
      if (candidateScore >= best) { continue; }
      var cumulative = 0.0;
      for (var other = 0u; other < used; other++) {
        if (entryScores[entryScoresOffset + first + other] <= candidateScore) {
          cumulative += entryWeights[entryWeightsOffset + first + other];
        }
      }
      if (cumulative >= level) { best = candidateScore; }
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
      })
    ];
  }
}
