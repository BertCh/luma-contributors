// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateNetworkViewBindings,
  validateNetworkOutputsDisjointFromInputs,
  validateNetworkWritableViews
} from '../network-view-utils';
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  getGPUGreatCircleRadius,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateMaximumCount,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getMovementDistanceWGSL} from '../../gpu-spatial-analysis/trajectory-staypoints/movement-common';

const OPERATION = 'GPURadiationFlowModel';

/**
 * Largest accepted zone count. The model is registered with `tier: 'reference'`: each of the
 * `zoneCount^2` output pairs sums the populations of the zones nearer to the origin than the
 * destination, a loop over all zones inside one invocation, so work grows with `zoneCount^3`
 * (about 1.1e9 distance evaluations at this cap). The scalable route is a per-origin segmented
 * sort by distance plus a prefix sum, which needs a float segmented scan that `GPUScan` (uint32
 * only) does not offer.
 */
export const GPU_RADIATION_FLOW_MODEL_MAXIMUM_ZONES = 1024;

/** Normalization of the radiation model. */
export type GPURadiationFlowModelNormalization =
  /** `T_ij = T_i m_i n_j / ((m_i + s_ij)(m_i + n_j + s_ij))`, the form of Simini et al. (2012). */
  | 'none'
  /**
   * The same flux divided by `1 - m_i / M` (`M` the total population), the finite-size correction
   * that makes each row sum to `T_i` when no two destinations are at the same distance from an origin.
   */
  | 'finite-size';

/** Properties for {@link GPURadiationFlowModel}. */
export type GPURadiationFlowModelProps = {
  /** Prefix for generated node IDs. Defaults to `'radiation-flow-model'`. */
  id?: string;
  /** Planar, or longitude and latitude great-circle (haversine). Defaults to planar. */
  spatialContext?: GPUSpatialContext;
  /** Zone centroids, one row per zone (`zoneCount` is derived from this view). */
  positions: GraphDataView<'float32x2'>;
  /**
   * Population (or any opportunity count) of each zone, `m_i` and `n_j` of the model. Non-negative.
   * Per-frame contents.
   */
  populations: GraphDataView<'float32'>;
  /**
   * Total number of trips leaving each zone, `T_i`, not counting trips that stay inside it.
   * Per-frame contents. The `zoneOutWeights` or the float-converted `zoneOutCounts` of a
   * `GPUFlowAggregation` built with `excludeSelfFlows: true` are the usual source; without that
   * option the zone totals include self flows and `T_i` is too large.
   */
  outflows: GraphDataView<'float32'>;
  /** Compile-time. Defaults to `'none'`. */
  normalization?: GPURadiationFlowModelNormalization;
  output: {
    /**
     * Expected flow from zone `i` to zone `j` at row `i * zoneCount + j`, `zoneCount^2` rows. The
     * diagonal is 0. This is the expectation of the model, not a random draw.
     */
    flows: GraphDataView<'float32'>;
  };
};

/**
 * Radiation model of human mobility (Simini, Gonzalez, Maritan and Barabasi, Nature 484, 2012).
 * A trip leaving zone `i` is absorbed by zone `j` with probability
 * `m_i n_j / ((m_i + s_ij)(m_i + n_j + s_ij))` where `m_i` and `n_j` are the populations of the
 * origin and destination and `s_ij` is the population of the zones strictly nearer to `i` than `j`
 * is, excluding `i` and `j` themselves. The expected flow is `T_i` times that probability. The
 * model has no fitted parameter.
 *
 * Ties: zones at exactly the distance `d_ij` do not count in `s_ij` (strict inequality). Where two
 * zones are at the same distance from an origin, implementations that sort the zones by distance
 * count one of them and not the other, depending on sort order, so results differ there.
 *
 * This is the expected flow matrix. scikit-mobility's `Radiation.generate` draws integer flows from
 * the probabilities; this contributor does not draw.
 *
 * Reference: tests use a binary64 per-origin sorted-prefix-sum reference written from the paper's
 * definition and check the closed-form row sum `1 - m_i / M`. scikit-mobility itself was not run.
 * Tier: reference, see {@link GPU_RADIATION_FLOW_MODEL_MAXIMUM_ZONES}.
 *
 * Per-frame: the contents of `positions`, `populations` and `outflows`. Compile-time: the zone
 * count, the spatial context and the normalization.
 */
export class GPURadiationFlowModel implements GPUCommandNodeProducer {
  /** Prefix for graph node IDs. */
  readonly id: string;
  /** Number of zones. */
  readonly zoneCount: number;
  /** Resolved spatial context. */
  readonly spatialContext: GPUSpatialContext;
  /** Validated properties. */
  readonly props: Readonly<GPURadiationFlowModelProps>;

  constructor(props: GPURadiationFlowModelProps) {
    const id = props.id ?? 'radiation-flow-model';
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT
    );
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
    this.zoneCount = props.positions.length;
    validateCapacity(id, 'zoneCount', this.zoneCount, 2);
    validateMaximumCount(id, 'zoneCount', this.zoneCount, GPU_RADIATION_FLOW_MODEL_MAXIMUM_ZONES);
    validateSinglePackedView(props.populations, ['float32'], `${id} populations`);
    validateViewLength(id, 'populations', props.populations, this.zoneCount);
    validateSinglePackedView(props.outflows, ['float32'], `${id} outflows`);
    validateViewLength(id, 'outflows', props.outflows, this.zoneCount);
    const normalization = props.normalization ?? 'none';
    validateRequirement(
      id,
      normalization === 'none' || normalization === 'finite-size',
      "normalization must be 'none' or 'finite-size'"
    );
    validateSinglePackedView(props.output.flows, ['float32'], `${id} output.flows`);
    validateViewLength(id, 'output.flows', props.output.flows, this.zoneCount * this.zoneCount);
    validateNetworkWritableViews(id, props.output);
    validateNetworkOutputsDisjointFromInputs(
      id,
      [props.output.flows],
      [props.positions, props.populations, props.outflows]
    );
    this.id = id;
    this.props = props;
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, zoneCount} = this;
    validateNetworkViewBindings(
      id,
      graph,
      [props.positions, props.populations, props.outflows, props.output.flows],
      {}
    );
    const finiteSize = props.normalization === 'finite-size';
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-flows`,
        operation: OPERATION,
        variant: finiteSize ? 'finite-size' : 'plain',
        invocationCount: zoneCount * zoneCount,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'populations', view: props.populations, type: 'f32', access: 'read'},
          {name: 'outflows', view: props.outflows, type: 'f32', access: 'read'},
          {name: 'flows', view: props.output.flows, type: 'f32', access: 'read_write'}
        ],
        declarations: `${getMovementDistanceWGSL(getGPUGreatCircleRadius(this.spatialContext))}
const ZONE_COUNT: u32 = ${zoneCount}u;`,
        body: /* wgsl */ `
  let origin = index / ZONE_COUNT;
  let destination = index % ZONE_COUNT;
  var flow = 0.0;
  if (origin != destination) {
    let distance = rowDistance(origin, destination);
    var between = 0.0;
    var total = 0.0;
    for (var zone = 0u; zone < ZONE_COUNT; zone = zone + 1u) {
      let population = populations[populationsOffset + zone];
      total = total + population;
      if (zone != origin && zone != destination && rowDistance(origin, zone) < distance) {
        between = between + population;
      }
    }
    let sourcePopulation = populations[populationsOffset + origin];
    let targetPopulation = populations[populationsOffset + destination];
    let denominator = (sourcePopulation + between) * (sourcePopulation + targetPopulation + between);
    if (denominator > 0.0) {
      flow = outflows[outflowsOffset + origin] * sourcePopulation * targetPopulation / denominator;
${
  finiteSize
    ? `      let remaining = 1.0 - sourcePopulation / total;
      flow = select(0.0, flow / remaining, remaining > 0.0);`
    : ''
}
    }
  }
  flows[flowsOffset + index] = flow;`
      })
    ];
  }
}
