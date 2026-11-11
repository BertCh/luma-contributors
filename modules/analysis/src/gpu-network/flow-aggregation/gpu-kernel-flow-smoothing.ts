// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  getGPUGreatCircleRadius,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateFiniteNumber,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getMovementDistanceWGSL} from '../../gpu-spatial-analysis/trajectory-staypoints/movement-common';

const OPERATION = 'GPUKernelFlowSmoothing';

/**
 * Largest accepted zone count. The contributor is registered with `tier: 'reference'`: the
 * neighbourhood radius of a zone compares every zone with every other (`zoneCount^2` per zone) and
 * the two smoothing passes each sum over all zones for each of the `zoneCount^2` flows, so work
 * grows with `zoneCount^3` (about 1.1e9 operations at this cap).
 */
export const GPU_KERNEL_FLOW_SMOOTHING_MAXIMUM_ZONES = 1024;

/** Number of float32 elements in a {@link GPUKernelFlowSmoothing} parameter view. */
export const GPU_KERNEL_FLOW_SMOOTHING_PARAMETER_LENGTH = 4;

/** Per-frame settings of {@link GPUKernelFlowSmoothing}. */
export type GPUKernelFlowSmoothingSettings = {
  /**
   * Weighted size `p` of a zone's neighbourhood, in the unit of `locationWeights`: the nearest
   * zones are taken until their weights sum to at least `p`. A value of 0 or less keeps only the
   * zone itself (and zones at distance 0), which reproduces the input.
   */
  neighborhoodSize: number;
  /**
   * Gaussian standard deviation as a fraction of the neighbourhood radius: `sigma = ratio * radius`.
   * Defaults to 0.5.
   */
  bandwidthRatio?: number;
};

/**
 * Packs the float parameters of {@link GPUKernelFlowSmoothing}.
 *
 * Layout: `[neighborhoodSize, bandwidthRatio, 0, 0]` as float32.
 *
 * @param settings Per-frame settings.
 * @param target Optional destination of at least 4 elements.
 */
export function getGPUKernelFlowSmoothingParameterValues(
  settings: GPUKernelFlowSmoothingSettings,
  target: Float32Array = new Float32Array(GPU_KERNEL_FLOW_SMOOTHING_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_KERNEL_FLOW_SMOOTHING_PARAMETER_LENGTH) {
    throw new Error(
      `${OPERATION} parameter target must hold ${GPU_KERNEL_FLOW_SMOOTHING_PARAMETER_LENGTH} elements`
    );
  }
  const bandwidthRatio = settings.bandwidthRatio ?? 0.5;
  validateFiniteNumber(OPERATION, 'neighborhoodSize', settings.neighborhoodSize);
  validateFiniteNumber(OPERATION, 'bandwidthRatio', bandwidthRatio, Number.MIN_VALUE);
  target.set([settings.neighborhoodSize, bandwidthRatio, 0, 0]);
  return target;
}

/** The flows to smooth: a dense zone-by-zone matrix, or the OD pairs of `GPUFlowAggregation`. */
export type GPUKernelFlowSmoothingFlows =
  | {
      kind: 'matrix';
      /** `zoneCount^2` rows, row `i * zoneCount + j` is the flow from zone `i` to zone `j`. */
      matrix: GraphDataView<'float32'>;
    }
  | {
      kind: 'pairs';
      /** Origin zone of each pair (`flowOriginZoneIds` of `GPUFlowAggregation`). */
      originZoneIds: GraphDataView<'uint32'>;
      /** Destination zone of each pair (`flowDestinationZoneIds`). */
      destinationZoneIds: GraphDataView<'uint32'>;
      /** Weight of each pair (`flowWeights`). Non-finite weights count as 0. */
      weights: GraphDataView<'float32'>;
      /**
       * One row: the number of valid pairs (`status.count` of `GPUFlowAggregation`). Pairs must be
       * distinct, as the aggregation's are; a repeated pair keeps one of its weights. Pairs with a
       * zone at or above `zoneCount` are ignored.
       */
      validPairCount: GraphDataView<'uint32'>;
    };

/** Properties for {@link GPUKernelFlowSmoothing}. */
export type GPUKernelFlowSmoothingProps = {
  /** Prefix for generated node IDs. Defaults to `'kernel-flow-smoothing'`. */
  id?: string;
  /** Planar, or longitude and latitude great-circle (haversine). Defaults to planar. */
  spatialContext?: GPUSpatialContext;
  /** Zone centroids, one row per zone (`zoneCount` is derived from this view). */
  positions: GraphDataView<'float32x2'>;
  /**
   * Weight of each zone in the neighbourhood size, for example its population or its number of
   * trips. Use ones for a neighbourhood of `p` zones. Per-frame contents.
   */
  locationWeights: GraphDataView<'float32'>;
  flows: GPUKernelFlowSmoothingFlows;
  /** Per-frame parameters, at least 4 float32 rows, from {@link getGPUKernelFlowSmoothingParameterValues}. */
  parameters: GraphDataView<'float32'>;
  output: {
    /** `zoneCount^2` smoothed flows, laid out like the input matrix. */
    smoothed: GraphDataView<'float32'>;
    /** Optional `zoneCount` rows: the neighbourhood radius of each zone. */
    radii?: GraphDataView<'float32'>;
  };
};

/**
 * Kernel smoothing of origin-destination flows: a flow is re-estimated as the kernel-weighted mean
 * of the flows between neighbouring origins and neighbouring destinations (a Nadaraya-Watson
 * estimator on the product of two adaptive zone kernels). The idea of smoothing flows over
 * neighbourhoods of both ends is that of Guo and Zhu (IEEE TVCG 20(12), 2014); this is not a
 * reproduction of their estimator, which was not available to check.
 *
 * The estimate is a mean, so it does not conserve the total flow: each weight row is normalised
 * to sum to 1, the columns are not. A zone that is a neighbour of many zones spreads its flows
 * into all of them and the total grows; a zone that is nobody's neighbour loses part of its
 * flows and the total shrinks. With zones at x = 0, 1 and 10, unit weights, neighbourhood size 2
 * and ratio 0.5, a flow of 100 inside the middle zone sums to 125.3 after smoothing and a flow of
 * 100 inside the far zone to 77.6. Rescale the output if totals must match.
 *
 * Definition used here:
 * 1. The neighbourhood of zone `i` is every zone within the smallest distance `h_i` at which the
 *    weights of the zones within `h_i` (zone `i` included) sum to at least `p`. Zones at distance
 *    exactly `h_i` belong to it. If all weights sum to less than `p`, `h_i` is the largest distance.
 * 2. Zone `k` of that neighbourhood has the Gaussian weight `K_ik = exp(-d_ik^2 / (2 sigma_i^2))`,
 *    `sigma_i = bandwidthRatio * h_i` (`K_ii = 1`); other zones have weight 0.
 * 3. `S_ij = sum_a sum_b K_ia K_jb M_ab / (sum_a K_ia * sum_b K_jb)`, a weighted mean of the
 *    flows `M_ab` over the pairs of neighbouring origins and neighbouring destinations. Pairs
 *    without a flow count as 0, so smoothing also spreads mass into empty pairs.
 *
 * The weighting is separable, so it is computed as two passes (rows, then columns).
 * The neighbourhood size, the kernel and the bandwidth ratio are this contributor's own choices.
 * Tests use a binary64 reference written from the definition above, a three-zone case worked by
 * hand, and closed-form limits (identity for a zero-size neighbourhood, the global mean for a
 * very wide kernel). Flow selection (removing duplicates) is not provided.
 *
 * Tier: reference, see {@link GPU_KERNEL_FLOW_SMOOTHING_MAXIMUM_ZONES}.
 */
export class GPUKernelFlowSmoothing implements GPUCommandNodeProducer {
  /** Prefix for graph node IDs. */
  readonly id: string;
  /** Number of zones. */
  readonly zoneCount: number;
  /** Resolved spatial context. */
  readonly spatialContext: GPUSpatialContext;
  /** Validated properties. */
  readonly props: Readonly<GPUKernelFlowSmoothingProps>;

  constructor(props: GPUKernelFlowSmoothingProps) {
    const id = props.id ?? 'kernel-flow-smoothing';
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT
    );
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
    this.zoneCount = props.positions.length;
    validateCapacity(id, 'zoneCount', this.zoneCount, 1);
    validateMaximumCount(id, 'zoneCount', this.zoneCount, GPU_KERNEL_FLOW_SMOOTHING_MAXIMUM_ZONES);
    validateSinglePackedView(props.locationWeights, ['float32'], `${id} locationWeights`);
    validateViewLength(id, 'locationWeights', props.locationWeights, this.zoneCount);
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    validateRequirement(
      id,
      props.parameters.length >= GPU_KERNEL_FLOW_SMOOTHING_PARAMETER_LENGTH,
      `parameters must contain ${GPU_KERNEL_FLOW_SMOOTHING_PARAMETER_LENGTH} float32 rows`
    );
    const {flows, output} = props;
    const matrixLength = this.zoneCount * this.zoneCount;
    if (flows.kind === 'matrix') {
      validateSinglePackedView(flows.matrix, ['float32'], `${id} flows.matrix`);
      validateViewLength(id, 'flows.matrix', flows.matrix, matrixLength);
    } else {
      validateRequirement(id, flows.kind === 'pairs', "flows.kind must be 'matrix' or 'pairs'");
      validateSinglePackedView(flows.originZoneIds, ['uint32'], `${id} flows.originZoneIds`);
      validateSinglePackedView(
        flows.destinationZoneIds,
        ['uint32'],
        `${id} flows.destinationZoneIds`
      );
      validateSinglePackedView(flows.weights, ['float32'], `${id} flows.weights`);
      validateSinglePackedView(flows.validPairCount, ['uint32'], `${id} flows.validPairCount`);
      validateViewLength(
        id,
        'flows.destinationZoneIds',
        flows.destinationZoneIds,
        flows.originZoneIds.length
      );
      validateViewLength(id, 'flows.weights', flows.weights, flows.originZoneIds.length);
      validateRequirement(
        id,
        flows.validPairCount.length >= 1,
        'flows.validPairCount must contain one uint32 row'
      );
    }
    validateSinglePackedView(output.smoothed, ['float32'], `${id} output.smoothed`);
    validateViewLength(id, 'output.smoothed', output.smoothed, matrixLength);
    if (output.radii) {
      validateSinglePackedView(output.radii, ['float32'], `${id} output.radii`);
      validateViewLength(id, 'output.radii', output.radii, this.zoneCount);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.smoothed, output.radii],
      [
        props.positions,
        props.locationWeights,
        props.parameters,
        ...(flows.kind === 'matrix'
          ? [flows.matrix]
          : [flows.originZoneIds, flows.destinationZoneIds, flows.weights, flows.validPairCount])
      ]
    );
    this.id = id;
    this.props = props;
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, zoneCount} = this;
    const {flows, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.locationWeights,
      props.parameters,
      output.smoothed,
      output.radii,
      ...(flows.kind === 'matrix'
        ? [flows.matrix]
        : [flows.originZoneIds, flows.destinationZoneIds, flows.weights, flows.validPairCount])
    ]);
    const matrixLength = zoneCount * zoneCount;
    const radii = output.radii ?? createTransientView(graph, `${id}-radii`, 'float32', zoneCount);
    const kernelWeights = createTransientView(graph, `${id}-kernel`, 'float32', matrixLength);
    const rowSmoothed = createTransientView(graph, `${id}-rows`, 'float32', matrixLength);
    const nodes: GPUCommandNode<Parameters>[] = [];
    let matrix: GraphDataView<'float32'>;
    if (flows.kind === 'matrix') {
      matrix = flows.matrix;
    } else {
      matrix = createTransientView(graph, `${id}-dense`, 'float32', matrixLength);
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-dense-reset`,
          operation: OPERATION,
          view: matrix,
          type: 'f32',
          value: '0.0'
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-dense-scatter`,
          operation: OPERATION,
          variant: 'scatter',
          invocationCount: flows.originZoneIds.length,
          bindings: [
            {name: 'originZoneIds', view: flows.originZoneIds, type: 'u32', access: 'read'},
            {
              name: 'destinationZoneIds',
              view: flows.destinationZoneIds,
              type: 'u32',
              access: 'read'
            },
            {name: 'weights', view: flows.weights, type: 'f32', access: 'read'},
            {name: 'pairCount', view: flows.validPairCount, type: 'u32', access: 'read'},
            {name: 'dense', view: matrix, type: 'f32', access: 'read_write'}
          ],
          body: /* wgsl */ `
  let origin = originZoneIds[originZoneIdsOffset + index];
  let destination = destinationZoneIds[destinationZoneIdsOffset + index];
  let weight = weights[weightsOffset + index];
  let isFinite = (bitcast<u32>(weight) & 0x7f800000u) != 0x7f800000u;
  if (index < pairCount[pairCountOffset] && origin < ${zoneCount}u && destination < ${zoneCount}u) {
    dense[denseOffset + origin * ${zoneCount}u + destination] = select(0.0, weight, isFinite);
  }`
        })
      );
    }
    const distanceSource = getMovementDistanceWGSL(getGPUGreatCircleRadius(this.spatialContext));
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-radii`,
        operation: OPERATION,
        variant: 'radii',
        invocationCount: zoneCount,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'locationWeights', view: props.locationWeights, type: 'f32', access: 'read'},
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'radii', view: radii, type: 'f32', access: 'read_write'}
        ],
        declarations: `${distanceSource}
const ZONE_COUNT: u32 = ${zoneCount}u;`,
        body: /* wgsl */ `
  let size = parameters[parametersOffset];
  var best = -1.0;
  var farthest = 0.0;
  for (var candidate = 0u; candidate < ZONE_COUNT; candidate = candidate + 1u) {
    let radius = rowDistance(index, candidate);
    farthest = max(farthest, radius);
    var weight = 0.0;
    for (var zone = 0u; zone < ZONE_COUNT; zone = zone + 1u) {
      if (rowDistance(index, zone) <= radius) {
        weight = weight + locationWeights[locationWeightsOffset + zone];
      }
    }
    if (weight >= size && (best < 0.0 || radius < best)) {
      best = radius;
    }
  }
  radii[radiiOffset + index] = select(farthest, best, best >= 0.0);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-kernel`,
        operation: OPERATION,
        variant: 'weights',
        invocationCount: matrixLength,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'radii', view: radii, type: 'f32', access: 'read'},
          {name: 'kernelWeights', view: kernelWeights, type: 'f32', access: 'read_write'}
        ],
        declarations: `${distanceSource}
const ZONE_COUNT: u32 = ${zoneCount}u;
const RADIUS_MEMBERSHIP_MARGIN: f32 = 1.000001;`,
        body: /* wgsl */ `
  let zone = index / ZONE_COUNT;
  let neighbour = index % ZONE_COUNT;
  let distance = rowDistance(zone, neighbour);
  let radius = radii[radiiOffset + zone];
  let sigma = radius * parameters[parametersOffset + 1u];
  var weight = 0.0;
  // The radius is a distance the radii kernel computed in another pipeline. A compiler may round
  // the same expression differently there (fused multiply-add), so the zone that defines the
  // radius is admitted with a margin of a few rounding steps instead of an exact comparison.
  if (distance <= radius * RADIUS_MEMBERSHIP_MARGIN) {
    weight = select(1.0, exp(-distance * distance / (2.0 * sigma * sigma)), sigma > 0.0);
  }
  kernelWeights[kernelWeightsOffset + index] = weight;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-rows`,
        operation: OPERATION,
        variant: 'origins',
        invocationCount: matrixLength,
        bindings: [
          {name: 'kernelWeights', view: kernelWeights, type: 'f32', access: 'read'},
          {name: 'matrix', view: matrix, type: 'f32', access: 'read'},
          {name: 'rowSmoothed', view: rowSmoothed, type: 'f32', access: 'read_write'}
        ],
        declarations: `const ZONE_COUNT: u32 = ${zoneCount}u;`,
        body: /* wgsl */ `
  let origin = index / ZONE_COUNT;
  let destination = index % ZONE_COUNT;
  var sum = 0.0;
  var total = 0.0;
  for (var neighbour = 0u; neighbour < ZONE_COUNT; neighbour = neighbour + 1u) {
    let weight = kernelWeights[kernelWeightsOffset + origin * ZONE_COUNT + neighbour];
    total = total + weight;
    if (weight != 0.0) {
      sum = sum + weight * matrix[matrixOffset + neighbour * ZONE_COUNT + destination];
    }
  }
  rowSmoothed[rowSmoothedOffset + index] = sum / total;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-columns`,
        operation: OPERATION,
        variant: 'destinations',
        invocationCount: matrixLength,
        bindings: [
          {name: 'kernelWeights', view: kernelWeights, type: 'f32', access: 'read'},
          {name: 'rowSmoothed', view: rowSmoothed, type: 'f32', access: 'read'},
          {name: 'smoothed', view: output.smoothed, type: 'f32', access: 'read_write'}
        ],
        declarations: `const ZONE_COUNT: u32 = ${zoneCount}u;`,
        body: /* wgsl */ `
  let origin = index / ZONE_COUNT;
  let destination = index % ZONE_COUNT;
  var sum = 0.0;
  var total = 0.0;
  for (var neighbour = 0u; neighbour < ZONE_COUNT; neighbour = neighbour + 1u) {
    let weight = kernelWeights[kernelWeightsOffset + destination * ZONE_COUNT + neighbour];
    total = total + weight;
    if (weight != 0.0) {
      sum = sum + weight * rowSmoothed[rowSmoothedOffset + origin * ZONE_COUNT + neighbour];
    }
  }
  smoothed[smoothedOffset + index] = sum / total;`
      })
    );
    return nodes;
  }
}
