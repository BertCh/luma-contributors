// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {fp64arithmetic} from '@luma.gl/shaders';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createTransientView, GPUReduction} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateGraphPorts,
  validateFiniteNumber,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPU_POINT_CLOUD_MAXIMUM_NEIGHBOR_CAPACITY} from './pointcloud-limits';
import {
  getPointCloudKernelNodes,
  validatePointCloudPackedViews,
  validatePointCloudScratchBindings
} from './pointcloud-kernel-nodes';
import {inferPointCloudNeighborCapacity} from './pointcloud-neighbor-capacity';

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** One float32 row: the standard-deviation multiplier. */
export const GPU_POINT_CLOUD_STATISTICAL_FILTER_PARAMETER_LENGTH = 1;

export type GPUPointCloudStatisticalFilterSettings = {
  /** Nonnegative multiplier of the population standard deviation; default 2. */
  standardDeviations?: number;
};

/** Pack per-frame outlier threshold settings for a caller-owned GPUParameterBuffer. */
export function getGPUPointCloudStatisticalFilterParameterValues(
  settings: GPUPointCloudStatisticalFilterSettings = {}
): Float32Array {
  validateFiniteNumber(
    'gpu-pointcloud-statistical-filter',
    'standardDeviations',
    settings.standardDeviations ?? 2,
    0
  );
  return new Float32Array([settings.standardDeviations ?? 2]);
}

export type GPUPointCloudStatisticalFilterOutput = {
  /** Mean retained neighbor distance in the neighborhood-query row domain; -1 means no neighborhood. */
  meanDistances: GraphDataView<'float32'>;
  /**
   * Neighborhood-query-aligned inlier flags: one when the row has at least one neighbor and its mean distance is
   * at most the global threshold, zero otherwise. Uses population standard deviation of valid mean
   * distances. There is no per-row status word: a row without neighbors (`meanDistances` is -1) is
   * an outlier by definition. If any supplied neighborhoodStatus reports invalid or unsupported
   * support, the global threshold is incomplete and every classification is suppressed to zero.
   * With reordered queryRows, scatter this mask through queryRows before passing it to SMRF.
   */
  validity: GraphDataView<'uint32'>;
  /** Optional single uint32: 0 if supplied neighborhoodStatus flags invalid/unsupported support, 1 otherwise. Omitted input assumes complete support. */
  thresholdComplete?: GraphDataView<'uint32'>;
};

export type GPUPointCloudStatisticalFilterProps = {
  /** Per-frame multiplier; overrides standardDeviations. */
  parameters?: GraphDataView<'float32'>;
  id?: string;
  squaredDistances: GraphDataView<'float32'>;
  counts: GraphDataView<'uint32'>;
  /** Optional query-aligned neighborhood flags. Invalid/unsupported queries do not contribute to moments. Truncated nearest-K neighborhoods remain intentional inputs. */
  neighborhoodStatus?: GraphDataView<'uint32'>;
  /** 1–64 slots per query, inferred from exact squaredDistances/counts lengths when nonempty. Zero query rows require an explicit capacity. */
  capacity?: number;
  standardDeviations?: number;
  /** Caller-owned result views; their contents are written by the graph. */
  output: GPUPointCloudStatisticalFilterOutput;
};

/**
 * Per-frame: input contents and parameters (standardDeviations). Compile-time: capacity and
 * view shapes. Without parameters, standardDeviations retains its constructor-time behavior.
 *
 * Global mean-neighbor-distance outlier filter (PDAL `filters.outlier`, statistical).
 *
 * The global mean and population variance of the valid mean distances come from three published
 * `GPUReduction` sums, so no kernel loops over rows. Moments are centered on the minimum supported
 * distance before scaling and reduction, preserving constant populations exactly. Unsupported
 * queries are excluded from moments and suppress all classifications; thresholdComplete exposes
 * that global loss. Truncation alone preserves intentional nearest-K behavior.
 * Capacity is 1–64 retained slots per query (GPU_POINT_CLOUD_MAXIMUM_NEIGHBOR_CAPACITY), across all routes.
 */
export class GPUPointCloudStatisticalFilter implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUPointCloudStatisticalFilterProps & {capacity: number};
  constructor(props: GPUPointCloudStatisticalFilterProps) {
    this.id = props.id ?? 'gpu-pointcloud-statistical-filter';
    validateGraphPorts(this.id, props, {
      squaredDistances: {required: true, viewType: 'data', formats: ['float32']},
      counts: {required: true, viewType: 'data', formats: ['uint32']},
      neighborhoodStatus: {required: false, viewType: 'data', formats: ['uint32']},
      parameters: {required: false, viewType: 'data', formats: ['float32']},
      'output.meanDistances': {required: true, viewType: 'data', formats: ['float32']},
      'output.validity': {required: true, viewType: 'data', formats: ['uint32']},
      'output.thresholdComplete': {required: false, viewType: 'data', formats: ['uint32']}
    });
    props = snapshotConstructionFields(props, [
      'parameters',
      'id',
      'squaredDistances',
      'counts',
      'neighborhoodStatus',
      'capacity',
      'standardDeviations',
      'output'
    ]);
    validateRequirement(
      this.id,
      props.output !== undefined && props.output !== null,
      'output must be given'
    );
    const capacity =
      props.capacity ??
      inferPointCloudNeighborCapacity(this.id, props.counts.length, props.squaredDistances.length);
    this.props = Object.freeze({
      ...props,
      capacity,
      output: snapshotConstructionFields(props.output, [
        'meanDistances',
        'validity',
        'thresholdComplete'
      ])
    });
    props = this.props;
    validatePointCloudPackedViews(
      this.id,
      {
        squaredDistances: props.squaredDistances,
        counts: props.counts,
        neighborhoodStatus: props.neighborhoodStatus,
        parameters: props.parameters
      },
      {
        meanDistances: props.output.meanDistances,
        validity: props.output.validity,
        thresholdComplete: props.output.thresholdComplete
      },
      [
        {
          inputs: {squaredDistances: props.squaredDistances, counts: props.counts},
          outputs: {meanDistances: props.output.meanDistances}
        },
        {
          inputs: {meanDistances: props.output.meanDistances},
          outputs: {validity: props.output.validity}
        }
      ]
    );
    const {
      output: {validity}
    } = props;
    validateRequirement(this.id, validity !== undefined, 'validity must be given');
    validateCapacity(this.id, 'capacity', capacity, 1, GPU_POINT_CLOUD_MAXIMUM_NEIGHBOR_CAPACITY);
    validateViewLength(
      this.id,
      'squaredDistances',
      props.squaredDistances,
      props.counts.length * capacity
    );
    validateViewLength(this.id, 'meanDistances', props.output.meanDistances, props.counts.length);
    validateViewLength(this.id, 'validity', validity, props.counts.length);
    if (props.neighborhoodStatus)
      validateViewLength(
        this.id,
        'neighborhoodStatus',
        props.neighborhoodStatus,
        props.counts.length
      );
    if (props.output.thresholdComplete)
      validateViewLength(this.id, 'thresholdComplete', props.output.thresholdComplete, 1);
    if (props.parameters) {
      validateViewLength(
        this.id,
        'parameters',
        props.parameters,
        GPU_POINT_CLOUD_STATISTICAL_FILTER_PARAMETER_LENGTH
      );
    }
    validateFiniteNumber(this.id, 'standardDeviations', props.standardDeviations ?? 2, 0);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {
      squaredDistances,
      counts,
      capacity,
      output: {meanDistances}
    } = this.props;
    const parameters = this.props.parameters;
    const neighborhoodStatus = this.props.neighborhoodStatus;
    const thresholdComplete = this.props.output.thresholdComplete;
    if (!counts.length)
      return thresholdComplete
        ? getPointCloudKernelNodes(graph, {
            id: `${this.id}-complete`,
            length: 1,
            inputs: {},
            outputs: {thresholdComplete},
            body: 'writeThresholdComplete(0u, 1u);'
          })
        : [];
    validateGraphChunkBindingRanges(graph, this.id, {
      squaredDistances,
      counts,
      meanDistances,
      validity: this.props.output.validity,
      parameters,
      neighborhoodStatus,
      thresholdComplete
    });
    validatePointCloudScratchBindings(graph, this.id, {
      'statistical filter row scratch': counts.length * 4,
      'statistical filter scalars': 4
    });
    const mask = this.props.output.validity;
    const nodes: GPUCommandNode<Parameters>[] = getPointCloudKernelNodes(graph, {
      id: `${this.id}-means`,
      length: counts.length,
      inputs: {squaredDistances, counts},
      outputs: {meanDistances},
      modules: [fp64arithmetic],
      body: `let support = min(readCounts(index), ${capacity}u);
if (support == 0u) { writeMeanDistances(index, -1.0); return; }
let distanceReference = sqrt_fp64(vec2f(readSquaredDistances(index * ${capacity}u), 0.0)).x;
var distanceSum = 0.0;
for (var slot = 0u; slot < support; slot++) { distanceSum += sqrt_fp64(vec2f(readSquaredDistances(index * ${capacity}u + slot), 0.0)).x - distanceReference; }
writeMeanDistances(index, distanceReference + distanceSum / f32(support));`
    });
    const validity = createTransientView(
      graph,
      `${this.id}-validity-flags`,
      'uint32',
      counts.length
    );
    const validCount = createTransientView(graph, `${this.id}-valid-count-total`, 'uint32', 1);
    const scaledValues = createTransientView(
      graph,
      `${this.id}-scaled-values`,
      'float32',
      counts.length
    );
    const mean = createTransientView(graph, `${this.id}-mean-value`, 'float32', 1);
    const varianceTerms = createTransientView(
      graph,
      `${this.id}-variance-terms`,
      'float32',
      counts.length
    );
    const variance = createTransientView(graph, `${this.id}-variance-value`, 'float32', 1);
    const threshold = createTransientView(graph, `${this.id}-threshold-value`, 'float32', 1);
    const reference = createTransientView(graph, `${this.id}-reference-value`, 'float32', 1);
    const supportLossFlags = neighborhoodStatus
      ? createTransientView(graph, `${this.id}-support-loss-flags`, 'uint32', counts.length)
      : undefined;
    const supportLoss = neighborhoodStatus
      ? createTransientView(graph, `${this.id}-support-loss`, 'uint32', 1)
      : undefined;
    const sum = (name: string, input: GraphDataView<'float32'>, output: GraphDataView<'float32'>) =>
      new GPUReduction({id: `${this.id}-${name}`, input, output, operation: 'sum'}).getCommandNodes(
        graph
      );
    nodes.push(
      ...getPointCloudKernelNodes(graph, {
        id: `${this.id}-validity`,
        length: counts.length,
        inputs: {meanDistances, ...(neighborhoodStatus ? {neighborhoodStatus} : {})},
        outputs: {validity},
        body: `let value = readMeanDistances(index); writeValidity(index, u32(value >= 0.0 && value <= bitcast<f32>(0x7f7fffffu) ${neighborhoodStatus ? '&& (readNeighborhoodStatus(index) & 5u) == 0u' : ''}));`
      }),
      ...new GPUReduction({
        id: `${this.id}-valid-count`,
        input: validity,
        output: validCount,
        operation: 'sum'
      }).getCommandNodes(graph),
      ...new GPUReduction({
        id: `${this.id}-reference`,
        input: meanDistances,
        mask: validity,
        output: reference,
        operation: 'min'
      }).getCommandNodes(graph),
      ...getPointCloudKernelNodes(graph, {
        id: `${this.id}-scaled`,
        length: counts.length,
        inputs: {meanDistances, validity, validCount, reference},
        outputs: {scaledValues},
        body: 'writeScaledValues(index, select(0.0, (readMeanDistances(index) - readReference(0u)) / f32(max(readValidCount(0u), 1u)), readValidity(index) != 0u));'
      }),
      ...sum('mean', scaledValues, mean),
      ...getPointCloudKernelNodes(graph, {
        id: `${this.id}-terms`,
        length: counts.length,
        inputs: {meanDistances, validity, validCount, mean, reference},
        outputs: {varianceTerms},
        // Scaling the difference by 1 / sqrt(count) before squaring keeps each term finite.
        body: `let scaledDifference = ((readMeanDistances(index) - readReference(0u)) - readMean(0u)) / sqrt(f32(max(readValidCount(0u), 1u)));
writeVarianceTerms(index, select(0.0, scaledDifference * scaledDifference, readValidity(index) != 0u));`
      }),
      ...sum('variance', varianceTerms, variance),
      ...getPointCloudKernelNodes(graph, {
        id: `${this.id}-threshold`,
        length: 1,
        inputs: {mean, variance, reference, ...(parameters ? {parameters} : {})},
        outputs: {threshold},
        body: `writeThreshold(0u, readReference(0u) + readMean(0u) + ${parameters ? 'readParameters(0u)' : getWGSLFloatLiteral(this.props.standardDeviations ?? 2)} * sqrt(max(readVariance(0u), 0.0)));`
      })
    );
    if (neighborhoodStatus && supportLossFlags && supportLoss)
      nodes.push(
        ...getPointCloudKernelNodes(graph, {
          id: `${this.id}-support-flags`,
          length: counts.length,
          inputs: {neighborhoodStatus},
          outputs: {supportLossFlags},
          body: 'writeSupportLossFlags(index, u32((readNeighborhoodStatus(index) & 5u) != 0u));'
        }),
        ...new GPUReduction({
          id: `${this.id}-support-loss-total`,
          input: supportLossFlags,
          output: supportLoss,
          operation: 'max'
        }).getCommandNodes(graph)
      );
    if (thresholdComplete)
      nodes.push(
        ...getPointCloudKernelNodes(graph, {
          id: `${this.id}-complete`,
          length: 1,
          inputs: supportLoss ? {supportLoss} : {},
          outputs: {thresholdComplete},
          body: `writeThresholdComplete(0u, ${supportLoss ? 'u32(readSupportLoss(0u) == 0u)' : '1u'});`
        })
      );
    nodes.push(
      ...getPointCloudKernelNodes(graph, {
        id: `${this.id}-classify`,
        length: counts.length,
        inputs: {meanDistances, threshold, ...(supportLoss ? {supportLoss} : {})},
        outputs: {mask},
        body: `let value = readMeanDistances(index);
writeMask(index, u32(value >= 0.0 && value <= readThreshold(0u) ${supportLoss ? '&& readSupportLoss(0u) == 0u' : ''}));`
      })
    );
    return nodes;
  }
}
