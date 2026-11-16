// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateNetworkViewBindings, validateNetworkWritableViews} from './network-view-utils';

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateMaximumCount
} from '../analysis-core/kernels/gpu-contributor-utils';
import {SCALED_QUOTIENT_WGSL} from '../analysis-core/kernels/scaled-quotient-wgsl';
import {IS_FINITE_FLOAT_WGSL} from '../analysis-core/kernels/wgsl-common';
import {
  validatePointProcessPositive,
  validatePointProcessView
} from '../gpu-spatial-analysis/point-process/point-process-utils';

/** Largest event count: bounds the per-query loop of the intensity kernel. */
const MAXIMUM_EVENT_COUNT = 65536;
/** Largest query count: bounds the per-event loop of the mass kernel. */
const MAXIMUM_QUERY_COUNT = 65536;
/** Largest distance matrix, 2^24 entries (64 MiB): the dense query-by-event table is read in full. */
const MAXIMUM_DISTANCE_COUNT = 16777216;

/** Edge-length quadrature-normalized network intensity. */
export type GPUNetworkKernelDensityProps = {
  id?: string;
  /** Query-major shortest path distances, queryCount * eventCount. +Infinity is disconnected. */
  distances: GraphDataView<'float32'>;
  eventCount: number;
  /** Network arc-length represented by each query. Zero allows extra evaluation-only locations. */
  quadratureLengths: GraphDataView<'float32'>;
  bandwidth: number;
  /** Query's coincident event row for leave-one-out, or 0xffffffff. Default includes all events. */
  excludedEventIds?: GraphDataView<'uint32'>;
  output: {
    intensity: GraphDataView<'float32'>;
    /** Integral of each uncorrected triangular kernel under the supplied edge-length quadrature. */
    kernelMass: GraphDataView<'float32'>;
    /** Per query: 0 valid, 1 invalid distance/length or an event with zero quadrature support. */
    status: GraphDataView<'uint32'>;
    /**
     * Optional named one-row status ports. `status` stays the per-query view, so the named ports
     * sit under `ports.status`.
     */
    ports?: {
      status: {
        /**
         * One-row `GPUInvalidStatusPort.invalidCount`: the number of queries whose `status` is 1.
         * Nothing is read back; it is a GPU reduction over `status`.
         */
        invalidCount?: GraphDataView<'uint32'>;
      };
    };
  };
};

/** The views the output writes, with the named status ports flattened. */
function getOutputViews(output: GPUNetworkKernelDensityProps['output']): GraphDataView[] {
  return [
    output.intensity,
    output.kernelMass,
    output.status,
    ...(output.ports?.status.invalidCount ? [output.ports.status.invalidCount] : [])
  ];
}

/**
 * Triangular shortest-path network KDE corrected by each event's total kernel mass over all
 * branches. Thus sum(queryLength * intensity) = eventCount when all events have support and
 * no self exclusions are requested. Junctions and terminal edges contribute their actual length;
 * disconnected components have zero cross contribution. This is a quadrature-normalized radial
 * kernel, not an equal-split heat kernel. Accuracy is controlled by caller quadrature resolution.
 * Distances are reusable resident GPUNetworkCostMatrix products in arc-length units. No routing
 * readback or planar-distance substitute occurs. O(queryCount * eventCount), O(eventCount) scratch.
 */
export class GPUNetworkKernelDensity implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUNetworkKernelDensityProps;
  constructor(props: GPUNetworkKernelDensityProps) {
    this.id = props.id ?? 'network-kernel-density';
    this.props = props;
    validateNetworkWritableViews(this.id, props.output, [
      {outputs: {intensity: props.output.intensity, status: props.output.status}},
      {
        outputs: {invalidCount: props.output.ports?.status.invalidCount},
        inputs: {status: props.output.status}
      }
    ]);
    const queryCount = props.quadratureLengths.length;
    const {id} = this;
    validateCapacity(id, 'eventCount', props.eventCount, 1, MAXIMUM_EVENT_COUNT);
    validateCapacity(id, 'quadratureLengths rows', queryCount, 1, MAXIMUM_QUERY_COUNT);
    validateMaximumCount(
      id,
      'distances rows (quadratureLengths rows times eventCount)',
      queryCount * props.eventCount,
      MAXIMUM_DISTANCE_COUNT
    );
    validatePointProcessPositive(props.bandwidth, id, 'bandwidth');
    validatePointProcessView(props.distances, 'float32', queryCount * props.eventCount);
    validatePointProcessView(props.quadratureLengths, 'float32', queryCount);
    if (props.excludedEventIds) {
      validatePointProcessView(props.excludedEventIds, 'uint32', queryCount);
    }
    validatePointProcessView(props.output.intensity, 'float32', queryCount);
    validatePointProcessView(props.output.kernelMass, 'float32', props.eventCount);
    validatePointProcessView(props.output.status, 'uint32', queryCount);
    if (props.output.ports?.status.invalidCount) {
      validatePointProcessView(props.output.ports.status.invalidCount, 'uint32', 1);
    }
    validateGraphOutputsDisjointFromInputs(this.id, getOutputViews(props.output), [
      props.distances,
      props.quadratureLengths,
      props.excludedEventIds
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const queryCount = props.quadratureLengths.length;
    validateNetworkViewBindings(id, graph, [
      props.distances,
      props.quadratureLengths,
      props.excludedEventIds,
      ...getOutputViews(props.output)
    ]);
    validateNetworkViewBindings(id, graph, [], {
      invalidEvents: props.eventCount * 4,
      shapeMass: props.eventCount * 4
    });
    const invalidEvents = createTransientView(
      graph,
      `${id}-invalid-events`,
      'uint32',
      props.eventCount
    );
    // The common 1 / bandwidth factor cancels in the corrected intensity. Keep its mass in
    // unscaled shape units so a representable final density survives an underflowing kernel.
    const shapeMass = createTransientView(graph, `${id}-shape-mass`, 'float32', props.eventCount);
    const declarations = `${IS_FINITE_FLOAT_WGSL}
${SCALED_QUOTIENT_WGSL}
fn kernelShape(distance: f32) -> f32 { return max(0.0, 1.0 - getScaledQuotient(distance, ${Math.fround(props.bandwidth)}f, 1.0, 0)); }
fn validDistance(distance: f32) -> bool { return distance >= 0.0 && (isFiniteFloat(distance) || bitcast<u32>(distance) == 0x7f800000u); }`;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-mass`,
        operation: 'GPUNetworkKernelDensity',
        variant: 'edge-length-mass',
        invocationCount: props.eventCount,
        bindings: [
          {name: 'distances', view: props.distances, type: 'f32', access: 'read'},
          {name: 'lengths', view: props.quadratureLengths, type: 'f32', access: 'read'},
          {name: 'mass', view: props.output.kernelMass, type: 'f32', access: 'read_write'},
          {name: 'shapeMass', view: shapeMass, type: 'f32', access: 'read_write'},
          {name: 'invalidEvents', view: invalidEvents, type: 'u32', access: 'read_write'}
        ],
        declarations,
        body: `var total = 0.0;
  var valid = true;
  for (var query = 0u; query < ${queryCount}u; query++) {
    let length = lengths[lengthsOffset + query];
    let distance = distances[distancesOffset + query * ${props.eventCount}u + index];
    valid = valid && validDistance(distance) && isFiniteFloat(length) && length >= 0.0;
    if (isFiniteFloat(distance) && length > 0.0) { total += length * kernelShape(distance); }
  }
  valid = valid && isFiniteFloat(total) && total > 0.0;
  shapeMass[shapeMassOffset + index] = total;
  var publishedMass = bitcast<f32>(0x7fc00000u | (index & 0u));
  if (valid) { publishedMass = getScaledQuotient(total, ${Math.fround(props.bandwidth)}f, 1.0, 0); }
  mass[massOffset + index] = publishedMass;
  invalidEvents[invalidEventsOffset + index] = select(1u, 0u, valid);`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-intensity`,
        operation: 'GPUNetworkKernelDensity',
        variant: 'normalized-intensity',
        invocationCount: queryCount,
        bindings: [
          {name: 'distances', view: props.distances, type: 'f32', access: 'read'},
          {name: 'shapeMass', view: shapeMass, type: 'f32', access: 'read'},
          {name: 'invalidEvents', view: invalidEvents, type: 'u32', access: 'read'},
          ...(props.excludedEventIds
            ? [
                {
                  name: 'excluded',
                  view: props.excludedEventIds,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'intensity', view: props.output.intensity, type: 'f32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations,
        body: `var total = 0.0;
  var valid = true;
  for (var event = 0u; event < ${props.eventCount}u; event++) {
    let eventValid = invalidEvents[invalidEventsOffset + event] == 0u;
    valid = valid && eventValid;
    ${props.excludedEventIds ? 'if (excluded[excludedOffset + index] == event) { continue; }' : ''}
    let distance = distances[distancesOffset + index * ${props.eventCount}u + event];
    if (eventValid && isFiniteFloat(distance)) { total += getScaledQuotient(kernelShape(distance), shapeMass[shapeMassOffset + event], 1.0, 0); }
  }
  valid = valid && isFiniteFloat(total);
  intensity[intensityOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), total, valid);
  status[statusOffset + index] = select(1u, 0u, valid);`
      })
    ];
    if (props.output.ports?.status.invalidCount) {
      nodes.push(
        ...new GPUReduction({
          id: `${id}-invalid-count`,
          input: props.output.status,
          output: props.output.ports.status.invalidCount,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
    }
    return nodes;
  }
}
