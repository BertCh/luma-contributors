// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../analysis-core/kernels/gpu-contributor-utils';
import {IS_FINITE_FLOAT_WGSL} from '../analysis-core/kernels/wgsl-common';
import {
  validatePointProcessPositive,
  validatePointProcessView
} from '../gpu-spatial-analysis/point-process/point-process-utils';

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
  };
};

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
    const queryCount = props.quadratureLengths.length;
    assert(Number.isInteger(props.eventCount) && props.eventCount > 0 && props.eventCount <= 65536);
    assert(queryCount > 0 && queryCount <= 65536 && queryCount * props.eventCount <= 16777216);
    validatePointProcessPositive(props.bandwidth);
    validatePointProcessView(props.distances, 'float32', queryCount * props.eventCount);
    validatePointProcessView(props.quadratureLengths, 'float32', queryCount);
    if (props.excludedEventIds) {
      validatePointProcessView(props.excludedEventIds, 'uint32', queryCount);
    }
    validatePointProcessView(props.output.intensity, 'float32', queryCount);
    validatePointProcessView(props.output.kernelMass, 'float32', props.eventCount);
    validatePointProcessView(props.output.status, 'uint32', queryCount);
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
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
    validateGraphViewsBelongToGraph(id, graph, [
      props.distances,
      props.quadratureLengths,
      props.excludedEventIds,
      ...Object.values(props.output)
    ]);
    const invalidEvents = createTransientView(
      graph,
      `${id}-invalid-events`,
      'uint32',
      props.eventCount
    );
    const declarations = `${IS_FINITE_FLOAT_WGSL}
fn kernel(distance: f32) -> f32 { return max(0.0, 1.0 - distance / ${Math.fround(props.bandwidth)}f) / ${Math.fround(props.bandwidth)}f; }
fn validDistance(distance: f32) -> bool { return distance >= 0.0 && (isFiniteFloat(distance) || bitcast<u32>(distance) == 0x7f800000u); }`;
    return [
      createWGSLKernelNode(graph, {
        id: `${id}-mass`,
        operation: 'GPUNetworkKernelDensity',
        variant: 'edge-length-mass',
        invocationCount: props.eventCount,
        bindings: [
          {name: 'distances', view: props.distances, type: 'f32', access: 'read'},
          {name: 'lengths', view: props.quadratureLengths, type: 'f32', access: 'read'},
          {name: 'mass', view: props.output.kernelMass, type: 'f32', access: 'read_write'},
          {name: 'invalidEvents', view: invalidEvents, type: 'u32', access: 'read_write'}
        ],
        declarations,
        body: `var total = 0.0;
  var valid = true;
  for (var query = 0u; query < ${queryCount}u; query++) {
    let length = lengths[lengthsOffset + query];
    let distance = distances[distancesOffset + query * ${props.eventCount}u + index];
    valid = valid && validDistance(distance) && isFiniteFloat(length) && length >= 0.0;
    if (isFiniteFloat(distance) && length > 0.0) { total += length * kernel(distance); }
  }
  valid = valid && isFiniteFloat(total) && total > 0.0;
  mass[massOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), total, valid);
  invalidEvents[invalidEventsOffset + index] = select(1u, 0u, valid);`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-intensity`,
        operation: 'GPUNetworkKernelDensity',
        variant: 'normalized-intensity',
        invocationCount: queryCount,
        bindings: [
          {name: 'distances', view: props.distances, type: 'f32', access: 'read'},
          {name: 'mass', view: props.output.kernelMass, type: 'f32', access: 'read'},
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
    valid = valid && invalidEvents[invalidEventsOffset + event] == 0u;
    ${props.excludedEventIds ? 'if (excluded[excludedOffset + index] == event) { continue; }' : ''}
    let distance = distances[distancesOffset + index * ${props.eventCount}u + event];
    if (isFiniteFloat(distance)) { total += kernel(distance) / mass[massOffset + event]; }
  }
  valid = valid && isFiniteFloat(total);
  intensity[intensityOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), total, valid);
  status[statusOffset + index] = select(1u, 0u, valid);`
      })
    ];
  }
}
