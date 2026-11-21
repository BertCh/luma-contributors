// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {snapshotConstructionFields} from '../analysis-core/kernels/construction-snapshot';
import {validateGraphChunkBindingRanges} from '../analysis-core/kernels/graph-chunk-dispatch';
import {createWGSLKernelNode} from '../analysis-core/kernels/wgsl-kernel-nodes';
import {
  isGraphDataView,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateMaximumCount,
  validateRequirement
} from '../analysis-core/kernels/gpu-contributor-utils';
import {SCALED_QUOTIENT_WGSL} from '../analysis-core/kernels/scaled-quotient-wgsl';
import {IS_FINITE_FLOAT_WGSL} from '../analysis-core/kernels/wgsl-common';
import {
  validatePointProcessPositive,
  validatePointProcessView
} from '../gpu-spatial-analysis/point-process/point-process-utils';
import {
  getNetworkPhysicalBuffer,
  validateNetworkViewBindings,
  validateNetworkWritableViews
} from './network-view-utils';

/** Largest event count: bounds the per-query loop of the intensity kernel. */
const MAXIMUM_EVENT_COUNT = 65536;
/** Largest query count: bounds the per-event loop of the mass kernel. */
const MAXIMUM_QUERY_COUNT = 65536;
/** Largest distance matrix, 2^24 entries (64 MiB): the dense query-by-event table is read in full. */
const MAXIMUM_DISTANCE_COUNT = 16777216;

/** Caller-owned destinations for network KDE results and diagnostics. */
export type GPUNetworkKernelDensityOutput = {
  intensity: GraphDataView<'float32'>;
  /** Integral of each uncorrected triangular kernel under the supplied edge-length quadrature. */
  kernelMass: GraphDataView<'float32'>;
  /** One word per query: 1 for a usable intensity, 0 for invalid or suppressed results. */
  validity: GraphDataView<'uint32'>;
  /** Optional scalar diagnostics. */
  status?: {
    /** Optional one-row GPU count of queries whose validity is 0. */
    invalidCount?: GraphDataView<'uint32'>;
    /**
     * Optional one-row flag: 1 when `costsComplete` is nonzero and `bandwidth` does not exceed
     * `costLimit` (when given), so every distance inside the kernel support is final;
     * otherwise 0. Requires `costsComplete`. Requesting this port retains incumbent intensity
     * and mass. Without it, supplied evidence of loss suppresses both to NaN and clears validity.
     */
    complete?: GraphDataView<'uint32'>;
  };
};

/** Edge-length quadrature-normalized network intensity. */
export type GPUNetworkKernelDensityProps = {
  id?: string;
  /**
   * Query-major shortest path distances, queryCount * eventCount. +Infinity is disconnected. A
   * `GPUNetworkCostMatrix` stopped at its round budget, or built with a `costLimit` below
   * `bandwidth`, also leaves +Infinity (or an overestimate) inside the kernel support: pass its
   * completeness word as `costsComplete` and its limit as `costLimit`, and optionally request
   * `output.status.complete` to retain uncertified incumbent values.
   */
  distances: GraphDataView<'float32'>;
  /**
   * Optional one-row completeness word of `distances`, for example the `output.status.converged`
   * of `GPUNetworkCostMatrix`: nonzero means every entry is final. Required by
   * `output.status.complete`.
   */
  costsComplete?: GraphDataView<'uint32'>;
  /**
   * Optional one-row cost limit the matrix was built with (the `costLimit` view of
   * `GPUNetworkCostMatrix`). The certificate is 0 when the limit
   * is below `bandwidth`: a distance inside the kernel support would then read as disconnected.
   * Pass it whenever the matrix ran with a cost limit.
   */
  costLimit?: GraphDataView<'float32'>;
  eventCount: number;
  /** Network arc-length represented by each query. Zero allows extra evaluation-only locations. */
  quadratureLengths: GraphDataView<'float32'>;
  bandwidth: number;
  /** Query's coincident event row for leave-one-out, or 0xffffffff. Default includes all events. */
  excludedEventIds?: GraphDataView<'uint32'>;
  output: GPUNetworkKernelDensityOutput;
};

/** The views the output writes, keyed by their public paths. */
function getOutputViews(output: GPUNetworkKernelDensityOutput) {
  return {
    'output.intensity': output.intensity,
    'output.kernelMass': output.kernelMass,
    'output.validity': output.validity,
    'output.status.invalidCount': output.status?.invalidCount,
    'output.status.complete': output.status?.complete
  };
}

/** The input views the graph reads, keyed by their public paths. */
function getInputViews(props: GPUNetworkKernelDensityProps) {
  return {
    distances: props.distances,
    quadratureLengths: props.quadratureLengths,
    excludedEventIds: props.excludedEventIds,
    costsComplete: props.costsComplete,
    costLimit: props.costLimit
  };
}

/**
 * Triangular shortest-path network KDE corrected by each event's total kernel mass over all
 * branches. Thus sum(queryLength * intensity) = eventCount when all events have support and
 * no self exclusions are requested. Junctions and terminal edges contribute their actual length;
 * disconnected components have zero cross contribution. This is a quadrature-normalized radial
 * kernel, not an equal-split heat kernel. Accuracy is controlled by caller quadrature resolution.
 * Distances are reusable resident GPUNetworkCostMatrix products in arc-length units. No routing
 * readback or planar-distance substitute occurs. O(queryCount * eventCount) work,
 * O(eventCount) scratch plus O(queryCount) when counting invalid queries.
 */
export class GPUNetworkKernelDensity implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUNetworkKernelDensityProps;
  constructor(props: GPUNetworkKernelDensityProps) {
    const captured = snapshotConstructionFields(props, [
      'id',
      'distances',
      'costsComplete',
      'costLimit',
      'eventCount',
      'quadratureLengths',
      'bandwidth',
      'excludedEventIds',
      'output'
    ]);
    const id = captured.id ?? 'network-kernel-density';
    validateRequirement(
      id,
      !('ports' in captured.output),
      'output.ports was removed; use output.status.invalidCount and output.status.complete, with per-query results in output.validity'
    );
    const output = snapshotConstructionFields(captured.output, [
      'intensity',
      'kernelMass',
      'validity',
      'status'
    ]);
    const {status} = output;
    validateRequirement(
      id,
      !isGraphDataView(status),
      'output.status is a named status map; use output.validity for the per-query view'
    );
    validateRequirement(
      id,
      status === undefined || (status !== null && typeof status === 'object'),
      'output.status must be an object when provided'
    );
    props = {
      ...captured,
      output: {
        ...output,
        ...(status === undefined
          ? {}
          : {status: snapshotConstructionFields(status, ['invalidCount', 'complete'])})
      }
    };
    this.id = id;
    this.props = props;
    validateNetworkWritableViews(this.id, props.output, [
      {outputs: {intensity: props.output.intensity, validity: props.output.validity}}
    ]);
    const queryCount = props.quadratureLengths.length;
    validateCapacity(id, 'eventCount', props.eventCount, 1, MAXIMUM_EVENT_COUNT);
    validateCapacity(id, 'quadratureLengths rows', queryCount, 1, MAXIMUM_QUERY_COUNT);
    validateMaximumCount(
      id,
      'distances rows (quadratureLengths rows times eventCount)',
      queryCount * props.eventCount,
      MAXIMUM_DISTANCE_COUNT
    );
    validatePointProcessPositive(props.bandwidth, id, 'bandwidth');
    validatePointProcessView(
      id,
      'distances',
      props.distances,
      ['float32'],
      queryCount * props.eventCount
    );
    validatePointProcessView(
      id,
      'quadratureLengths',
      props.quadratureLengths,
      ['float32'],
      queryCount
    );
    if (props.excludedEventIds) {
      validatePointProcessView(
        id,
        'excludedEventIds',
        props.excludedEventIds,
        ['uint32'],
        queryCount
      );
    }
    validatePointProcessView(
      id,
      'output.intensity',
      props.output.intensity,
      ['float32'],
      queryCount
    );
    validatePointProcessView(
      id,
      'output.kernelMass',
      props.output.kernelMass,
      ['float32'],
      props.eventCount
    );
    validatePointProcessView(id, 'output.validity', props.output.validity, ['uint32'], queryCount);
    const invalidCount = props.output.status?.invalidCount;
    if (invalidCount !== undefined) {
      validatePointProcessView(id, 'output.status.invalidCount', invalidCount, ['uint32'], 1);
    }
    validateCompletenessPorts(
      id,
      props.costsComplete,
      props.costLimit,
      props.output.status?.complete
    );
    validateGraphOutputsDisjointFromInputs(id, getOutputViews(props.output), getInputViews(props));
    validateGraphOutputsDisjointFromInputs(
      id,
      {'output.status.invalidCount': invalidCount},
      {'output.validity': props.output.validity}
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const queryCount = props.quadratureLengths.length;
    const complete = props.output.status?.complete;
    const invalidCount = props.output.status?.invalidCount;
    const bindings = {...getInputViews(props), ...getOutputViews(props.output)};
    const physicalHandles = new Map<unknown, GraphDataView['buffer']>();
    for (const [path, view] of Object.entries(bindings)) {
      if (!view) continue;
      const physical = getNetworkPhysicalBuffer(view);
      const previous = physicalHandles.get(physical);
      validateRequirement(
        id,
        !previous || previous === view.buffer,
        `${path} conflicts with another view of the same physical buffer; a physical buffer must be imported once`
      );
      physicalHandles.set(physical, view.buffer);
    }
    validateGraphChunkBindingRanges(graph, id, bindings);
    validateNetworkViewBindings(id, graph, [], {
      invalidEvents: props.eventCount * 4,
      shapeMass: props.eventCount * 4,
      ...(invalidCount ? {invalidQueries: queryCount * 4} : {})
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
          {name: 'validity', view: props.output.validity, type: 'u32', access: 'read_write'}
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
  validity[validityOffset + index] = select(0u, 1u, valid);`
      })
    ];
    // Positive float32 values have the same order as their unsigned IEEE words. Read the
    // horizon as u32 so neither a NaN comparison nor GPU flush-to-zero can certify lost support.
    const bandwidthBits = new Uint32Array(Float32Array.of(props.bandwidth).buffer)[0];
    const horizonDeclarations = `fn hasSufficientNetworkDensityHorizon(horizonBits:u32)->bool {
  return horizonBits >= ${bandwidthBits}u && horizonBits <= 0x7f800000u;
}`;
    if (!complete && (props.costsComplete || props.costLimit)) {
      // A separate parallel pass preserves the numerical kernels and rechecks resident evidence
      // on every replay. Each output has its own pass so packed disjoint outputs remain legal.
      const evidenceBindings = [
        ...(props.costsComplete
          ? [
              {
                name: 'costsComplete',
                view: props.costsComplete,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : []),
        ...(props.costLimit
          ? [
              {
                name: 'costLimit',
                view: props.costLimit,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : [])
      ];
      const isComplete = [
        ...(props.costsComplete ? ['costsComplete[costsCompleteOffset] != 0u'] : []),
        ...(props.costLimit
          ? ['hasSufficientNetworkDensityHorizon(costLimit[costLimitOffset])']
          : [])
      ].join(' && ');
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-suppress-mass`,
          operation: 'GPUNetworkKernelDensity',
          variant: 'suppress-uncertified-mass',
          declarations: props.costLimit ? horizonDeclarations : undefined,
          invocationCount: props.eventCount,
          bindings: [
            ...evidenceBindings,
            {name: 'mass', view: props.output.kernelMass, type: 'f32', access: 'read_write'}
          ],
          body: `if (!(${isComplete})) { mass[massOffset + index] = bitcast<f32>(0x7fc00000u | (index & 0u)); }`
        }),
        createWGSLKernelNode(graph, {
          id: `${id}-suppress-intensity`,
          operation: 'GPUNetworkKernelDensity',
          variant: 'suppress-uncertified-intensity',
          declarations: props.costLimit ? horizonDeclarations : undefined,
          invocationCount: queryCount,
          bindings: [
            ...evidenceBindings,
            {name: 'intensity', view: props.output.intensity, type: 'f32', access: 'read_write'},
            {name: 'validity', view: props.output.validity, type: 'u32', access: 'read_write'}
          ],
          body: `if (!(${isComplete})) {
  intensity[intensityOffset + index] = bitcast<f32>(0x7fc00000u | (index & 0u));
  validity[validityOffset + index] = 0u;
}`
        })
      );
    }
    if (complete && props.costsComplete) {
      // The triangular kernel is zero at and beyond `bandwidth`, so the distances are complete
      // for this estimate exactly when the matrix is and its limit reaches the bandwidth.
      // Share the integer horizon predicate with suppression, including subnormals and NaNs.
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-complete`,
          operation: 'GPUNetworkKernelDensity',
          variant: props.costLimit ? 'complete-limit' : 'complete',
          declarations: props.costLimit ? horizonDeclarations : undefined,
          invocationCount: 1,
          bindings: [
            {name: 'costsComplete', view: props.costsComplete, type: 'u32', access: 'read'},
            ...(props.costLimit
              ? [
                  {
                    name: 'costLimit',
                    view: props.costLimit,
                    type: 'u32' as const,
                    access: 'read' as const
                  }
                ]
              : []),
            {name: 'complete', view: complete, type: 'u32', access: 'read_write'}
          ],
          body: `let isComplete = costsComplete[costsCompleteOffset] != 0u${
            props.costLimit
              ? ' && hasSufficientNetworkDensityHorizon(costLimit[costLimitOffset])'
              : ''
          };
  complete[completeOffset] = select(0u, 1u, isComplete);`
        })
      );
    }
    if (invalidCount) {
      const invalidQueries = createTransientView(
        graph,
        `${id}-invalid-queries`,
        'uint32',
        queryCount
      );
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-mark-invalid-queries`,
          operation: 'GPUNetworkKernelDensity',
          variant: 'mark-invalid-queries',
          invocationCount: queryCount,
          bindings: [
            {name: 'validity', view: props.output.validity, type: 'u32', access: 'read'},
            {name: 'invalidQueries', view: invalidQueries, type: 'u32', access: 'read_write'}
          ],
          body: `invalidQueries[invalidQueriesOffset + index] = select(0u, 1u, validity[validityOffset + index] == 0u);`
        }),
        ...new GPUReduction({
          id: `${id}-invalid-count`,
          input: invalidQueries,
          output: invalidCount,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
    }
    return nodes;
  }
}

/** Validates the optional matrix completeness inputs and the `complete` port they drive. */
function validateCompletenessPorts(
  id: string,
  costsComplete: GraphDataView<'uint32'> | undefined,
  costLimit: GraphDataView<'float32'> | undefined,
  complete: GraphDataView<'uint32'> | undefined
): void {
  for (const [name, view, format] of [
    ['costsComplete', costsComplete, 'uint32'],
    ['costLimit', costLimit, 'float32'],
    ['output.status.complete', complete, 'uint32']
  ] as const) {
    if (view !== undefined) {
      validatePointProcessView(id, name, view, [format], 1);
    }
  }
  validateRequirement(
    id,
    !complete || Boolean(costsComplete),
    'output.status.complete requires costsComplete, the matrix completeness word'
  );
}
