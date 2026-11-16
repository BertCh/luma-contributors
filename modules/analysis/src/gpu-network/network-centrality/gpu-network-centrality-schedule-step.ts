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
  validateCapacity,
  validateCommandNodeBudget,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GPUNetworkCentrality, type GPUNetworkCentralityProps} from './gpu-network-centrality';
import {validateCentralityLogicalRanges} from './network-centrality-bindings';

/** Pivot row counts and row bases retain the uint32 indexing envelope of network inputs. */
const NETWORK_CENTRALITY_SCHEDULE_MAXIMUM_PIVOT_COUNT = 0xffffffff;

/** One host submission, with a fixed padded pivot capacity to preserve the compiled graph. */
export type NetworkCentralityScheduleStep = {
  /** First row of the original pivot vector to copy into this submission's pivot buffer. */
  pivotOffset: number;
  /** Number of real rows to copy; fill remaining lanes with an invalid node (0xffffffff). */
  pivotCount: number;
  /** Fixed pivot buffer length and lane count for every submission. */
  laneCount: number;
  /** Write this value into the contributor's reset view before submitting the step. */
  reset: number;
};

/**
 * Lazily partitions any uint32-sized pivot vector into bounded graph submissions, without
 * allocating a plan proportional to the number of pivots. Compile one
 * GPUNetworkCentralityScheduleStep with pivotNodes.length = laneCount. For each descriptor,
 * upload its pivots (pad the last batch with 0xffffffff), update reset, then encode and submit
 * the same compiled graph on the same queue. Retain/copy the first pivotCount per-pivot outputs
 * before overwriting them on the next step. Node/edge betweenness and status remain GPU-resident.
 * Only reset=1 starts a new complete run; do not interleave runs using the same output buffers.
 *
 * For sampled estimates use sampling: 'sampled' and betweennessScale = nodeCount / the ORIGINAL
 * pivotCount, not the padded capacity or the step's pivotCount. This helper does not pick pivots.
 * There is no distance/dependency continuation across steps: each step must converge under the
 * existing maximumIterations bound. For longer paths use createNetworkCentralityContinuation.
 * Inspect the final accumulated status before using results.
 */
export function planNetworkCentralitySchedule(props: {
  pivotCount: number;
  laneCount: number;
}): Iterable<NetworkCentralityScheduleStep> & {readonly stepCount: number} {
  const id = 'network-centrality-schedule';
  validateCapacity(
    id,
    'pivotCount',
    props.pivotCount,
    1,
    NETWORK_CENTRALITY_SCHEDULE_MAXIMUM_PIVOT_COUNT
  );
  validateCapacity(id, 'laneCount', props.laneCount, 1, props.pivotCount);
  const {pivotCount, laneCount} = props;
  return {
    stepCount: Math.ceil(pivotCount / laneCount),
    *[Symbol.iterator]() {
      for (let pivotOffset = 0; pivotOffset < pivotCount; pivotOffset += laneCount) {
        yield {
          pivotOffset,
          pivotCount: Math.min(laneCount, pivotCount - pivotOffset),
          laneCount,
          reset: Number(pivotOffset === 0)
        };
      }
    }
  };
}

export type GPUNetworkCentralityScheduleStepProps = GPUNetworkCentralityProps & {
  /** One per-frame word: nonzero replaces accumulated outputs, zero adds/merges this batch. */
  reset: GraphDataView<'uint32'>;
};

/**
 * A reusable one-batch Brandes/closeness graph for a host-driven centrality schedule. It keeps
 * command nodes independent of the total source count: each encoding searches just
 * pivotNodes.length lanes. Node/edge totals add in submission order, converged uses min, and
 * failure/approximation flags use max. Per-pivot outputs describe only the current batch.
 *
 * Import persistent caller-owned buffers for accumulated outputs; reset must be nonzero on the
 * first encoding of a run. Runtime pivots and reset never change shader sources. Keep the same
 * laneCount, betweennessScale and other configuration for the entire schedule. Exact all-node
 * results have the parent contributor's tolerance; batching only changes the grouping of sums.
 * This public contributor lets hosts encode/queue bounded batches beside other contributors,
 * including several submissions without readbacks. The per-batch round budget still applies;
 * createNetworkCentralityContinuation supplies the host-controlled long-path alternative.
 * CPU oracle: computeNetworkCentralityOnCPU, or the committed networkx fixture.
 */
export class GPUNetworkCentralityScheduleStep implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUNetworkCentralityScheduleStepProps;
  readonly estimatedCommandNodeCount: number;
  private readonly contributor: GPUNetworkCentrality;

  constructor(props: GPUNetworkCentralityScheduleStepProps) {
    this.id = props.id ?? 'network-centrality-schedule-step';
    this.props = props;
    validateSinglePackedView(props.reset, ['uint32'], `${this.id} reset`);
    validateCapacity(this.id, 'reset.length', props.reset.length, 1, 1);
    validateRequirement(
      this.id,
      props.laneCount === undefined || props.laneCount === props.pivotNodes.length,
      'laneCount must equal pivotNodes.length (one fixed-capacity batch per encoding)'
    );
    this.contributor = new GPUNetworkCentrality(
      {
        ...props,
        id: this.id,
        laneCount: props.pivotNodes.length
      },
      true
    );
    const accumulated = [
      props.output.betweenness,
      props.output.edgeBetweenness,
      ...Object.values(props.output.status ?? {})
    ].filter(view => view !== undefined);
    validateCentralityLogicalRanges(
      this.id,
      {
        betweenness: props.output.betweenness,
        edgeBetweenness: props.output.edgeBetweenness,
        closeness: props.output.closeness,
        harmonicCloseness: props.output.harmonicCloseness,
        reachedCounts: props.output.reachedCounts,
        ...props.output.status
      },
      {reset: props.reset}
    );
    // Only each accumulated destination and reset coexist in a merge dispatch. Per-pivot
    // fields and different scalar statuses may share aligned prefixes across separate kernels.
    for (const destination of accumulated)
      if (destination.length > 0)
        validateDisjointWritableViews(
          this.id,
          {destination, reset: props.reset},
          {strictAlignedBindings: true}
        );
    this.estimatedCommandNodeCount =
      this.contributor.estimatedCommandNodeCount + accumulated.length;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    // All checks precede even the partial-output allocations. The inner contributor has exactly
    // these shapes, with zero-offset fresh scratch replacing the accumulated output bindings.
    this.contributor.validateGraphBindings(graph);
    validateGraphChunkBindingRanges(graph, id, {reset: props.reset});
    const output = {...props.output, status: {...props.output.status}};
    const merges: GPUCommandNode<Parameters>[] = [];
    const createPartial = <Format extends 'float32' | 'uint32'>(
      name: string,
      destination: GraphDataView<Format>,
      mode: 'sum' | 'min' | 'max'
    ) => {
      const partial = createTransientView(
        graph,
        `${id}-partial-${name}`,
        destination.format,
        destination.length
      );
      const type = destination.format === 'float32' ? 'f32' : 'u32';
      const initial = mode === 'min' ? '1u' : type === 'f32' ? '0.0' : '0u';
      const next = 'partialValues[partialValuesOffset + index]';
      merges.push(
        createWGSLKernelNode(graph, {
          id: `${id}-merge-${name}`,
          operation: 'GPUNetworkCentralityScheduleStep',
          variant: `${type}-${mode}`,
          invocationCount: destination.length,
          bindings: [
            {name: 'partialValues', view: partial, type, access: 'read'},
            {name: 'totals', view: destination, type, access: 'read_write'},
            {name: 'reset', view: props.reset, type: 'u32', access: 'read'}
          ],
          body: `var previous = ${initial};
  if (reset[resetOffset] == 0u) { previous = totals[totalsOffset + index]; }
  totals[totalsOffset + index] = ${mode === 'sum' ? `previous + ${next}` : `${mode}(previous, ${next})`};`
        })
      );
      return partial;
    };
    if (output.betweenness)
      output.betweenness = createPartial('betweenness', output.betweenness, 'sum');
    if (output.edgeBetweenness)
      output.edgeBetweenness = createPartial('edge-betweenness', output.edgeBetweenness, 'sum');
    for (const name of [
      'converged',
      'zeroWeightCycle',
      'iterationLimitReached',
      'approximation'
    ] as const) {
      const destination = output.status[name];
      if (destination)
        output.status[name] = createPartial(
          name,
          destination,
          name === 'converged' ? 'min' : 'max'
        );
    }
    return [
      ...new GPUNetworkCentrality({
        ...props,
        id: `${id}-partial`,
        laneCount: props.pivotNodes.length,
        output
      }).getCommandNodes(graph),
      ...merges
    ];
  }
}
