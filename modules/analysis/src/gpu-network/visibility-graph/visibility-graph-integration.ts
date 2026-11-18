// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUNetworkTopology} from '../../analysis-core/contracts/network-topology';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {createVisibilityGraphKernelNode} from '../../gpu-spatial-analysis/urban-morphology/visibility-graph-kernel';
import {GPUNetworkReachability} from '../network-reachability/gpu-network-reachability';
import {
  createCentralityBatchInfoNode,
  createCentralityExpandSeedsNode
} from '../network-centrality/network-centrality-passes';
import {alignElementCount, getSubView} from '../network-view-utils';

/** Adds all-viewpoint integration using symmetric CSR distances, with batch-sized frontier scratch. @internal */
export function addVisibilityGraphIntegration<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    /** Last centrality command: integration reuses scratch only after it finishes. */
    after?: string;
    topology: GPUNetworkTopology;
    pivots: GraphDataView<'uint32'>;
    laneCount: number;
    maximumIterations: number;
    localIterations: number;
    degree: GraphDataView<'uint32'>;
    metrics: GraphDataView<'float32x4'>;
    integration: GraphDataView<'float32x4'>;
    extendedIntegration: GraphDataView<'float32x4'>;
    converged: GraphDataView<'uint32'>;
    iterationLimitReached: GraphDataView<'uint32'>;
  }
): void {
  const {id, topology, pivots, laneCount} = props;
  let previousNodeId = props.after;
  const addNodes = (nodes: GPUCommandNode<Parameters> | readonly GPUCommandNode<Parameters>[]) => {
    previousNodeId = addVisibilityGraphMetricNodes(
      graph,
      'type' in nodes ? [nodes] : nodes,
      previousNodeId
    );
  };
  const count = topology.offsets.length - 1;
  const batchCount = pivots.length / laneCount;
  const seedStride = alignElementCount(laneCount);
  const seeds = createTransientView(graph, `${id}-seeds`, 'uint32', batchCount * seedStride);
  const batchInfo = createTransientView(
    graph,
    `${id}-batch-info`,
    'uint32',
    batchCount * alignElementCount(1)
  );
  const costs = createTransientView(graph, `${id}-costs`, 'float32', laneCount * count);
  // Per viewpoint: sampled reachable count excluding self, hop sum, observed maximum, sampled self.
  const totals = createTransientView(graph, `${id}-totals`, 'float32x4', count);
  const batchConverged = createTransientView(graph, `${id}-batch-converged`, 'uint32', 1);
  addNodes([
    createCentralityExpandSeedsNode(graph, {
      id: `${id}-expand-seeds`,
      nodeCount: count,
      laneCount,
      batchCount,
      seedStride,
      pivotNodes: pivots,
      expandedSeeds: seeds
    }),
    createCentralityBatchInfoNode(graph, {
      id: `${id}-batch-info`,
      laneCount,
      batchCount,
      batchInfo
    }),
    createVisibilityGraphKernelNode(graph, {
      id: `${id}-clear`,
      operation: id,
      invocationCount: count * 4,
      bindings: [{name: 'totals', view: totals, type: 'f32', access: 'read_write'}],
      body: 'totals[totalsOffset+index]=0.0;'
    })
  ]);
  for (let batch = 0; batch < batchCount; batch++) {
    addNodes(
      new GPUNetworkReachability({
        id: `${id}-batch-${batch}-reachability`,
        topology,
        laneCount,
        startNodes: getSubView(graph, seeds, batch * seedStride, laneCount),
        maximumIterations: Math.min(props.maximumIterations, count),
        localIterations: props.localIterations,
        output: {costs, status: {converged: batchConverged}}
      }).getCommandNodes(graph)
    );
    addNodes(
      createVisibilityGraphKernelNode(graph, {
        id: `${id}-batch-${batch}-accumulate`,
        operation: id,
        invocationCount: count,
        declarations: IS_FINITE_VALUE_WGSL,
        bindings: [
          {name: 'costs', view: costs, type: 'f32', access: 'read'},
          {name: 'pivots', view: pivots, type: 'u32', access: 'read'},
          {
            name: 'batchInfo',
            view: getSubView(graph, batchInfo, batch * alignElementCount(1), 1),
            type: 'u32',
            access: 'read'
          },
          {name: 'totals', view: totals, type: 'f32', access: 'read_write'}
        ],
        // Batch identity is runtime data; every batch uses identical WGSL and pipeline constants.
        body: `var reached=totals[totalsOffset+4u*index];var depth=totals[totalsOffset+4u*index+1u];
var maximum=totals[totalsOffset+4u*index+2u];var sampledSelf=totals[totalsOffset+4u*index+3u];
for(var lane=0u;lane<${laneCount}u;lane++){
 let distance=costs[costsOffset+lane*${count}u+index];
 let isSelf=pivots[pivotsOffset+batchInfo[batchInfoOffset]+lane]==index;
 sampledSelf+=f32(isSelf);
 if(!isSelf&&isFiniteValue(distance)){reached+=1.0;depth+=distance;maximum=max(maximum,distance);}
}
totals[totalsOffset+4u*index]=reached;totals[totalsOffset+4u*index+1u]=depth;
totals[totalsOffset+4u*index+2u]=maximum;totals[totalsOffset+4u*index+3u]=sampledSelf;`
      })
    );
    addNodes(
      createVisibilityGraphKernelNode(graph, {
        id: `${id}-batch-${batch}-status`,
        operation: id,
        invocationCount: 1,
        bindings: [
          {name: 'batchConverged', view: batchConverged, type: 'u32', access: 'read'},
          {name: 'converged', view: props.converged, type: 'u32', access: 'read_write'},
          {
            name: 'iterationLimitReached',
            view: props.iterationLimitReached,
            type: 'u32',
            access: 'read_write'
          }
        ],
        body: `let finished=batchConverged[batchConvergedOffset];converged[convergedOffset]&=finished;
iterationLimitReached[iterationLimitReachedOffset]|=1u-finished;`
      })
    );
  }
  addNodes(
    createVisibilityGraphKernelNode(graph, {
      id: `${id}-finalize`,
      operation: id,
      invocationCount: count,
      declarations: GET_NAN_WGSL,
      bindings: [
        {name: 'totals', view: totals, type: 'f32', access: 'read'},
        {name: 'degree', view: props.degree, type: 'u32', access: 'read'},
        {name: 'metrics', view: props.metrics, type: 'f32', access: 'read_write'},
        {name: 'integration', view: props.integration, type: 'f32', access: 'read_write'},
        {name: 'extended', view: props.extendedIntegration, type: 'f32', access: 'read_write'}
      ],
      body: `let observations=f32(${pivots.length}u)-totals[totalsOffset+4u*index+3u];
let scale=f32(${count - 1}u)/max(1.0,observations);
let reachable=totals[totalsOffset+4u*index]*scale;let total=totals[totalsOffset+4u*index+1u]*scale;
let maximum=totals[totalsOffset+4u*index+2u];let size=reachable+1.0;let missing=getNaN();
var mean=missing;var asymmetry=missing;var real=missing;var hillier=missing;
if(reachable>0.0){mean=total/reachable;}
if(size>2.0){asymmetry=2.0*(mean-1.0)/(size-2.0);let normalization=2.0*(size*(log2((size+2.0)/3.0)-1.0)+1.0)/((size-1.0)*(size-2.0));real=asymmetry/normalization;hillier=1.0/real;}
metrics[metricsOffset+4u*index]=f32(degree[degreeOffset+index]);metrics[metricsOffset+4u*index+1u]=reachable;
metrics[metricsOffset+4u*index+2u]=total;metrics[metricsOffset+4u*index+3u]=select(missing,reachable/total,total>0.0);
integration[integrationOffset+4u*index]=mean;integration[integrationOffset+4u*index+1u]=asymmetry;
integration[integrationOffset+4u*index+2u]=real;integration[integrationOffset+4u*index+3u]=hillier;
var pyramid=missing;var teklenburg=missing;var penn=missing;
if(size>2.0&&mean>1.0){let normalization=2.0*(size-log2(size)-1.0)/((size-1.0)*(size-2.0));pyramid=normalization/asymmetry;if(total-size+1.0>1.0){teklenburg=log(0.5*(size-2.0))/log(total-size+1.0);}let radius=min(maximum,size-1.0);let greatest=radius*(size-0.5*(radius+1.0));let least=size-1.0;if(greatest>least){penn=(greatest-total)/(greatest-least);}}
extended[extendedOffset+4u*index]=pyramid;extended[extendedOffset+4u*index+1u]=teklenburg;
extended[extendedOffset+4u*index+2u]=penn;extended[extendedOffset+4u*index+3u]=maximum;`
    })
  );
}

/**
 * Preserve the metric schedule across batches so frontier scratch has disjoint lifetimes.
 * Buffer hazards alone do not order a later batch's queue writes against the previous batch's
 * final cost reads: the compiler may eagerly seed later queues, retaining one large queue per
 * batch. Explicit command dependencies let the published allocator reuse those physical buffers.
 * @internal
 */
export function addVisibilityGraphMetricNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  nodes: readonly GPUCommandNode<Parameters>[],
  after?: string
): string | undefined {
  let previousNodeId = after;
  graph.add(
    nodes.map(node => {
      const ordered = previousNodeId
        ? {...node, dependsOn: [...(node.dependsOn ?? []), previousNodeId]}
        : node;
      previousNodeId = node.id;
      return ordered;
    })
  );
  return previousNodeId;
}
