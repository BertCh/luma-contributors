// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GraphDataView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {RASTER_FINITE_WGSL, rasterBinding} from './raster-kernel-utils';
import type {GPURasterRegionGraphProps} from './gpu-raster-region-graph';

/** Each non-isolated component selects an outgoing edge and the component count at least halves. */
export function getRasterRegionBoruvkaRoundCount(
  regionCount: number,
  maximumIterations: number
): number {
  return Math.min(maximumIterations, Math.ceil(Math.log2(regionCount)));
}

/** Sparse raster-boundary edges, parallel component minima and monotone atomic union-find. */
export function createRasterRegionBoruvkaNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPURasterRegionGraphProps,
  id: string,
  parents: GraphDataView<'uint32'>
): GPUCommandNode<Parameters>[] {
  const count = props.grid.width * props.grid.height;
  const regions = props.regionCount;
  // Two slots per cell: right and down. Invalid/background/self edges use the missing-key sentinel.
  // Duplicates preserve raster adjacency without a sort or a quadratic scratch allocation.
  const first = createTransientView(graph, `${id}-edge-first`, 'uint32', count * 2);
  const second = createTransientView(graph, `${id}-edge-second`, 'uint32', count * 2);
  const minimumWeight = createTransientView(graph, `${id}-minimum-weight`, 'uint32', regions);
  const minimumEdge = createTransientView(graph, `${id}-minimum-edge`, 'uint32', regions);
  const atomic = (name: string, view: GraphDataView<'uint32'>) => ({
    name,
    view,
    type: 'atomic<u32>' as const,
    access: 'read_write' as const
  });
  const rootDeclarations = `fn findRoot(region: u32) -> u32 {
  var root = region;
  loop {
    let next = atomicLoad(&parents[parentsOffset + root]); if (next == root) { return root; }
    // Atomic path halving is safe while other invocations union: all links strictly decrease.
    let grandparent = atomicLoad(&parents[parentsOffset + next]);
    atomicMin(&parents[parentsOffset + root], grandparent); root = next;
  }
}`;
  const edgeBindings = [
    rasterBinding('first', first),
    rasterBinding('second', second),
    rasterBinding('weights', props.output.weights),
    atomic('parents', parents)
  ];
  const edgeBody = `let firstRegion = first[firstOffset + index]; let secondRegion = second[secondOffset + index];
if (firstRegion == 0xffffffffu) { return; }
let weight = weights[weightsOffset + firstRegion * ${regions}u + secondRegion];
if (!finite(weight) || weight > ${getWGSLFloatLiteral(props.mergeThreshold)}) { return; }
let firstRoot = findRoot(firstRegion); let secondRoot = findRoot(secondRegion);
if (firstRoot == secondRoot) { return; }`;
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-edge-list`,
      operation: 'GPURasterRegionGraph',
      invocationCount: count,
      bindings: [
        rasterBinding('labels', props.labels),
        rasterBinding('first', first, true),
        rasterBinding('second', second, true)
      ],
      body: `let region = labels[labelsOffset + index];
for (var direction = 0u; direction < 2u; direction++) {
  let edge = index * 2u + direction;
  first[firstOffset + edge] = 0xffffffffu; second[secondOffset + edge] = 0xffffffffu;
  let neighbor = index + select(1u, ${props.grid.width}u, direction == 1u);
  if (region >= ${regions}u || neighbor >= ${count}u || (direction == 0u && index % ${props.grid.width}u == ${props.grid.width - 1}u)) { continue; }
  let otherRegion = labels[labelsOffset + neighbor];
  if (otherRegion < ${regions}u && otherRegion != region) { first[firstOffset + edge] = region; second[secondOffset + edge] = otherRegion; }
}`
    })
  ];
  const rounds = getRasterRegionBoruvkaRoundCount(regions, props.maximumIterations);
  for (let round = 0; round < rounds; round++) {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-boruvka-clear-${round}`,
        operation: 'GPURasterRegionGraph',
        invocationCount: regions,
        bindings: [
          rasterBinding('minimumWeight', minimumWeight, true),
          rasterBinding('minimumEdge', minimumEdge, true)
        ],
        body: 'minimumWeight[minimumWeightOffset + index] = 0xffffffffu; minimumEdge[minimumEdgeOffset + index] = 0xffffffffu;'
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-boruvka-weight-${round}`,
        operation: 'GPURasterRegionGraph',
        invocationCount: count * 2,
        bindings: [...edgeBindings, atomic('minimumWeight', minimumWeight)],
        declarations: `${RASTER_FINITE_WGSL}${rootDeclarations}`,
        body: `${edgeBody}
let weightBits = bitcast<u32>(weight);
atomicMin(&minimumWeight[minimumWeightOffset + firstRoot], weightBits);
atomicMin(&minimumWeight[minimumWeightOffset + secondRoot], weightBits);`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-boruvka-edge-${round}`,
        operation: 'GPURasterRegionGraph',
        invocationCount: count * 2,
        bindings: [
          ...edgeBindings,
          rasterBinding('minimumWeight', minimumWeight),
          atomic('minimumEdge', minimumEdge)
        ],
        declarations: `${RASTER_FINITE_WGSL}${rootDeclarations}`,
        body: `${edgeBody}
let weightBits = bitcast<u32>(weight);
if (weightBits == minimumWeight[minimumWeightOffset + firstRoot]) { atomicMin(&minimumEdge[minimumEdgeOffset + firstRoot], index); }
if (weightBits == minimumWeight[minimumWeightOffset + secondRoot]) { atomicMin(&minimumEdge[minimumEdgeOffset + secondRoot], index); }`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-boruvka-union-${round}`,
        operation: 'GPURasterRegionGraph',
        invocationCount: regions,
        bindings: [
          rasterBinding('first', first),
          rasterBinding('second', second),
          rasterBinding('minimumEdge', minimumEdge),
          atomic('parents', parents)
        ],
        declarations: rootDeclarations,
        body: `let edge = minimumEdge[minimumEdgeOffset + index]; if (edge == 0xffffffffu) { return; }
loop {
  let firstRoot = findRoot(first[firstOffset + edge]); let secondRoot = findRoot(second[secondOffset + edge]);
  if (firstRoot == secondRoot) { break; }
  let lower = min(firstRoot, secondRoot); let upper = max(firstRoot, secondRoot);
  let exchanged = atomicCompareExchangeWeak(&parents[parentsOffset + upper], upper, lower);
  if (exchanged.exchanged) { break; }
}`
      })
    );
  }
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-boruvka-status-clear`,
      operation: 'GPURasterRegionGraph',
      invocationCount: 1,
      bindings: [rasterBinding('pending', props.output.status.iterationLimitReached, true)],
      body: 'pending[pendingOffset] = 0u;'
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-boruvka-pending`,
      operation: 'GPURasterRegionGraph',
      invocationCount: count * 2,
      bindings: [...edgeBindings, atomic('pending', props.output.status.iterationLimitReached)],
      declarations: `${RASTER_FINITE_WGSL}${rootDeclarations}`,
      body: `${edgeBody}\natomicOr(&pending[pendingOffset], 1u);`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-boruvka-mapping`,
      operation: 'GPURasterRegionGraph',
      invocationCount: regions,
      bindings: [atomic('parents', parents), rasterBinding('mapping', props.output.mapping, true)],
      declarations: rootDeclarations,
      body: 'mapping[mappingOffset + index] = findRoot(index);'
    })
  );
  return nodes;
}
