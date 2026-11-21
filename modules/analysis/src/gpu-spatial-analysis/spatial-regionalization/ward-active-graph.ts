// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getBoundedDispatchLayout,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  type WGSLDispatchGate
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {ROUNDED_DIFFERENCE_WGSL} from '../spatial-regression/rounded-difference-wgsl';
import {validateGPUStorageBindingByteLength} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUSpatialWeights} from '../spatial-weights/index';

/** Contracted undirected adjacency; every unique edge owns one cost and two incidence links. */
export type WardActiveAdjacency = {
  heads: GraphDataView<'uint32'>;
  edges: GraphDataView<'uint32x2'>;
  links: GraphDataView<'uint32x2'>;
  costs: GraphDataView<'float32'>;
};

/** GPU-resident contraction. Canonical endpoints and total-order argmins make hash insertion order immaterial. */
export class WardActiveGraph<Parameters> {
  readonly parents: GraphDataView<'uint32'>;
  readonly heads: GraphDataView<'uint32'>;
  readonly links: GraphDataView<'uint32x2'>;
  readonly changed: GraphDataView<'uint32'>;
  private readonly costs: GraphDataView<'float32'>[];
  private readonly previousEdges: GraphDataView<'uint32'>;
  readonly capacity: number;
  private readonly endpoints: GraphDataView<'uint32x2'>[];
  private readonly positions: GraphDataView<'uint32'>[];
  private readonly hash: GraphDataView<'uint32'>;
  private readonly counts: GraphDataView<'uint32'>;
  private readonly dispatch: WGSLDispatchGate<Parameters>;
  private readonly hashCapacity: number;
  private readonly dispatchWGSL: string;

  constructor(
    readonly graph: GPUCommandGraph<Parameters>,
    readonly id: string,
    readonly rowCount: number,
    readonly weights: GPUSpatialWeights
  ) {
    this.capacity = Math.max(1, weights.neighbors.length);
    this.hashCapacity = this.capacity * 2;
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'contracted edge hash',
      this.hashCapacity * 4,
      'reduce the adjacency slot capacity'
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'contracted edges',
      this.capacity * 8,
      'reduce the adjacency slot capacity'
    );
    const make = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    this.parents = make('parents', rowCount);
    this.heads = make('heads', rowCount);
    this.endpoints = [0, 1].map(index =>
      createTransientView(graph, `${id}-edges-${index}`, 'uint32x2', this.capacity)
    );
    this.positions = [0, 1].map(index => make(`hash-positions-${index}`, this.capacity));
    this.links = createTransientView(graph, `${id}-edge-links`, 'uint32x2', this.capacity);
    this.costs = [0, 1].map(index =>
      createTransientView(graph, `${id}-edge-costs-${index}`, 'float32', this.capacity)
    );
    this.changed = make('changed-regions', rowCount);
    this.previousEdges = make('previous-edges', this.capacity);
    this.hash = make('edge-hash', this.hashCapacity);
    this.counts = make('edge-counts', 2);
    this.dispatch = createWGSLDispatchGate(graph, {
      id: `${id}-edge-dispatch`,
      operation: 'GPUWardRegions',
      shapes: [[this.capacity, 256]]
    });
    const layout = getBoundedDispatchLayout(
      'GPUWardRegions',
      this.capacity,
      256,
      graph.device.limits.maxComputeWorkgroupsPerDimension
    );
    this.dispatchWGSL = `let workgroups = (activeCount + 255u) / 256u;
  dispatch[dispatchOffset] = min(workgroups, ${layout.x}u);
  dispatch[dispatchOffset + 1u] = select(1u, min((workgroups + ${layout.x - 1}u) / ${layout.x}u, ${layout.y}u), workgroups > ${layout.x}u);
  dispatch[dispatchOffset + 2u] = max(1u, (workgroups + ${layout.x * layout.y - 1}u) / ${layout.x * layout.y}u);`;
  }

  getInitializeNodes(): GPUCommandNode<Parameters>[] {
    const {graph, id, weights, capacity, hashCapacity} = this;
    return [
      createWGSLKernelNode(graph, {
        id: `${id}-initialize-edge-storage`,
        operation: 'GPUWardRegions',
        variant: 'initialize-edge-storage',
        invocationCount: Math.max(capacity, hashCapacity),
        bindings: [
          {name: 'hash', view: this.hash, type: 'u32', access: 'read_write'},
          {name: 'positions', view: this.positions[0], type: 'u32', access: 'read_write'},
          {name: 'edges', view: this.endpoints[0], type: 'u32', access: 'read_write'},
          {name: 'initialCosts', view: this.costs[0], type: 'f32', access: 'read_write'},
          {name: 'counts', view: this.counts, type: 'u32', access: 'read_write'}
        ],
        declarations: GET_NAN_WGSL,
        body: `if (index < ${hashCapacity}u) { hash[hashOffset + index] = 0xffffffffu; }
  if (index < ${capacity}u) { initialCosts[initialCostsOffset + index] = getNaN(); positions[positionsOffset + index] = 0xffffffffu; edges[edgesOffset + 2u * index] = 0xffffffffu; edges[edgesOffset + 2u * index + 1u] = 0xffffffffu; }
  if (index == 0u) { counts[countsOffset] = ${weights.neighbors.length}u; counts[countsOffset + 1u] = 0u; }`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-initialize-edges`,
        operation: 'GPUWardRegions',
        variant: 'initialize-edges',
        invocationCount: this.rowCount,
        bindings: [
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'edges', view: this.endpoints[0], type: 'u32', access: 'read_write'},
          {name: 'changed', view: this.changed, type: 'u32', access: 'read_write'},
          {name: 'parents', view: this.parents, type: 'u32', access: 'read_write'}
        ],
        body: `parents[parentsOffset + index] = index; changed[changedOffset + index] = 0u;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let other = neighbors[neighborsOffset + slot];
    if (other < ${this.rowCount}u) { edges[edgesOffset + 2u * slot] = min(index, other); edges[edgesOffset + 2u * slot + 1u] = max(index, other); }
  }`
      })
    ];
  }

  getRoundNodes(
    iteration: number,
    gate: WGSLDispatchGate<Parameters>,
    regions: GraphDataView<'float32'>,
    columnCount: number,
    rawFeatures = false
  ): {nodes: GPUCommandNode<Parameters>[]; adjacency: WardActiveAdjacency} {
    const {graph, capacity, rowCount, heads, links} = this;
    const prefix = `${this.id}-contract-${iteration}`;
    const source = iteration % 2;
    const destination = 1 - source;
    const edges = this.endpoints[destination];
    const costs = this.costs[destination];
    const regionStride = rawFeatures ? 2 * columnCount + 2 : columnCount + 1;
    const edgeKernel = {
      operation: 'GPUWardRegions',
      invocationCount: capacity,
      workgroupSize: 256,
      ...this.dispatch.getGate(0)
    };
    const countBinding = {name: 'counts', view: this.counts, type: 'u32', access: 'read'} as const;
    const nodes = [
      createWGSLKernelNode(graph, {
        id: `${prefix}-begin`,
        operation: 'GPUWardRegions',
        variant: 'contract-begin',
        invocationCount: 1,
        bindings: [
          {name: 'mergeDispatch', view: gate.view, type: 'u32', access: 'read'},
          {name: 'counts', view: this.counts, type: 'u32', access: 'read_write'},
          this.dispatch.binding
        ],
        body: `counts[countsOffset + 1u] = 0u;
  let activeCount = select(0u, counts[countsOffset], mergeDispatch[mergeDispatchOffset] != 0u);
  ${this.dispatchWGSL}`
      }),
      createWGSLKernelNode(graph, {
        id: `${prefix}-clear-hash`,
        variant: 'contract-clear-hash',
        ...edgeKernel,
        bindings: [
          countBinding,
          {name: 'positions', view: this.positions[source], type: 'u32', access: 'read'},
          {name: 'hash', view: this.hash, type: 'u32', access: 'read_write'}
        ],
        body: `if (index >= counts[countsOffset]) { return; }
  let slot = positions[positionsOffset + index];
  if (slot != 0xffffffffu) { hash[hashOffset + slot] = 0xffffffffu; }`
      }),
      createWGSLKernelNode(graph, {
        id: `${prefix}-map`,
        variant: 'contract-map',
        ...edgeKernel,
        bindings: [
          countBinding,
          {name: 'edges', view: this.endpoints[source], type: 'u32', access: 'read_write'},
          {name: 'changed', view: this.changed, type: 'u32', access: 'read'},
          {name: 'costs', view: this.costs[source], type: 'f32', access: 'read_write'},
          {name: 'parents', view: this.parents, type: 'u32', access: 'read'}
        ],
        declarations: GET_NAN_WGSL,
        body: `if (index >= counts[countsOffset]) { return; }
  let first = edges[edgesOffset + 2u * index]; let second = edges[edgesOffset + 2u * index + 1u];
  if (first == 0xffffffffu) { return; }
  if (changed[changedOffset + first] != 0u || changed[changedOffset + second] != 0u) { costs[costsOffset + index] = getNaN(); }
  // Each edge held active roots before the preceding merge, so only one parent hop is needed.
  let firstRoot = parents[parentsOffset + first]; let secondRoot = parents[parentsOffset + second];
  edges[edgesOffset + 2u * index] = min(firstRoot, secondRoot);
  edges[edgesOffset + 2u * index + 1u] = max(firstRoot, secondRoot);`
      }),
      createWGSLKernelNode(graph, {
        id: `${prefix}-clear-heads`,
        operation: 'GPUWardRegions',
        variant: 'contract-clear-heads',
        invocationCount: rowCount,
        ...gate.getGate(0),
        bindings: [
          {name: 'changed', view: this.changed, type: 'u32', access: 'read_write'},
          {name: 'heads', view: heads, type: 'u32', access: 'read_write'}
        ],
        body: 'heads[headsOffset + index] = 0xffffffffu; changed[changedOffset + index] = 0u;'
      }),
      createWGSLKernelNode(graph, {
        id: `${prefix}-unique`,
        variant: 'contract-unique',
        ...edgeKernel,
        bindings: [
          countBinding,
          {name: 'edges', view: this.endpoints[source], type: 'u32', access: 'read'},
          {name: 'positions', view: this.positions[source], type: 'u32', access: 'read_write'},
          {name: 'hash', view: this.hash, type: 'atomic<u32>', access: 'read_write'}
        ],
        body: `if (index >= counts[countsOffset]) { return; }
  positions[positionsOffset + index] = 0xffffffffu;
  let first = edges[edgesOffset + 2u * index]; let second = edges[edgesOffset + 2u * index + 1u];
  if (first == 0xffffffffu || first == second) { return; }
  var slot = ((first * 0x9e3779b9u) ^ (second * 0x85ebca6bu)) % ${this.hashCapacity}u;
  loop {
    let result = atomicCompareExchangeWeak(&hash[hashOffset + slot], 0xffffffffu, index);
    if (result.exchanged) { positions[positionsOffset + index] = slot; break; }
    // Weak CAS may fail spuriously without a competing insertion.
    if (result.old_value == 0xffffffffu) { continue; }
    let previous = result.old_value;
    if (edges[edgesOffset + 2u * previous] == first && edges[edgesOffset + 2u * previous + 1u] == second) { break; }
    slot = (slot + 1u) % ${this.hashCapacity}u;
  }`
      }),
      createWGSLKernelNode(graph, {
        id: `${prefix}-compact`,
        variant: 'contract-compact',
        ...edgeKernel,
        bindings: [
          {name: 'counts', view: this.counts, type: 'atomic<u32>', access: 'read_write'},
          {name: 'source', view: this.endpoints[source], type: 'u32', access: 'read'},
          {name: 'positions', view: this.positions[source], type: 'u32', access: 'read'},
          {name: 'edges', view: edges, type: 'u32', access: 'read_write'},
          {
            name: 'nextPositions',
            view: this.positions[destination],
            type: 'u32',
            access: 'read_write'
          },
          {name: 'previousEdges', view: this.previousEdges, type: 'u32', access: 'read_write'},
          {name: 'heads', view: heads, type: 'atomic<u32>', access: 'read_write'},
          {name: 'links', view: links, type: 'u32', access: 'read_write'}
        ],
        body: `if (index >= atomicLoad(&counts[countsOffset])) { return; }
  let hashSlot = positions[positionsOffset + index];
  if (hashSlot == 0xffffffffu) { return; }
  let destination = atomicAdd(&counts[countsOffset + 1u], 1u);
  previousEdges[previousEdgesOffset + destination] = index;
  let first = source[sourceOffset + 2u * index]; let second = source[sourceOffset + 2u * index + 1u];
  edges[edgesOffset + 2u * destination] = first; edges[edgesOffset + 2u * destination + 1u] = second;
  nextPositions[nextPositionsOffset + destination] = hashSlot;
  links[linksOffset + 2u * destination] = atomicExchange(&heads[headsOffset + first], destination);
  links[linksOffset + 2u * destination + 1u] = atomicExchange(&heads[headsOffset + second], destination);`
      }),
      createWGSLKernelNode(graph, {
        id: `${prefix}-commit`,
        operation: 'GPUWardRegions',
        variant: 'contract-commit',
        invocationCount: 1,
        bindings: [
          {name: 'mergeDispatch', view: gate.view, type: 'u32', access: 'read'},
          {name: 'counts', view: this.counts, type: 'u32', access: 'read_write'},
          this.dispatch.binding
        ],
        body: `if (mergeDispatch[mergeDispatchOffset] != 0u) { counts[countsOffset] = counts[countsOffset + 1u]; }
  let activeCount = select(0u, counts[countsOffset], mergeDispatch[mergeDispatchOffset] != 0u);
  ${this.dispatchWGSL}`
      }),
      createWGSLKernelNode(graph, {
        id: `${prefix}-costs`,
        variant: 'contract-costs',
        ...edgeKernel,
        declarations: rawFeatures ? ROUNDED_DIFFERENCE_WGSL : '',
        bindings: [
          countBinding,
          {name: 'edges', view: edges, type: 'u32', access: 'read'},
          {name: 'regions', view: regions, type: 'f32', access: 'read'},
          {name: 'previousEdges', view: this.previousEdges, type: 'u32', access: 'read'},
          {name: 'previousCosts', view: this.costs[source], type: 'f32', access: 'read'},
          {name: 'costs', view: costs, type: 'f32', access: 'read_write'}
        ],
        body: `if (index >= counts[countsOffset]) { return; }
  let previous = previousCosts[previousCostsOffset + previousEdges[previousEdgesOffset + index]];
  if ((bitcast<u32>(previous) & 0x7fffffffu) <= 0x7f800000u) { costs[costsOffset + index] = previous; return; }
  let first = edges[edgesOffset + 2u * index] * ${regionStride}u;
  let second = edges[edgesOffset + 2u * index + 1u] * ${regionStride}u;
  let firstCount = regions[regionsOffset + first]; let secondCount = regions[regionsOffset + second];
  ${rawFeatures ? `let runtimeZero = bitcast<u32>(regions[regionsOffset + first + ${regionStride - 1}u]);` : ''}
  var increase = 0.0;
  for (var column = 1u; column <= ${columnCount}u; column++) {
    ${
      rawFeatures
        ? `let originDifference = getRoundedDifference(regions[regionsOffset + first + ${columnCount}u + column], regions[regionsOffset + second + ${columnCount}u + column], runtimeZero);
    let difference = originDifference + getRoundedDifference(regions[regionsOffset + first + column] / firstCount, regions[regionsOffset + second + column] / secondCount, runtimeZero);`
        : 'let difference = regions[regionsOffset + first + column] / firstCount - regions[regionsOffset + second + column] / secondCount;'
    }
    increase += firstCount * secondCount / (firstCount + secondCount) * difference * difference;
  }
  costs[costsOffset + index] = increase;`
      })
    ];
    return {nodes, adjacency: {heads, edges, links, costs}};
  }

  getPublishNodes(output: GraphDataView<'uint32'>): GPUCommandNode<Parameters>[] {
    // Separate source/destination buffers make pointer doubling race free even on deep merge trees.
    const nodes: GPUCommandNode<Parameters>[] = [];
    let source = this.parents;
    let destination = output;
    for (let step = 0; step < Math.max(1, Math.ceil(Math.log2(this.rowCount))); step++) {
      nodes.push(
        createWGSLKernelNode(this.graph, {
          id: `${this.id}-publish-${step}`,
          operation: 'GPUWardRegions',
          variant: 'publish-labels',
          invocationCount: this.rowCount,
          bindings: [
            {name: 'source', view: source, type: 'u32', access: 'read'},
            {name: 'destination', view: destination, type: 'u32', access: 'read_write'}
          ],
          body: 'let parent = source[sourceOffset + index]; destination[destinationOffset + index] = source[sourceOffset + parent];'
        })
      );
      [source, destination] = [destination, source];
    }
    if (source !== output)
      nodes.push(
        createWGSLKernelNode(this.graph, {
          id: `${this.id}-publish-copy`,
          operation: 'GPUWardRegions',
          variant: 'publish-copy',
          invocationCount: this.rowCount,
          bindings: [
            {name: 'source', view: source, type: 'u32', access: 'read'},
            {name: 'destination', view: output, type: 'u32', access: 'read_write'}
          ],
          body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
        })
      );
    return nodes;
  }
}
