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
  createWGSLDispatchGate,
  createWGSLKernelNode,
  type WGSLKernelBinding,
  type WGSLKernelProps
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUNetworkStructureProps} from './gpu-network-structure';

type WordView = GraphDataView<'uint32'>;

/** Reserve NONE and keep the auxiliary vertex IDs and binary-lifting offsets in uint32. */
export const GPU_NETWORK_STRUCTURE_MAXIMUM_AUXILIARY_COUNT = 0xfffffffe;

/** @internal Shape and conservative command-node bound, including all gated pointer jumps. */
export function getNetworkStructureResilienceShape(nodeCount: number, edgeCount: number) {
  const treeLevels = Math.max(1, Math.ceil(Math.log2(nodeCount)));
  const edgeRounds = Math.ceil(Math.log2(Math.max(edgeCount, 2)));
  const auxiliaryCount = edgeCount + nodeCount * treeLevels;
  const componentRounds = Math.ceil(Math.log2(Math.max(auxiliaryCount, 2)));
  // Count every emitted family, including zero-invocation edge nodes. The fixed part
  // includes optional articulation publication, so this remains an upper bound for all outputs.
  const fixedNodeCount = 18;
  const forestNodeCount = treeLevels * (7 + 3 * treeLevels);
  const tourNodeCount = 4 * edgeRounds;
  const liftingNodeCount = 6 * treeLevels;
  const auxiliaryNodeCount = componentRounds * (7 + 3 * componentRounds);
  const estimatedCommandNodeCount =
    fixedNodeCount + forestNodeCount + tourNodeCount + liftingNodeCount + auxiliaryNodeCount;
  return {treeLevels, edgeRounds, auxiliaryCount, componentRounds, estimatedCommandNodeCount};
}

/** @internal Build the resilience stage after GPUNetworkStructure validates its props and bindings. */
export function createNetworkStructureResilienceNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUNetworkStructureProps,
  nodeCount: number,
  edgeCount: number
): readonly GPUCommandNode<Parameters>[] {
  return new NetworkStructureResilience(id, props, nodeCount, edgeCount).getCommandNodes(graph);
}

/**
 * Internal parallel resilience stage; metadata and binding sizes are validated by the wrapper.
 *
 * Boruvka's unique minimum-slot choices build a spanning forest. An Euler tour ranks its arcs;
 * binary lifting then represents every fundamental cycle by O(log V) path segments. Active
 * segments connect their two halves, so connectivity of this implicit auxiliary graph is exactly
 * the equivalence closure of edges sharing a fundamental cycle (the biconnected edge blocks).
 * This is the tree/auxiliary-connectivity formulation of parallel biconnectivity; it needs neither
 * DFS nor a depth-sized loop. Scratch is O(E + V log V). All props are compile-time.
 * The auxiliary edges remain implicit: the published connectivity primitive requires materialized
 * adjacency, which here would add O(E log V) slots. Path segments enumerate those domain-specific
 * constraints in logarithmic work per original edge instead.
 *
 * In each connectivity round every nonisolated component either hooks or receives a hook;
 * consequently the number of unfinished components halves. Full pointer jumping after each
 * round makes ceil(log2 N) rounds sufficient, with no convergence approximation or readback.
 * Loops retain singleton blocks and never participate in articulation classification.
 */
class NetworkStructureResilience implements GPUCommandNodeProducer {
  constructor(
    readonly id: string,
    readonly props: GPUNetworkStructureProps,
    readonly nodeCount: number,
    readonly edgeCount: number
  ) {}

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount, edgeCount} = this;
    const {treeLevels, edgeRounds, auxiliaryCount, componentRounds} =
      getNetworkStructureResilienceShape(nodeCount, edgeCount);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const view = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', Math.max(length, 1));
    const read = (name: string, buffer: WordView): WGSLKernelBinding => ({
      name,
      view: buffer,
      type: 'u32',
      access: 'read'
    });
    const write = (name: string, buffer: WordView, atomic = false): WGSLKernelBinding => ({
      name,
      view: buffer,
      type: atomic ? 'atomic<u32>' : 'u32',
      access: 'read_write'
    });
    const declarations = `const VERTEX_COUNT: u32 = ${nodeCount}u;
const EDGE_COUNT: u32 = ${edgeCount}u;
const LEVEL_COUNT: u32 = ${treeLevels}u;
const NONE: u32 = 0xffffffffu;`;
    const append = (
      name: string,
      length: number,
      bindings: WGSLKernelBinding[],
      body: string,
      extraDeclarations = '',
      gate?: Pick<WGSLKernelProps<Parameters>, 'condition' | 'extraResources'>
    ) => {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-${name}`,
          operation: 'GPUNetworkStructure',
          variant: name.replace(/-\d+/g, ''),
          invocationCount: length,
          bindings,
          declarations: `${declarations}\n${extraDeclarations}`,
          body,
          ...gate
        })
      );
    };
    // Reuse one scratch allocation across connectivity rounds. A GPU flag suppresses all
    // remaining jumps after the parent table has become flat; copies share the same gate.
    const compress = (
      name: string,
      length: number,
      rounds: number,
      parents: WordView,
      scratch: WordView,
      outerDispatch?: WordView
    ) => {
      const pending = view(`${name}-pending`, 1);
      const dispatch = createWGSLDispatchGate<Parameters>(graph, {
        id: `${id}-${name}-dispatch`,
        operation: 'GPUNetworkStructure',
        shapes: [[length]]
      });
      append(
        `${name}-seed`,
        1,
        [
          write('pending', pending),
          ...(outerDispatch ? [read('outerDispatch', outerDispatch)] : [])
        ],
        outerDispatch
          ? 'pending[pendingOffset] = select(0u, 1u, outerDispatch[outerDispatchOffset] != 0u);'
          : 'pending[pendingOffset] = 1u;'
      );
      for (let round = 0; round < rounds; round++) {
        append(
          `${name}-gate-${round}`,
          1,
          [write('pending', pending), dispatch.binding],
          `
if (pending[pendingOffset] != 0u) { ${dispatch.armWGSL} }
else { ${dispatch.disarmWGSL} }
pending[pendingOffset] = 0u;`
        );
        append(
          `${name}-jump-${round}`,
          length,
          [read('parents', parents), write('scratch', scratch), write('pending', pending, true)],
          `
let parent = parents[parentsOffset + index];
let ancestor = parents[parentsOffset + parent];
scratch[scratchOffset + index] = ancestor;
if (ancestor != parent) { atomicOr(&pending[pendingOffset], 1u); }`,
          '',
          dispatch.getGate(0)
        );
        append(
          `${name}-copy-${round}`,
          length,
          [read('scratch', scratch), write('parents', parents)],
          'parents[parentsOffset + index] = scratch[scratchOffset + index];',
          '',
          dispatch.getGate(0)
        );
      }
    };
    const sources = view('sources', edgeCount * 2);
    const tree = view('tree', edgeCount);
    const components = view('components', nodeCount);
    const parents = view('parents', nodeCount);
    const vertexScratch = view('vertex-scratch', nodeCount);
    const choices = view('choices', nodeCount);
    const reverse = props.reverseEdgeSlots!;
    const adjacency = [
      read('offsets', props.topology.offsets),
      read('neighbors', props.topology.neighbors)
    ];
    append(
      'initialize-slots',
      edgeCount,
      [read('neighbors', props.topology.neighbors), write('sources', sources), write('tree', tree)],
      `
sources[sourcesOffset + index * 2u] = NONE;
sources[sourcesOffset + index * 2u + 1u] = neighbors[neighborsOffset + index];
tree[treeOffset + index] = 0u;`
    );
    append(
      'initialize-vertices',
      nodeCount,
      [
        read('offsets', props.topology.offsets),
        write('sources', sources),
        write('components', components)
      ],
      `
components[componentsOffset + index] = index;
let first = min(offsets[offsetsOffset + index], EDGE_COUNT);
let last = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
for (var edge = first; edge < last; edge++) { sources[sourcesOffset + edge * 2u] = index; }`
    );
    const edgeBindings = [read('sources', sources), read('reverse', reverse)];
    const validEdge = `let source = sources[sourcesOffset + index * 2u];
let destination = sources[sourcesOffset + index * 2u + 1u];
let twin = reverse[reverseOffset + index];
if (source >= VERTEX_COUNT || destination >= VERTEX_COUNT || twin >= EDGE_COUNT || index > twin) { return; }`;
    const forestPending = view('forest-pending', 1);
    const forestGate = createWGSLDispatchGate<Parameters>(graph, {
      id: `${id}-forest-dispatch`,
      operation: 'GPUNetworkStructure',
      shapes: [[nodeCount], [edgeCount]]
    });
    append('forest-seed', 1, [write('pending', forestPending)], 'pending[pendingOffset] = 1u;');
    for (let round = 0; round < treeLevels; round++) {
      append(
        `forest-gate-${round}`,
        1,
        [write('pending', forestPending), forestGate.binding],
        `
if (pending[pendingOffset] != 0u) { ${forestGate.armWGSL} }
else { ${forestGate.disarmWGSL} }
pending[pendingOffset] = 0u;`
      );
      append(
        `forest-clear-${round}`,
        nodeCount,
        [read('components', components), write('parents', parents), write('choices', choices)],
        `
parents[parentsOffset + index] = components[componentsOffset + index];
choices[choicesOffset + index] = NONE;`,
        '',
        forestGate.getGate(0)
      );
      append(
        `forest-choose-${round}`,
        edgeCount,
        [...edgeBindings, read('components', components), write('choices', choices, true)],
        `
${validEdge}
let first = components[componentsOffset + source];
let second = components[componentsOffset + destination];
if (first != second) {
  atomicMin(&choices[choicesOffset + first], index);
  atomicMin(&choices[choicesOffset + second], index);
}`,
        '',
        forestGate.getGate(1)
      );
      append(
        `forest-hook-${round}`,
        edgeCount,
        [
          ...edgeBindings,
          read('components', components),
          read('choices', choices),
          write('tree', tree),
          write('pending', forestPending, true)
        ],
        `
${validEdge}
let first = components[componentsOffset + source];
let second = components[componentsOffset + destination];
if (first != second && (choices[choicesOffset + first] == index || choices[choicesOffset + second] == index)) {
  tree[treeOffset + index] = 1u;
  tree[treeOffset + twin] = 1u;
  atomicOr(&pending[pendingOffset], 1u);
}`,
        '',
        forestGate.getGate(1)
      );
      // Minimum-ranked incident choices contain only mutual two-cycles. Break each at
      // its lower root, then jump the entire chosen forest, including high-ID stars.
      append(
        `forest-link-${round}`,
        nodeCount,
        [
          read('components', components),
          read('choices', choices),
          read('sources', sources),
          write('parents', parents)
        ],
        `
let root = components[componentsOffset + index];
var parent = root;
if (root == index) {
  let edge = choices[choicesOffset + index];
  if (edge != NONE) {
    let first = components[componentsOffset + sources[sourcesOffset + edge * 2u]];
    let second = components[componentsOffset + sources[sourcesOffset + edge * 2u + 1u]];
    let neighbor = select(first, second, first == index);
    if (choices[choicesOffset + neighbor] != edge || index > neighbor) { parent = neighbor; }
  }
}
parents[parentsOffset + index] = parent;`,
        '',
        forestGate.getGate(0)
      );
      compress(
        `forest-compress-${round}`,
        nodeCount,
        treeLevels,
        parents,
        vertexScratch,
        forestGate.view
      );
      append(
        `forest-publish-${round}`,
        nodeCount,
        [read('parents', parents), write('components', components)],
        'components[componentsOffset + index] = parents[parentsOffset + index];',
        '',
        forestGate.getGate(0)
      );
    }
    // One tour successor per directed tree slot. The CSR-neighbourhood loop visits each slot
    // once overall, including at high-degree vertices; it does not search once per tree edge.
    const successor = view('successor', edgeCount);
    const tourMinimum = view('tour-minimum', edgeCount);
    const edgeScratch = view('edge-scratch', edgeCount * 2);
    const tourDistance = view('tour-distance', edgeCount);
    append(
      'tour-clear',
      edgeCount,
      [write('successor', successor), write('minimum', tourMinimum)],
      `
successor[successorOffset + index] = index;
minimum[minimumOffset + index] = index;`
    );
    append(
      'tour-link',
      nodeCount,
      [
        read('offsets', props.topology.offsets),
        read('tree', tree),
        read('reverse', reverse),
        write('successor', successor)
      ],
      `
var first = NONE; var previous = NONE;
let begin = min(offsets[offsetsOffset + index], EDGE_COUNT);
let end = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
for (var edge = begin; edge < end; edge++) {
  if (tree[treeOffset + edge] == 0u) { continue; }
  if (first == NONE) { first = edge; }
  if (previous != NONE) { successor[successorOffset + reverse[reverseOffset + previous]] = edge; }
  previous = edge;
}
if (previous != NONE) { successor[successorOffset + reverse[reverseOffset + previous]] = first; }`
    );
    for (let round = 0; round < edgeRounds; round++) {
      append(
        `tour-minimum-${round}`,
        edgeCount,
        [read('successor', successor), read('minimum', tourMinimum), write('scratch', edgeScratch)],
        `
let next = successor[successorOffset + index];
scratch[scratchOffset + index * 2u] = successor[successorOffset + next];
scratch[scratchOffset + index * 2u + 1u] = min(minimum[minimumOffset + index], minimum[minimumOffset + next]);`
      );
      append(
        `tour-minimum-copy-${round}`,
        edgeCount,
        [
          write('successor', successor),
          write('minimum', tourMinimum),
          read('scratch', edgeScratch)
        ],
        `
successor[successorOffset + index] = scratch[scratchOffset + index * 2u];
minimum[minimumOffset + index] = scratch[scratchOffset + index * 2u + 1u];`
      );
    }
    // Rebuild successors, then cut at the component's minimum tour arc. That arc is the tail
    // (distance zero); suffix distances compare opposite arcs and orient the forest consistently.
    append(
      'tour-relink-clear',
      edgeCount,
      [write('successor', successor)],
      'successor[successorOffset + index] = index;'
    );
    append(
      'tour-relink',
      nodeCount,
      [
        read('offsets', props.topology.offsets),
        read('tree', tree),
        read('reverse', reverse),
        write('successor', successor)
      ],
      `
var first = NONE; var previous = NONE;
let begin = min(offsets[offsetsOffset + index], EDGE_COUNT);
let end = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
for (var edge = begin; edge < end; edge++) {
  if (tree[treeOffset + edge] == 0u) { continue; }
  if (first == NONE) { first = edge; }
  if (previous != NONE) { successor[successorOffset + reverse[reverseOffset + previous]] = edge; }
  previous = edge;
}
if (previous != NONE) { successor[successorOffset + reverse[reverseOffset + previous]] = first; }`
    );
    append(
      'tour-cut',
      edgeCount,
      [
        write('successor', successor),
        read('minimum', tourMinimum),
        write('distance', tourDistance)
      ],
      `
if (minimum[minimumOffset + index] == index) { successor[successorOffset + index] = index; distance[distanceOffset + index] = 0u; }
else { distance[distanceOffset + index] = 1u; }`
    );
    for (let round = 0; round < edgeRounds; round++) {
      append(
        `tour-rank-${round}`,
        edgeCount,
        [
          read('successor', successor),
          read('distance', tourDistance),
          write('scratch', edgeScratch)
        ],
        `
let next = successor[successorOffset + index];
scratch[scratchOffset + index * 2u] = successor[successorOffset + next];
scratch[scratchOffset + index * 2u + 1u] = distance[distanceOffset + index] + distance[distanceOffset + next];`
      );
      append(
        `tour-rank-copy-${round}`,
        edgeCount,
        [
          write('successor', successor),
          write('distance', tourDistance),
          read('scratch', edgeScratch)
        ],
        `
successor[successorOffset + index] = scratch[scratchOffset + index * 2u];
distance[distanceOffset + index] = scratch[scratchOffset + index * 2u + 1u];`
      );
    }
    const ancestors = view('ancestors', nodeCount * treeLevels);
    const parentEdges = view('parent-edges', nodeCount);
    const depth = view('depth', nodeCount);
    append(
      'tree-clear',
      nodeCount,
      [write('ancestors', ancestors), write('parentEdges', parentEdges), write('depth', depth)],
      `
ancestors[ancestorsOffset + index] = index;
parentEdges[parentEdgesOffset + index] = NONE;
depth[depthOffset + index] = 0u;`
    );
    // The tail arc returns to the root; the opposite arc leaves it and has larger distance.
    append(
      'tree-orient',
      edgeCount,
      [
        ...edgeBindings,
        read('tree', tree),
        read('distance', tourDistance),
        write('ancestors', ancestors),
        write('parentEdges', parentEdges),
        write('depth', depth)
      ],
      `
${validEdge}
if (tree[treeOffset + index] == 0u) { return; }
var child = source; var parent = destination;
if (distance[distanceOffset + index] > distance[distanceOffset + twin]) { child = destination; parent = source; }
ancestors[ancestorsOffset + child] = parent;
parentEdges[parentEdgesOffset + child] = index;
depth[depthOffset + child] = 1u;`
    );
    // Level is GPU state, rather than shader text or a pipeline override. Every level
    // binds the same complete table and uses the same pair of pipelines.
    const levelState = view('level-state', 1);
    append(
      'tree-level-seed',
      1,
      [write('levelState', levelState)],
      'levelState[levelStateOffset] = 1u;'
    );
    for (let level = 1; level < treeLevels; level++) {
      append(
        `tree-lift-${level}`,
        nodeCount,
        [write('ancestors', ancestors), read('levelState', levelState)],
        `
let level = levelState[levelStateOffset];
let previous = (level - 1u) * VERTEX_COUNT;
let next = level * VERTEX_COUNT;
let parent = ancestors[ancestorsOffset + previous + index];
ancestors[ancestorsOffset + next + index] = ancestors[ancestorsOffset + previous + parent];`
      );
      append(
        `tree-level-advance-${level}`,
        1,
        [write('levelState', levelState)],
        'levelState[levelStateOffset] += 1u;'
      );
    }
    // Compute depths independently; preserve the ancestor table used by cycle decomposition.
    append(
      'depth-seed',
      nodeCount,
      [read('ancestors', ancestors), write('parents', parents)],
      'parents[parentsOffset + index] = ancestors[ancestorsOffset + index];'
    );
    const depthScratch = view('depth-scratch', nodeCount);
    for (let round = 0; round < treeLevels; round++) {
      append(
        `depth-rank-${round}`,
        nodeCount,
        [
          read('parents', parents),
          read('depth', depth),
          write('scratch', vertexScratch),
          write('depthScratch', depthScratch)
        ],
        `
let parent = parents[parentsOffset + index];
scratch[scratchOffset + index] = parents[parentsOffset + parent];
depthScratch[depthScratchOffset + index] = depth[depthOffset + index] + depth[depthOffset + parent];`
      );
      append(
        `depth-copy-${round}`,
        nodeCount,
        [
          write('parents', parents),
          write('depth', depth),
          read('scratch', vertexScratch),
          read('depthScratch', depthScratch)
        ],
        `
parents[parentsOffset + index] = scratch[scratchOffset + index];
depth[depthOffset + index] = depthScratch[depthScratchOffset + index];`
      );
    }
    const activeSegments = view('activeSegments', nodeCount * treeLevels);
    const labels = view('labels', auxiliaryCount);
    const auxiliaryParents = view('auxiliary-parents', auxiliaryCount);
    const auxiliaryScratch = view('auxiliary-scratch', auxiliaryCount);
    append(
      'auxiliary-clear',
      auxiliaryCount,
      [write('labels', labels)],
      'labels[labelsOffset + index] = index;'
    );
    append(
      'segments-clear',
      nodeCount * treeLevels,
      [write('activeSegments', activeSegments)],
      'activeSegments[activeSegmentsOffset + index] = 0u;'
    );
    append(
      'segments-seed',
      nodeCount,
      [read('parentEdges', parentEdges), write('labels', labels)],
      `
let edge = parentEdges[parentEdgesOffset + index];
if (edge != NONE) { labels[labelsOffset + EDGE_COUNT + index] = edge; }`
    );
    // Bounded logarithmic path decomposition; callback code is substituted at construction,
    // without a per-edge list allocation. A selected segment covers exactly 2^level tree arcs.
    const cycleBindings = [
      ...edgeBindings,
      read('tree', tree),
      read('ancestors', ancestors),
      read('depth', depth)
    ];
    const cycleBody = (visit: string) => `
${validEdge}
if (source == destination || tree[treeOffset + index] != 0u) { return; }
var first = source; var second = destination;
if (depth[depthOffset + first] < depth[depthOffset + second]) { first = destination; second = source; }
let difference = depth[depthOffset + first] - depth[depthOffset + second];
for (var level = 0u; level < LEVEL_COUNT; level++) {
  if ((difference & (1u << level)) != 0u) {
    let segment = level * VERTEX_COUNT + first;
    ${visit}
    first = ancestors[ancestorsOffset + segment];
  }
}
if (first == second) { return; }
for (var remaining = LEVEL_COUNT; remaining > 0u; remaining--) {
  let level = remaining - 1u;
  let firstAncestor = ancestors[ancestorsOffset + level * VERTEX_COUNT + first];
  let secondAncestor = ancestors[ancestorsOffset + level * VERTEX_COUNT + second];
  if (firstAncestor != secondAncestor) {
    { let segment = level * VERTEX_COUNT + first; ${visit} }
    { let segment = level * VERTEX_COUNT + second; ${visit} }
    first = firstAncestor; second = secondAncestor;
  }
}
{ let segment = first; ${visit} }
{ let segment = second; ${visit} }`;
    append(
      'cycles-activate',
      edgeCount,
      [...cycleBindings, write('activeSegments', activeSegments, true)],
      cycleBody('atomicStore(&activeSegments[activeSegmentsOffset + segment], 1u);')
    );
    append(
      'segment-level-seed',
      1,
      [write('levelState', levelState)],
      'levelState[levelStateOffset] = LEVEL_COUNT - 1u;'
    );
    for (let level = treeLevels - 1; level > 0; level--) {
      append(
        `segments-activate-${level}`,
        nodeCount,
        [
          read('ancestors', ancestors),
          write('activeSegments', activeSegments, true),
          read('levelState', levelState)
        ],
        `
let level = levelState[levelStateOffset];
let segment = level * VERTEX_COUNT + index;
if (atomicLoad(&activeSegments[activeSegmentsOffset + segment]) != 0u) {
  let half = (level - 1u) * VERTEX_COUNT;
  let middle = ancestors[ancestorsOffset + half + index];
  atomicStore(&activeSegments[activeSegmentsOffset + half + index], 1u);
  atomicStore(&activeSegments[activeSegmentsOffset + half + middle], 1u);
}`
      );
      append(
        `segment-level-retreat-${level}`,
        1,
        [write('levelState', levelState)],
        'levelState[levelStateOffset] -= 1u;'
      );
    }
    const auxiliaryPending = view('auxiliary-pending', 1);
    const auxiliaryGate = createWGSLDispatchGate<Parameters>(graph, {
      id: `${id}-auxiliary-dispatch`,
      operation: 'GPUNetworkStructure',
      shapes: [[auxiliaryCount], [edgeCount], [nodeCount * treeLevels]]
    });
    const hook = (first: string, second: string) => `{
let firstLabel = labels[labelsOffset + ${first}];
let secondLabel = labels[labelsOffset + ${second}];
if (firstLabel != secondLabel) {
  atomicMin(&choices[choicesOffset + firstLabel], secondLabel);
  atomicMin(&choices[choicesOffset + secondLabel], firstLabel);
  atomicOr(&pending[pendingOffset], 1u);
}}`;
    append(
      'auxiliary-seed',
      1,
      [write('pending', auxiliaryPending)],
      'pending[pendingOffset] = 1u;'
    );
    for (let round = 0; round < componentRounds; round++) {
      append(
        `auxiliary-gate-${round}`,
        1,
        [write('pending', auxiliaryPending), auxiliaryGate.binding],
        `
if (pending[pendingOffset] != 0u) { ${auxiliaryGate.armWGSL} }
else { ${auxiliaryGate.disarmWGSL} }
pending[pendingOffset] = 0u;`
      );
      append(
        `auxiliary-parents-${round}`,
        auxiliaryCount,
        [write('choices', auxiliaryScratch)],
        'choices[choicesOffset + index] = NONE;',
        '',
        auxiliaryGate.getGate(0)
      );
      const hookBindings = [
        read('labels', labels),
        write('choices', auxiliaryScratch, true),
        write('pending', auxiliaryPending, true)
      ];
      append(
        `cycles-hook-${round}`,
        edgeCount,
        [...cycleBindings, ...hookBindings],
        cycleBody(hook('index', 'EDGE_COUNT + segment')),
        '',
        auxiliaryGate.getGate(1)
      );
      append(
        `segments-hook-${round}`,
        nodeCount * treeLevels,
        [
          read('activeSegments', activeSegments),
          read('ancestors', ancestors),
          read('parentEdges', parentEdges),
          ...hookBindings
        ],
        `
if (activeSegments[activeSegmentsOffset + index] == 0u) { return; }
let level = index / VERTEX_COUNT;
let vertex = index % VERTEX_COUNT;
if (level == 0u) {
  let edge = parentEdges[parentEdgesOffset + vertex];
  if (edge != NONE) { ${hook('EDGE_COUNT + index', 'edge')} }
} else {
  let half = (level - 1u) * VERTEX_COUNT;
  let middle = ancestors[ancestorsOffset + half + vertex];
  ${hook('EDGE_COUNT + index', 'EDGE_COUNT + half + vertex')}
  ${hook('EDGE_COUNT + index', 'EDGE_COUNT + half + middle')}
}`,
        '',
        auxiliaryGate.getGate(2)
      );
      // Minimum-neighbour choices on an undirected graph have only mutual two-cycles:
      // a longer cycle would require strictly decreasing component IDs.
      append(
        `auxiliary-link-${round}`,
        auxiliaryCount,
        [
          read('labels', labels),
          read('choices', auxiliaryScratch),
          write('parents', auxiliaryParents)
        ],
        `
let root = labels[labelsOffset + index];
var parent = root;
if (root == index) {
  let neighbor = choices[choicesOffset + index];
  if (neighbor != NONE && (choices[choicesOffset + neighbor] != index || index > neighbor)) { parent = neighbor; }
}
parents[parentsOffset + index] = parent;`,
        '',
        auxiliaryGate.getGate(0)
      );
      compress(
        `auxiliary-compress-${round}`,
        auxiliaryCount,
        componentRounds,
        auxiliaryParents,
        auxiliaryScratch,
        auxiliaryGate.view
      );
      append(
        `auxiliary-publish-${round}`,
        auxiliaryCount,
        [read('parents', auxiliaryParents), write('labels', labels)],
        'labels[labelsOffset + index] = parents[parentsOffset + index];',
        '',
        auxiliaryGate.getGate(0)
      );
    }
    const blocks = props.output.biconnectedComponents ?? view('blocks', edgeCount);
    const covered = view('covered', edgeCount);
    append(
      'covered-clear',
      edgeCount,
      [write('covered', covered)],
      'covered[coveredOffset + index] = 0u;'
    );
    append(
      'covered-tree',
      nodeCount,
      [
        read('parentEdges', parentEdges),
        read('activeSegments', activeSegments),
        write('covered', covered),
        read('reverse', reverse)
      ],
      `
let edge = parentEdges[parentEdgesOffset + index];
if (edge != NONE) {
  let value = activeSegments[activeSegmentsOffset + index];
  covered[coveredOffset + edge] = value;
  covered[coveredOffset + reverse[reverseOffset + edge]] = value;
}`
    );
    append(
      'publish-blocks',
      edgeCount,
      [
        ...edgeBindings,
        read('labels', labels),
        write('blocks', blocks),
        ...(props.output.bridges
          ? [read('tree', tree), read('covered', covered), write('bridges', props.output.bridges)]
          : [])
      ],
      `
let source = sources[sourcesOffset + index * 2u];
let destination = sources[sourcesOffset + index * 2u + 1u];
let twin = reverse[reverseOffset + index];
let valid = source < VERTEX_COUNT && destination < VERTEX_COUNT && twin < EDGE_COUNT;
var label = NONE;
if (valid) { label = labels[labelsOffset + min(index, twin)]; }
blocks[blocksOffset + index] = label;
${props.output.bridges ? 'bridges[bridgesOffset + index] = select(0u, 1u, valid && tree[treeOffset + index] != 0u && covered[coveredOffset + index] == 0u);' : ''}`
    );
    if (props.output.articulationPoints) {
      append(
        'publish-articulation',
        nodeCount,
        [
          ...adjacency,
          read('blocks', blocks),
          write('articulation', props.output.articulationPoints)
        ],
        `
var firstBlock = NONE; var articulationFlag = 0u;
let first = min(offsets[offsetsOffset + index], EDGE_COUNT);
let last = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
for (var edge = first; edge < last; edge++) {
  let blockLabel = blocks[blocksOffset + edge];
  if (neighbors[neighborsOffset + edge] == index || blockLabel == NONE) { continue; }
  if (firstBlock == NONE) { firstBlock = blockLabel; }
  else if (blockLabel != firstBlock) { articulationFlag = 1u; }
}
articulation[articulationOffset + index] = articulationFlag;`
      );
    }
    return nodes;
  }
}
