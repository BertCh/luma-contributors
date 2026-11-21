// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRefinementOutputLength} from './refinement-output-validation';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateOutlineOutputPasses} from '../outline-geometry/validate-outline-output-passes';
import {validateRefinementScratchByteLengths} from './refinement-storage-validation';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {EXACT_ORIENTATION_WGSL} from '../../analysis-core/kernels/exact-orientation-wgsl';
import {
  validateSinglePackedView,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getReadPointWGSL} from '../shared/spatial-wgsl-point-helpers';

/** GEOS canonical comparison order; polygon children are exterior ring then holes. */
export const GPU_CANONICAL_GEOMETRY_TYPE = {
  point: 0,
  multiPoint: 1,
  lineString: 2,
  linearRing: 3,
  multiLineString: 4,
  polygon: 5,
  multiPolygon: 6,
  collection: 7
} as const;

/** A postorder geometry tree. Every child ID is smaller than its parent ID; maximum nesting is 64. */
export type GPUCanonicalGeometryTree = {
  positions: GraphDataView<'float32x2'>;
  /** [type, firstVertexOrChildSlot, vertexOrChildCount, stableSourceId]. Leaves: types 0,2,3. */
  nodes: GraphDataView<'uint32x4'>;
  /** Child node IDs, grouped by parent. Empty components remain explicit zero-count nodes. */
  children: GraphDataView<'uint32'>;
};
export type GPUNormalizeGeometryProps = GPUCanonicalGeometryTree & {
  id?: string;
  /**
   * Compile-time. Deepest allowed chain of nested nodes, counting the leaf (1 to 64). A deeper
   * node reports status 2. Collection ordering runs one dependency level per round, so a smaller
   * bound for known shallow trees cuts the number of dispatches. Defaults to 64.
   */
  maximumNestingDepth?: number;
  output: {
    positions: GraphDataView<'float32x2'>;
    children: GraphDataView<'uint32'>;
    /** Per node: 0 success, 1 nonfinite leaf, 2 invalid tree/range/depth, uncertain orientation or invalid child. */
    status: GraphDataView<'uint32'>;
  };
};

/** Comparison of normalized subtrees (GEOS canonical order). Needs `nodes`, `normalized` and `ordered` bindings. */
const COMPARE_NODES_WGSL = `fn normalizedPoint(vertex: u32) -> vec2f { return vec2f(normalized[normalizedOffset + vertex * 2u], normalized[normalizedOffset + vertex * 2u + 1u]); }
fn comparePoint(left: vec2f, right: vec2f) -> i32 {
  if (left.x != right.x) { return select(1, -1, left.x < right.x); }
  if (left.y != right.y) { return select(1, -1, left.y < right.y); }
  return 0;
}
// Iterative lexicographic tree comparison: no recursion, hashes or host traversal.
fn compareNodes(leftRoot: u32, rightRoot: u32) -> i32 {
  var leftStack: array<u32, 64>; var rightStack: array<u32, 64>; var childStack: array<u32, 64>;
  var depth = 0u; var left = leftRoot; var right = rightRoot;
  loop {
    let leftType = field(left, 0u); let rightType = field(right, 0u);
    if (leftType != rightType) { return select(1, -1, leftType < rightType); }
    let leftFirst = field(left, 1u); let rightFirst = field(right, 1u);
    let leftCount = field(left, 2u); let rightCount = field(right, 2u);
    if (isLeaf(leftType)) {
      if (leftCount != rightCount) { return select(1, -1, leftCount < rightCount); }
      for (var vertex = 0u; vertex < min(leftCount, rightCount); vertex++) {
        let comparison = comparePoint(normalizedPoint(leftFirst + vertex), normalizedPoint(rightFirst + vertex));
        if (comparison != 0) { return comparison; }
      }
      if (leftCount != rightCount) { return select(1, -1, leftCount < rightCount); }
    } else if (min(leftCount, rightCount) > 0u) {
      leftStack[depth] = left; rightStack[depth] = right; childStack[depth] = 1u; depth++;
      left = ordered[orderedOffset + leftFirst]; right = ordered[orderedOffset + rightFirst]; continue;
    } else if (leftCount != rightCount) { return select(1, -1, leftCount < rightCount); }
    var advanced = false;
    loop {
      if (depth == 0u) { return 0; }
      let parentLeft = leftStack[depth - 1u]; let parentRight = rightStack[depth - 1u];
      let nextChild = childStack[depth - 1u];
      let countLeft = field(parentLeft, 2u); let countRight = field(parentRight, 2u);
      if (nextChild < min(countLeft, countRight)) {
        childStack[depth - 1u]++; left = ordered[orderedOffset + field(parentLeft, 1u) + nextChild]; right = ordered[orderedOffset + field(parentRight, 1u) + nextChild]; advanced = true; break;
      }
      if (countLeft != countRight) { return select(1, -1, countLeft < countRight); }
      depth--;
    }
    if (!advanced) { return 0; }
  }
  return 0;
}`;

/** Normalizes points, lines, rings, polygons, multis and nested collections without flattening their structure. */
/** Canonical comparisons use 64-entry private traversal stacks. */
const MAXIMUM_NESTING_DEPTH = 64;

export class GPUNormalizeGeometry implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUNormalizeGeometryProps;
  constructor(props: GPUNormalizeGeometryProps) {
    this.id = props.id ?? 'normalize-geometry';
    props = {...props, output: {...props.output}};
    this.props = props;
    validateCapacity(
      this.id,
      'maximumNestingDepth',
      props.maximumNestingDepth ?? MAXIMUM_NESTING_DEPTH,
      1,
      MAXIMUM_NESTING_DEPTH
    );
    validateSinglePackedView(props.positions, ['float32x2'], `${this.id} positions`, {
      id: this.id,
      contributor: 'GPUNormalizeGeometry'
    });
    validateSinglePackedView(props.nodes, ['uint32x4'], `${this.id} nodes`, {
      id: this.id,
      contributor: 'GPUNormalizeGeometry'
    });
    validateSinglePackedView(props.children, ['uint32'], `${this.id} children`, {
      id: this.id,
      contributor: 'GPUNormalizeGeometry'
    });
    validateSinglePackedView(props.output.positions, ['float32x2'], `${this.id} output.positions`, {
      id: this.id,
      contributor: 'GPUNormalizeGeometry'
    });
    validateSinglePackedView(props.output.children, ['uint32'], `${this.id} output.children`, {
      id: this.id,
      contributor: 'GPUNormalizeGeometry'
    });
    validateSinglePackedView(props.output.status, ['uint32'], `${this.id} output.status`, {
      id: this.id,
      contributor: 'GPUNormalizeGeometry'
    });
    validateRefinementOutputLength(
      {id: this.id, contributor: 'GPUNormalizeGeometry'},
      'output.positions',
      props.output.positions,
      props.positions.length,
      'positions.length'
    );
    validateRefinementOutputLength(
      {id: this.id, contributor: 'GPUNormalizeGeometry'},
      'output.children',
      props.output.children,
      props.children.length,
      'children.length'
    );
    validateRefinementOutputLength(
      {id: this.id, contributor: 'GPUNormalizeGeometry'},
      'output.status',
      props.output.status,
      props.nodes.length,
      'nodes.length'
    );
    validateOutlineOutputPasses(this.id, props.output, [
      ['positions', 'status'],
      ['children', 'status']
    ]);
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.positions,
      props.nodes,
      props.children
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphChunkBindingRanges(
      graph,
      id,
      Object.fromEntries(
        [props.positions, props.nodes, props.children, ...Object.values(props.output)].map(
          (view, index) => [`binding.${index}`, view]
        )
      )
    );
    validateRefinementScratchByteLengths(graph, id, {
      nodeScratch: props.nodes.length * 4,
      childScratch: props.children.length * 4
    });
    const holeFlags = createTransientView(
      graph,
      `${id}-hole-flags`,
      'uint32',
      Math.max(props.nodes.length, 1)
    );
    return [
      createWGSLKernelNode(graph, {
        id: `${id}-copy-children`,
        operation: 'GPUNormalizeGeometry',
        variant: 'copy-children',
        invocationCount: props.children.length,
        bindings: [
          {name: 'children', view: props.children, type: 'u32', access: 'read'},
          {name: 'ordered', view: props.output.children, type: 'u32', access: 'read_write'}
        ],
        body: 'ordered[orderedOffset + index] = children[childrenOffset + index];'
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-copy-positions`,
        operation: 'GPUNormalizeGeometry',
        variant: 'copy-positions',
        invocationCount: props.positions.length * 2,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'normalized', view: props.output.positions, type: 'f32', access: 'read_write'}
        ],
        body: 'normalized[normalizedOffset + index] = positions[positionsOffset + index];'
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-hole-init`,
        operation: 'GPUNormalizeGeometry',
        variant: 'hole-init',
        invocationCount: props.nodes.length,
        bindings: [{name: 'holeFlags', view: holeFlags, type: 'u32', access: 'read_write'}],
        body: 'holeFlags[holeFlagsOffset + index] = 0u;'
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-hole-mark`,
        operation: 'GPUNormalizeGeometry',
        variant: 'hole-mark',
        invocationCount: props.nodes.length,
        bindings: [
          {name: 'nodes', view: props.nodes, type: 'u32', access: 'read'},
          {name: 'children', view: props.children, type: 'u32', access: 'read'},
          {name: 'holeFlags', view: holeFlags, type: 'atomic<u32>', access: 'read_write'}
        ],
        // Every non-first child of a polygon is a hole ring; marking is O(children) in total.
        body: `if (nodes[nodesOffset + index * 4u] != 5u) { return; }
let first = nodes[nodesOffset + index * 4u + 1u]; let count = nodes[nodesOffset + index * 4u + 2u];
let limit = min(count, ${props.children.length}u - min(first, ${props.children.length}u));
for (var child = 1u; child < limit; child++) {
  let holeNode = children[childrenOffset + first + child];
  if (holeNode < index) { atomicStore(&holeFlags[holeFlagsOffset + holeNode], 1u); }
}`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-leaves`,
        operation: 'GPUNormalizeGeometry',
        variant: 'leaves',
        invocationCount: props.nodes.length,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'nodes', view: props.nodes, type: 'u32', access: 'read'},
          {name: 'holeFlags', view: holeFlags, type: 'u32', access: 'read'},
          {name: 'normalized', view: props.output.positions, type: 'f32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `${EXACT_ORIENTATION_WGSL}
fn field(node: u32, component: u32) -> u32 { return nodes[nodesOffset + node * 4u + component]; }
fn isLeaf(kind: u32) -> bool { return kind == 0u || kind == 2u || kind == 3u; }
${getReadPointWGSL('point', 'positions')}
fn comparePoint(left: vec2f, right: vec2f) -> i32 {
  if (left.x != right.x) { return select(1, -1, left.x < right.x); }
  if (left.y != right.y) { return select(1, -1, left.y < right.y); }
  return 0;
}`,
        // Leaves only read raw positions and write their own vertex range, so they are independent.
        body: `let node = index;
let kind = field(node, 0u); let first = field(node, 1u); let count = field(node, 2u);
if (kind > 7u) { status[statusOffset + node] = 2u; return; }
if (!isLeaf(kind)) { return; }
status[statusOffset + node] = 0u;
{
    if (first > ${props.positions.length}u || count > ${props.positions.length}u - first || (kind == 0u && count > 1u)) { status[statusOffset + node] = 2u; return; }
    for (var vertex = 0u; vertex < count; vertex++) { if (any((bitcast<vec2u>(point(first + vertex)) & vec2u(0x7f800000u)) == vec2u(0x7f800000u))) { status[statusOffset + node] = 1u; } }
    if (status[statusOffset + node] != 0u || count < 2u) { return; }
    let closed = all(point(first) == point(first + count - 1u));
    var rotation = 0u; var reverse = false;
    if (closed || kind == 3u) {
      let uniqueCount = count - select(0u, 1u, closed);
      for (var vertex = 0u; vertex < uniqueCount; vertex++) {
        if (comparePoint(point(first + vertex), point(first + rotation)) < 0) { rotation = vertex; }
      }
      // The extreme-vertex predicate avoids shoelace cancellation on translated rings.
      let center = point(first + rotation);
      var previous = center; var next = center;
      for (var step = 1u; step < uniqueCount; step++) {
        if (all(previous == center)) { previous = point(first + (rotation + uniqueCount - step) % uniqueCount); }
        if (all(next == center)) { next = point(first + (rotation + step) % uniqueCount); }
        if (any(previous != center) && any(next != center)) { break; }
      }
      var winding = 0;
      if (any(previous != center) && any(next != center)) { winding = orientSign(previous, center, next); }
      if (winding == 2) { status[statusOffset + node] = 2u; return; }
      // Holes are the non-first children of polygons, marked once by the hole-mark kernel.
      let hole = kind == 3u && holeFlags[holeFlagsOffset + node] != 0u;
      reverse = select(winding > 0, winding < 0, hole);
      for (var vertex = 0u; vertex < count; vertex++) {
        let offset = vertex % uniqueCount;
        let source = (rotation + select(offset, uniqueCount - offset, reverse)) % uniqueCount;
        let value = point(first + source); normalized[normalizedOffset + (first + vertex) * 2u] = value.x; normalized[normalizedOffset + (first + vertex) * 2u + 1u] = value.y;
      }
    } else {
      for (var vertex = 0u; vertex < count; vertex++) { let comparison = comparePoint(point(first + vertex), point(first + count - 1u - vertex)); if (comparison != 0) { reverse = comparison > 0; break; } }
      if (reverse) { for (var vertex = 0u; vertex < count; vertex++) { let value = point(first + count - 1u - vertex); normalized[normalizedOffset + (first + vertex) * 2u] = value.x; normalized[normalizedOffset + (first + vertex) * 2u + 1u] = value.y; } }
    }
}`
      }),
      ...this._getCollectionNodes(graph)
    ];
  }

  /**
   * Orders the children of collections level by level. A collection is processed in the round after
   * the last of its children, so rounds run in parallel across nodes, and inside a collection every
   * child finds its own rank by counting the siblings that sort before it. Leaves are round 1.
   */
  private _getCollectionNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const maximumDepth = props.maximumNestingDepth ?? 64;
    const nodeCount = props.nodes.length;
    const childCount = props.children.length;
    const view = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', Math.max(length, 1));
    const rounds = view('rounds', nodeCount);
    const heights = view('heights', nodeCount);
    const scratchChildren = view('scratch-children', childCount);
    const slotParents = view('slot-parents', childCount);
    const roundState = view('round-state', 1);
    const nodeFields = `fn field(node: u32, component: u32) -> u32 { return nodes[nodesOffset + node * 4u + component]; }
fn isLeaf(kind: u32) -> bool { return kind == 0u || kind == 2u || kind == 3u; }`;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-round-init`,
        operation: 'GPUNormalizeGeometry',
        variant: 'round-init',
        invocationCount: nodeCount,
        bindings: [
          {name: 'nodes', view: props.nodes, type: 'u32', access: 'read'},
          {name: 'rounds', view: rounds, type: 'u32', access: 'read_write'},
          {name: 'heights', view: heights, type: 'u32', access: 'read_write'},
          {name: 'roundState', view: roundState, type: 'u32', access: 'read_write'}
        ],
        declarations: nodeFields,
        // Leaves (and unknown kinds, which the leaf pass failed) are finished in round 1.
        body: `if (index == 0u) { roundState[roundStateOffset] = 2u; }
let kind = field(index, 0u);
let finished = kind > 7u || isLeaf(kind);
rounds[roundsOffset + index] = select(0u, 1u, finished);
heights[heightsOffset + index] = select(0u, 1u, finished);`
      })
    ];
    if (childCount > 0) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-slot-parent-init`,
          operation: 'GPUNormalizeGeometry',
          variant: 'slot-parent-init',
          invocationCount: childCount,
          bindings: [{name: 'slotParents', view: slotParents, type: 'u32', access: 'read_write'}],
          body: 'slotParents[slotParentsOffset + index] = 0xffffffffu;'
        }),
        createWGSLKernelNode(graph, {
          id: `${id}-slot-parent-mark`,
          operation: 'GPUNormalizeGeometry',
          variant: 'slot-parent-mark',
          invocationCount: nodeCount,
          bindings: [
            {name: 'nodes', view: props.nodes, type: 'u32', access: 'read'},
            {name: 'slotParents', view: slotParents, type: 'u32', access: 'read_write'}
          ],
          declarations: nodeFields,
          // Valid trees give every collection its own range of child slots.
          body: `let kind = field(index, 0u);
if (kind > 7u || isLeaf(kind)) { return; }
let first = field(index, 1u); let count = field(index, 2u);
if (first > ${childCount}u || count > ${childCount}u - first) { return; }
for (var child = 0u; child < count; child++) { slotParents[slotParentsOffset + first + child] = index; }`
        })
      );
    }
    const lastRound = maximumDepth + 1;
    for (let round = 2; round <= lastRound; round++) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-round-${round}-nodes`,
          operation: 'GPUNormalizeGeometry',
          variant: 'collection-round-nodes',
          invocationCount: nodeCount,
          bindings: [
            {name: 'nodes', view: props.nodes, type: 'u32', access: 'read'},
            {name: 'ordered', view: props.output.children, type: 'u32', access: 'read_write'},
            {name: 'scratch', view: scratchChildren, type: 'u32', access: 'read'},
            {name: 'rounds', view: rounds, type: 'atomic<u32>', access: 'read_write'},
            {name: 'roundState', view: roundState, type: 'u32', access: 'read'},
            {name: 'heights', view: heights, type: 'u32', access: 'read_write'},
            {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
          ],
          declarations: `${nodeFields}
const MAXIMUM_DEPTH: u32 = ${maximumDepth}u;`,
          // A node sorted in the previous round copies its ranked children back; a pending node
          // whose children all finished in earlier rounds computes its status and height.
          body: `let ROUND = roundState[roundStateOffset];
let kind = field(index, 0u); let first = field(index, 1u); let count = field(index, 2u);
if (kind > 7u || isLeaf(kind)) { return; }
let finishedRound = atomicLoad(&rounds[roundsOffset + index]);
if (finishedRound != 0u) {
  if (finishedRound == ROUND - 1u && status[statusOffset + index] == 0u) {
    for (var child = select(0u, 1u, kind == 5u); child < count; child++) {
      ordered[orderedOffset + first + child] = scratch[scratchOffset + first + child];
    }
  }
  return;
}
let valid = first <= ${childCount}u && count <= ${childCount}u - first;
var ready = true;
if (valid) {
  for (var child = 0u; child < count; child++) {
    let childNode = ordered[orderedOffset + first + child];
    if (childNode < index) {
      let childRound = atomicLoad(&rounds[roundsOffset + childNode]);
      if (childRound == 0u || childRound >= ROUND) { ready = false; }
    }
  }
}
if (!ready) { return; }
var failed = !valid;
var height = 1u;
if (valid) {
  for (var child = 0u; child < count; child++) {
    let childNode = ordered[orderedOffset + first + child];
    if (childNode >= index || status[statusOffset + childNode] != 0u) { failed = true; } else { height = max(height, heights[heightsOffset + childNode] + 1u); }
  }
}
if (height > MAXIMUM_DEPTH) { failed = true; }
status[statusOffset + index] = select(0u, 2u, failed);
heights[heightsOffset + index] = height;
atomicStore(&rounds[roundsOffset + index], ROUND);`
        })
      );
      if (childCount > 0) {
        nodes.push(
          createWGSLKernelNode(graph, {
            id: `${id}-round-${round}-rank`,
            operation: 'GPUNormalizeGeometry',
            variant: 'collection-round-rank',
            invocationCount: childCount,
            bindings: [
              {name: 'nodes', view: props.nodes, type: 'u32', access: 'read'},
              {name: 'normalized', view: props.output.positions, type: 'f32', access: 'read'},
              {name: 'ordered', view: props.output.children, type: 'u32', access: 'read'},
              {name: 'scratch', view: scratchChildren, type: 'u32', access: 'read_write'},
              {name: 'slotParents', view: slotParents, type: 'u32', access: 'read'},
              {name: 'rounds', view: rounds, type: 'u32', access: 'read'},
              {name: 'roundState', view: roundState, type: 'u32', access: 'read'},
              {name: 'status', view: props.output.status, type: 'u32', access: 'read'}
            ],
            declarations: `${nodeFields}
${COMPARE_NODES_WGSL}`,
            // Descending order, ties kept in source order: a child's rank is the number of
            // siblings that compare greater, or equal with a smaller position.
            body: `let ROUND = roundState[roundStateOffset];
let parent = slotParents[slotParentsOffset + index];
if (parent == 0xffffffffu || rounds[roundsOffset + parent] != ROUND || status[statusOffset + parent] != 0u) { return; }
let first = field(parent, 1u); let count = field(parent, 2u);
let start = select(0u, 1u, field(parent, 0u) == 5u);
let position = index - first;
if (position < start) { return; }
let selected = ordered[orderedOffset + index];
var rank = 0u;
for (var other = start; other < count; other++) {
  if (other == position) { continue; }
  let comparison = compareNodes(ordered[orderedOffset + first + other], selected);
  if (comparison > 0 || (comparison == 0 && other < position)) { rank++; }
}
scratch[scratchOffset + first + start + rank] = selected;`
          })
        );
      }
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-round-${round}-advance`,
          operation: 'GPUNormalizeGeometry',
          variant: 'collection-round-advance',
          invocationCount: 1,
          bindings: [{name: 'roundState', view: roundState, type: 'u32', access: 'read_write'}],
          body: 'roundState[roundStateOffset] += 1u;'
        })
      );
    }
    // Collections that never finished (cycles or chains deeper than the bound) are invalid, and
    // the last round's ranked children are copied back.
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-collections-finish`,
        operation: 'GPUNormalizeGeometry',
        variant: 'collection-finish',
        invocationCount: nodeCount,
        bindings: [
          {name: 'nodes', view: props.nodes, type: 'u32', access: 'read'},
          {name: 'ordered', view: props.output.children, type: 'u32', access: 'read_write'},
          {name: 'scratch', view: scratchChildren, type: 'u32', access: 'read'},
          {name: 'rounds', view: rounds, type: 'u32', access: 'read'},
          {name: 'roundState', view: roundState, type: 'u32', access: 'read'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: nodeFields,
        body: `let kind = field(index, 0u); let first = field(index, 1u); let count = field(index, 2u);
if (kind > 7u || isLeaf(kind)) { return; }
let finishedRound = rounds[roundsOffset + index];
if (finishedRound == 0u) { status[statusOffset + index] = 2u; return; }
if (finishedRound + 1u == roundState[roundStateOffset] && status[statusOffset + index] == 0u) {
  for (var child = select(0u, 1u, kind == 5u); child < count; child++) {
    ordered[orderedOffset + first + child] = scratch[scratchOffset + first + child];
  }
}`
      })
    );
    return nodes;
  }
}
