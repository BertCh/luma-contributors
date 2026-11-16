// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateOutlineOutputPasses} from '../outline-geometry/validate-outline-output-passes';
import {validateRefinementScratchByteLengths} from './refinement-storage-validation';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {
  GPUDelaunayTessellation,
  getDelaunayMaximumTriangleCount,
  validateDelaunayScratch
} from '../delaunay-tessellation/gpu-delaunay-tessellation';
import type {GPUConcaveHullProps} from '../concave-hull/gpu-concave-hull';
import {
  validateSinglePackedView,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  getGraphOutputViews
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createTriangleEdgeAdjacencyNodes} from '../shared/triangle-edge-adjacency';
import {getReadPointWGSL} from '../shared/spatial-wgsl-point-helpers';

const ERODE_WORKGROUP_SIZE = 256;

export type GPURatioConcaveHullProps = Omit<GPUConcaveHullProps, 'maximumCircumradius'> & {
  /** Fraction of the range of Delaunay edge lengths, in [0, 1]. */
  ratio: number;
  /** Allow interior triangle erosion to start holes. Default false. */
  allowHoles?: boolean;
  /** Maximum triangle removals; default the triangle capacity. */
  maximumIterations?: number;
};

/**
 * Edge-length-ratio hull from Delaunay border erosion. A removable border triangle
 * has one exposed edge and an interior opposite vertex, preserving connectivity
 * and all input vertices. Hole seeds have three interior vertices. A triangle is a
 * candidate while its exposed edge is strictly longer than
 * `minLength + ratio * (maxLength - minLength)`. At ratio one this is the convex
 * hull. Collinear/point sets have an empty area boundary. `converged` also covers
 * the removal budget; incomplete masks are not final hulls.
 *
 * **Removal order.** Erosion runs in rounds. Each round removes, in parallel, every
 * candidate that is the largest (exposed edge length, then lowest triangle row)
 * among the candidates around all three of its vertices, and rounds repeat until
 * no candidate is left. The result is a connected hull over every input vertex in
 * which no removable border triangle exceeds the threshold. It is not the hull of
 * GEOS and JTS `ConcaveHull` (`shapely.concave_hull`), which remove one triangle
 * at a time, always the largest: a triangle that is only the largest in its own
 * neighbourhood can go here before a larger removal elsewhere reaches that
 * neighbourhood and would have made it unremovable. The two hulls agree at ratio
 * one and on the committed Shapely fixtures (five and thirty-two points), and
 * otherwise can keep different triangles (nine points are enough; both walks keep
 * seven triangles there, with areas 37.66 against 31.75). GEOS also removes an
 * edge exactly at the threshold.
 */
export class GPURatioConcaveHull implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURatioConcaveHullProps;
  constructor(props: GPURatioConcaveHullProps) {
    this.id = props.id ?? 'ratio-concave-hull';
    this.props = props;
    const capacity = getDelaunayMaximumTriangleCount(props.positions.length);
    if (!(Number.isFinite(props.ratio) && props.ratio >= 0 && props.ratio <= 1)) {
      throw new Error('Invalid hull ratio');
    }
    validateCapacity(this.id, 'maximumIterations', props.maximumIterations ?? capacity, 0);
    validateSinglePackedView(props.positions, ['float32x2'], this.id);
    validateSinglePackedView(props.output.triangles, ['uint32x3'], this.id);
    validateSinglePackedView(props.output.boundaryEdges, ['uint32x2'], this.id);
    for (const view of [
      props.output.triangleValidity,
      props.output.edgeValidity,
      props.output.status.converged
    ]) {
      validateSinglePackedView(view, ['uint32'], this.id);
    }
    if (
      props.output.triangles.length !== capacity ||
      props.output.triangleValidity.length !== capacity ||
      props.output.boundaryEdges.length !== capacity * 3 ||
      props.output.edgeValidity.length !== capacity * 3 ||
      props.output.status.converged.length !== 1
    ) {
      throw new Error('Hull output capacity mismatch');
    }
    validateOutlineOutputPasses(
      this.id,
      {...props.output, status: undefined, converged: props.output.status.converged},
      [['triangleValidity', 'converged'], ['boundaryEdges', 'edgeValidity'], ['triangles']]
    );
    validateGraphOutputsDisjointFromInputs(this.id, getGraphOutputViews(props.output), [
      props.positions
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output} = props;
    const capacity = output.triangles.length;
    validateGraphChunkBindingRanges(
      graph,
      id,
      Object.fromEntries(
        [props.positions, ...getGraphOutputViews(output)].map((view, index) => [
          `binding.${index}`,
          view
        ])
      )
    );
    const topologyCapacity = validateDelaunayScratch(graph.device, id, props.positions.length);
    validateRefinementScratchByteLengths(graph, id, {
      delaunayTopology: topologyCapacity * 16,
      adjacencyAndSort: capacity * 3 * 4,
      work: (3 * props.positions.length + capacity + 4) * 4
    });
    const count = createTransientView(graph, `${id}-count`, 'uint32', 1);
    const requiredCount = createTransientView(graph, `${id}-required`, 'uint32', 1);
    const overflow = createTransientView(graph, `${id}-overflow`, 'uint32', 1);
    const adjacency = createTransientView(graph, `${id}-adjacency`, 'uint32', capacity * 3);
    const vertexCount = props.positions.length;
    // Scratch words: boundary counts, candidate maxima and winners per vertex, candidate keys per triangle, round state.
    const work = createTransientView(graph, `${id}-work`, 'uint32', 3 * vertexCount + capacity + 4);
    const nodes = [
      ...new GPUDelaunayTessellation({
        id: `${id}-delaunay`,
        positions: props.positions,
        spatialContext: props.spatialContext,
        output: {
          triangles: output.triangles,
          status: {count, requiredCount, overflow, converged: output.status.converged}
        }
      }).getCommandNodes(graph)
    ];
    const triangleAccess = `const CAPACITY: u32 = ${capacity}u;
fn triangle(row: u32) -> vec3u { return vec3u(triangles[trianglesOffset + row * 3u], triangles[trianglesOffset + row * 3u + 1u], triangles[trianglesOffset + row * 3u + 2u]); }`;
    nodes.push(
      // Adjacency comes from one stable sort of undirected edge keys instead of a triangle scan per edge.
      ...createTriangleEdgeAdjacencyNodes(graph, {
        id: `${id}-adjacency`,
        operation: 'GPURatioConcaveHull',
        pointCount: vertexCount,
        triangles: output.triangles,
        triangleCount: count,
        adjacency
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-validity`,
        operation: 'GPURatioConcaveHull',
        variant: 'initial-validity',
        invocationCount: capacity,
        bindings: [
          {name: 'count', view: count, type: 'u32', access: 'read'},
          {name: 'validity', view: output.triangleValidity, type: 'u32', access: 'read_write'}
        ],
        body: 'validity[validityOffset + index] = select(0u, 1u, index < count[countOffset]);'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-erode`,
        operation: 'GPURatioConcaveHull',
        variant: props.allowHoles ? 'ratio-erosion-holes' : 'ratio-erosion',
        // One cooperative workgroup runs the erosion rounds with barriers between phases.
        invocationCount: ERODE_WORKGROUP_SIZE,
        workgroupSize: ERODE_WORKGROUP_SIZE,
        guardIndex: false,
        bindings: [
          {name: 'triangles', view: output.triangles, type: 'u32', access: 'read'},
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'count', view: count, type: 'u32', access: 'read'},
          {name: 'adjacency', view: adjacency, type: 'u32', access: 'read'},
          {name: 'validity', view: output.triangleValidity, type: 'u32', access: 'read_write'},
          {name: 'converged', view: output.status.converged, type: 'u32', access: 'read_write'},
          {name: 'work', view: work, type: 'atomic<u32>', access: 'read_write'}
        ],
        declarations: `${triangleAccess}
const VERTEX_COUNT: u32 = ${vertexCount}u;
const WORKGROUP_SIZE: u32 = ${ERODE_WORKGROUP_SIZE}u;
const BOUNDARY_BASE: u32 = 0u;
const MAXIMUM_BASE: u32 = ${vertexCount}u;
const WINNER_BASE: u32 = ${2 * vertexCount}u;
const CANDIDATE_BASE: u32 = ${3 * vertexCount}u;
const STATE_BASE: u32 = ${3 * vertexCount + capacity}u;
const WINNER_MARK: u32 = 0xffffffffu;
override RATIO: f32;
override MAXIMUM_ITERATIONS: u32;
var<workgroup> stopFlag: u32;
var<workgroup> holeMode: u32;
var<workgroup> invalid: atomic<u32>;
var<workgroup> minimumBits: atomic<u32>;
var<workgroup> maximumBits: atomic<u32>;
${getReadPointWGSL('point', 'positions')}
fn isFiniteBits(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn edgeLength(current: vec3u, edge: u32) -> f32 { return distance(point(current[edge]), point(current[(edge + 1u) % 3u])); }
fn exposed(row: u32, edge: u32) -> bool {
  let neighbor = adjacency[adjacencyOffset + row * 3u + edge];
  if (neighbor == 0xffffffffu) { return true; }
  return validity[validityOffset + neighbor] == 0u;
}
fn isBoundaryVertex(vertex: u32) -> bool { return atomicLoad(&work[workOffset + BOUNDARY_BASE + vertex]) != 0u; }
fn synchronize() { storageBarrier(); workgroupBarrier(); }
fn offerCandidate(row: u32, current: vec3u, size: f32, threshold: f32) {
  if (size > threshold) {
    let bits = bitcast<u32>(size);
    atomicStore(&work[workOffset + CANDIDATE_BASE + row], bits);
    for (var corner = 0u; corner < 3u; corner++) { atomicMax(&work[workOffset + MAXIMUM_BASE + current[corner]], bits); }
    atomicAdd(&work[workOffset + STATE_BASE + 2u], 1u);
  }
}`,
        constants: {
          RATIO: Math.fround(props.ratio),
          MAXIMUM_ITERATIONS: props.maximumIterations ?? capacity
        },
        // Rounds remove, in parallel, every candidate that is the largest (size, then lowest row)
        // around all three of its vertices, so removals in one round never share a vertex.
        body: `let lane = localInvocationIndex;
let triangleCount = min(count[countOffset], ${capacity}u);
if (lane == 0u) {
  stopFlag = select(0u, 1u, converged[convergedOffset] == 0u);
  atomicStore(&invalid, 0u); atomicStore(&minimumBits, 0x7f800000u); atomicStore(&maximumBits, 0u);
}
synchronize();
if (workgroupUniformLoad(&stopFlag) == 0u) {
  for (var vertex = lane; vertex < VERTEX_COUNT; vertex += WORKGROUP_SIZE) {
    let value = point(vertex);
    if (!isFiniteBits(value.x) || !isFiniteBits(value.y)) { atomicStore(&invalid, 1u); }
  }
  synchronize();
  for (var row = lane; row < triangleCount; row += WORKGROUP_SIZE) {
    let current = triangle(row);
    for (var edge = 0u; edge < 3u; edge++) {
      let edgeSize = edgeLength(current, edge);
      if (!isFiniteBits(edgeSize)) { atomicStore(&invalid, 1u); } else {
        atomicMin(&minimumBits, bitcast<u32>(edgeSize)); atomicMax(&maximumBits, bitcast<u32>(edgeSize));
      }
    }
  }
  synchronize();
  if (lane == 0u) {
    if (atomicLoad(&invalid) != 0u) { converged[convergedOffset] = 0u; stopFlag = 1u; }
    atomicStore(&work[workOffset + STATE_BASE + 1u], 0u);
  }
  synchronize();
  let minimumLength = bitcast<f32>(atomicLoad(&minimumBits));
  let threshold = minimumLength + RATIO * (bitcast<f32>(atomicLoad(&maximumBits)) - minimumLength);
  loop {
    if (workgroupUniformLoad(&stopFlag) != 0u) { break; }
    for (var vertex = lane; vertex < VERTEX_COUNT; vertex += WORKGROUP_SIZE) {
      atomicStore(&work[workOffset + BOUNDARY_BASE + vertex], 0u);
      atomicStore(&work[workOffset + MAXIMUM_BASE + vertex], 0u);
      atomicStore(&work[workOffset + WINNER_BASE + vertex], 0xffffffffu);
    }
    for (var row = lane; row < triangleCount; row += WORKGROUP_SIZE) { atomicStore(&work[workOffset + CANDIDATE_BASE + row], 0u); }
    if (lane == 0u) { atomicStore(&work[workOffset + STATE_BASE], 0u); atomicStore(&work[workOffset + STATE_BASE + 2u], 0u); }
    synchronize();
    // Count exposed edges per vertex: a vertex with a nonzero count lies on the current boundary.
    for (var row = lane; row < triangleCount; row += WORKGROUP_SIZE) {
      if (validity[validityOffset + row] == 0u) { continue; }
      let current = triangle(row);
      for (var edge = 0u; edge < 3u; edge++) {
        if (exposed(row, edge)) {
          atomicAdd(&work[workOffset + BOUNDARY_BASE + current[edge]], 1u);
          atomicAdd(&work[workOffset + BOUNDARY_BASE + current[(edge + 1u) % 3u]], 1u);
        }
      }
    }
    synchronize();
    // Border candidates: one exposed edge and an interior opposite vertex.
    for (var row = lane; row < triangleCount; row += WORKGROUP_SIZE) {
      if (validity[validityOffset + row] == 0u) { continue; }
      let current = triangle(row);
      var exposedCount = 0u; var exposedEdge = 0u;
      for (var edge = 0u; edge < 3u; edge++) { if (exposed(row, edge)) { exposedCount++; exposedEdge = edge; } }
      if (exposedCount == 1u && !isBoundaryVertex(current[(exposedEdge + 2u) % 3u])) {
        offerCandidate(row, current, edgeLength(current, exposedEdge), threshold);
      }
    }
    synchronize();
    if (lane == 0u) { holeMode = ${props.allowHoles ? 'select(0u, 1u, atomicLoad(&work[workOffset + STATE_BASE + 2u]) == 0u)' : '0u'}; }
    synchronize();
    if (workgroupUniformLoad(&holeMode) != 0u) {
      // No border candidate remains: seed holes from triangles whose three vertices are interior.
      for (var row = lane; row < triangleCount; row += WORKGROUP_SIZE) {
        if (validity[validityOffset + row] == 0u) { continue; }
        let current = triangle(row);
        if (isBoundaryVertex(current.x) || isBoundaryVertex(current.y) || isBoundaryVertex(current.z)) { continue; }
        offerCandidate(row, current, max(edgeLength(current, 0u), max(edgeLength(current, 1u), edgeLength(current, 2u))), threshold);
      }
      synchronize();
    }
    for (var row = lane; row < triangleCount; row += WORKGROUP_SIZE) {
      let bits = atomicLoad(&work[workOffset + CANDIDATE_BASE + row]);
      if (bits == 0u) { continue; }
      let current = triangle(row);
      if (atomicLoad(&work[workOffset + MAXIMUM_BASE + current.x]) == bits && atomicLoad(&work[workOffset + MAXIMUM_BASE + current.y]) == bits && atomicLoad(&work[workOffset + MAXIMUM_BASE + current.z]) == bits) {
        for (var corner = 0u; corner < 3u; corner++) { atomicMin(&work[workOffset + WINNER_BASE + current[corner]], row); }
      }
    }
    synchronize();
    for (var row = lane; row < triangleCount; row += WORKGROUP_SIZE) {
      if (atomicLoad(&work[workOffset + CANDIDATE_BASE + row]) == 0u) { continue; }
      let current = triangle(row);
      if (atomicLoad(&work[workOffset + WINNER_BASE + current.x]) == row && atomicLoad(&work[workOffset + WINNER_BASE + current.y]) == row && atomicLoad(&work[workOffset + WINNER_BASE + current.z]) == row) {
        atomicStore(&work[workOffset + CANDIDATE_BASE + row], WINNER_MARK);
        atomicAdd(&work[workOffset + STATE_BASE], 1u);
      }
    }
    synchronize();
    if (lane == 0u) {
      let winners = atomicLoad(&work[workOffset + STATE_BASE]);
      let removals = atomicLoad(&work[workOffset + STATE_BASE + 1u]);
      if (winners == 0u) { stopFlag = 1u; }
      else if (removals + winners > MAXIMUM_ITERATIONS) { converged[convergedOffset] = 0u; stopFlag = 1u; }
      else { atomicStore(&work[workOffset + STATE_BASE + 1u], removals + winners); }
    }
    synchronize();
    if (workgroupUniformLoad(&stopFlag) == 0u) {
      for (var row = lane; row < triangleCount; row += WORKGROUP_SIZE) {
        if (atomicLoad(&work[workOffset + CANDIDATE_BASE + row]) == WINNER_MARK) { validity[validityOffset + row] = 0u; }
      }
    }
    synchronize();
  }
}`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-boundary`,
        operation: 'GPURatioConcaveHull',
        variant: 'boundary',
        invocationCount: capacity * 3,
        bindings: [
          {name: 'triangles', view: output.triangles, type: 'u32', access: 'read'},
          {name: 'adjacency', view: adjacency, type: 'u32', access: 'read'},
          {name: 'validity', view: output.triangleValidity, type: 'u32', access: 'read'},
          {name: 'edges', view: output.boundaryEdges, type: 'u32', access: 'read_write'},
          {name: 'edgeValidity', view: output.edgeValidity, type: 'u32', access: 'read_write'}
        ],
        declarations: triangleAccess,
        body: `let row = index / 3u; let edge = index % 3u; let current = triangle(row);
let neighbor = adjacency[adjacencyOffset + index];
var boundary = validity[validityOffset + row] != 0u;
if (neighbor != 0xffffffffu && validity[validityOffset + neighbor] != 0u) { boundary = false; }
edges[edgesOffset + index * 2u] = current[edge]; edges[edgesOffset + index * 2u + 1u] = current[(edge + 1u) % 3u];
edgeValidity[edgeValidityOffset + index] = select(0u, 1u, boundary);`
      })
    );
    return nodes;
  }
}
