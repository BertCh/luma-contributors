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
  GPUDelaunayTessellation,
  getDelaunayMaximumTriangleCount,
  getDelaunayScratchTriangleCapacity,
  validateDelaunayScratch
} from '../delaunay-tessellation/gpu-delaunay-tessellation';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  getGraphOutputViews,
  validateCapacity,
  validateFiniteNumber,
  validateSinglePackedView,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {EXACT_ORIENTATION_WGSL} from '../../analysis-core/kernels/exact-orientation-wgsl';
import {TRIANGLE_CIRCUMCIRCLE_WGSL} from '../shared/triangle-circumcircle-wgsl';
import {
  createTriangleEdgeAdjacencyNodes,
  TRIANGLE_ADJACENCY_NONE
} from '../shared/triangle-edge-adjacency';

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {createPackedGroupHullNodes} from './packed-group-hull';

/** Three edges of two uint32 words per reserved triangle use uint32 word addresses. */
const CONCAVE_HULL_MAXIMUM_TRIANGLE_RESERVATION = Math.floor(0xffffffff / 6);

function getConcaveHullOutputViews(
  output: GPUConcaveHullProps['output'] | GPUGroupConcaveHullPackedOutput
): GraphDataView[] {
  return [
    ...getGraphOutputViews(output),
    ...('groups' in output
      ? [
          output.groups.triangleOffsets,
          output.groups.triangleCounts,
          output.groups.edgeOffsets,
          output.groups.edgeCounts
        ]
      : [])
  ];
}

function validateConcaveHullRadius(id: string, value: number): void {
  validateFiniteNumber(id, 'maximumCircumradius', value, 0);
  validateRequirement(
    id,
    Math.fround(value) > 0,
    'maximumCircumradius must be greater than 0 after float32 conversion'
  );
}

export type GPUConcaveHullProps = {
  id?: string;
  positions: GraphDataView<'float32x2'>;
  spatialContext?: GPUSpatialContext;
  /** Maximum circumradius of retained Delaunay triangles in planar coordinate units. */
  maximumCircumradius: number;
  output: {
    /** Dense oriented boundary edges, each pair references source positions; edgeValidity masks padding. */
    boundaryEdges: GraphDataView<'uint32x2'>;
    edgeValidity: GraphDataView<'uint32'>;
    /** Retained Delaunay triangles; triangleValidity masks removed triangles and padding. */
    triangles: GraphDataView<'uint32x3'>;
    triangleValidity: GraphDataView<'uint32'>;
    /** One row; zero if tessellation did not finish. */
    status: {converged: GraphDataView<'uint32'>};
  };
};

/**
 * Checks the fixed output reservation: `triangleCapacity` triangle rows and three boundary-edge
 * rows per triangle (the Delaunay bound per group), `convergedRows` status rows, and one buffer
 * per output so no kernel aliases another's writes.
 */
function validateConcaveHullOutput(
  id: string,
  output: GPUConcaveHullProps['output'],
  triangleCapacity: number,
  convergedRows: number
): void {
  validateCapacity(
    id,
    'triangle reservation',
    triangleCapacity,
    0,
    CONCAVE_HULL_MAXIMUM_TRIANGLE_RESERVATION
  );
  validateSinglePackedView(output.triangles, ['uint32x3'], `${id} output.triangles`);
  validateSinglePackedView(output.triangleValidity, ['uint32'], `${id} output.triangleValidity`);
  validateSinglePackedView(output.boundaryEdges, ['uint32x2'], `${id} output.boundaryEdges`);
  validateSinglePackedView(output.edgeValidity, ['uint32'], `${id} output.edgeValidity`);
  validateSinglePackedView(output.status.converged, ['uint32'], `${id} output.status.converged`);
  validateViewLength(id, 'output.triangles', output.triangles, triangleCapacity);
  validateViewLength(id, 'output.triangleValidity', output.triangleValidity, triangleCapacity);
  validateViewLength(id, 'output.boundaryEdges', output.boundaryEdges, 3 * triangleCapacity);
  validateViewLength(id, 'output.edgeValidity', output.edgeValidity, 3 * triangleCapacity);
  validateRequirement(
    id,
    output.status.converged.length >= convergedRows,
    `output.status.converged must have at least ${convergedRows} rows`
  );
  const outputViews = getConcaveHullOutputViews(output);
  validateRequirement(
    id,
    new Set(outputViews.map(view => view.buffer)).size === outputViews.length,
    'output views must use distinct buffers'
  );
}

/**
 * Planar alpha-complex hull: retain triangles whose circumradius does not exceed the requested
 * radius, then expose their oriented boundary. Holes and disconnected components are preserved.
 * This deliberately uses radius-based alpha-shape semantics, not GEOS's ratio-based concave hull.
 * Isolated points and lines produce no area boundary. Capacity is 2*n-5 triangles and three edges
 * per triangle. Boundary edges can be passed to ring assembly without geometric intersection work.
 * A triangle-relative power-of-two-scaled circumcircle avoids intermediate overflow/underflow;
 * unrepresentable radii and degenerate triangles are not retained.
 */
export class GPUConcaveHull implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUConcaveHullProps;
  constructor(props: GPUConcaveHullProps) {
    this.id = props.id ?? 'concave-hull';
    this.props = props;
    const capacity = getDelaunayMaximumTriangleCount(props.positions.length);
    validateSinglePackedView(props.positions, ['float32x2'], `${this.id} positions`);
    resolveGPUSpatialContext(this.id, props.spatialContext);
    validateConcaveHullRadius(this.id, props.maximumCircumradius);
    getDelaunayScratchTriangleCapacity(this.id, props.positions.length);
    validateConcaveHullOutput(this.id, props.output, capacity, 1);
    validateGraphOutputsDisjointFromInputs(this.id, getGraphOutputViews(props.output), [
      props.positions
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    validateGraphChunkBindingRanges(graph, id, {
      positions: props.positions,
      ...Object.fromEntries(
        getGraphOutputViews(output).map((view, index) => [`output ${index}`, view])
      )
    });
    validateDelaunayScratch(graph.device, id, props.positions.length);
    // Edge-key sort padding is at most twice its uint32 half-edge input; boundaryEdges is
    // already a larger 24-byte-per-triangle binding and bounds adjacency/sort scratch.
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'edge adjacency scratch',
      Math.max(output.boundaryEdges.length, 1) * 4,
      'reduce positions rows'
    );
    const count = createTransientView(graph, `${id}-count`, 'uint32', 1);
    const requiredCount = createTransientView(graph, `${id}-required`, 'uint32', 1);
    const overflow = createTransientView(graph, `${id}-overflow`, 'uint32', 1);
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
    const adjacency = createTransientView(
      graph,
      `${id}-adjacency`,
      'uint32',
      output.boundaryEdges.length
    );
    nodes.push(
      createRetainTrianglesNode(graph, {
        id: `${id}-retain`,
        positions: props.positions,
        triangles: output.triangles,
        count,
        converged: output.status.converged,
        validity: output.triangleValidity,
        maximumCircumradius: props.maximumCircumradius
      }),
      ...createTriangleEdgeAdjacencyNodes(graph, {
        id: `${id}-adjacency`,
        operation: 'GPUConcaveHull',
        pointCount: props.positions.length,
        triangles: output.triangles,
        triangleCount: count,
        validity: output.triangleValidity,
        adjacency
      }),
      createBoundaryEdgesNode(graph, {
        id: `${id}-boundary`,
        triangles: output.triangles,
        validity: output.triangleValidity,
        adjacency,
        boundaryEdges: output.boundaryEdges,
        edgeValidity: output.edgeValidity
      })
    );
    return nodes;
  }
}

/**
 * Marks the triangles whose circumradius is at most `maximumCircumradius`. With `groupIds`,
 * `converged` has one row per group and a triangle takes the row of its first vertex's group.
 */
function createRetainTrianglesNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView<'float32x2'>;
    triangles: GraphDataView<'uint32x3'>;
    count: GraphDataView<'uint32'>;
    converged: GraphDataView<'uint32'>;
    validity: GraphDataView<'uint32'>;
    maximumCircumradius: number;
    groupIds?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode(graph, {
    id: props.id,
    operation: 'GPUConcaveHull',
    invocationCount: props.triangles.length,
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'triangles', view: props.triangles, type: 'u32', access: 'read'},
      {name: 'count', view: props.count, type: 'u32', access: 'read'},
      {name: 'converged', view: props.converged, type: 'u32', access: 'read'},
      {name: 'validity', view: props.validity, type: 'u32', access: 'read_write'},
      ...(props.groupIds
        ? [{name: 'groupIds', view: props.groupIds, type: 'u32' as const, access: 'read' as const}]
        : [])
    ],
    // The radius threshold is a pipeline override so changing it does not rebuild the module.
    declarations: `${EXACT_ORIENTATION_WGSL}
${TRIANGLE_CIRCUMCIRCLE_WGSL}
override MAXIMUM_CIRCUMRADIUS: f32;
fn position(row: u32) -> vec2f { return vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]); }`,
    constants: {MAXIMUM_CIRCUMRADIUS: Math.fround(props.maximumCircumradius)},
    body: `var retain = false;
if (index < count[countOffset]${
      props.groupIds
        ? ` && converged[convergedOffset + groupIds[groupIdsOffset + triangles[trianglesOffset + index * 3u]]] != 0u`
        : ' && converged[convergedOffset] != 0u'
    }) {
  let triangle = vec3<u32>(triangles[trianglesOffset + index * 3u], triangles[trianglesOffset + index * 3u + 1u], triangles[trianglesOffset + index * 3u + 2u]);
  let first = position(triangle.x); let second = position(triangle.y); let third = position(triangle.z);
  let circle = getTriangleCircumcircle(first, second, third);
  retain = circle.orientation != 0 && circle.radius <= MAXIMUM_CIRCUMRADIUS;
}
validity[validityOffset + index] = u32(retain);`
  });
}

/** Marks the boundary edges of the retained triangles (those with no retained neighbor). */
function createBoundaryEdgesNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    triangles: GraphDataView<'uint32x3'>;
    validity: GraphDataView<'uint32'>;
    adjacency: GraphDataView<'uint32'>;
    boundaryEdges: GraphDataView<'uint32x2'>;
    edgeValidity: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode(graph, {
    id: props.id,
    operation: 'GPUConcaveHull',
    invocationCount: props.boundaryEdges.length,
    bindings: [
      {name: 'triangles', view: props.triangles, type: 'u32', access: 'read'},
      {name: 'validity', view: props.validity, type: 'u32', access: 'read'},
      {name: 'adjacency', view: props.adjacency, type: 'u32', access: 'read'},
      {name: 'edges', view: props.boundaryEdges, type: 'u32', access: 'read_write'},
      {name: 'edgeValidity', view: props.edgeValidity, type: 'u32', access: 'read_write'}
    ],
    // A retained triangle edge is on the boundary when no retained triangle shares it.
    body: `let triangleIndex = index / 3u; let edgeIndex = index % 3u;
var boundary = false; var edge = vec2<u32>(0u);
if (triangleIndex < ${props.triangles.length}u && validity[validityOffset + triangleIndex] != 0u) {
  let triangle = vec3<u32>(triangles[trianglesOffset + triangleIndex * 3u], triangles[trianglesOffset + triangleIndex * 3u + 1u], triangles[trianglesOffset + triangleIndex * 3u + 2u]);
  edge = vec2<u32>(triangle[edgeIndex], triangle[(edgeIndex + 1u) % 3u]);
  boundary = adjacency[adjacencyOffset + index] == ${TRIANGLE_ADJACENCY_NONE}u;
}
edges[edgesOffset + index * 2u] = edge.x; edges[edgesOffset + index * 2u + 1u] = edge.y;
edgeValidity[edgeValidityOffset + index] = u32(boundary);`
  });
}

/** Compact group-major alpha triangles and boundary edges, with ranges for every group. */
export type GPUGroupConcaveHullPackedOutput = GPUConcaveHullProps['output'] & {
  /** Boundary-edge count; requiredCount equals count and overflow is zero with the full reservation. */
  status: GPUConcaveHullProps['output']['status'] & {
    count: GraphDataView<'uint32'>;
    requiredCount: GraphDataView<'uint32'>;
    overflow: GraphDataView<'uint32'>;
  };
  groups: {
    triangleOffsets: GraphDataView<'uint32'>;
    triangleCounts: GraphDataView<'uint32'>;
    edgeOffsets: GraphDataView<'uint32'>;
    edgeCounts: GraphDataView<'uint32'>;
  };
};

export type GPUGroupConcaveHullProps = Omit<GPUConcaveHullProps, 'output'> & {
  groupIds: GraphDataView<'uint32'>;
  /** Explicit group capacity; groups outside [0,groupCount) are omitted. */
  groupCount: number;
} & (
    | {layout?: 'fixed'; output: GPUConcaveHullProps['output']}
    | {layout: 'packed'; output: GPUGroupConcaveHullPackedOutput}
  );

/**
 * Independent alpha complexes for GPU-resident labels. The default fixed layout reserves 2*n-5
 * triangles per group and three boundary slots per triangle; converged has one row per group. All groups are triangulated
 * by one group-aware {@link GPUDelaunayTessellation} (sites are keyed by group, so no triangle
 * mixes groups and one pipeline serves every group), then retained and outlined once in a packed
 * list and scattered into the per-group slots. Work is O(n log n) in the point count; only the
 * fixed output reservation is O(groupCount*n). With layout 'packed', the reservation is only
 * 2*n-5 triangles total and three edges per triangle; retained triangles and boundary edges are
 * compact group-major prefixes, with group offsets/counts (including empty groups). Validity is
 * one in each compact prefix and zero in padding. Nonconverged groups contribute no geometry.
 * In the fixed layout, triangles of group `g` start at slot `g*(2*n-5)` in ascending canonical order, and its edges at
 * `3*` that slot.
 */
export class GPUGroupConcaveHull implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUGroupConcaveHullProps;
  constructor(props: GPUGroupConcaveHullProps) {
    // Keep validated topology and output maps independent of later caller mutations; borrow views.
    props =
      props.layout === 'packed'
        ? {
            ...props,
            output: {
              ...props.output,
              status: {...props.output.status},
              groups: {...props.output.groups}
            }
          }
        : {...props, output: {...props.output, status: {...props.output.status}}};
    this.id = props.id ?? 'group-concave-hull';
    this.props = props;
    validateCapacity(this.id, 'groupCount', props.groupCount);
    validateSinglePackedView(props.positions, ['float32x2'], `${this.id} positions`);
    validateSinglePackedView(props.groupIds, ['uint32'], `${this.id} groupIds`);
    resolveGPUSpatialContext(this.id, props.spatialContext);
    validateConcaveHullRadius(this.id, props.maximumCircumradius);
    getDelaunayScratchTriangleCapacity(this.id, props.positions.length, props.groupCount);
    validateRequirement(
      this.id,
      props.groupIds.length === props.positions.length,
      `groupIds must have exactly ${props.positions.length} rows (one per position, as GPUDelaunayTessellation requires)`
    );
    validateRequirement(
      this.id,
      props.layout === undefined || props.layout === 'fixed' || props.layout === 'packed',
      "layout must be 'fixed' or 'packed'"
    );
    const capacity =
      getDelaunayMaximumTriangleCount(props.positions.length) *
      (props.layout === 'packed' ? 1 : props.groupCount);
    if (props.layout === 'packed') {
      validateRequirement(
        this.id,
        Boolean(props.output.groups),
        'output.groups is required for packed layout'
      );
      for (const name of [
        'triangleOffsets',
        'triangleCounts',
        'edgeOffsets',
        'edgeCounts'
      ] as const) {
        const view: GraphDataView<'uint32'> = props.output.groups[name];
        validateRequirement(
          this.id,
          Boolean(view),
          `output.groups.${name} is required for packed layout`
        );
        validateSinglePackedView(view, ['uint32'], `${this.id} output.groups.${name}`);
        validateViewLength(this.id, `output.groups.${name}`, view, props.groupCount);
      }
      for (const name of ['count', 'requiredCount', 'overflow'] as const) {
        const view: GraphDataView<'uint32'> = props.output.status[name];
        validateRequirement(
          this.id,
          Boolean(view),
          `output.status.${name} is required for packed layout`
        );
        validateSinglePackedView(view, ['uint32'], `${this.id} output.status.${name}`);
        validateViewLength(this.id, `output.status.${name}`, view, 1);
      }
    }
    validateConcaveHullOutput(this.id, props.output, capacity, props.groupCount);
    validateGraphOutputsDisjointFromInputs(this.id, getConcaveHullOutputViews(props.output), [
      props.positions,
      props.groupIds
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output, groupCount} = props;
    const pointCount = props.positions.length;
    const capacity = getDelaunayMaximumTriangleCount(pointCount);
    validateGraphChunkBindingRanges(graph, id, {
      positions: props.positions,
      groupIds: props.groupIds,
      ...Object.fromEntries(
        getConcaveHullOutputViews(output).map((view, index) => [`output ${index}`, view])
      )
    });
    validateDelaunayScratch(graph.device, id, pointCount, groupCount);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'packed boundary scratch',
      Math.max(3 * capacity, 1) * 8,
      'reduce positions rows'
    );
    const scalar = (name: string, length = 1) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const count = scalar('count');
    const requiredCount = scalar('required');
    const overflow = scalar('overflow');
    const groupCounts = scalar('group-counts', groupCount);
    const groupOffsets = scalar('group-offsets', groupCount);
    const groupConverged = scalar('group-converged', groupCount);
    // Packed (group-major) triangles of every group.
    const packedTriangles = createTransientView(
      graph,
      `${id}-packed-triangles`,
      'uint32x3',
      capacity
    );
    const packedValidity = scalar('packed-validity', capacity);
    const packedEdges = createTransientView(graph, `${id}-packed-edges`, 'uint32x2', 3 * capacity);
    const packedEdgeValidity = scalar('packed-edge-validity', 3 * capacity);
    const adjacency = scalar('adjacency', 3 * capacity);
    const nodes: GPUCommandNode<Parameters>[] = [
      ...new GPUDelaunayTessellation({
        id: `${id}-delaunay`,
        positions: props.positions,
        groups: {ids: props.groupIds, count: groupCount},
        spatialContext: props.spatialContext,
        output: {
          triangles: packedTriangles,
          status: {count, requiredCount, overflow},
          groups: {
            triangleCounts: groupCounts,
            triangleOffsets: groupOffsets,
            status: {converged: groupConverged}
          }
        }
      }).getCommandNodes(graph)
    ];
    if (capacity > 0) {
      nodes.push(
        createRetainTrianglesNode(graph, {
          id: `${id}-retain`,
          positions: props.positions,
          triangles: packedTriangles,
          count,
          converged: groupConverged,
          validity: packedValidity,
          maximumCircumradius: props.maximumCircumradius,
          groupIds: props.groupIds
        }),
        ...createTriangleEdgeAdjacencyNodes(graph, {
          id: `${id}-adjacency`,
          operation: 'GPUGroupConcaveHull',
          pointCount,
          triangles: packedTriangles,
          triangleCount: count,
          validity: packedValidity,
          adjacency
        }),
        createBoundaryEdgesNode(graph, {
          id: `${id}-boundary`,
          triangles: packedTriangles,
          validity: packedValidity,
          adjacency,
          boundaryEdges: packedEdges,
          edgeValidity: packedEdgeValidity
        })
      );
    }
    if (props.layout === 'packed') {
      nodes.push(
        ...createPackedGroupHullNodes(graph, {
          id,
          groupCount,
          triangles: packedTriangles,
          triangleValidity: packedValidity,
          edges: packedEdges,
          edgeValidity: packedEdgeValidity,
          groupOffsets,
          groupCounts,
          groupConverged,
          output: props.output
        })
      );
      return nodes;
    }
    // Per-group slots: clear, then scatter the packed lists to `group * capacity + rank`.
    if (capacity > 0) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-clear-triangles`,
          operation: 'GPUGroupConcaveHull',
          invocationCount: output.triangles.length,
          bindings: [
            {name: 'triangles', view: output.triangles, type: 'u32', access: 'read_write'},
            {name: 'valid', view: output.triangleValidity, type: 'u32', access: 'read_write'}
          ],
          body: `triangles[trianglesOffset + index * 3u] = 0u; triangles[trianglesOffset + index * 3u + 1u] = 0u; triangles[trianglesOffset + index * 3u + 2u] = 0u;
valid[validOffset + index] = 0u;`
        }),
        createWGSLKernelNode(graph, {
          id: `${id}-clear-edges`,
          operation: 'GPUGroupConcaveHull',
          invocationCount: output.boundaryEdges.length,
          bindings: [
            {name: 'edges', view: output.boundaryEdges, type: 'u32', access: 'read_write'},
            {name: 'valid', view: output.edgeValidity, type: 'u32', access: 'read_write'}
          ],
          body: `edges[edgesOffset + index * 2u] = 0u; edges[edgesOffset + index * 2u + 1u] = 0u;
valid[validOffset + index] = 0u;`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-publish-converged`,
        operation: 'GPUGroupConcaveHull',
        invocationCount: groupCount,
        bindings: [
          {name: 'source', view: groupConverged, type: 'u32', access: 'read'},
          {name: 'converged', view: output.status.converged, type: 'u32', access: 'read_write'}
        ],
        body: 'converged[convergedOffset + index] = source[sourceOffset + index];'
      })
    );
    if (capacity > 0) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-triangles-publish`,
          operation: 'GPUGroupConcaveHull',
          invocationCount: capacity,
          bindings: [
            {name: 'groups', view: props.groupIds, type: 'u32', access: 'read'},
            {name: 'offsets', view: groupOffsets, type: 'u32', access: 'read'},
            {name: 'count', view: count, type: 'u32', access: 'read'},
            {name: 'triangles', view: packedTriangles, type: 'u32', access: 'read'},
            {name: 'valid', view: packedValidity, type: 'u32', access: 'read'},
            {name: 'outputTriangles', view: output.triangles, type: 'u32', access: 'read_write'},
            {name: 'outputValid', view: output.triangleValidity, type: 'u32', access: 'read_write'}
          ],
          body: `if (index >= count[countOffset]) { return; }
let group = groups[groupsOffset + triangles[trianglesOffset + index * 3u]];
let slot = group * ${capacity}u + index - offsets[offsetsOffset + group];
for (var vertex = 0u; vertex < 3u; vertex++) {
  outputTriangles[outputTrianglesOffset + slot * 3u + vertex] = triangles[trianglesOffset + index * 3u + vertex];
}
outputValid[outputValidOffset + slot] = valid[validOffset + index];`
        }),
        createWGSLKernelNode(graph, {
          id: `${id}-edges-publish`,
          operation: 'GPUGroupConcaveHull',
          invocationCount: 3 * capacity,
          bindings: [
            {name: 'groups', view: props.groupIds, type: 'u32', access: 'read'},
            {name: 'offsets', view: groupOffsets, type: 'u32', access: 'read'},
            {name: 'count', view: count, type: 'u32', access: 'read'},
            {name: 'triangles', view: packedTriangles, type: 'u32', access: 'read'},
            {name: 'edges', view: packedEdges, type: 'u32', access: 'read'},
            {name: 'valid', view: packedEdgeValidity, type: 'u32', access: 'read'},
            {name: 'outputEdges', view: output.boundaryEdges, type: 'u32', access: 'read_write'},
            {name: 'outputValid', view: output.edgeValidity, type: 'u32', access: 'read_write'}
          ],
          body: `let triangle = index / 3u;
if (triangle >= count[countOffset]) { return; }
let group = groups[groupsOffset + triangles[trianglesOffset + triangle * 3u]];
let slot = (group * ${capacity}u + triangle - offsets[offsetsOffset + group]) * 3u + index % 3u;
outputEdges[outputEdgesOffset + slot * 2u] = edges[edgesOffset + index * 2u];
outputEdges[outputEdgesOffset + slot * 2u + 1u] = edges[edgesOffset + index * 2u + 1u];
outputValid[outputValidOffset + slot] = valid[validOffset + index];`
        })
      );
    }
    return nodes;
  }
}
