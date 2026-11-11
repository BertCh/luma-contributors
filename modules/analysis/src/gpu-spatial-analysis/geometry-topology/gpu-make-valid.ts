// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {
  GPUPackedGeometryPort,
  GPUPolygonGeometryPort
} from '../../analysis-core/contracts/index';
import {
  validateCapacity,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validatePowerOfTwo
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {EXACT_ORIENTATION_WGSL} from '../../analysis-core/kernels/exact-orientation-wgsl';
import {EXACT_GEOMETRY_WGSL} from '../../analysis-core/kernels/exact-geometry-wgsl';
import {validateLineNodingBounds} from './gpu-line-noding';
import {GPUPolygonize, type GPUPolygonizeProps} from './gpu-polygonize';
import {
  GPU_TOPOLOGY_NONE,
  type GPUPolygonizeOutput,
  type GPUTopologyPrecisionPolicy
} from './topology-types';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';

const OPERATION = 'GPUMakeValid';

/** Properties for {@link GPUMakeValid}. */
export type GPUMakeValidProps = {
  id?: string;
  /**
   * Planar only (default: native planar coordinates). A longitude-latitude context throws at
   * construction: project positions to a planar frame first, for example with
   * `createGPULocalFrame` (`contracts/local-frame`) and `GPULocalFrameProjection`
   * (`geometry-measures`), and pass the frame's context.
   */
  spatialContext?: GPUSpatialContext;
  /** Polygon boundaries to node and rebuild into bounded faces. */
  polygons: GPUPackedGeometryPort<GPUPolygonGeometryPort> & {
    /** Optional source feature ID per feature, carried to `output.polygons.sourceIds`. */
    sourceIds?: GraphDataView<'uint32'>;
  };
  /**
   * Capacity of the exact intersection event list used by noding. The ring count plus four times
   * this capacity is at most {@link getGPULineSplitMaximumPieceBound} (4194304 on a default
   * device), and `polygons.positions` has at most
   * {@link getGPUSegmentIntersectionMaximumVertexCount} rows.
   */
  intersectionCapacity: number;
  precision: GPUTopologyPrecisionPolicy;
  /** Optional hot-pixel snap rounding before geometry reconstruction. */
  snapRounding?: GPUPolygonizeProps['snapRounding'];
  /** Caller-owned outputs; `output.status.uncertainCount` is the optional uncertain-pair counter. */
  output: GPUPolygonizeOutput;
  spatialSort?: boolean;
  leafCapacity?: number;
};

/**
 * Topology repair that rebuilds valid polygons from arbitrary, possibly invalid polygon
 * boundaries, independently for each source feature.
 *
 * Every ring of a feature is noded and polygonized into bounded faces, then a face is kept only
 * when the source rings of its feature cover it an odd number of times: the fill rule is source
 * parity (even-odd), the rule of the GEOS linework `make_valid` method that shapely uses. A valid
 * polygon with a hole therefore comes back as the annulus without a second polygon filling the
 * hole, an island inside a hole is kept as its own polygon, and the doubly covered lens of
 * mutually crossing rings is dropped. Bow ties and self-crossing rings are split into polygon
 * features. {@link GPUPolygonize} alone keeps every bounded face (shapely `polygonize`
 * semantics); the parity filter is specific to this operator.
 *
 * Rings close implicitly (the repository convention): the edge from a ring's last vertex back to
 * its first is always part of the boundary, and a ring that repeats its first vertex is read the
 * same way. `output.nodedSegments.sourceEdgeIds` are rows of `polygons.positions` (the edge's
 * start vertex; the closing edge reports the ring's last row).
 *
 * Parity is evaluated exactly against the noded segments of the feature. For each face the
 * operator looks at the first edge of the face's shell (counter-clockwise, face on its left) and
 * counts crossings of a ray cast from an infinitesimal point inside the face next to that edge.
 * The point is symbolic (infinitesimally left of the edge's midpoint, the shared probe of
 * `utils/exact-geometry-wgsl.ts`), so it is strictly inside the face by construction and no
 * rounded sample coordinate is needed.
 * Every sign comes from `orientSign`; a centroid is not used because it can fall outside a
 * concave face or inside one of its holes. The probe is brute force: each face scans every noded
 * segment (`faces * nodedSegments` tests), one thread per face.
 *
 * Status: `count` is the number of kept polygons, `requiredCount` the unclamped ring-assembly
 * requirement (the ring capacity needed, an upper bound on the polygons written), `overflow` flags
 * noding or ring capacity loss and `candidateOverflow` only an incomplete noding candidate list.
 * If an orientation cannot be certified (non-finite coordinates or a span of more than 200 binary
 * orders of magnitude) the face is kept. Open and cut remnants stay in `output.diagnostics`.
 */
export class GPUMakeValid implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUMakeValidProps;
  /** Resolved spatial context (planar only). */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUMakeValidProps) {
    this.id = props.id ?? 'make-valid';
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    this.props = props;
    if (props.polygons.kind !== 'polygons') {
      throw new Error(`${this.id} polygons.kind must be polygons`);
    }
    if (props.polygons.ringOffsets.length < 2) {
      throw new Error(`${this.id} polygons require at least one ring`);
    }
    // Delegates are built later, in getCommandNodes: check forwarded options now, under this id.
    validateCapacity(this.id, 'intersectionCapacity', props.intersectionCapacity);
    if (props.leafCapacity !== undefined) {
      validatePowerOfTwo(this.id, 'leafCapacity', props.leafCapacity);
    }
    const {polygons: faces} = props.output;
    const ringCapacity = faces.ringOffsets.length - 1;
    validateSinglePackedView(
      faces.positions,
      ['float32x2'],
      `${this.id} output.polygons.positions`
    );
    for (const [name, view] of [
      ['ringOffsets', faces.ringOffsets],
      ['polygonOffsets', faces.polygonOffsets],
      ['featureOffsets', faces.featureOffsets],
      ['sourceIds', faces.sourceIds]
    ] as const) {
      validatePackedUint32View(view, `${this.id} output.polygons.${name}`);
    }
    if (
      ringCapacity < 1 ||
      faces.polygonOffsets.length !== ringCapacity + 1 ||
      faces.featureOffsets.length !== ringCapacity + 1 ||
      faces.sourceIds.length !== ringCapacity
    ) {
      throw new Error(
        `${this.id} polygon offset and source capacities must share one ring capacity`
      );
    }
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {polygons} = props;
    const ringCount = polygons.ringOffsets.length - 1;
    const ringSourceIds = createTransientView(graph, `${id}-ring-source-ids`, 'uint32', ringCount);
    const sourceBinding = polygons.sourceIds
      ? [
          {
            name: 'inputSourceIds',
            view: polygons.sourceIds,
            type: 'u32' as const,
            access: 'read' as const
          }
        ]
      : [];
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-ring-sources`,
        operation: OPERATION,
        variant: 'ring-sources',
        bindings: [
          {name: 'featureOffsets', view: polygons.featureOffsets, type: 'u32', access: 'read'},
          {name: 'polygonOffsets', view: polygons.polygonOffsets, type: 'u32', access: 'read'},
          ...sourceBinding,
          {name: 'ringSourceIds', view: ringSourceIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: ringCount,
        declarations: `const FEATURE_COUNT: u32 = ${polygons.featureOffsets.length - 1}u;
const POLYGON_COUNT: u32 = ${polygons.polygonOffsets.length - 1}u;`,
        body: `var polygonLow = 0u;
  var polygonHigh = POLYGON_COUNT;
  while (polygonLow + 1u < polygonHigh) {
    let middle = (polygonLow + polygonHigh) / 2u;
    if (polygonOffsets[polygonOffsetsOffset + middle] <= index) { polygonLow = middle; } else { polygonHigh = middle; }
  }
  var featureLow = 0u;
  var featureHigh = FEATURE_COUNT;
  while (featureLow + 1u < featureHigh) {
    let middle = (featureLow + featureHigh) / 2u;
    if (featureOffsets[featureOffsetsOffset + middle] <= polygonLow) { featureLow = middle; } else { featureHigh = middle; }
  }
  ringSourceIds[ringSourceIdsOffset + index] = ${polygons.sourceIds ? 'inputSourceIds[inputSourceIdsOffset + featureLow]' : 'featureLow'};`
      })
    ];
    const {output} = props;
    const ringCapacity = output.polygons.ringOffsets.length - 1;
    const vertexCapacity = output.polygons.positions.length;
    const segmentCapacity = output.nodedSegments.endpoints.length;
    validateGraphViewsBelongToGraph(id, graph, [
      polygons.positions,
      polygons.featureOffsets,
      polygons.polygonOffsets,
      polygons.ringOffsets,
      polygons.sourceIds,
      output.polygons.positions,
      output.polygons.featureOffsets,
      output.polygons.polygonOffsets,
      output.polygons.ringOffsets,
      output.polygons.sourceIds,
      output.nodedSegments.endpoints,
      output.nodedSegments.sourceRingIds,
      ...Object.values(output.nodedSegments.status),
      ...Object.values(output.status)
    ]);
    // Noding consumes open line paths, while polygon rings close implicitly: one repeated first
    // vertex per ring makes the closing edge part of the arrangement (as GPUPolygonOverlay does).
    // A ring that already repeats its first vertex gains a zero-length edge, which noding drops.
    const vertexCount = polygons.positions.length;
    const closedVertexCount = vertexCount + ringCount;
    validateLineNodingBounds(
      id,
      graph.device,
      closedVertexCount,
      props.polygons.ringOffsets.length - 1,
      props.intersectionCapacity
    );
    const closedPositions = createTransientView(
      graph,
      `${id}-ring-closure-positions`,
      'float32x2',
      closedVertexCount
    );
    const closedOffsets = createTransientView(
      graph,
      `${id}-ring-closure-offsets`,
      'uint32',
      ringCount + 1
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-ring-closure-offsets`,
        operation: OPERATION,
        variant: 'ring-closure-offsets',
        bindings: [
          {name: 'ringOffsets', view: polygons.ringOffsets, type: 'u32', access: 'read'},
          {name: 'closedOffsets', view: closedOffsets, type: 'u32', access: 'read_write'}
        ],
        invocationCount: ringCount + 1,
        body: 'closedOffsets[closedOffsetsOffset + index] = ringOffsets[ringOffsetsOffset + index] + index;'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-ring-closure-positions`,
        operation: OPERATION,
        variant: 'ring-closure-positions',
        bindings: [
          {name: 'positions', view: polygons.positions, type: 'f32', access: 'read'},
          {name: 'ringOffsets', view: polygons.ringOffsets, type: 'u32', access: 'read'},
          {name: 'closedPositions', view: closedPositions, type: 'f32', access: 'read_write'}
        ],
        invocationCount: closedVertexCount,
        declarations: `const RING_COUNT: u32 = ${ringCount}u;
const VERTEX_COUNT: u32 = ${vertexCount}u;`,
        // Ring r owns closed rows [ringOffsets[r] + r, ringOffsets[r + 1] + r + 1): its vertices,
        // then its first vertex again. An empty ring's one row, and rows after an empty last ring,
        // own no vertex: they get the first stored vertex (the origin when there is none), never
        // `positions[ringOffsets[last]]`, which no ring owns and which holds whatever the buffer
        // held (the noder reads every row of a line, including a one-row line). Rows before
        // `ringOffsets[0]` resolve to ring 0 and repeat its first vertex, never `positions[index]`,
        // which no ring owns either.
        body: `var ring = 0u;
  var high = RING_COUNT;
  while (ring + 1u < high) {
    let middle = (ring + high) / 2u;
    if (ringOffsets[ringOffsetsOffset + middle] + middle <= index) { ring = middle; } else { high = middle; }
  }
  let ringStart = ringOffsets[ringOffsetsOffset + ring];
  let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
  var point = vec2f(0.0);
  var source = ringStart;
  var hasSource = ringEnd > ringStart;
  if (hasSource) {
    source = index - ring;
    if (index < ringStart || source >= ringEnd) { source = ringStart; }
  } else {
    source = ringOffsets[ringOffsetsOffset];
    hasSource = ringOffsets[ringOffsetsOffset + RING_COUNT] > source;
  }
  if (hasSource) {
    source = min(source, VERTEX_COUNT - 1u);
    point = vec2f(positions[positionsOffset + 2u * source], positions[positionsOffset + 2u * source + 1u]);
  }
  closedPositions[closedPositionsOffset + 2u * index] = point.x;
  closedPositions[closedPositionsOffset + 2u * index + 1u] = point.y;`
      })
    );

    // GPUPolygonize emits every bounded face into transient storage; the parity filter below
    // compacts the faces it keeps into the caller's polygon port.
    const faces = {
      kind: 'polygons' as const,
      positions: createTransientView(graph, `${id}-face-positions`, 'float32x2', vertexCapacity),
      featureOffsets: createTransientView(
        graph,
        `${id}-face-feature-offsets`,
        'uint32',
        ringCapacity + 1
      ),
      polygonOffsets: createTransientView(
        graph,
        `${id}-face-polygon-offsets`,
        'uint32',
        ringCapacity + 1
      ),
      ringOffsets: createTransientView(
        graph,
        `${id}-face-ring-offsets`,
        'uint32',
        ringCapacity + 1
      ),
      sourceIds: createTransientView(graph, `${id}-face-source-ids`, 'uint32', ringCapacity)
    };
    const polygonizeProps: GPUPolygonizeProps = {
      id: `${id}-polygonize`,
      spatialContext: this.spatialContext,
      lines: {
        kind: 'lines',
        positions: closedPositions,
        lineOffsets: closedOffsets,
        sourceIds: ringSourceIds
      },
      groupIds: ringSourceIds,
      intersectionCapacity: props.intersectionCapacity,
      precision: props.precision,
      snapRounding: props.snapRounding,
      output: {...output, polygons: faces},
      spatialSort: props.spatialSort,
      leafCapacity: props.leafCapacity
    };
    nodes.push(...new GPUPolygonize(polygonizeProps).getCommandNodes(graph));
    // Noding numbered edges by rows of the closed copy. Report rows of polygons.positions instead:
    // ring r's rows were shifted by r, so its closing edge gets the row of its last vertex.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-source-edge-rows`,
        operation: OPERATION,
        variant: 'source-edge-rows',
        bindings: [
          {name: 'count', view: output.nodedSegments.status.count, type: 'u32', access: 'read'},
          {
            name: 'sourceRingIds',
            view: output.nodedSegments.sourceRingIds,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'sourceEdgeIds',
            view: output.nodedSegments.sourceEdgeIds,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: output.nodedSegments.endpoints.length,
        body: `let edge = sourceEdgeIds[sourceEdgeIdsOffset + index];
  let ring = sourceRingIds[sourceRingIdsOffset + index];
  if (index < count[countOffset] && edge != ${GPU_TOPOLOGY_NONE}u && ring != ${GPU_TOPOLOGY_NONE}u) {
    sourceEdgeIds[sourceEdgeIdsOffset + index] = edge - ring;
  }`
      })
    );

    const none = `${GPU_TOPOLOGY_NONE}u`;
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const read = (name: string, view: GraphDataView, type: 'u32' | 'f32' = 'u32') => ({
      name,
      view,
      type,
      access: 'read' as const
    });
    const write = (name: string, view: GraphDataView, type: 'u32' | 'f32' = 'u32') => ({
      name,
      view,
      type,
      access: 'read_write' as const
    });
    const kernel = (
      variant: string,
      bindings: readonly WGSLKernelBinding[],
      invocationCount: number,
      body: string,
      declarations?: string
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${variant}`,
        operation: OPERATION,
        variant,
        bindings,
        invocationCount,
        body,
        declarations
      });
    const last = ringCapacity - 1;

    const nodedGroups = u32('noded-groups', segmentCapacity);
    const keep = u32('keep', ringCapacity);
    const ringCounts = u32('ring-counts', ringCapacity);
    const vertexCounts = u32('vertex-counts', ringCapacity);
    const keepScan = u32('keep-scan', ringCapacity);
    const ringScan = u32('ring-scan', ringCapacity);
    const vertexScan = u32('vertex-scan', ringCapacity);
    const keptIds = u32('kept-ids', ringCapacity);
    const keptVertexStarts = u32('kept-vertex-starts', ringCapacity + 1);

    nodes.push(
      kernel(
        'noded-groups',
        [
          read('count', output.nodedSegments.status.count),
          read('nodedRings', output.nodedSegments.sourceRingIds),
          read('ringSourceIds', ringSourceIds),
          write('nodedGroups', nodedGroups)
        ],
        segmentCapacity,
        `var group = ${none};
  if (index < count[countOffset]) { group = ringSourceIds[ringSourceIdsOffset + nodedRings[nodedRingsOffset + index]]; }
  nodedGroups[nodedGroupsOffset + index] = group;`
      ),
      kernel(
        'face-parity',
        [
          read('nodedEndpoints', output.nodedSegments.endpoints, 'f32'),
          read('nodedGroups', nodedGroups),
          read('facePositions', faces.positions, 'f32'),
          read('faceRingOffsets', faces.ringOffsets),
          read('facePolygonOffsets', faces.polygonOffsets),
          read('faceSourceIds', faces.sourceIds),
          write('keep', keep)
        ],
        ringCapacity,
        `let group = faceSourceIds[faceSourceIdsOffset + index];
  var keepFace = 0u;
  if (group != ${none}) {
    let shell = facePolygonOffsets[facePolygonOffsetsOffset + index];
    let start = faceRingOffsets[faceRingOffsetsOffset + shell];
    let end = faceRingOffsets[faceRingOffsetsOffset + shell + 1u];
    if (end - start >= 3u) {
    // First edge of the counter-clockwise shell: the face lies on its left. The probe is the
    // symbolic point infinitesimally left of the edge's midpoint (utils/exact-geometry-wgsl.ts).
    let a = vec2f(facePositions[facePositionsOffset + 2u * start], facePositions[facePositionsOffset + 2u * start + 1u]);
    let b = vec2f(facePositions[facePositionsOffset + 2u * start + 2u], facePositions[facePositionsOffset + 2u * start + 3u]);
    let probe = ExactProbe(0.5 * (a + b), a, b, 1);
    var parity = 0u;
    var uncertain = false;
    for (var segment = 0u; segment < SEGMENT_CAPACITY; segment++) {
      if (nodedGroups[nodedGroupsOffset + segment] != group) { continue; }
      let base = nodedEndpointsOffset + 4u * segment;
      let first = vec2f(nodedEndpoints[base], nodedEndpoints[base + 1u]);
      let second = vec2f(nodedEndpoints[base + 2u], nodedEndpoints[base + 3u]);
      // The single call site of the exact helpers in this kernel.
      let crossing = exactProbeCrossing(probe, first, second);
      if ((u32(crossing.y) & EXACT_FLAG_UNCERTAIN) != 0u) { uncertain = true; }
      if (crossing.x != 0) { parity = parity ^ 1u; }
    }
    keepFace = select(0u, 1u, parity == 1u || uncertain);
    }
  }
  keep[keepOffset + index] = keepFace;`,
        `const SEGMENT_CAPACITY: u32 = ${segmentCapacity}u;
${EXACT_ORIENTATION_WGSL}
${EXACT_GEOMETRY_WGSL}`
      ),
      kernel(
        'face-counts',
        [
          read('keep', keep),
          read('faceRingOffsets', faces.ringOffsets),
          read('facePolygonOffsets', faces.polygonOffsets),
          write('ringCounts', ringCounts),
          write('vertexCounts', vertexCounts)
        ],
        ringCapacity,
        `let firstRing = facePolygonOffsets[facePolygonOffsetsOffset + index];
  let endRing = facePolygonOffsets[facePolygonOffsetsOffset + index + 1u];
  let isKept = keep[keepOffset + index] != 0u;
  ringCounts[ringCountsOffset + index] = select(0u, endRing - firstRing, isKept);
  vertexCounts[vertexCountsOffset + index] = select(0u, faceRingOffsets[faceRingOffsetsOffset + endRing] - faceRingOffsets[faceRingOffsetsOffset + firstRing], isKept);`
      ),
      ...new GPUScan({
        id: `${id}-keep-scan`,
        input: keep,
        output: keepScan,
        mode: 'inclusive'
      }).getCommandNodes(graph),
      ...new GPUScan({
        id: `${id}-ring-scan`,
        input: ringCounts,
        output: ringScan,
        mode: 'inclusive'
      }).getCommandNodes(graph),
      ...new GPUScan({
        id: `${id}-vertex-scan`,
        input: vertexCounts,
        output: vertexScan,
        mode: 'inclusive'
      }).getCommandNodes(graph),
      kernel(
        'kept-ids',
        [read('keep', keep), read('keepScan', keepScan), write('keptIds', keptIds)],
        ringCapacity,
        `if (keep[keepOffset + index] != 0u) { keptIds[keptIdsOffset + keepScan[keepScanOffset + index] - 1u] = index; }`
      ),
      kernel(
        'polygon-layout',
        [
          read('keepScan', keepScan),
          read('ringScan', ringScan),
          read('ringCounts', ringCounts),
          read('keptIds', keptIds),
          read('faceSourceIds', faces.sourceIds),
          write('featureOffsets', output.polygons.featureOffsets),
          write('polygonOffsets', output.polygons.polygonOffsets),
          write('sourceIds', output.polygons.sourceIds)
        ],
        ringCapacity + 1,
        `let kept = keepScan[keepScanOffset + ${last}u];
  featureOffsets[featureOffsetsOffset + index] = min(index, kept);
  var ringStart = ringScan[ringScanOffset + ${last}u];
  if (index < kept) {
    let face = keptIds[keptIdsOffset + index];
    ringStart = ringScan[ringScanOffset + face] - ringCounts[ringCountsOffset + face];
  }
  polygonOffsets[polygonOffsetsOffset + index] = ringStart;
  if (index < ${ringCapacity}u) {
    var group = ${none};
    if (index < kept) { group = faceSourceIds[faceSourceIdsOffset + keptIds[keptIdsOffset + index]]; }
    sourceIds[sourceIdsOffset + index] = group;
  }`
      ),
      kernel(
        'vertex-starts',
        [
          read('keepScan', keepScan),
          read('vertexScan', vertexScan),
          read('vertexCounts', vertexCounts),
          read('keptIds', keptIds),
          write('keptVertexStarts', keptVertexStarts)
        ],
        ringCapacity + 1,
        `let kept = keepScan[keepScanOffset + ${last}u];
  var start = vertexScan[vertexScanOffset + ${last}u];
  if (index < kept) {
    let face = keptIds[keptIdsOffset + index];
    start = vertexScan[vertexScanOffset + face] - vertexCounts[vertexCountsOffset + face];
  }
  keptVertexStarts[keptVertexStartsOffset + index] = start;`
      ),
      kernel(
        'ring-offsets',
        [
          read('keepScan', keepScan),
          read('keptIds', keptIds),
          read('polygonOffsets', output.polygons.polygonOffsets),
          read('keptVertexStarts', keptVertexStarts),
          read('facePolygonOffsets', faces.polygonOffsets),
          read('faceRingOffsets', faces.ringOffsets),
          write('ringOffsets', output.polygons.ringOffsets)
        ],
        ringCapacity + 1,
        `let kept = keepScan[keepScanOffset + ${last}u];
  if (index >= polygonOffsets[polygonOffsetsOffset + ${ringCapacity}u]) {
    ringOffsets[ringOffsetsOffset + index] = keptVertexStarts[keptVertexStartsOffset + ${ringCapacity}u];
    return;
  }
  var low = 0u;
  var high = kept;
  while (low + 1u < high) {
    let middle = low + (high - low) / 2u;
    if (polygonOffsets[polygonOffsetsOffset + middle] <= index) { low = middle; } else { high = middle; }
  }
  let firstRing = facePolygonOffsets[facePolygonOffsetsOffset + keptIds[keptIdsOffset + low]];
  let sourceRing = firstRing + (index - polygonOffsets[polygonOffsetsOffset + low]);
  ringOffsets[ringOffsetsOffset + index] = keptVertexStarts[keptVertexStartsOffset + low] + faceRingOffsets[faceRingOffsetsOffset + sourceRing] - faceRingOffsets[faceRingOffsetsOffset + firstRing];`
      ),
      kernel(
        'gather-vertices',
        [
          read('keepScan', keepScan),
          read('keptIds', keptIds),
          read('keptVertexStarts', keptVertexStarts),
          read('facePolygonOffsets', faces.polygonOffsets),
          read('faceRingOffsets', faces.ringOffsets),
          read('facePositions', faces.positions, 'f32'),
          write('positions', output.polygons.positions, 'f32')
        ],
        vertexCapacity,
        `if (index >= keptVertexStarts[keptVertexStartsOffset + ${ringCapacity}u]) { return; }
  var low = 0u;
  var high = keepScan[keepScanOffset + ${last}u];
  while (low + 1u < high) {
    let middle = low + (high - low) / 2u;
    if (keptVertexStarts[keptVertexStartsOffset + middle] <= index) { low = middle; } else { high = middle; }
  }
  let firstRing = facePolygonOffsets[facePolygonOffsetsOffset + keptIds[keptIdsOffset + low]];
  let source = faceRingOffsets[faceRingOffsetsOffset + firstRing] + (index - keptVertexStarts[keptVertexStartsOffset + low]);
  positions[positionsOffset + 2u * index] = facePositions[facePositionsOffset + 2u * source];
  positions[positionsOffset + 2u * index + 1u] = facePositions[facePositionsOffset + 2u * source + 1u];`
      ),
      kernel(
        'publish-count',
        [read('keepScan', keepScan), write('count', output.status.count)],
        1,
        `count[countOffset] = keepScan[keepScanOffset + ${last}u];`
      )
    );
    return nodes;
  }
}
