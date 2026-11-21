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
  validateRequirement,
  validateGPUStorageBindingByteLength,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validatePowerOfTwo
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {EXACT_ORIENTATION_WGSL} from '../../analysis-core/kernels/exact-orientation-wgsl';
import {EXACT_GEOMETRY_WGSL} from '../../analysis-core/kernels/exact-geometry-wgsl';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import {getMakeValidSegmentIndexNodes} from './make-valid-segment-index';
import {getBinarySearchWGSL} from '../../analysis-core/kernels/binary-search-wgsl';
import {validateLineNodingBounds} from './gpu-line-noding';
import {GPUPolygonize, type GPUPolygonizeProps} from './gpu-polygonize';
import {
  validatePolygonizeStatus,
  validateSupportedTopologyStatus,
  validateTopologyStatusScalars,
  validateTopologyStatusAliases,
  snapshotTopologyStatus
} from './topology-status';
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
 * Every sign comes from the shared exact helpers; a centroid is not used because it can fall
 * outside a concave face or inside one of its holes. Segments are stably sorted by height, then
 * by exact uint32 source group. Each face binary-searches its group's leaf range before traversing
 * the height BVH, so unrelated groups are pruned independently of their IDs' numeric spacing.
 * Inclusive height bounds retain edges touching the symbolic ray.
 *
 * Status: `count` is the number of kept features, `requiredRingCount` the unclamped ring-assembly
 * requirement (the ring capacity needed, an upper bound on the polygons written), `overflow` flags
 * final ring/vertex capacity loss and `candidateOverflow` any upstream noding loss. Internal
 * groups are feature rows; every uint32 caller identity, including 0xffffffff, is preserved in
 * the active output provenance prefix.
 * If an orientation cannot be certified (non-finite coordinates or a span of more than 200 binary
 * orders of magnitude) the face is kept. Open and cut remnants stay in `output.diagnostics`.
 * Only cardinality, overflow and uncertainty status destinations are published. Inherited
 * invalidity, iteration and approximation destinations are refused at construction; omit them
 * and use `candidateOverflow`, `uncertainCount` and the edge diagnostics instead.
 */
export class GPUMakeValid implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUMakeValidProps;
  /** Resolved spatial context (planar only). */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUMakeValidProps) {
    const snapshot = snapshotConstructionFields(props, [
      'id',
      'spatialContext',
      'polygons',
      'intersectionCapacity',
      'precision',
      'snapRounding',
      'output',
      'spatialSort',
      'leafCapacity'
    ]);
    this.id = snapshot.id ?? 'make-valid';
    this.spatialContext = snapshotConstructionSpatialContext(
      resolveGPUSpatialContext(this.id, snapshot.spatialContext)
    );
    const output = snapshotConstructionFields(snapshot.output, [
      'nodedSegments',
      'directedEdges',
      'polygons',
      'diagnostics',
      'status'
    ]);
    // Capture rejected fields as well as supported ones, so validation sees the caller's
    // complete declared status vocabulary before any unsupported field could be discarded.
    const snapshotStatus = snapshotTopologyStatus;
    props = Object.freeze({
      ...snapshot,
      spatialContext: this.spatialContext,
      polygons: snapshotConstructionFields(snapshot.polygons, [
        'kind',
        'positions',
        'featureOffsets',
        'polygonOffsets',
        'ringOffsets',
        'sourceIds'
      ]),
      precision: snapshotConstructionFields(snapshot.precision, [
        'vertexTolerance',
        'predicates',
        'storage'
      ]),
      snapRounding:
        snapshot.snapRounding &&
        snapshotConstructionFields(snapshot.snapRounding, ['gridSize', 'lattice']),
      output: Object.freeze({
        nodedSegments: Object.freeze({
          ...snapshotConstructionFields(output.nodedSegments, [
            'endpoints',
            'sourceFeatureIds',
            'sourceRingIds',
            'sourceEdgeIds',
            'sourceStartParameters',
            'sourceEndParameters'
          ]),
          status: snapshotStatus(output.nodedSegments.status)
        }),
        directedEdges: Object.freeze({
          ...snapshotConstructionFields(output.directedEdges, [
            'endpoints',
            'sourceSegmentIds',
            'sourceFeatureIds',
            'twinIds',
            'ringIds',
            'flags'
          ]),
          status: snapshotStatus(output.directedEdges.status)
        }),
        polygons: snapshotConstructionFields(output.polygons, [
          'kind',
          'positions',
          'featureOffsets',
          'polygonOffsets',
          'ringOffsets',
          'sourceIds'
        ]),
        diagnostics: snapshotConstructionFields(output.diagnostics, [
          'edgeClasses',
          'closedRingCount',
          'openChainCount',
          'cutEdgeCount',
          'dangleCount',
          'unusedEdgeCount'
        ]),
        status: snapshotStatus(output.status)
      })
    });
    this.props = props;
    validateRequirement(
      this.id,
      !('requiredRingCount' in props.output.directedEdges.status),
      'output.directedEdges.status.requiredRingCount is final-only; use output.directedEdges.status.requiredCount or output.status.requiredRingCount'
    );
    for (const [path, status] of [
      ['output.status', props.output.status],
      ['output.nodedSegments.status', props.output.nodedSegments.status],
      ['output.directedEdges.status', props.output.directedEdges.status]
    ] as const) {
      validateSupportedTopologyStatus(this.id, status, path);
      if (path !== 'output.status') validateTopologyStatusScalars(this.id, status, path);
    }
    validatePolygonizeStatus(this.id, props.output.status);
    validateTopologyStatusAliases(this.id, props.output);
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
    const parityLeafCapacity =
      2 ** Math.ceil(Math.log2(props.output.nodedSegments.endpoints.length));
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'parity segment records',
      props.output.nodedSegments.endpoints.length * 5 * Uint32Array.BYTES_PER_ELEMENT,
      'reduce output.nodedSegments.endpoints.length'
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'parity BVH nodes',
      (2 * parityLeafCapacity - 1) * 2 * Float32Array.BYTES_PER_ELEMENT,
      'reduce output.nodedSegments.endpoints.length'
    );
    const ringCount = polygons.ringOffsets.length - 1;
    // Noding consumes open line paths, while polygon rings close implicitly: one repeated first
    // vertex per ring makes the closing edge part of the arrangement (as GPUPolygonOverlay does).
    // A ring that already repeats its first vertex gains a zero-length edge, which noding drops.
    const vertexCount = polygons.positions.length;
    const closedVertexCount = vertexCount + ringCount;
    validateLineNodingBounds(
      id,
      graph.device,
      closedVertexCount,
      ringCount,
      props.intersectionCapacity
    );
    const ringSourceIds = createTransientView(graph, `${id}-ring-source-ids`, 'uint32', ringCount);
    const ringFeatureRows = createTransientView(
      graph,
      `${id}-ring-feature-rows`,
      'uint32',
      ringCount
    );
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
          {name: 'ringSourceIds', view: ringSourceIds, type: 'u32', access: 'read_write'},
          {name: 'ringFeatureRows', view: ringFeatureRows, type: 'u32', access: 'read_write'}
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
  ringSourceIds[ringSourceIdsOffset + index] = ${polygons.sourceIds ? 'inputSourceIds[inputSourceIdsOffset + featureLow]' : 'featureLow'};
  ringFeatureRows[ringFeatureRowsOffset + index] = featureLow;`
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
      groupIds: ringFeatureRows,
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

    // Pack endpoints and source group together so traversal stays within eight storage bindings.
    const nodedRecords = u32('noded-records', segmentCapacity * 5);
    const faceProbes = createTransientView(graph, `${id}-face-probes`, 'float32x4', ringCapacity);
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
        'parity-segments',
        [
          read('count', output.nodedSegments.status.count),
          read('endpoints', output.nodedSegments.endpoints, 'f32'),
          read('nodedRings', output.nodedSegments.sourceRingIds),
          read('ringFeatureRows', ringFeatureRows),
          write('records', nodedRecords)
        ],
        segmentCapacity,
        `var group = ${none};
  var first = vec2f(0.0);
  var second = vec2f(0.0);
  if (index < count[countOffset]) {
    group = ringFeatureRows[ringFeatureRowsOffset + nodedRings[nodedRingsOffset + index]];
    first = vec2f(endpoints[endpointsOffset + 4u * index], endpoints[endpointsOffset + 4u * index + 1u]);
    second = vec2f(endpoints[endpointsOffset + 4u * index + 2u], endpoints[endpointsOffset + 4u * index + 3u]);
  }
  records[recordsOffset + 5u * index] = bitcast<u32>(first.x);
  records[recordsOffset + 5u * index + 1u] = bitcast<u32>(first.y);
  records[recordsOffset + 5u * index + 2u] = bitcast<u32>(second.x);
  records[recordsOffset + 5u * index + 3u] = bitcast<u32>(second.y);
  records[recordsOffset + 5u * index + 4u] = group;
`
      ),
      kernel(
        'face-probes',
        [
          read('facePositions', faces.positions, 'f32'),
          read('faceRingOffsets', faces.ringOffsets),
          read('facePolygonOffsets', faces.polygonOffsets),
          read('faceSourceIds', faces.sourceIds),
          write('probes', faceProbes, 'f32')
        ],
        ringCapacity,
        `var first = vec2f(0.0);
  var second = vec2f(0.0);
  if (faceSourceIds[faceSourceIdsOffset + index] != ${none}) {
    let shell = facePolygonOffsets[facePolygonOffsetsOffset + index];
    let start = faceRingOffsets[faceRingOffsetsOffset + shell];
    let end = faceRingOffsets[faceRingOffsetsOffset + shell + 1u];
    if (end - start >= 3u) {
      first = vec2f(facePositions[facePositionsOffset + 2u * start], facePositions[facePositionsOffset + 2u * start + 1u]);
      second = vec2f(facePositions[facePositionsOffset + 2u * start + 2u], facePositions[facePositionsOffset + 2u * start + 3u]);
    }
  }
  probes[probesOffset + 4u * index] = first.x;
  probes[probesOffset + 4u * index + 1u] = first.y;
  probes[probesOffset + 4u * index + 2u] = second.x;
  probes[probesOffset + 4u * index + 3u] = second.y;`
      )
    );
    const parityIndex = getMakeValidSegmentIndexNodes<Parameters>(graph, {
      id: `${id}-parity`,
      operation: OPERATION,
      records: nodedRecords,
      leafCapacity: parityLeafCapacity
    });
    const {bvh} = parityIndex;
    nodes.push(
      ...parityIndex.nodes,
      kernel(
        'face-parity',
        [
          read('records', parityIndex.records),
          read('sortedGroups', parityIndex.groups),
          read('probes', faceProbes, 'f32'),
          read('faceSourceIds', faces.sourceIds),
          read('nodeMinima', bvh.nodeMinima, 'f32'),
          read('nodeMaxima', bvh.nodeMaxima, 'f32'),
          read('leafIds', bvh.leafIds),
          write('keep', keep)
        ],
        ringCapacity,
        `let group = faceSourceIds[faceSourceIdsOffset + index];
  var keepFace = 0u;
  if (group != ${none}) {
    let a = vec2f(probes[probesOffset + 4u * index], probes[probesOffset + 4u * index + 1u]);
    let b = vec2f(probes[probesOffset + 4u * index + 2u], probes[probesOffset + 4u * index + 3u]);
    if (any(a != b)) {
    // First edge of the counter-clockwise shell: the face lies on its left. The probe is the
    // symbolic point infinitesimally left of the edge's midpoint (utils/exact-geometry-wgsl.ts).
    let probe = ExactProbe(0.5 * (a + b), a, b, 1);
    var parity = 0u;
    var uncertain = false;
    let groupFirst = lowerBoundSortedGroups(0u, SEGMENT_CAPACITY, group);
    let groupLast = upperBoundSortedGroups(groupFirst, SEGMENT_CAPACITY, group);
    var node = 0u;
    loop {
      let level = firstLeadingBit(node + 1u);
      let span = LEAF_CAPACITY >> level;
      let nodeFirst = (node + 1u - (1u << level)) * span;
      let nodeLast = nodeFirst + span;
      let minimum = vec2f(nodeMinima[nodeMinimaOffset + 2u * node], nodeMinima[nodeMinimaOffset + 2u * node + 1u]);
      let maximum = vec2f(nodeMaxima[nodeMaximaOffset + 2u * node], nodeMaxima[nodeMaximaOffset + 2u * node + 1u]);
      // Use the endpoint interval, not the rounded midpoint: a one-step-tall probe may round
      // onto either endpoint. Equality stays a candidate for the infinitesimal side decision.
      // Keep both sides of the ray: an uncertifiable orientation left of the probe used to
      // retain the face too. Height rejection alone preserves that conservative fallback.
      if (nodeFirst < groupLast && nodeLast > groupFirst && minimum.y <= max(a.y, b.y) && maximum.y >= min(a.y, b.y)) {
        if (node < INTERNAL_NODE_COUNT) {
          node = 2u * node + 1u;
          continue;
        }
        let segment = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
        if (segment < SEGMENT_CAPACITY && records[recordsOffset + 5u * segment + 4u] == group) {
          let base = recordsOffset + 5u * segment;
          let first = vec2f(bitcast<f32>(records[base]), bitcast<f32>(records[base + 1u]));
          let second = vec2f(bitcast<f32>(records[base + 2u]), bitcast<f32>(records[base + 3u]));
          // The single call site of the exact helpers in this kernel.
          let crossing = exactProbeCrossing(probe, first, second);
          if ((u32(crossing.y) & EXACT_FLAG_UNCERTAIN) != 0u) { uncertain = true; }
          if (crossing.x != 0) { parity = parity ^ 1u; }
        }
      }
      loop {
        if (node == 0u || (node & 1u) == 1u) { break; }
        node = (node - 1u) / 2u;
      }
      if (node == 0u) { break; }
      node++;
    }
    keepFace = select(0u, 1u, parity == 1u || uncertain);
    }
  }
  keep[keepOffset + index] = keepFace;`,
        `const SEGMENT_CAPACITY: u32 = ${segmentCapacity}u;
const INTERNAL_NODE_COUNT: u32 = ${bvh.internalNodeCount}u;
const LEAF_CAPACITY: u32 = ${parityLeafCapacity}u;
${getBinarySearchWGSL('sortedGroups')}
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
    if (polygons.sourceIds) {
      nodes.push(
        kernel(
          'publish-source-ids',
          [
            read('keepScan', keepScan),
            read('ringFeatureRows', ringFeatureRows),
            read('ringSourceIds', ringSourceIds),
            write('sourceIds', output.polygons.sourceIds)
          ],
          ringCapacity,
          `if (index < keepScan[keepScanOffset + ${last}u]) {
  let feature = sourceIds[sourceIdsOffset + index];
  var low = 0u;
  var high = ${ringCount}u;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (ringFeatureRows[ringFeatureRowsOffset + middle] < feature) { low = middle + 1u; }
    else { high = middle; }
  }
  sourceIds[sourceIdsOffset + index] = ringSourceIds[ringSourceIdsOffset + low];
}`
        )
      );
    }
    return nodes;
  }
}
