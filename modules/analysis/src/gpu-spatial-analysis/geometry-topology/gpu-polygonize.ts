// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  snapshotLineFeatureGeometry,
  getLineFeatureLookupWGSL,
  type GPULineFeatureGeometry
} from '../../analysis-core/contracts/line-feature-provenance';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

import {
  createTransientView,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validatePowerOfTwo
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPUSegmentRingAssembly,
  GPU_SEGMENT_RING_ASSEMBLY_FLAG_CANCELLED,
  GPU_SEGMENT_RING_ASSEMBLY_FLAG_DANGLING,
  GPU_SEGMENT_RING_ASSEMBLY_NONE
} from '../../analysis-core/ring-assembly/index';
import {GPULineNoding, validateLineNodingBounds} from './gpu-line-noding';
import {GPUSnapRoundedNoding} from './gpu-snap-rounded-noding';
import {createPolygonizeDanglePruningNodes} from './polygonize-dangle-pruning';
import {
  validatePolygonizeStatus,
  validateSupportedTopologyStatus,
  validateTopologyStatusScalars,
  snapshotTopologyStatus,
  validateTopologyStatusAliases
} from './topology-status';
import {
  GPU_POLYGONIZE_EDGE_CLASS,
  GPU_TOPOLOGY_NONE,
  type GPUPolygonizeOutput,
  type GPUTopologyPrecisionPolicy
} from './topology-types';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';

const OPERATION = 'GPUPolygonize';

/** Properties for {@link GPUPolygonize}. */
export type GPUPolygonizeProps = {
  id?: string;
  /**
   * Planar only (default: native planar coordinates). A longitude-latitude context throws at
   * construction: project positions to a planar frame first, for example with
   * `createGPULocalFrame` (`contracts/local-frame`) and `GPULocalFrameProjection`
   * (`geometry-measures`), and pass the frame's context.
   */
  spatialContext?: GPUSpatialContext;
  /** Arbitrary finite planar linework; rings need not be pre-noded or consistently oriented. */
  lines: GPULineFeatureGeometry;
  /** Optional polygonization group per logical feature. Without grouping offsets, one group per line and a shared default group; with offsets, default groups are feature rows. */
  groupIds?: GraphDataView<'uint32'>;
  /**
   * Capacity of the exact intersection event list used by noding. The line count plus four times
   * this capacity is at most {@link getGPULineSplitMaximumPieceBound} (4194304 on a default
   * device), and `lines.positions` has at most {@link getGPUSegmentIntersectionMaximumVertexCount}
   * rows.
   */
  intersectionCapacity: number;
  /** Explicit precision and vertex identity policy. */
  precision: GPUTopologyPrecisionPolicy;
  /** Optional hot-pixel snap rounding before topology assembly; lattice reports the actual grid. */
  snapRounding?: {gridSize: number; lattice: GraphDataView<'float32'>};
  /**
   * Caller-owned noding, half-edge, polygon and diagnostic outputs. `output.status.uncertainCount`
   * is the optional uncertain-predicate counter.
   */
  output: GPUPolygonizeOutput;
  spatialSort?: boolean;
  leafCapacity?: number;
};

/**
 * General bounded polygonization of planar linework.
 *
 * The operation first nodes every crossing, touch and collinear overlap, expands every atomic
 * segment into two directed edges, recursively peels degree-one edges, then applies the ring
 * assembler's deterministic left-face traversal. Peeled edges remain in the source diagnostics.
 * Equivalent directed edges within a group share one traversal owner, while every
 * noded segment and its provenance remains in the output and diagnostics. Clockwise unbounded faces
 * and zero-area walks are excluded from the polygon port. Every polygon is one canonical polygon
 * feature and `polygons.sourceIds` carries the source group of its shell. Diagnostics remain
 * aligned with noded source segments.
 *
 * Every bounded face is emitted (shapely `polygonize` semantics), including a face that fills the
 * hole of a holed ring; no fill rule is applied. Use {@link GPUMakeValid} to keep faces by source
 * parity. Status: `count` is the number of features written, `requiredRingCount` the unclamped
 * ring-assembly requirement (the polygon ring capacity needed), `overflow` final ring/vertex
 * capacity loss and `candidateOverflow` any upstream noding loss. Each supplied `uncertainCount`
 * (top-level, noded or directed) receives the same count on every submission.
 */
export class GPUPolygonize implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUPolygonizeProps;
  /** Resolved spatial context (planar only). */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUPolygonizeProps) {
    this.id = props.id ?? 'polygonize';
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    const snapshot = snapshotConstructionFields(props, [
      'id',
      'spatialContext',
      'lines',
      'groupIds',
      'intersectionCapacity',
      'precision',
      'snapRounding',
      'output',
      'spatialSort',
      'leafCapacity'
    ]);
    const outputSnapshot = snapshotConstructionFields(snapshot.output, [
      'nodedSegments',
      'directedEdges',
      'polygons',
      'diagnostics',
      'status'
    ]);
    const nodedSnapshot = snapshotConstructionFields(outputSnapshot.nodedSegments, [
      'endpoints',
      'sourceFeatureIds',
      'sourceRingIds',
      'sourceEdgeIds',
      'sourceStartParameters',
      'sourceEndParameters',
      'status'
    ]);
    const directedSnapshot = snapshotConstructionFields(outputSnapshot.directedEdges, [
      'endpoints',
      'sourceSegmentIds',
      'sourceFeatureIds',
      'twinIds',
      'ringIds',
      'flags',
      'status'
    ]);
    const nodedStatus = snapshotTopologyStatus(nodedSnapshot.status);
    const directedStatus = snapshotTopologyStatus(directedSnapshot.status);
    const statusSnapshot = snapshotTopologyStatus(outputSnapshot.status);
    validatePolygonizeStatus(this.id, statusSnapshot);
    validateTopologyStatusScalars(this.id, nodedStatus, 'output.nodedSegments.status');
    validateTopologyStatusScalars(this.id, directedStatus, 'output.directedEdges.status');
    validateSupportedTopologyStatus(this.id, statusSnapshot, 'output.status');
    validateSupportedTopologyStatus(this.id, nodedStatus, 'output.nodedSegments.status');
    validateSupportedTopologyStatus(this.id, directedStatus, 'output.directedEdges.status');
    props = Object.freeze({
      ...snapshot,
      lines: snapshotLineFeatureGeometry(this.id, snapshot.lines),
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
          ...nodedSnapshot,
          status: nodedStatus
        }),
        directedEdges: Object.freeze({
          ...directedSnapshot,
          status: directedStatus
        }),
        polygons: snapshotConstructionFields(outputSnapshot.polygons, [
          'kind',
          'positions',
          'featureOffsets',
          'polygonOffsets',
          'ringOffsets',
          'sourceIds'
        ]),
        diagnostics: snapshotConstructionFields(outputSnapshot.diagnostics, [
          'edgeClasses',
          'closedRingCount',
          'openChainCount',
          'cutEdgeCount',
          'dangleCount',
          'unusedEdgeCount'
        ]),
        status: statusSnapshot
      })
    });
    this.props = props;
    validateTopologyStatusAliases(this.id, props.output);
    const {id} = this;
    const {output, precision} = props;
    if (!(precision.vertexTolerance > 0) || !Number.isFinite(precision.vertexTolerance)) {
      throw new Error(`${id} precision.vertexTolerance must be finite and positive`);
    }
    if (precision.predicates !== 'exact-or-uncertain' || precision.storage !== 'float32') {
      throw new Error(`${id} supports exact-or-uncertain predicates with float32 storage`);
    }
    // Delegates are built later, in getCommandNodes: check forwarded options now, under this id.
    validateCapacity(id, 'intersectionCapacity', props.intersectionCapacity);
    if (props.leafCapacity !== undefined) {
      validatePowerOfTwo(id, 'leafCapacity', props.leafCapacity);
    }
    if (props.groupIds) {
      validatePackedUint32View(props.groupIds, `${id} groupIds`);
      if (
        props.groupIds.length !==
        (props.lines.featureOffsets ?? props.lines.lineOffsets).length - 1
      ) {
        throw new Error(
          `${id} groupIds length must equal the logical feature count (line count without featureOffsets)`
        );
      }
    }
    const segmentCapacity = output.nodedSegments.endpoints.length;
    const directedCapacity = output.directedEdges.endpoints.length;
    if (directedCapacity !== segmentCapacity * 2) {
      throw new Error(`${id} directed edge capacity must be twice the noded segment capacity`);
    }
    for (const [name, view] of [
      ['sourceSegmentIds', output.directedEdges.sourceSegmentIds],
      ['sourceFeatureIds', output.directedEdges.sourceFeatureIds],
      ['twinIds', output.directedEdges.twinIds],
      ['ringIds', output.directedEdges.ringIds],
      ['flags', output.directedEdges.flags]
    ] as const) {
      validatePackedUint32View(view, `${id} output.directedEdges.${name}`);
      if (view.length !== directedCapacity) {
        throw new Error(`${id} directedEdges.${name} length must equal endpoints length`);
      }
    }
    validateSinglePackedView(
      output.directedEdges.endpoints,
      ['float32x4'],
      `${id} output.directedEdges.endpoints`
    );
    validatePackedUint32View(output.diagnostics.edgeClasses, `${id} diagnostics.edgeClasses`);
    if (output.diagnostics.edgeClasses.length !== segmentCapacity) {
      throw new Error(`${id} diagnostics.edgeClasses length must equal noded segment capacity`);
    }
    for (const [name, view] of Object.entries(output.diagnostics)) {
      if (name !== 'edgeClasses') {
        validatePackedUint32View(view, `${id} diagnostics.${name}`);
        if (view.length < 1) {
          throw new Error(`${id} diagnostics.${name} must contain one row`);
        }
      }
    }
    const ringCapacity = output.polygons.ringOffsets.length - 1;
    if (
      ringCapacity < 1 ||
      output.polygons.polygonOffsets.length !== ringCapacity + 1 ||
      output.polygons.featureOffsets.length !== ringCapacity + 1 ||
      output.polygons.sourceIds.length !== ringCapacity
    ) {
      throw new Error(`${id} polygon offset and source capacities must share one ring capacity`);
    }
    if (output.polygons.kind !== 'polygons') {
      throw new Error(`${id} output.polygons.kind must be polygons`);
    }
    if (!output.status.requiredRingCount || !output.status.candidateOverflow) {
      throw new Error(`${id} output.status requires requiredRingCount and candidateOverflow`);
    }
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.lines.positions,
      props.lines.lineOffsets,
      props.lines.sourceIds,
      props.lines.featureOffsets,
      props.groupIds,
      output.nodedSegments.endpoints,
      output.nodedSegments.sourceFeatureIds,
      output.nodedSegments.sourceRingIds,
      output.nodedSegments.sourceEdgeIds,
      output.nodedSegments.sourceStartParameters,
      output.nodedSegments.sourceEndParameters,
      ...Object.values(output.nodedSegments.status),
      output.directedEdges.endpoints,
      output.directedEdges.sourceSegmentIds,
      output.directedEdges.sourceFeatureIds,
      output.directedEdges.twinIds,
      output.directedEdges.ringIds,
      output.directedEdges.flags,
      ...Object.values(output.directedEdges.status),
      output.polygons.positions,
      output.polygons.featureOffsets,
      output.polygons.polygonOffsets,
      output.polygons.ringOffsets,
      output.polygons.sourceIds,
      output.diagnostics.edgeClasses,
      ...Object.values(output.diagnostics).filter(view => view !== output.diagnostics.edgeClasses),
      ...Object.values(output.status)
    ]);
    validateLineNodingBounds(
      id,
      graph.device,
      props.lines.positions.length,
      props.lines.lineOffsets.length - 1,
      props.intersectionCapacity
    );
    const nodes: GPUCommandNode<Parameters>[] = [];
    const uncertaintyOutputs = [
      ...new Set(
        [
          output.status.uncertainCount,
          output.nodedSegments.status.uncertainCount,
          output.directedEdges.status.uncertainCount
        ].filter((view): view is GraphDataView<'uint32'> => Boolean(view))
      )
    ];
    const uncertainCount = uncertaintyOutputs.length
      ? createTransientView(graph, `${id}-uncertain-count`, 'uint32', 1)
      : undefined;
    const nodingProps = {
      id: `${id}-noding`,
      spatialContext: this.spatialContext,
      lines: props.lines,
      intersectionCapacity: props.intersectionCapacity,
      output: {
        ...output.nodedSegments,
        status: {...output.nodedSegments.status, uncertainCount}
      },
      spatialSort: props.spatialSort,
      leafCapacity: props.leafCapacity
    };
    const noding = props.snapRounding
      ? new GPUSnapRoundedNoding({
          ...nodingProps,
          gridSize: props.snapRounding.gridSize,
          output: {
            ...nodingProps.output,
            lattice: props.snapRounding.lattice,
            latticeEndpoints: createTransientView(
              graph,
              `${id}-lattice-endpoints`,
              'float32x4',
              output.nodedSegments.endpoints.length
            )
          }
        })
      : new GPULineNoding(nodingProps);
    nodes.push(...noding.getCommandNodes(graph));

    const segmentCapacity = output.nodedSegments.endpoints.length;
    const directedCapacity = output.directedEdges.endpoints.length;
    const directedGroups = createTransientView(
      graph,
      `${id}-directed-groups`,
      'uint32',
      directedCapacity
    );
    const groupBindings =
      props.groupIds && !props.lines.featureOffsets
        ? [
            {
              name: 'inputGroups',
              view: props.groupIds,
              type: 'u32' as const,
              access: 'read' as const
            }
          ]
        : [];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-directed-edge-geometry`,
        operation: OPERATION,
        variant: 'directed-edge-geometry',
        bindings: [
          {name: 'count', view: output.nodedSegments.status.count, type: 'u32', access: 'read'},
          {
            name: 'sourceEndpoints',
            view: output.nodedSegments.endpoints,
            type: 'f32',
            access: 'read'
          },
          {
            name: 'endpoints',
            view: output.directedEdges.endpoints,
            type: 'f32',
            access: 'read_write'
          }
        ],
        invocationCount: directedCapacity,
        body: `let source = index / 2u;
  let isActive = source < count[countOffset];
  let input = sourceEndpointsOffset + source * 4u;
  let outputBase = endpointsOffset + index * 4u;
  if (isActive && (index & 1u) == 0u) {
    endpoints[outputBase] = sourceEndpoints[input]; endpoints[outputBase + 1u] = sourceEndpoints[input + 1u];
    endpoints[outputBase + 2u] = sourceEndpoints[input + 2u]; endpoints[outputBase + 3u] = sourceEndpoints[input + 3u];
  } else if (isActive) {
    endpoints[outputBase] = sourceEndpoints[input + 2u]; endpoints[outputBase + 1u] = sourceEndpoints[input + 3u];
    endpoints[outputBase + 2u] = sourceEndpoints[input]; endpoints[outputBase + 3u] = sourceEndpoints[input + 1u];
  } else {
    endpoints[outputBase] = 0.0; endpoints[outputBase + 1u] = 0.0;
    endpoints[outputBase + 2u] = 0.0; endpoints[outputBase + 3u] = 0.0;
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-directed-edge-provenance`,
        operation: OPERATION,
        variant: 'directed-edge-provenance',
        bindings: [
          {name: 'count', view: output.nodedSegments.status.count, type: 'u32', access: 'read'},
          {
            name: 'sourceFeatures',
            view: output.nodedSegments.sourceFeatureIds,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'sourceRings',
            view: output.nodedSegments.sourceRingIds,
            type: 'u32',
            access: 'read'
          },
          ...groupBindings,
          {
            name: 'sourceSegmentIds',
            view: output.directedEdges.sourceSegmentIds,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'sourceFeatureIds',
            view: output.directedEdges.sourceFeatureIds,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'twinIds', view: output.directedEdges.twinIds, type: 'u32', access: 'read_write'},
          {name: 'groups', view: directedGroups, type: 'u32', access: 'read_write'}
        ],
        invocationCount: directedCapacity,
        declarations: `const NONE: u32 = ${GPU_TOPOLOGY_NONE}u;`,
        body: `let source = index / 2u;
  let isActive = source < count[countOffset];
  sourceSegmentIds[sourceSegmentIdsOffset + index] = select(NONE, source, isActive);
  sourceFeatureIds[sourceFeatureIdsOffset + index] = select(NONE, sourceFeatures[sourceFeaturesOffset + source], isActive);
  twinIds[twinIdsOffset + index] = select(NONE, index ^ 1u, isActive);
  let sourceRing = sourceRings[sourceRingsOffset + source];
  groups[groupsOffset + index] = select(NONE, ${props.groupIds && !props.lines.featureOffsets ? 'inputGroups[inputGroupsOffset + sourceRing]' : '0u'}, isActive);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-directed-edge-status`,
        operation: OPERATION,
        variant: 'directed-edge-status',
        bindings: [
          {name: 'count', view: output.nodedSegments.status.count, type: 'u32', access: 'read'},
          {
            name: 'requiredCount',
            view: output.nodedSegments.status.requiredCount,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'candidateOverflow',
            view: output.nodedSegments.status.candidateOverflow,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'directedCount',
            view: output.directedEdges.status.count,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'directedRequired',
            view: output.directedEdges.status.requiredCount,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'directedOverflow',
            view: output.directedEdges.status.overflow,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'directedCandidateOverflow',
            view: output.directedEdges.status.candidateOverflow,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: 1,
        body: `directedCount[directedCountOffset] = 2u * count[countOffset];
  directedRequired[directedRequiredOffset] = 2u * requiredCount[requiredCountOffset];
  directedOverflow[directedOverflowOffset] = select(0u, 1u, 2u * requiredCount[requiredCountOffset] > ${directedCapacity}u);
  directedCandidateOverflow[directedCandidateOverflowOffset] = candidateOverflow[candidateOverflowOffset];`
      })
    );

    if (props.lines.featureOffsets)
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-feature-groups`,
          operation: OPERATION,
          invocationCount: directedCapacity,
          bindings: [
            {name: 'parts', view: output.nodedSegments.sourceRingIds, type: 'u32', access: 'read'},
            {name: 'count', view: output.nodedSegments.status.count, type: 'u32', access: 'read'},
            {name: 'features', view: props.lines.featureOffsets, type: 'u32', access: 'read'},
            ...(props.groupIds
              ? [{name: 'inputGroups', view: props.groupIds, type: 'u32', access: 'read'} as const]
              : []),
            {name: 'groups', view: directedGroups, type: 'u32', access: 'read_write'}
          ],
          declarations: getLineFeatureLookupWGSL(props.lines),
          body: `var group = 0xffffffffu;
if (index / 2u < count[countOffset]) {
  let feature = lineFeature(parts[partsOffset + index / 2u]);
  group = ${props.groupIds ? 'inputGroups[inputGroupsOffset + feature]' : 'feature'};
}
groups[groupsOffset + index] = group;`
        })
      );
    const ringCapacity = output.polygons.ringOffsets.length - 1;
    const vertexCapacity = output.polygons.positions.length;
    const transient = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);
    const ringPositions = transient('ring-positions', 'float32x2', vertexCapacity);
    const ringOffsets = transient('ring-offsets', 'uint32', ringCapacity + 1);
    const ringAreas = transient('ring-areas', 'float32', ringCapacity);
    const ringIsHole = transient('ring-is-hole', 'uint32', ringCapacity);
    const ringShells = transient('ring-shells', 'uint32', ringCapacity);
    const ringCount = transient('ring-count', 'uint32', 1);
    const ringRequiredCount = transient('ring-required-count', 'uint32', 1);
    const ringOverflow = transient('ring-overflow', 'uint32', 1);
    const faceInput = createPolygonizeDanglePruningNodes(graph, {
      id: `${id}-dangles`,
      endpoints: output.directedEdges.endpoints,
      groups: directedGroups,
      count: output.directedEdges.status.count,
      vertexTolerance: props.precision.vertexTolerance
    });
    nodes.push(...faceInput.nodes);
    nodes.push(
      ...new GPUSegmentRingAssembly({
        id: `${id}-faces`,
        spatialContext: this.spatialContext,
        endpoints: faceInput.endpoints,
        count: output.directedEdges.status.count,
        groups: directedGroups,
        pairedOpposites: true,
        vertexTolerance: props.precision.vertexTolerance,
        interiorSide: 'left',
        normalizeWinding: true,
        splitTouchingRings: true,
        output: {
          positions: ringPositions,
          ringOffsets,
          ringAreas,
          ringIsHole,
          ringShells,
          segmentRings: output.directedEdges.ringIds,
          segmentFlags: output.directedEdges.flags,
          status: {count: ringCount, requiredCount: ringRequiredCount, overflow: ringOverflow},
          polygons: output.polygons
        }
      }).getCommandNodes(graph)
    );

    const diagnosticCounters = transient('diagnostic-counters', 'uint32', 5);
    const edgeClass = GPU_POLYGONIZE_EDGE_CLASS;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-diagnostic-clear`,
        operation: OPERATION,
        variant: 'diagnostic-clear',
        bindings: [{name: 'counters', view: diagnosticCounters, type: 'u32', access: 'read_write'}],
        invocationCount: 5,
        body: 'counters[countersOffset + index] = 0u;'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-diagnostics`,
        operation: OPERATION,
        variant: 'diagnostics',
        bindings: [
          {
            name: 'segmentCount',
            view: output.nodedSegments.status.count,
            type: 'u32',
            access: 'read'
          },
          {name: 'ringIds', view: output.directedEdges.ringIds, type: 'u32', access: 'read'},
          {name: 'flags', view: output.directedEdges.flags, type: 'u32', access: 'read'},
          {name: 'ringIsHole', view: ringIsHole, type: 'u32', access: 'read'},
          {name: 'ringShells', view: ringShells, type: 'u32', access: 'read'},
          {name: 'ringAreas', view: ringAreas, type: 'f32', access: 'read'},
          {
            name: 'edgeClasses',
            view: output.diagnostics.edgeClasses,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'counters', view: diagnosticCounters, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: segmentCapacity,
        // Workgroup-local counters, then five global atomics per workgroup instead of per segment.
        guardIndex: false,
        declarations: `var<workgroup> localCounts: array<atomic<u32>, 5>;
const NONE: u32 = ${GPU_SEGMENT_RING_ASSEMBLY_NONE}u;
const FLAG_CANCELLED: u32 = ${GPU_SEGMENT_RING_ASSEMBLY_FLAG_CANCELLED}u;
const FLAG_DANGLING: u32 = ${GPU_SEGMENT_RING_ASSEMBLY_FLAG_DANGLING}u;`,
        body: `if (localInvocationIndex < 5u) { atomicStore(&localCounts[localInvocationIndex], 0u); }
  workgroupBarrier();
  var classification = 0u;
  if (index < INVOCATION_COUNT && index < segmentCount[segmentCountOffset]) {
    let first = 2u * index;
    let firstRing = ringIds[ringIdsOffset + first];
    let secondRing = ringIds[ringIdsOffset + first + 1u];
    let firstBounded = firstRing != NONE && abs(ringAreas[ringAreasOffset + firstRing]) > 0.0 && (ringIsHole[ringIsHoleOffset + firstRing] == 0u || ringShells[ringShellsOffset + firstRing] != NONE);
    let secondBounded = secondRing != NONE && abs(ringAreas[ringAreasOffset + secondRing]) > 0.0 && (ringIsHole[ringIsHoleOffset + secondRing] == 0u || ringShells[ringShellsOffset + secondRing] != NONE);
    let used = firstBounded || secondBounded;
    let flagBits = flags[flagsOffset + first] | flags[flagsOffset + first + 1u];
    let dangle = (flagBits & FLAG_DANGLING) != 0u;
    let cancelled = (flagBits & FLAG_CANCELLED) != 0u;
    if (used) {
      classification = ${edgeClass.closedRing}u;
      atomicAdd(&localCounts[0u], 1u);
    } else {
      classification = ${edgeClass.unused}u;
      atomicAdd(&localCounts[4u], 1u);
      if (!cancelled) {
        classification = classification | ${edgeClass.openChain}u;
        atomicAdd(&localCounts[1u], 1u);
      }
      if (dangle) {
        classification = classification | ${edgeClass.dangle}u;
        atomicAdd(&localCounts[3u], 1u);
      } else if (!cancelled) {
        classification = classification | ${edgeClass.cut}u;
        atomicAdd(&localCounts[2u], 1u);
      }
    }
  }
  if (index < INVOCATION_COUNT) { edgeClasses[edgeClassesOffset + index] = classification; }
  workgroupBarrier();
  if (localInvocationIndex < 5u) {
    let localCount = atomicLoad(&localCounts[localInvocationIndex]);
    if (localCount > 0u) { atomicAdd(&counters[countersOffset + localInvocationIndex], localCount); }
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish-diagnostics`,
        operation: OPERATION,
        variant: 'publish-diagnostics',
        bindings: [
          {name: 'counters', view: diagnosticCounters, type: 'u32', access: 'read'},
          {
            name: 'closedRingCount',
            view: output.diagnostics.closedRingCount,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'openChainCount',
            view: output.diagnostics.openChainCount,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'cutEdgeCount',
            view: output.diagnostics.cutEdgeCount,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'dangleCount',
            view: output.diagnostics.dangleCount,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'unusedEdgeCount',
            view: output.diagnostics.unusedEdgeCount,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: 1,
        body: `closedRingCount[closedRingCountOffset] = counters[countersOffset];
  openChainCount[openChainCountOffset] = counters[countersOffset + 1u];
  cutEdgeCount[cutEdgeCountOffset] = counters[countersOffset + 2u];
  dangleCount[dangleCountOffset] = counters[countersOffset + 3u];
  unusedEdgeCount[unusedEdgeCountOffset] = counters[countersOffset + 4u];`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish-status`,
        operation: OPERATION,
        variant: 'publish-status',
        bindings: [
          {name: 'ringOverflow', view: ringOverflow, type: 'u32', access: 'read'},
          {name: 'ringRequiredCount', view: ringRequiredCount, type: 'u32', access: 'read'},
          {
            name: 'featureOffsets',
            view: output.polygons.featureOffsets,
            type: 'u32',
            access: 'read'
          },
          {name: 'count', view: output.status.count, type: 'u32', access: 'read_write'},
          {
            name: 'requiredRingCount',
            view: output.status.requiredRingCount,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'overflow', view: output.status.overflow, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `const RING_CAPACITY: u32 = ${ringCapacity}u;`,
        // requiredRingCount is the unclamped ring-assembly requirement (every face ring, including
        // holes), the number to size the polygon ring capacity with.
        body: `count[countOffset] = featureOffsets[featureOffsetsOffset + RING_CAPACITY];
  requiredRingCount[requiredRingCountOffset] = ringRequiredCount[ringRequiredCountOffset];
  overflow[overflowOffset] = ringOverflow[ringOverflowOffset];`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish-candidate-overflow`,
        operation: OPERATION,
        variant: 'publish-candidate-overflow',
        bindings: [
          {
            name: 'nodingOverflow',
            view: output.nodedSegments.status.overflow,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'nodingCandidateOverflow',
            view: output.nodedSegments.status.candidateOverflow,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'candidateOverflow',
            view: output.status.candidateOverflow,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: 1,
        // Noded capacity loss is intermediate loss for the polygon consumer.
        body: 'candidateOverflow[candidateOverflowOffset] = nodingCandidateOverflow[nodingCandidateOverflowOffset] | nodingOverflow[nodingOverflowOffset];'
      })
    );
    uncertaintyOutputs.forEach((view, index) => {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-publish-uncertain-${index}`,
          operation: OPERATION,
          variant: 'publish-uncertain',
          invocationCount: 1,
          bindings: [
            {name: 'uncertainCount', view: uncertainCount!, type: 'u32', access: 'read'},
            {name: 'destination', view, type: 'u32', access: 'read_write'}
          ],
          body: 'destination[destinationOffset] = uncertainCount[uncertainCountOffset];'
        })
      );
    });
    return nodes;
  }
}
