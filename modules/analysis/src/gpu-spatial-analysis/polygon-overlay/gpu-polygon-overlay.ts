// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUPolygonOverlayFillRule, GPUPolygonOverlayProps} from './polygon-overlay-types';
import {validatePolygons, validateStatus, validateOverlayOptions} from './overlay-input-validation';
import {getPolygonOverlayPlan} from './overlay-planning';
import {createOverlayAssemblyCompletenessNode} from './overlay-assembly-completeness';
import {
  createOverlayFeatureSourceNodes,
  validateOverlayFeatureSourceOutput
} from './overlay-feature-sources';
import {createOverlaySourceIdentityNodes} from './overlay-source-identities';

import {snapshotOverlayStatusFields, getOverlayOutputViews} from './overlay-storage-validation';
import {clamp} from '@math.gl/core';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  createTransientView,
  GPUScan,
  validatePackedUint32View,
  type GPUBVH,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createStableLexicographicIndexSortNodes} from '../../analysis-core/kernels/stable-lexicographic-index-sort';
import {
  getNextPowerOfTwo,
  validateCapacity,
  validateSinglePackedView,
  validateRequirement,
  validateFiniteNumber,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';

import {
  GPUSnapRoundedNoding,
  type GPUSnapRoundedSegmentPort
} from '../geometry-topology/gpu-snap-rounded-noding';

import {getSegmentBVHNodes} from '../segment-intersection/segment-bvh';
import {GPUSegmentRingAssembly} from '../../analysis-core/ring-assembly/index';
import {
  resolveGPUSpatialContext,
  resolveGPUSecondarySpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  createPointerJumpNodes,
  getFaceRayShootBodyWGSL,
  getFaceRayShootDeclarationsWGSL,
  getInsideWindingWGSL,
  getPointerJumpRoundCount
} from './overlay-face-labelling-wgsl';

const OPERATION = 'GPUPolygonOverlay';
/** Positive shader thresholds must survive WebGPU subnormal flushing and f32 conversion. */
const MINIMUM_VERTEX_TOLERANCE = 2 ** -126;
/** Snap rounding rounds up to a power-of-two cell, whose largest finite f32 is 2^127. */
const MAXIMUM_VERTEX_TOLERANCE = 2 ** 127;

/** Bits of a sorted lattice coordinate: lattice values are at most 2^22 + 1, the sentinel 2^23 - 1. */
const LATTICE_KEY_BITS = 23;

/** `readLattice(row)`: the canonical `vec4f(lo, hi)` of a sorted piece row. Needs a `sortedLattice` binding. */
const READ_LATTICE_WGSL = /* wgsl */ `
fn readLattice(row: u32) -> vec4f {
  let base = sortedLatticeOffset + row * 4u;
  return vec4f(sortedLattice[base], sortedLattice[base + 1u], sortedLattice[base + 2u], sortedLattice[base + 3u]);
}`;

/** Value step of a pointer-jump chain: the parent's contribution (`COMPONENTS` words per group). */
const JUMP_INIT_BODY_WGSL = `let parent = parents[parentsOffset + index];
  for (var component = 0u; component < COMPONENTS; component++) {
    var value = 0u;
    if (parent != NO_GROUP) { value = delta[deltaOffset + parent * COMPONENTS + component]; }
    values[valuesOffset + index * COMPONENTS + component] = value;
  }`;

export type {
  GPUPolygonOverlayOperation,
  GPUPolygonOverlayFillRule,
  GPUPolygonOverlayCapacity,
  GPUPolygonOverlayBoundaryPort,
  GPUPolygonOverlayOutput,
  GPUPolygonOverlayProps
} from './polygon-overlay-types';

/**
 * General polygon Boolean overlay over a snap-rounded planar arrangement.
 *
 * Both operands are noded together by snap rounding (`GPUSnapRoundedNoding`): vertices and
 * crossings are rounded to the lattice described at `vertexTolerance`, and every edge is rerouted
 * through the hot pixels it passes, so the pieces meet only at shared lattice vertices and
 * coincident pieces are bit-identical. Faces are then labelled by winding number, not by testing
 * each piece against every ring: coincident pieces form groups, one ray toward +x per group finds
 * the nearest group to its right (exact predicates on the lattice, BVH walk), and pointer jumping
 * along those parents sums the winding of each polygon (level 1) and the number of covering
 * polygons of each operand (level 2) beside every group. A group is kept when the Boolean result
 * differs across it, oriented with the retained surface on its left. Ring assembly matches
 * vertices exactly on the lattice and produces compact GeoArrow-style polygons, including holes.
 * `dissolve` applies the same rule to the union of `left`, so shared and overlapping internal
 * boundaries disappear.
 *
 * Operand membership uses the `fillRule` (default even-odd shells and holes; `'positive'`,
 * `'negative'` and `'nonzero'` keep regions a raw boundary covers twice, which `GPUBufferSurface`
 * relies on). When rounding makes an operand invalid (for example, retraced spikes), this winding
 * rule on the snap-rounded linework still defines the result; GEOS polygon overlay on invalid
 * operands need not agree.
 *
 * Output and arrangement capacities are explicit. `candidateOverflow` means noding was incomplete,
 * including non-finite input, pairs the exact predicates could not certify, open assembly segments
 * and continuation conflicts; `overflow` covers
 * capacity loss in the emitted noded segments, boundary, rings or vertices. Both flags must be zero
 * before accepting geometry or provenance. Boundary provenance is one row per group of coincident
 * pieces, from the first piece of the group in lattice order, and remains aligned with the compact
 * selected edge list. Results are exact on the lattice: no node drift, no f32 crossing points.
 * For valid pointwise lattice-rounded operands, the result equals GEOS OverlayNG with `grid_size`
 * applied to those operands (shapely on `set_precision` inputs).
 *
 * The optional `output.status.requiredRingCount` is the unclamped number of rings (shells plus
 * holes) the assembly needed, so callers can size the ring capacity of `output.geometry`; the
 * optional `ringCount` is the number of rings written;
 * `output.boundary.status.requiredCount` is the unclamped selected-edge count. Optional
 * `output.status.nodedSegmentRequiredCount` reports the noder's unclamped piece requirement.
 * When boundary overflow is zero, positions need at most selected edges plus required rings
 * (one closing vertex per ring). Candidate overflow must be resolved before trusting the plan.
 */
export class GPUPolygonOverlay implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUPolygonOverlayProps;
  /** Resolved spatial context (planar only). */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved inside rule. */
  readonly fillRule: GPUPolygonOverlayFillRule;

  constructor(props: GPUPolygonOverlayProps) {
    const captured = snapshotConstructionFields(props, [
      'id',
      'spatialContext',
      'rightSpatialContext',
      'left',
      'right',
      'operation',
      'fillRule',
      'capacity',
      'vertexTolerance',
      'output',
      'spatialSort',
      'leafCapacity'
    ]);
    this.id = captured.id ?? 'polygon-overlay';
    const captureOperand = (operand: GPUPolygonOverlayProps['left']) =>
      snapshotConstructionFields(operand, [
        'kind',
        'positions',
        'ringOffsets',
        'polygonOffsets',
        'featureOffsets',
        'sourceIds'
      ]);
    const left = captureOperand(captured.left);
    const right = captured.right && captureOperand(captured.right);
    const outputFields = snapshotConstructionFields(captured.output, [
      'geometry',
      'boundary',
      'status',
      'featureSources'
    ]);
    const capturedBoundary = snapshotConstructionFields(outputFields.boundary, [
      'endpoints',
      'operandIds',
      'sourceFeatureIds',
      'status'
    ]);
    const capturedOutput = {...outputFields, boundary: capturedBoundary};
    const statusSnapshot = snapshotOverlayStatusFields(this.id, capturedOutput);
    props = Object.freeze({
      ...captured,
      left,
      right,
      output: Object.freeze({
        ...capturedOutput,
        status: statusSnapshot.status,
        boundary: Object.freeze({
          ...capturedBoundary,
          status: statusSnapshot.boundaryStatus
        })
      })
    });
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    if (props.rightSpatialContext !== undefined && !props.right) {
      throw new Error(`${this.id} rightSpatialContext needs right`);
    }
    resolveGPUSecondarySpatialContext(
      this.id,
      'right',
      props.rightSpatialContext,
      this.spatialContext
    );
    this.props = props;
    validateFiniteNumber(
      this.id,
      'vertexTolerance',
      props.vertexTolerance,
      0,
      MAXIMUM_VERTEX_TOLERANCE
    );
    validateFiniteNumber(
      this.id,
      'vertexTolerance converted to float32',
      Math.fround(props.vertexTolerance),
      MINIMUM_VERTEX_TOLERANCE,
      MAXIMUM_VERTEX_TOLERANCE
    );
    this.fillRule = props.fillRule ?? 'even-odd';
    const {id} = this;
    const {output} = props;
    validateOverlayOptions(id, props.operation, props.fillRule);
    validatePolygons(id, 'left', props.left);
    if (props.operation === 'dissolve') {
      if (props.right) {
        throw new Error(`${id} dissolve accepts only the left operand`);
      }
    } else if (!props.right) {
      throw new Error(`${id} ${props.operation} requires the right operand`);
    }
    if (props.right) {
      validatePolygons(id, 'right', props.right);
    }
    if (!(props.vertexTolerance > 0) || !Number.isFinite(props.vertexTolerance)) {
      throw new Error(`${id} vertexTolerance must be finite and positive`);
    }
    for (const [name, value] of Object.entries(props.capacity)) {
      validateCapacity(id, `capacity.${name}`, value);
    }
    validatePolygons(id, 'output.geometry', output.geometry);
    validateRequirement(
      id,
      (output.geometry as {sourceIds?: unknown}).sourceIds === undefined,
      'output.geometry.sourceIds is not written: an overlay polygon has no single source row'
    );
    const ringCapacity = output.geometry.ringOffsets.length - 1;
    if (
      output.geometry.polygonOffsets.length !== ringCapacity + 1 ||
      output.geometry.featureOffsets.length !== ringCapacity + 1
    ) {
      throw new Error(`${id} output polygon offsets must share one ring capacity`);
    }
    validateOverlayFeatureSourceOutput(id, output);
    const boundary = output.boundary;
    validateSinglePackedView(boundary.endpoints, ['float32x4'], `${id} output.boundary.endpoints`);
    for (const [name, view] of [
      ['operandIds', boundary.operandIds],
      ['sourceFeatureIds', boundary.sourceFeatureIds]
    ] as const) {
      validatePackedUint32View(view, `${id} output.boundary.${name}`);
      if (view.length !== boundary.endpoints.length) {
        throw new Error(`${id} output.boundary.${name} length must equal endpoints length`);
      }
    }
    validateStatus(id, 'output.status', output.status);
    validateStatus(id, 'output.boundary.status', boundary.status);
    validateGraphOutputsDisjointFromInputs(id, getOverlayOutputViews(output), [
      props.left.sourceIds,
      props.right?.sourceIds
    ]);
  }

  /** Checks every graph/device constraint without declaring scratch or command nodes. */
  preflight<Parameters>(graph: GPUCommandGraph<Parameters>): GPUPolygonOverlayPlan {
    return getPolygonOverlayPlan(graph, this.id, this.props);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {left, right, output} = props;
    this.preflight(graph);

    const rightVertexCount = right?.positions.length ?? 0;
    const leftRingCount = left.ringOffsets.length - 1;
    const rightRingCount = right ? right.ringOffsets.length - 1 : 0;
    // Noding consumes open line paths, while polygon rings close implicitly. Add one repeated
    // first vertex per ring so the closing boundary edge participates in the arrangement.
    const leftExpandedVertexCount = left.positions.length + leftRingCount;
    const combinedVertexCount = leftExpandedVertexCount + rightVertexCount + rightRingCount;
    const combinedRingCount = leftRingCount + rightRingCount;
    const combinedPositions = createTransientView(
      graph,
      `${id}-combined-positions`,
      'float32x2',
      combinedVertexCount
    );
    const combinedOffsets = createTransientView(
      graph,
      `${id}-combined-offsets`,
      'uint32',
      combinedRingCount + 1
    );
    const combinedSources = createTransientView(
      graph,
      `${id}-combined-sources`,
      'uint32',
      combinedRingCount
    );
    const nodes: GPUCommandNode<Parameters>[] = [];
    const rightPositionBinding = right
      ? [
          {
            name: 'rightPositions',
            view: right.positions,
            type: 'f32' as const,
            access: 'read' as const
          }
        ]
      : [];
    const combinePositionsNode = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-combine-positions`,
      operation: OPERATION,
      variant: 'combine-positions',
      bindings: [
        {name: 'leftPositions', view: left.positions, type: 'f32', access: 'read'},
        ...rightPositionBinding,
        {name: 'leftRingOffsets', view: left.ringOffsets, type: 'u32', access: 'read'},
        ...(right
          ? [
              {
                name: 'rightRingOffsets',
                view: right.ringOffsets,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : []),
        {name: 'combinedOffsets', view: combinedOffsets, type: 'u32', access: 'read'},
        {name: 'positions', view: combinedPositions, type: 'f32', access: 'read_write'}
      ],
      invocationCount: combinedVertexCount,
      // Rows no ring owns (the closing slot of an empty ring, rows past the last ring) are
      // padding. The noder measures its lattice frame over every row and flags every non-finite
      // row, so padding must hold a stored vertex: never `positions[ringOffsets[last]]`, which no
      // ring owns and which holds whatever the buffer held before (a reused transient, an unwritten
      // fixed-capacity result row).
      declarations: `const LEFT_RING_COUNT: u32 = ${leftRingCount}u;
const RIGHT_RING_COUNT: u32 = ${rightRingCount}u;
const RING_COUNT: u32 = ${combinedRingCount}u;
fn getLeftVertex(row: u32) -> vec2f {
  return vec2f(leftPositions[leftPositionsOffset + row * 2u], leftPositions[leftPositionsOffset + row * 2u + 1u]);
}
${
  right
    ? `fn getRightVertex(row: u32) -> vec2f {
  return vec2f(rightPositions[rightPositionsOffset + row * 2u], rightPositions[rightPositionsOffset + row * 2u + 1u]);
}`
    : ''
}
// The first stored vertex of the left operand, else of the right operand, else the origin (both
// operands are empty and the noder sees no segment).
fn getPaddingVertex() -> vec2f {
  let leftBegin = leftRingOffsets[leftRingOffsetsOffset];
  if (leftRingOffsets[leftRingOffsetsOffset + LEFT_RING_COUNT] > leftBegin) { return getLeftVertex(leftBegin); }
  ${
    right
      ? `let rightBegin = rightRingOffsets[rightRingOffsetsOffset];
  if (rightRingOffsets[rightRingOffsetsOffset + RIGHT_RING_COUNT] > rightBegin) { return getRightVertex(rightBegin); }`
      : ''
  }
  return vec2f(0.0);
}`,
      body: `var combinedRing = 0u;
  var high = RING_COUNT;
  while (combinedRing + 1u < high) {
    let middle = (combinedRing + high) / 2u;
    if (combinedOffsets[combinedOffsetsOffset + middle] <= index) { combinedRing = middle; } else { high = middle; }
  }
  let fromLeft = combinedRing < LEFT_RING_COUNT;
  let ring = select(combinedRing - LEFT_RING_COUNT, combinedRing, fromLeft);
  let local = index - combinedOffsets[combinedOffsetsOffset + combinedRing];
  var point = vec2f(0.0);
  var sourceBegin = 0u;
  var sourceEnd = 0u;
  if (fromLeft) {
    sourceBegin = leftRingOffsets[leftRingOffsetsOffset + ring];
    sourceEnd = leftRingOffsets[leftRingOffsetsOffset + ring + 1u];
  }${
    right
      ? ` else {
    sourceBegin = rightRingOffsets[rightRingOffsetsOffset + ring];
    sourceEnd = rightRingOffsets[rightRingOffsetsOffset + ring + 1u];
  }`
      : ''
  }
  if (sourceEnd > sourceBegin) {
    // Ring vertices, then the ring's first vertex for the closing slot and any row after it.
    let sourceRow = select(sourceBegin + local, sourceBegin, local >= sourceEnd - sourceBegin);
    ${right ? 'if (fromLeft) { point = getLeftVertex(sourceRow); } else { point = getRightVertex(sourceRow); }' : 'point = getLeftVertex(sourceRow);'}
  } else {
    point = getPaddingVertex();
  }
  positions[positionsOffset + index * 2u] = point.x;
  positions[positionsOffset + index * 2u + 1u] = point.y;`
    });

    const rightTopologyBindings = right
      ? [
          {
            name: 'rightFeatureOffsets',
            view: right.featureOffsets,
            type: 'u32' as const,
            access: 'read' as const
          },
          {
            name: 'rightPolygonOffsets',
            view: right.polygonOffsets,
            type: 'u32' as const,
            access: 'read' as const
          },
          {
            name: 'rightRingOffsets',
            view: right.ringOffsets,
            type: 'u32' as const,
            access: 'read' as const
          }
        ]
      : [];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-combine-topology`,
        operation: OPERATION,
        variant: 'combine-topology',
        bindings: [
          {name: 'leftFeatureOffsets', view: left.featureOffsets, type: 'u32', access: 'read'},
          {name: 'leftPolygonOffsets', view: left.polygonOffsets, type: 'u32', access: 'read'},
          {name: 'leftRingOffsets', view: left.ringOffsets, type: 'u32', access: 'read'},
          ...rightTopologyBindings,
          {name: 'offsets', view: combinedOffsets, type: 'u32', access: 'read_write'},
          {name: 'sources', view: combinedSources, type: 'u32', access: 'read_write'}
        ],
        invocationCount: combinedRingCount + 1,
        declarations: `const LEFT_RING_COUNT: u32 = ${leftRingCount}u;
const LEFT_EXPANDED_VERTEX_COUNT: u32 = ${leftExpandedVertexCount}u;
const LEFT_POLYGON_COUNT: u32 = ${left.polygonOffsets.length - 1}u;
const RIGHT_POLYGON_COUNT: u32 = ${right ? right.polygonOffsets.length - 1 : 0}u;`,
        body: `if (index <= LEFT_RING_COUNT) {
    offsets[offsetsOffset + index] = leftRingOffsets[leftRingOffsetsOffset + index] + index;
  }${
    right
      ? ` else {
    let rightRing = index - LEFT_RING_COUNT;
    offsets[offsetsOffset + index] = LEFT_EXPANDED_VERTEX_COUNT + rightRingOffsets[rightRingOffsetsOffset + rightRing] + rightRing;
  }`
      : ''
  }
  if (index >= ${combinedRingCount}u) { return; }
  let fromLeft = index < LEFT_RING_COUNT;
  let ring = select(index - LEFT_RING_COUNT, index, fromLeft);
  var polygon = 0u;
  var high = select(RIGHT_POLYGON_COUNT, LEFT_POLYGON_COUNT, fromLeft);
  while (polygon + 1u < high) {
    let middle = (polygon + high) / 2u;
    var start = 0u;
    if (fromLeft) { start = leftPolygonOffsets[leftPolygonOffsetsOffset + middle]; }
    ${right ? 'else { start = rightPolygonOffsets[rightPolygonOffsetsOffset + middle]; }' : ''}
    if (start <= ring) { polygon = middle; } else { high = middle; }
  }
  var feature = 0u;
  high = select(${right ? `${right.featureOffsets.length - 1}u` : '0u'}, ${left.featureOffsets.length - 1}u, fromLeft);
  while (feature + 1u < high) {
    let middle = (feature + high) / 2u;
    var start = 0u;
    if (fromLeft) { start = leftFeatureOffsets[leftFeatureOffsetsOffset + middle]; }
    ${right ? 'else { start = rightFeatureOffsets[rightFeatureOffsetsOffset + middle]; }' : ''}
    if (start <= polygon) { feature = middle; } else { high = middle; }
  }
  sources[sourcesOffset + index] = feature | select(0x80000000u, 0u, fromLeft);`
      })
    );
    nodes.push(combinePositionsNode);
    // Global polygon index of every combined ring (left polygons, then right polygons offset by the
    // left count). Separate from `combine-topology`, which already binds eight storage buffers.
    const ringPolygons = createTransientView(
      graph,
      `${id}-ring-polygons`,
      'uint32',
      combinedRingCount
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-ring-polygons`,
        operation: OPERATION,
        variant: 'ring-polygons',
        bindings: [
          {name: 'leftPolygonOffsets', view: left.polygonOffsets, type: 'u32', access: 'read'},
          ...(right
            ? [
                {
                  name: 'rightPolygonOffsets',
                  view: right.polygonOffsets,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'ringPolygons', view: ringPolygons, type: 'u32', access: 'read_write'}
        ],
        invocationCount: combinedRingCount,
        declarations: `const LEFT_RING_COUNT: u32 = ${leftRingCount}u;
const LEFT_POLYGON_COUNT: u32 = ${left.polygonOffsets.length - 1}u;
const RIGHT_POLYGON_COUNT: u32 = ${right ? right.polygonOffsets.length - 1 : 0}u;`,
        body: `let fromLeft = index < LEFT_RING_COUNT;
  let ring = select(index - LEFT_RING_COUNT, index, fromLeft);
  var polygon = 0u;
  var high = select(RIGHT_POLYGON_COUNT, LEFT_POLYGON_COUNT, fromLeft);
  while (polygon + 1u < high) {
    let middle = (polygon + high) / 2u;
    var start = 0u;
    if (fromLeft) { start = leftPolygonOffsets[leftPolygonOffsetsOffset + middle]; }
    ${right ? 'else { start = rightPolygonOffsets[rightPolygonOffsetsOffset + middle]; }' : ''}
    if (start <= ring) { polygon = middle; } else { high = middle; }
  }
  ringPolygons[ringPolygonsOffset + index] = polygon + select(LEFT_POLYGON_COUNT, 0u, fromLeft);`
      })
    );

    const noded = this._createNodedPort(graph);
    nodes.push(
      ...new GPUSnapRoundedNoding({
        id: `${id}-noding`,
        spatialContext: this.spatialContext,
        lines: {
          kind: 'lines',
          positions: combinedPositions,
          lineOffsets: combinedOffsets,
          sourceIds: combinedSources
        },
        intersectionCapacity: props.capacity.intersections,
        gridSize: props.vertexTolerance,
        output: {
          ...noded,
          status: {...noded.status, uncertainCount: output.status.uncertainCount}
        },
        spatialSort: props.spatialSort,
        leafCapacity: props.leafCapacity
      }).getCommandNodes(graph)
    );

    // ---- Face labelling on the lattice arrangement (see overlay-face-labelling-wgsl.ts). ----
    const capacity = props.capacity.nodedSegments;
    const transient = <Format extends 'uint32' | 'float32' | 'float32x2' | 'float32x4'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);
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
        declarations,
        body
      });
    const read = (name: string, view: GraphDataView, type: 'u32' | 'f32' | 'i32' = 'u32') =>
      ({name, view, type, access: 'read'}) as const;
    const write = (name: string, view: GraphDataView, type: 'u32' | 'f32' | 'i32' = 'u32') =>
      ({name, view, type, access: 'read_write'}) as const;

    // 1. Canonical lattice pieces: lo -> hi in (y, x) order, direction sign, identity rows.
    const pieceLattice = transient('piece-lattice', 'float32x4', capacity);
    const pieceSigma = transient('piece-sigma', 'uint32', capacity);
    const identity = transient('piece-identity', 'uint32', capacity);
    nodes.push(
      kernel(
        'piece-canonical',
        [
          read('segmentCount', noded.status.count),
          read('latticeEndpoints', noded.latticeEndpoints, 'f32'),
          write('pieceLattice', pieceLattice, 'f32'),
          write('pieceSigma', pieceSigma),
          write('identity', identity)
        ],
        capacity,
        `identity[identityOffset + index] = index;
  var lattice = vec4f(-1.0);
  var sigma = 0u;
  if (index < min(segmentCount[segmentCountOffset], CAPACITY)) {
    let base = latticeEndpointsOffset + index * 4u;
    let first = vec2f(latticeEndpoints[base], latticeEndpoints[base + 1u]);
    let second = vec2f(latticeEndpoints[base + 2u], latticeEndpoints[base + 3u]);
    // Lattice values are finite integers >= 0; non-finite or collapsed pieces carry no boundary.
    if (first.x >= 0.0 && first.y >= 0.0 && second.x >= 0.0 && second.y >= 0.0 && any(first != second)) {
      let firstIsLow = first.y < second.y || (first.y == second.y && first.x < second.x);
      lattice = select(vec4f(second, first), vec4f(first, second), firstIsLow);
      // +1 when the piece runs lo -> hi, -1 (as a wrapping u32) otherwise.
      sigma = select(0xffffffffu, 1u, firstIsLow);
    }
  }
  let output = pieceLatticeOffset + index * 4u;
  pieceLattice[output] = lattice.x;
  pieceLattice[output + 1u] = lattice.y;
  pieceLattice[output + 2u] = lattice.z;
  pieceLattice[output + 3u] = lattice.w;
  pieceSigma[pieceSigmaOffset + index] = sigma;`,
        `const CAPACITY: u32 = ${capacity}u;`
      )
    );

    // 2. Sort keys: (loY, loX, hiY, hiX, polygon); collapsed pieces sort last (loY sentinel).
    const keyLowY = transient('key-low-y', 'uint32', capacity);
    const keyLowX = transient('key-low-x', 'uint32', capacity);
    const keyHighY = transient('key-high-y', 'uint32', capacity);
    const keyHighX = transient('key-high-x', 'uint32', capacity);
    const keyPolygon = transient('key-polygon', 'uint32', capacity);
    nodes.push(
      kernel(
        'piece-keys',
        [
          read('pieceLattice', pieceLattice, 'f32'),
          read('sourceRingIds', noded.sourceRingIds),
          read('ringPolygons', ringPolygons),
          write('keyLowY', keyLowY),
          write('keyLowX', keyLowX),
          write('keyHighY', keyHighY),
          write('keyHighX', keyHighX),
          write('keyPolygon', keyPolygon)
        ],
        capacity,
        `let base = pieceLatticeOffset + index * 4u;
  let lattice = vec4f(pieceLattice[base], pieceLattice[base + 1u], pieceLattice[base + 2u], pieceLattice[base + 3u]);
  var lowY = KEY_SENTINEL;
  var lowX = 0u;
  var highY = 0u;
  var highX = 0u;
  var polygon = 0u;
  if (lattice.x >= 0.0) {
    lowY = u32(lattice.y);
    lowX = u32(lattice.x);
    highY = u32(lattice.w);
    highX = u32(lattice.z);
    let ring = sourceRingIds[sourceRingIdsOffset + index];
    if (ring < RING_COUNT) { polygon = ringPolygons[ringPolygonsOffset + ring]; }
  }
  keyLowY[keyLowYOffset + index] = lowY;
  keyLowX[keyLowXOffset + index] = lowX;
  keyHighY[keyHighYOffset + index] = highY;
  keyHighX[keyHighXOffset + index] = highX;
  keyPolygon[keyPolygonOffset + index] = polygon;`,
        `const RING_COUNT: u32 = ${combinedRingCount}u;
// Above every lattice value (at most 2^22 + 1), within the 23 sorted bits.
const KEY_SENTINEL: u32 = 0x7fffffu;`
      )
    );
    const polygonCount =
      left.polygonOffsets.length - 1 + (right ? right.polygonOffsets.length - 1 : 0);
    const sorted = createStableLexicographicIndexSortNodes<Parameters>(graph, {
      id: `${id}-piece-sort`,
      operation: OPERATION,
      indices: identity,
      keys: [
        {view: keyLowY, keyBits: LATTICE_KEY_BITS},
        {view: keyLowX, keyBits: LATTICE_KEY_BITS},
        {view: keyHighY, keyBits: LATTICE_KEY_BITS},
        {view: keyHighX, keyBits: LATTICE_KEY_BITS},
        {
          view: keyPolygon,
          keyBits: clamp(Math.ceil(Math.log2(polygonCount + 1)), 1, 31)
        }
      ]
    });
    nodes.push(...sorted.nodes);

    // 3. Pieces in group order.
    const sortedLattice = transient('sorted-lattice', 'float32x4', capacity);
    const sortedSigma = transient('sorted-sigma', 'uint32', capacity);
    const sortedPolygon = transient('sorted-polygon', 'uint32', capacity);
    nodes.push(
      kernel(
        'sort-gather',
        [
          read('sortedIndices', sorted.sortedIndices),
          read('pieceLattice', pieceLattice, 'f32'),
          read('pieceSigma', pieceSigma),
          read('keyPolygon', keyPolygon),
          write('sortedLattice', sortedLattice, 'f32'),
          write('sortedSigma', sortedSigma),
          write('sortedPolygon', sortedPolygon)
        ],
        capacity,
        `let piece = min(sortedIndices[sortedIndicesOffset + index], CAPACITY - 1u);
  for (var component = 0u; component < 4u; component++) {
    sortedLattice[sortedLatticeOffset + index * 4u + component] = pieceLattice[pieceLatticeOffset + piece * 4u + component];
  }
  sortedSigma[sortedSigmaOffset + index] = pieceSigma[pieceSigmaOffset + piece];
  sortedPolygon[sortedPolygonOffset + index] = keyPolygon[keyPolygonOffset + piece];`,
        `const CAPACITY: u32 = ${capacity}u;`
      )
    );

    // 4. Group boundaries: level 1 = (lo, hi, polygon) runs, level 2 = (lo, hi) runs.
    const flagsOne = transient('flags-one', 'uint32', capacity);
    const flagsTwo = transient('flags-two', 'uint32', capacity);
    nodes.push(
      kernel(
        'group-flags',
        [
          read('sortedLattice', sortedLattice, 'f32'),
          read('sortedPolygon', sortedPolygon),
          write('flagsOne', flagsOne),
          write('flagsTwo', flagsTwo)
        ],
        capacity,
        `let current = readLattice(index);
  var one = 0u;
  var two = 0u;
  if (current.x >= 0.0) {
    if (index == 0u) {
      one = 1u;
      two = 1u;
    } else {
      let differs = any(readLattice(index - 1u) != current);
      two = select(0u, 1u, differs);
      one = select(0u, 1u, differs || sortedPolygon[sortedPolygonOffset + index - 1u] != sortedPolygon[sortedPolygonOffset + index]);
    }
  }
  flagsOne[flagsOneOffset + index] = one;
  flagsTwo[flagsTwoOffset + index] = two;`,
        READ_LATTICE_WGSL
      )
    );
    const ordinalsOne = transient('ordinals-one', 'uint32', capacity);
    const ordinalsTwo = transient('ordinals-two', 'uint32', capacity);
    const sigmaPrefix = transient('sigma-prefix', 'uint32', capacity);
    for (const [name, input, scanOutput] of [
      ['ordinals-one', flagsOne, ordinalsOne],
      ['ordinals-two', flagsTwo, ordinalsTwo],
      ['sigma-prefix', sortedSigma, sigmaPrefix]
    ] as const) {
      nodes.push(
        ...new GPUScan({
          id: `${id}-${name}`,
          input,
          output: scanOutput,
          mode: 'inclusive'
        }).getCommandNodes(graph)
      );
    }
    const counts = transient('group-counts', 'uint32', 2);
    nodes.push(
      kernel(
        'group-counts',
        [
          read('ordinalsOne', ordinalsOne),
          read('ordinalsTwo', ordinalsTwo),
          write('counts', counts)
        ],
        1,
        `counts[countsOffset] = ordinalsOne[ordinalsOneOffset + ${capacity - 1}u];
  counts[countsOffset + 1u] = ordinalsTwo[ordinalsTwoOffset + ${capacity - 1}u];`
      )
    );
    const headsOne = transient('heads-one', 'uint32', capacity);
    const headsTwo = transient('heads-two', 'uint32', capacity);
    const firstOne = transient('first-one', 'uint32', capacity);
    nodes.push(
      kernel(
        'group-heads',
        [
          read('flagsOne', flagsOne),
          read('flagsTwo', flagsTwo),
          read('ordinalsOne', ordinalsOne),
          read('ordinalsTwo', ordinalsTwo),
          write('headsOne', headsOne),
          write('headsTwo', headsTwo),
          write('firstOne', firstOne)
        ],
        capacity,
        `if (flagsOne[flagsOneOffset + index] != 0u) {
    headsOne[headsOneOffset + ordinalsOne[ordinalsOneOffset + index] - 1u] = index;
  }
  if (flagsTwo[flagsTwoOffset + index] != 0u) {
    let group = ordinalsTwo[ordinalsTwoOffset + index] - 1u;
    headsTwo[headsTwoOffset + group] = index;
    firstOne[firstOneOffset + group] = ordinalsOne[ordinalsOneOffset + index] - 1u;
  }`
      )
    );

    // 5. Group tables: lattice edge, BVH bounds, and for level 1 polygon and winding step.
    const buildGroupTable = (level: 1 | 2, heads: GraphDataView<'uint32'>) => {
      const groupLattice = transient(`group-lattice-${level}`, 'float32x4', capacity);
      const minima = transient(`group-minima-${level}`, 'float32x2', capacity);
      const maxima = transient(`group-maxima-${level}`, 'float32x2', capacity);
      nodes.push(
        kernel(
          `group-lattice-${level}`,
          [
            read('counts', counts),
            read('heads', heads),
            read('sortedLattice', sortedLattice, 'f32'),
            write('groupLattice', groupLattice, 'f32'),
            write('minima', minima, 'f32'),
            write('maxima', maxima, 'f32')
          ],
          capacity,
          `var edge = vec4f(0.0);
  var minimum = vec2f(FAR, FAR);
  var maximum = vec2f(-FAR, -FAR);
  if (index < counts[countsOffset + ${level - 1}u]) {
    edge = readLattice(heads[headsOffset + index]);
    minimum = vec2f(min(edge.x, edge.z), edge.y);
    maximum = vec2f(max(edge.x, edge.z), edge.w);
  }
  for (var component = 0u; component < 4u; component++) {
    groupLattice[groupLatticeOffset + index * 4u + component] = edge[component];
  }
  minima[minimaOffset + index * 2u] = minimum.x;
  minima[minimaOffset + index * 2u + 1u] = minimum.y;
  maxima[maximaOffset + index * 2u] = maximum.x;
  maxima[maximaOffset + index * 2u + 1u] = maximum.y;`,
          `const FAR: f32 = 3.0e38;
${READ_LATTICE_WGSL}`
        )
      );
      // An inverted box (minimum > maximum) is skipped by the BVH build and the ray walk.
      const {bvh, nodes: bvhNodes} = getSegmentBVHNodes(graph, {
        id: `${id}-group-bvh-${level}`,
        operation: OPERATION,
        minima,
        maxima,
        leafCapacity: Math.max(2, getNextPowerOfTwo(capacity)),
        spatialSort: false
      });
      nodes.push(...bvhNodes);
      return {groupLattice, bvh};
    };
    const levelOne = buildGroupTable(1, headsOne);
    const groupPolygons = transient('group-polygons', 'uint32', capacity);
    const groupDelta = transient('group-delta', 'uint32', capacity);
    nodes.push(
      kernel(
        'group-winding-steps',
        [
          read('counts', counts),
          read('heads', headsOne),
          read('sigmaPrefix', sigmaPrefix),
          read('sortedPolygon', sortedPolygon),
          write('groupPolygons', groupPolygons),
          write('groupDelta', groupDelta)
        ],
        capacity,
        `var polygon = 0u;
  var delta = 0u;
  let groupCount = counts[countsOffset];
  if (index < groupCount) {
    let start = heads[headsOffset + index];
    var end = ${capacity}u;
    if (index + 1u < groupCount) { end = heads[headsOffset + index + 1u]; }
    // Sum of the direction signs over the run, from the prefix sums (wrapping, so exact as i32).
    var before = 0u;
    if (start > 0u) { before = sigmaPrefix[sigmaPrefixOffset + start - 1u]; }
    delta = sigmaPrefix[sigmaPrefixOffset + end - 1u] - before;
    polygon = sortedPolygon[sortedPolygonOffset + start];
  }
  groupPolygons[groupPolygonsOffset + index] = polygon;
  groupDelta[groupDeltaOffset + index] = delta;`
      )
    );

    // 6. Level 1: winding of each polygon beside its own edges.
    const parentsOne = transient('parents-one', 'uint32', capacity);
    const valuesOne = transient('values-one', 'uint32', capacity);
    nodes.push(
      this._createRayShootNode(graph, 1, levelOne, counts, groupPolygons, parentsOne),
      kernel(
        'jump-init-1',
        [read('parents', parentsOne), read('delta', groupDelta), write('values', valuesOne)],
        capacity,
        JUMP_INIT_BODY_WGSL.replaceAll('COMPONENTS', '1u'),
        'const NO_GROUP: u32 = 0xffffffffu;'
      )
    );
    const jumpOne = createPointerJumpNodes<Parameters>(graph, {
      id: `${id}-jump-1`,
      operation: OPERATION,
      parents: parentsOne,
      values: valuesOne,
      componentCount: 1,
      rounds: getPointerJumpRoundCount(capacity)
    });
    nodes.push(...jumpOne.nodes);

    // 7. Coverage change across each level-2 group, per operand.
    const coverageA = transient('coverage-a', 'uint32', capacity);
    const coverageB = transient('coverage-b', 'uint32', capacity);
    const leftPolygonCount = left.polygonOffsets.length - 1;
    nodes.push(
      kernel(
        'group-coverage',
        [
          read('counts', counts),
          read('groupPolygons', groupPolygons),
          read('groupDelta', groupDelta, 'i32'),
          read('windings', jumpOne.values, 'i32'),
          write('coverageA', coverageA, 'i32'),
          write('coverageB', coverageB, 'i32')
        ],
        capacity,
        `var changeA = 0;
  var changeB = 0;
  if (index < counts[countsOffset]) {
    let right = windings[windingsOffset + index];
    let left = right + groupDelta[groupDeltaOffset + index];
    let change = select(0, 1, insideWinding(left)) - select(0, 1, insideWinding(right));
    if (groupPolygons[groupPolygonsOffset + index] < LEFT_POLYGON_COUNT) { changeA = change; } else { changeB = change; }
  }
  coverageA[coverageAOffset + index] = changeA;
  coverageB[coverageBOffset + index] = changeB;`,
        `const LEFT_POLYGON_COUNT: u32 = ${leftPolygonCount}u;
${getInsideWindingWGSL(this.fillRule)}`
      )
    );
    const coveragePrefixA = transient('coverage-prefix-a', 'uint32', capacity);
    const coveragePrefixB = transient('coverage-prefix-b', 'uint32', capacity);
    for (const [name, input, scanOutput] of [
      ['coverage-prefix-a', coverageA, coveragePrefixA],
      ['coverage-prefix-b', coverageB, coveragePrefixB]
    ] as const) {
      nodes.push(
        ...new GPUScan({
          id: `${id}-${name}`,
          input,
          output: scanOutput,
          mode: 'inclusive'
        }).getCommandNodes(graph)
      );
    }
    const groupCoverageDelta = transient('group-coverage-delta', 'uint32', 2 * capacity);
    nodes.push(
      kernel(
        'group-coverage-delta',
        [
          read('counts', counts),
          read('firstOne', firstOne),
          read('prefixA', coveragePrefixA),
          read('prefixB', coveragePrefixB),
          write('coverageDelta', groupCoverageDelta)
        ],
        capacity,
        `var deltaA = 0u;
  var deltaB = 0u;
  if (index < counts[countsOffset + 1u]) {
    let first = firstOne[firstOneOffset + index];
    var end = counts[countsOffset];
    if (index + 1u < counts[countsOffset + 1u]) { end = firstOne[firstOneOffset + index + 1u]; }
    var beforeA = 0u;
    var beforeB = 0u;
    if (first > 0u) {
      beforeA = prefixA[prefixAOffset + first - 1u];
      beforeB = prefixB[prefixBOffset + first - 1u];
    }
    deltaA = prefixA[prefixAOffset + end - 1u] - beforeA;
    deltaB = prefixB[prefixBOffset + end - 1u] - beforeB;
  }
  coverageDelta[coverageDeltaOffset + index * 2u] = deltaA;
  coverageDelta[coverageDeltaOffset + index * 2u + 1u] = deltaB;`
      )
    );

    // 8. Level 2: coverage count of each operand beside every edge group.
    const levelTwo = buildGroupTable(2, headsTwo);
    const parentsTwo = transient('parents-two', 'uint32', capacity);
    const valuesTwo = transient('values-two', 'uint32', 2 * capacity);
    nodes.push(
      this._createRayShootNode(graph, 2, levelTwo, counts, undefined, parentsTwo),
      kernel(
        'jump-init-2',
        [
          read('parents', parentsTwo),
          read('delta', groupCoverageDelta),
          write('values', valuesTwo)
        ],
        capacity,
        JUMP_INIT_BODY_WGSL.replaceAll('COMPONENTS', '2u'),
        'const NO_GROUP: u32 = 0xffffffffu;'
      )
    );
    const jumpTwo = createPointerJumpNodes<Parameters>(graph, {
      id: `${id}-jump-2`,
      operation: OPERATION,
      parents: parentsTwo,
      values: valuesTwo,
      componentCount: 2,
      rounds: getPointerJumpRoundCount(capacity)
    });
    nodes.push(...jumpTwo.nodes);

    // 9. Classification: keep a group when the Boolean result differs across it.
    const selected = transient('selected', 'uint32', capacity);
    const orientations = transient('orientations', 'uint32', capacity);
    const selectionScan = transient('selection-scan', 'uint32', capacity);
    nodes.push(
      kernel(
        'classify-boundary',
        [
          read('counts', counts),
          read('rightCoverage', jumpTwo.values, 'i32'),
          read('coverageDelta', groupCoverageDelta, 'i32'),
          write('selected', selected),
          write('orientations', orientations)
        ],
        capacity,
        `var keep = false;
  var reverse = false;
  if (index < counts[countsOffset + 1u]) {
    let rightA = rightCoverage[rightCoverageOffset + index * 2u];
    let rightB = rightCoverage[rightCoverageOffset + index * 2u + 1u];
    let leftA = rightA + coverageDelta[coverageDeltaOffset + index * 2u];
    let leftB = rightB + coverageDelta[coverageDeltaOffset + index * 2u + 1u];
    let leftResult = ${this._getOperationExpression('(leftA > 0)', '(leftB > 0)')};
    let rightResult = ${this._getOperationExpression('(rightA > 0)', '(rightB > 0)')};
    keep = leftResult != rightResult;
    // The edge runs lo -> hi with its left side on the left: flip it when the result is on the right.
    reverse = keep && rightResult;
  }
  selected[selectedOffset + index] = select(0u, 1u, keep);
  orientations[orientationsOffset + index] = select(0u, 1u, reverse);`
      ),
      ...new GPUScan({
        id: `${id}-selection-scan`,
        input: selected,
        output: selectionScan,
        mode: 'inclusive'
      }).getCommandNodes(graph)
    );

    const boundary = output.boundary;
    const boundaryGroups = transient('boundary-groups', 'uint32', boundary.endpoints.length);
    const latticeBoundary = transient('lattice-boundary', 'float32x4', boundary.endpoints.length);
    nodes.push(
      kernel(
        'emit-boundary',
        [
          read('selected', selected),
          read('scan', selectionScan),
          read('orientations', orientations),
          read('heads', headsTwo),
          read('sortedLattice', sortedLattice, 'f32'),
          read('lattice', noded.lattice, 'f32'),
          write('endpoints', boundary.endpoints, 'f32'),
          write('latticeEndpoints', latticeBoundary, 'f32')
        ],
        capacity,
        `if (selected[selectedOffset + index] == 0u) { return; }
  let outputRow = scan[scanOffset + index] - 1u;
  if (outputRow >= BOUNDARY_CAPACITY) { return; }
  let edge = readLattice(heads[headsOffset + index]);
  let reverse = orientations[orientationsOffset + index] != 0u;
  let start = select(edge.xy, edge.zw, reverse);
  let end = select(edge.zw, edge.xy, reverse);
  let origin = vec2f(lattice[latticeOffset], lattice[latticeOffset + 1u]);
  let cell = lattice[latticeOffset + 2u];
  let outputBase = endpointsOffset + outputRow * 4u;
  let latticeBase = latticeEndpointsOffset + outputRow * 4u;
  latticeEndpoints[latticeBase] = start.x;
  latticeEndpoints[latticeBase + 1u] = start.y;
  latticeEndpoints[latticeBase + 2u] = end.x;
  latticeEndpoints[latticeBase + 3u] = end.y;
  // World position of a lattice point: origin + k * cell (cell is a power of two).
  endpoints[outputBase] = origin.x + start.x * cell;
  endpoints[outputBase + 1u] = origin.y + start.y * cell;
  endpoints[outputBase + 2u] = origin.x + end.x * cell;
  endpoints[outputBase + 3u] = origin.y + end.y * cell;`,
        `const BOUNDARY_CAPACITY: u32 = ${boundary.endpoints.length}u;
${READ_LATTICE_WGSL}`
      ),
      kernel(
        'boundary-groups',
        [write('groups', boundaryGroups)],
        boundary.endpoints.length,
        'groups[groupsOffset + index] = 0u;'
      ),
      kernel(
        'emit-boundary-provenance',
        [
          read('selected', selected),
          read('scan', selectionScan),
          read('heads', headsTwo),
          read('sortedIndices', sorted.sortedIndices),
          read('packedSources', noded.sourceFeatureIds),
          write('operandIds', boundary.operandIds),
          write('sourceFeatureIds', boundary.sourceFeatureIds)
        ],
        capacity,
        `if (selected[selectedOffset + index] == 0u) { return; }
  let outputRow = scan[scanOffset + index] - 1u;
  if (outputRow >= BOUNDARY_CAPACITY) { return; }
  // The first piece of the group in sort order owns its provenance.
  let piece = sortedIndices[sortedIndicesOffset + heads[headsOffset + index]];
  let packed = packedSources[packedSourcesOffset + piece];
  operandIds[operandIdsOffset + outputRow] = packed >> 31u;
  sourceFeatureIds[sourceFeatureIdsOffset + outputRow] = packed & 0x7fffffffu;`,
        `const BOUNDARY_CAPACITY: u32 = ${boundary.endpoints.length}u;`
      ),
      kernel(
        'boundary-status',
        [
          read('scan', selectionScan),
          read('nodingOverflow', noded.status.overflow),
          read('nodingCandidateOverflow', noded.status.candidateOverflow),
          write('count', boundary.status.count),
          write('requiredCount', boundary.status.requiredCount),
          write('overflow', boundary.status.overflow),
          write('candidateOverflow', boundary.status.candidateOverflow)
        ],
        1,
        `let required = scan[scanOffset + NODED_CAPACITY - 1u];
  let candidateIncomplete = nodingCandidateOverflow[nodingCandidateOverflowOffset];
  requiredCount[requiredCountOffset] = required;
  count[countOffset] = min(required, BOUNDARY_CAPACITY);
  overflow[overflowOffset] = select(0u, 1u, required > BOUNDARY_CAPACITY || nodingOverflow[nodingOverflowOffset] != 0u);
  candidateOverflow[candidateOverflowOffset] = candidateIncomplete;`,
        `const BOUNDARY_CAPACITY: u32 = ${boundary.endpoints.length}u;
const NODED_CAPACITY: u32 = ${capacity}u;`
      )
    );

    // 10. Ring assembly on the lattice (exact vertex matching), then back to world coordinates.
    const ringCapacity = output.geometry.ringOffsets.length - 1;
    const ringPositions = transient(
      'ring-positions',
      'float32x2',
      output.geometry.positions.length
    );
    const ringOffsets = transient('ring-offsets', 'uint32', ringCapacity + 1);
    const ringCount = transient('ring-count', 'uint32', 1);
    const ringRequiredCount = transient('ring-required-count', 'uint32', 1);
    const ringOverflow = transient('ring-overflow', 'uint32', 1);
    const openSegmentCount = transient('assembly-open-count', 'uint32', 1);
    const segmentFlags = transient('assembly-flags', 'uint32', boundary.endpoints.length);
    // Optional ring-level status (rule 8 names), copied from the assembly's ring scalars.
    const ringStatusOutputs = [
      ['ringCount', 'ringCountInput', output.status.ringCount, ringCount],
      [
        'requiredRingCount',
        'ringRequiredCountInput',
        output.status.requiredRingCount,
        ringRequiredCount
      ]
    ] as const;
    const ringStatusBindings = ringStatusOutputs.flatMap(([name, inputName, view, source]) =>
      view ? [read(inputName, source), write(name, view)] : []
    );
    const ringStatusBody = ringStatusOutputs
      .flatMap(([name, inputName, view]) =>
        view ? [`${name}[${name}Offset] = ${inputName}[${inputName}Offset];`] : []
      )
      .join('\n  ');
    const ringStatusVariant = ringStatusOutputs.some(([, , view]) => view)
      ? `-${ringStatusOutputs.map(([, , view]) => (view ? 1 : 0)).join('')}`
      : '';
    const featureSourceMapping = output.featureSources
      ? {
          segmentRings: transient(
            'source-segment-rings',
            'uint32',
            output.boundary.endpoints.length
          ),
          ringShells: transient('source-ring-shells', 'uint32', ringCapacity),
          ringAreas: transient('source-ring-areas', 'float32', ringCapacity)
        }
      : undefined;
    nodes.push(
      ...new GPUSegmentRingAssembly({
        id: `${id}-assemble`,
        spatialContext: this.spatialContext,
        endpoints: latticeBoundary,
        count: boundary.status.count,
        groups: boundaryGroups,
        // Distinct lattice points are at least 1 apart: matching at 1/4 is exact.
        vertexTolerance: 0.25,
        interiorSide: 'left',
        normalizeWinding: true,
        // Selection emits exactly one directed edge per coincident group; no opposite pair
        // survives to assembly, so another cancellation search would only repeat that work.
        cancelOpposingSegments: false,
        splitTouchingRings: true,
        output: {
          positions: ringPositions,
          ringOffsets,
          segmentFlags,
          status: {
            count: ringCount,
            requiredCount: ringRequiredCount,
            overflow: ringOverflow,
            openSegmentCount
          },
          polygons: output.geometry,
          ...featureSourceMapping
        }
      }).getCommandNodes(graph)
    );
    if (output.featureSources && featureSourceMapping)
      nodes.push(
        ...createOverlayFeatureSourceNodes(graph, {
          id,
          ...featureSourceMapping,
          ringCount,
          boundaryCount: boundary.status.count,
          sourceRows: boundary.sourceFeatureIds,
          operandIds: boundary.operandIds,
          output: output.featureSources
        })
      );
    nodes.push(
      kernel(
        'lattice-to-world',
        [
          read('lattice', noded.lattice, 'f32'),
          read('ringOffsets', output.geometry.ringOffsets),
          write('positions', output.geometry.positions, 'f32')
        ],
        Math.max(1, output.geometry.positions.length),
        `if (index >= ringOffsets[ringOffsetsOffset + RING_CAPACITY]) { return; }
  let cell = lattice[latticeOffset + 2u];
  positions[positionsOffset + index * 2u] = lattice[latticeOffset] + positions[positionsOffset + index * 2u] * cell;
  positions[positionsOffset + index * 2u + 1u] = lattice[latticeOffset + 1u] + positions[positionsOffset + index * 2u + 1u] * cell;`,
        `const RING_CAPACITY: u32 = ${ringCapacity}u;`
      ),
      kernel(
        'publish',
        [
          read('featureOffsets', output.geometry.featureOffsets),
          read('boundaryOverflow', boundary.status.overflow),
          read('candidateOverflowInput', boundary.status.candidateOverflow),
          read('ringOverflow', ringOverflow),
          write('count', output.status.count),
          write('overflow', output.status.overflow),
          write('candidateOverflow', output.status.candidateOverflow)
        ],
        1,
        `let featureCount = featureOffsets[featureOffsetsOffset + RING_CAPACITY];
  let incomplete = boundaryOverflow[boundaryOverflowOffset] | ringOverflow[ringOverflowOffset];
  count[countOffset] = featureCount;
  overflow[overflowOffset] = incomplete;
  candidateOverflow[candidateOverflowOffset] = candidateOverflowInput[candidateOverflowInputOffset];`,
        `const RING_CAPACITY: u32 = ${ringCapacity}u;`
      )
    );
    nodes.push(
      createOverlayAssemblyCompletenessNode(graph, {
        id,
        segmentFlags,
        segmentCount: boundary.status.count,
        openSegmentCount,
        candidateOverflow: output.status.candidateOverflow
      })
    );
    // A separate node, so the publish kernel stays within the storage-binding limit.
    if (ringStatusBindings.length > 0) {
      nodes.push(kernel(`ring-status${ringStatusVariant}`, ringStatusBindings, 1, ringStatusBody));
    }
    nodes.push(
      ...createOverlaySourceIdentityNodes(graph, {
        id,
        operation: OPERATION,
        rows: boundary.sourceFeatureIds,
        operands: boundary.operandIds,
        count: boundary.status.count,
        leftIds: left.sourceIds,
        rightIds: right?.sourceIds
      })
    );
    return nodes;
  }

  private _createRayShootNode<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    level: 1 | 2,
    table: {groupLattice: GraphDataView<'float32x4'>; bvh: GPUBVH},
    counts: GraphDataView<'uint32'>,
    groupPolygons: GraphDataView<'uint32'> | undefined,
    parents: GraphDataView<'uint32'>
  ): GPUCommandNode<Parameters> {
    const {bvh} = table;
    return createWGSLKernelNode<Parameters>(graph, {
      id: `${this.id}-ray-shoot-${level}`,
      operation: OPERATION,
      variant: `ray-shoot-${level}`,
      bindings: [
        {name: 'counts', view: counts, type: 'u32', access: 'read'},
        {name: 'groupLattice', view: table.groupLattice, type: 'f32', access: 'read'},
        ...(groupPolygons
          ? [
              {
                name: 'groupPolygons',
                view: groupPolygons,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : []),
        {name: 'nodeMinima', view: bvh.nodeMinima, type: 'f32', access: 'read'},
        {name: 'nodeMaxima', view: bvh.nodeMaxima, type: 'f32', access: 'read'},
        {name: 'leafIds', view: bvh.leafIds, type: 'u32', access: 'read'},
        {name: 'parents', view: parents, type: 'u32', access: 'read_write'}
      ],
      invocationCount: parents.length,
      declarations: getFaceRayShootDeclarationsWGSL({
        internalNodeCount: bvh.internalNodeCount,
        countRow: level - 1,
        samePolygon: level === 1
      }),
      body: getFaceRayShootBodyWGSL({samePolygon: level === 1})
    });
  }

  private _createNodedPort<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): GPUSnapRoundedSegmentPort {
    const {id, props} = this;
    const rows = props.capacity.nodedSegments;
    const scalar = (name: string) => createTransientView(graph, `${id}-${name}`, 'uint32', 1);
    return {
      endpoints: createTransientView(graph, `${id}-noded-endpoints`, 'float32x4', rows),
      latticeEndpoints: createTransientView(
        graph,
        `${id}-noded-lattice-endpoints`,
        'float32x4',
        rows
      ),
      lattice: createTransientView(graph, `${id}-lattice`, 'float32', 4),
      sourceFeatureIds: createTransientView(graph, `${id}-noded-features`, 'uint32', rows),
      sourceRingIds: createTransientView(graph, `${id}-noded-rings`, 'uint32', rows),
      sourceEdgeIds: createTransientView(graph, `${id}-noded-edges`, 'uint32', rows),
      sourceStartParameters: createTransientView(graph, `${id}-noded-start`, 'float32', rows),
      sourceEndParameters: createTransientView(graph, `${id}-noded-end`, 'float32', rows),
      status: {
        count: scalar('noded-count'),
        requiredCount:
          props.output.status.nodedSegmentRequiredCount ?? scalar('noded-required-count'),
        overflow: scalar('noded-overflow'),
        candidateOverflow: scalar('noded-candidate-overflow')
      }
    };
  }

  private _getOperationExpression(left: string, right: string): string {
    switch (this.props.operation) {
      case 'intersection':
        return `${left} && ${right}`;
      case 'union':
        return `${left} || ${right}`;
      case 'difference':
        return `${left} && !${right}`;
      case 'symmetric-difference':
        return `${left} != ${right}`;
      case 'dissolve':
        return left;
    }
  }
}

/** Validates resident views and device limits without graph mutation. Use planGPUPolygonOverlay
 * for dimension-only planning before allocating recipe inputs/outputs. */
export function preflightGPUPolygonOverlay<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUPolygonOverlayProps
): GPUPolygonOverlayPlan {
  return new GPUPolygonOverlay(props).preflight(graph);
}

/** Validated overlay allocation counts; no resources have been registered by preflight. */
export type GPUPolygonOverlayPlan = Readonly<{
  expandedVertexCount: number;
  inputRingCount: number;
  hotPixelCapacity: number;
  nodedSegmentCapacity: number;
  boundaryCapacity: number;
  outputRingCapacity: number;
  outputVertexCapacity: number;
}>;
