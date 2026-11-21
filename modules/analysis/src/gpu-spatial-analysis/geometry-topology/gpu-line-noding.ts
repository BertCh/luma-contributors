// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  snapshotLineFeatureGeometry,
  createLineFeatureProvenanceNode,
  type GPULineFeatureGeometry
} from '../../analysis-core/contracts/line-feature-provenance';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

import {
  createTransientView,
  GPUScan,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import type {Device} from '@luma.gl/core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateSinglePackedView,
  validatePowerOfTwo
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPULineSplit,
  getGPULineSplitMaximumPieceBound,
  GPU_LINE_SPLIT_SELF_CONTACT
} from '../line-split/index';
import {getGPUSegmentIntersectionMaximumVertexCount} from '../segment-intersection/index';
import {GPU_TOPOLOGY_NONE, type GPUNodedSegmentPort} from './topology-types';
import {validateSupportedTopologyStatus} from './topology-status';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';

const OPERATION = 'GPULineNoding';

/** Properties for {@link GPULineNoding}. */
export type GPULineNodingProps = {
  /** Prefix for generated node and transient IDs. */
  id?: string;
  /**
   * Planar only (default: native planar coordinates). A longitude-latitude context throws at
   * construction: project positions to a planar frame first, for example with
   * `createGPULocalFrame` (`contracts/local-frame`) and `GPULocalFrameProjection`
   * (`geometry-measures`), and pass the frame's context.
   */
  spatialContext?: GPUSpatialContext;
  /** Finite planar line parts; optional featureOffsets groups parts. sourceIds, when present, has one ID per logical feature. */
  lines: GPULineFeatureGeometry;
  /**
   * Capacity of the exact segment-intersection event list. The line count plus four times this
   * capacity is at most {@link getGPULineSplitMaximumPieceBound} (4194304 on a default
   * device), and `lines.positions` has at most {@link getGPUSegmentIntersectionMaximumVertexCount}
   * rows.
   */
  intersectionCapacity: number;
  /**
   * Caller-owned atomic noded segments and provenance. `output.status.uncertainCount` is the
   * optional uncertain-predicate counter.
   */
  output: GPUNodedSegmentPort;
  spatialSort?: boolean;
  leafCapacity?: number;
};

/**
 * Throws unless a noding of `vertexCount` vertices on `lineCount` lines with `intersectionCapacity`
 * pairs fits the internal line split: the segment table holds at most
 * {@link getGPUSegmentIntersectionMaximumVertexCount} vertices and the piece table at most
 * {@link getGPULineSplitMaximumPieceBound} bounded pieces (`lines + 4 * intersectionCapacity`).
 * Used by every contributor that nodes lines, under its own ID.
 *
 * @internal
 */
export function validateLineNodingBounds(
  id: string,
  device: Device,
  vertexCount: number,
  lineCount: number,
  intersectionCapacity: number
): void {
  validateMaximumCount(
    id,
    'lines.positions rows',
    vertexCount,
    getGPUSegmentIntersectionMaximumVertexCount(device)
  );
  validateMaximumCount(
    id,
    'lines + 4 x intersectionCapacity',
    lineCount + 4 * intersectionCapacity,
    getGPULineSplitMaximumPieceBound(device)
  );
}

/**
 * Nodes arbitrary finite planar linework and emits one atomic row for every segment between
 * consecutive distinct noded vertices (a repeated vertex yields no zero-length row). Crossings, touches and both ends of collinear overlaps reuse the
 * exact-or-uncertain predicates of `GPUSegmentIntersection` through `GPULineSplit`.
 *
 * `overflow` reports only final segment capacity. `candidateOverflow` reports incomplete split
 * input (intersection capacity, uncertifiable pairs or non-finite vertices). `requiredCount` is
 * exact only when `candidateOverflow` is zero. Inspect `uncertainCount` to distinguish uncertainty
 * from a capacity that can be increased.
 */
export class GPULineNoding implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPULineNodingProps;
  /** Resolved spatial context (planar only). */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPULineNodingProps) {
    this.id = props.id ?? 'line-noding';
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    const snapshot = snapshotConstructionFields(props, [
      'id',
      'spatialContext',
      'lines',
      'intersectionCapacity',
      'output',
      'spatialSort',
      'leafCapacity'
    ]);
    const outputSnapshot = snapshotConstructionFields(snapshot.output, [
      'endpoints',
      'sourceFeatureIds',
      'sourceRingIds',
      'sourceEdgeIds',
      'sourceStartParameters',
      'sourceEndParameters',
      'status'
    ]);
    validateSupportedTopologyStatus(this.id, outputSnapshot.status, 'output.status');
    props = Object.freeze({
      ...snapshot,
      lines: snapshotLineFeatureGeometry(this.id, snapshot.lines),
      output: Object.freeze({
        ...outputSnapshot,
        status: snapshotConstructionFields(outputSnapshot.status, [
          'count',
          'requiredCount',
          'overflow',
          'candidateOverflow',
          'uncertainCount'
        ])
      })
    });
    this.props = props;
    const {id} = this;
    const {lines, output} = props;
    if (lines.kind !== 'lines') {
      throw new Error(`${id} lines must be linestring geometry`);
    }
    validateSinglePackedView(lines.positions, ['float32x2'], `${id} lines.positions`);
    validatePackedUint32View(lines.lineOffsets, `${id} lines.lineOffsets`);
    if (lines.lineOffsets.length < 2 || lines.positions.length < 1) {
      throw new Error(`${id} lines require at least one line and one vertex`);
    }
    validateCapacity(id, 'intersectionCapacity', props.intersectionCapacity);
    // Delegates are built later, in getCommandNodes: check forwarded options now, under this id.
    if (props.leafCapacity !== undefined) {
      validatePowerOfTwo(id, 'leafCapacity', props.leafCapacity);
    }
    const capacity = output.endpoints.length;
    validateSinglePackedView(output.endpoints, ['float32x4'], `${id} output.endpoints`);
    for (const [name, view] of [
      ['sourceFeatureIds', output.sourceFeatureIds],
      ['sourceRingIds', output.sourceRingIds],
      ['sourceEdgeIds', output.sourceEdgeIds]
    ] as const) {
      validatePackedUint32View(view, `${id} output.${name}`);
      if (view.length !== capacity) {
        throw new Error(`${id} output.${name} length must equal output.endpoints length`);
      }
    }
    for (const [name, view] of [
      ['sourceStartParameters', output.sourceStartParameters],
      ['sourceEndParameters', output.sourceEndParameters]
    ] as const) {
      validateSinglePackedView(view, ['float32'], `${id} output.${name}`);
      if (view.length !== capacity) {
        throw new Error(`${id} output.${name} length must equal output.endpoints length`);
      }
    }
    for (const [name, view] of Object.entries(output.status)) {
      if (view) {
        validatePackedUint32View(view, `${id} output.status.${name}`);
        if (view.length < 1) {
          throw new Error(`${id} output.status.${name} must contain one row`);
        }
      }
    }
    if (!output.status.requiredCount || !output.status.candidateOverflow) {
      throw new Error(`${id} output.status requires requiredCount and candidateOverflow`);
    }
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {lines, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      lines.positions,
      lines.lineOffsets,
      lines.sourceIds,
      lines.featureOffsets,
      output.endpoints,
      output.sourceFeatureIds,
      output.sourceRingIds,
      output.sourceEdgeIds,
      output.sourceStartParameters,
      output.sourceEndParameters,
      ...Object.values(output.status)
    ]);
    validateLineNodingBounds(
      id,
      graph.device,
      lines.positions.length,
      lines.lineOffsets.length - 1,
      props.intersectionCapacity
    );
    const pairCapacity = props.intersectionCapacity;
    const lineCount = lines.lineOffsets.length - 1;
    const pieceCapacity = lineCount + pairCapacity * 4;
    const pieceVertexCapacity = lines.positions.length + pairCapacity * 8;
    const transient = <Format extends 'uint32' | 'float32x2'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);
    const piecePositions = transient('piece-positions', 'float32x2', pieceVertexCapacity);
    const pieceOffsets = transient('piece-offsets', 'uint32', pieceCapacity + 1);
    const pieceSources = transient('piece-sources', 'uint32', pieceCapacity);
    const pieceCount = transient('piece-count', 'uint32', 1);
    const pieceRequiredCount = transient('piece-required-count', 'uint32', 1);
    const pieceVertexCount = transient('piece-vertex-count', 'uint32', 1);
    const pieceRequiredVertexCount = transient('piece-required-vertex-count', 'uint32', 1);
    const pieceSourceEdges = transient('piece-source-edges', 'uint32', pieceVertexCapacity);
    // Incomplete split output: piece capacity, uncertain pairs or non-finite vertices.
    const splitIncomplete = transient('split-incomplete', 'uint32', 1);
    const nodes: GPUCommandNode<Parameters>[] = [];
    nodes.push(
      ...new GPULineSplit({
        id: `${id}-split`,
        lines: {kind: 'lines', positions: lines.positions, lineOffsets: lines.lineOffsets},
        intersectionCapacity: pairCapacity,
        spatialSort: props.spatialSort,
        leafCapacity: props.leafCapacity,
        output: {
          status: {uncertainCount: output.status.uncertainCount},
          pieces: {
            geometry: {kind: 'lines', positions: piecePositions, lineOffsets: pieceOffsets},
            sourceIds: createTransientView(
              graph,
              `${id}-piece-identities`,
              'uint32',
              pieceCapacity
            ),
            sourcePartRows: pieceSources,
            status: {
              count: pieceCount,
              requiredCount: pieceRequiredCount,
              overflow: splitIncomplete,
              candidateOverflow: output.status.candidateOverflow,
              vertexCount: pieceVertexCount,
              requiredVertexCount: pieceRequiredVertexCount
            },
            vertexSourceEdges: pieceSourceEdges
          }
        }
      }).getCommandNodes(graph)
    );

    const segmentFlags = transient('segment-flags', 'uint32', pieceVertexCapacity);
    const segmentScan = transient('segment-scan', 'uint32', pieceVertexCapacity);
    const commonBindings: WGSLKernelBinding[] = [
      {name: 'pieceOffsets', view: pieceOffsets, type: 'u32', access: 'read'},
      {name: 'pieceCount', view: pieceCount, type: 'u32', access: 'read'},
      {name: 'pieceVertexCount', view: pieceVertexCount, type: 'u32', access: 'read'}
    ];
    const pieceLookupWGSL = `
fn pieceOfVertex(vertex: u32) -> u32 {
  let count = pieceCount[pieceCountOffset];
  var low = 0u;
  var high = count;
  while (low + 1u < high) {
    let middle = (low + high) / 2u;
    if (pieceOffsets[pieceOffsetsOffset + middle] <= vertex) { low = middle; } else { high = middle; }
  }
  return low;
}`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-segment-flags`,
        operation: OPERATION,
        variant: 'segment-flags',
        bindings: [
          ...commonBindings,
          {name: 'piecePositions', view: piecePositions, type: 'f32', access: 'read'},
          {name: 'flags', view: segmentFlags, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pieceVertexCapacity,
        declarations: pieceLookupWGSL,
        // A repeated vertex is not a segment: a zero-length row would be a self-loop in the
        // arrangement, which strands the rings through its vertex in polygonization.
        body: `var valid = false;
  if (index < pieceVertexCount[pieceVertexCountOffset] && pieceCount[pieceCountOffset] > 0u) {
    let piece = pieceOfVertex(index);
    valid = index + 1u < pieceOffsets[pieceOffsetsOffset + piece + 1u];
    if (valid) {
      let base = piecePositionsOffset + index * 2u;
      valid = piecePositions[base] != piecePositions[base + 2u] || piecePositions[base + 1u] != piecePositions[base + 3u];
    }
  }
  flags[flagsOffset + index] = select(0u, 1u, valid);`
      }),
      ...new GPUScan({
        id: `${id}-segment-scan`,
        input: segmentFlags,
        output: segmentScan,
        mode: 'inclusive'
      }).getCommandNodes(graph)
    );

    const outputCapacity = output.endpoints.length;
    const sourceIdBinding: WGSLKernelBinding[] =
      lines.sourceIds && !lines.featureOffsets
        ? [{name: 'inputSourceIds', view: lines.sourceIds, type: 'u32', access: 'read'}]
        : [];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize`,
        operation: OPERATION,
        variant: 'initialize',
        bindings: [
          {name: 'endpoints', view: output.endpoints, type: 'f32', access: 'read_write'},
          {
            name: 'sourceFeatureIds',
            view: output.sourceFeatureIds,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'sourceRingIds', view: output.sourceRingIds, type: 'u32', access: 'read_write'},
          {name: 'sourceEdgeIds', view: output.sourceEdgeIds, type: 'u32', access: 'read_write'},
          {
            name: 'startParameters',
            view: output.sourceStartParameters,
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'endParameters',
            view: output.sourceEndParameters,
            type: 'f32',
            access: 'read_write'
          }
        ],
        invocationCount: outputCapacity,
        body: `let endpoint = endpointsOffset + index * 4u;
  endpoints[endpoint] = 0.0; endpoints[endpoint + 1u] = 0.0;
  endpoints[endpoint + 2u] = 0.0; endpoints[endpoint + 3u] = 0.0;
  sourceFeatureIds[sourceFeatureIdsOffset + index] = ${GPU_TOPOLOGY_NONE}u;
  sourceRingIds[sourceRingIdsOffset + index] = ${GPU_TOPOLOGY_NONE}u;
  sourceEdgeIds[sourceEdgeIdsOffset + index] = ${GPU_TOPOLOGY_NONE}u;
  startParameters[startParametersOffset + index] = 0.0;
  endParameters[endParametersOffset + index] = 0.0;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write-geometry`,
        operation: OPERATION,
        variant: 'write-geometry',
        bindings: [
          {name: 'flags', view: segmentFlags, type: 'u32', access: 'read'},
          {name: 'scan', view: segmentScan, type: 'u32', access: 'read'},
          {name: 'piecePositions', view: piecePositions, type: 'f32', access: 'read'},
          {name: 'endpoints', view: output.endpoints, type: 'f32', access: 'read_write'}
        ],
        invocationCount: pieceVertexCapacity,
        declarations: `const OUTPUT_CAPACITY: u32 = ${outputCapacity}u;
fn piecePoint(vertex: u32) -> vec2f {
  return vec2f(piecePositions[piecePositionsOffset + vertex * 2u], piecePositions[piecePositionsOffset + vertex * 2u + 1u]);
}`,
        body: `if (flags[flagsOffset + index] == 0u) { return; }
  let outputRow = scan[scanOffset + index] - 1u;
  if (outputRow >= OUTPUT_CAPACITY) { return; }
  let a = piecePoint(index);
  let b = piecePoint(index + 1u);
  let endpoint = endpointsOffset + outputRow * 4u;
  endpoints[endpoint] = a.x; endpoints[endpoint + 1u] = a.y;
  endpoints[endpoint + 2u] = b.x; endpoints[endpoint + 3u] = b.y;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write-owners`,
        operation: OPERATION,
        variant: 'write-owners',
        bindings: [
          {name: 'pieceOffsets', view: pieceOffsets, type: 'u32', access: 'read'},
          {name: 'pieceCount', view: pieceCount, type: 'u32', access: 'read'},
          {name: 'flags', view: segmentFlags, type: 'u32', access: 'read'},
          {name: 'scan', view: segmentScan, type: 'u32', access: 'read'},
          {name: 'pieceSources', view: pieceSources, type: 'u32', access: 'read'},
          ...sourceIdBinding,
          {
            name: 'sourceFeatureIds',
            view: output.sourceFeatureIds,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'sourceRingIds', view: output.sourceRingIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pieceVertexCapacity,
        declarations: `${pieceLookupWGSL}
const OUTPUT_CAPACITY: u32 = ${outputCapacity}u;`,
        body: `if (flags[flagsOffset + index] == 0u) { return; }
  let outputRow = scan[scanOffset + index] - 1u;
  if (outputRow >= OUTPUT_CAPACITY) { return; }
  let piece = pieceOfVertex(index);
  let line = pieceSources[pieceSourcesOffset + piece];
  sourceFeatureIds[sourceFeatureIdsOffset + outputRow] = ${lines.sourceIds && !lines.featureOffsets ? 'inputSourceIds[inputSourceIdsOffset + line]' : 'line'};
  sourceRingIds[sourceRingIdsOffset + outputRow] = line;`
      }),
      // The splitter records the original outgoing edge. Geometry-based reassignment loses
      // traversal identity on retraces and on midpoints that round to a shared vertex.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write-edge-provenance`,
        operation: OPERATION,
        variant: 'write-edge-provenance',
        bindings: [
          {name: 'flags', view: segmentFlags, type: 'u32', access: 'read'},
          {name: 'scan', view: segmentScan, type: 'u32', access: 'read'},
          {name: 'pieceSourceEdges', view: pieceSourceEdges, type: 'u32', access: 'read'},
          {name: 'sourceEdgeIds', view: output.sourceEdgeIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pieceVertexCapacity,
        body: `if (flags[flagsOffset + index] == 0u) { return; }
  let outputRow = scan[scanOffset + index] - 1u;
  if (outputRow >= ${outputCapacity}u) { return; }
  sourceEdgeIds[sourceEdgeIdsOffset + outputRow] = pieceSourceEdges[pieceSourceEdgesOffset + index] & ~${GPU_LINE_SPLIT_SELF_CONTACT}u;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write-parameters`,
        operation: OPERATION,
        variant: 'write-parameters',
        bindings: [
          {name: 'flags', view: segmentFlags, type: 'u32', access: 'read'},
          {name: 'scan', view: segmentScan, type: 'u32', access: 'read'},
          {name: 'piecePositions', view: piecePositions, type: 'f32', access: 'read'},
          {name: 'inputPositions', view: lines.positions, type: 'f32', access: 'read'},
          {name: 'sourceEdgeIds', view: output.sourceEdgeIds, type: 'u32', access: 'read'},
          {
            name: 'startParameters',
            view: output.sourceStartParameters,
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'endParameters',
            view: output.sourceEndParameters,
            type: 'f32',
            access: 'read_write'
          }
        ],
        invocationCount: pieceVertexCapacity,
        declarations: `const OUTPUT_CAPACITY: u32 = ${outputCapacity}u;
fn inputPoint(vertex: u32) -> vec2f {
  return vec2f(inputPositions[inputPositionsOffset + vertex * 2u], inputPositions[inputPositionsOffset + vertex * 2u + 1u]);
}
fn piecePoint(vertex: u32) -> vec2f {
  return vec2f(piecePositions[piecePositionsOffset + vertex * 2u], piecePositions[piecePositionsOffset + vertex * 2u + 1u]);
}
fn differenceExponent(delta: vec2f) -> u32 {
  return max(bitcast<u32>(delta.x) & 0x7fffffffu, bitcast<u32>(delta.y) & 0x7fffffffu) >> 23u;
}
fn scaleDifference(value: vec2f, power: i32) -> vec2f {
  return vec2f(ldexp(value.x, power), ldexp(value.y, power));
}
fn magnitudePower(magnitude: u32) -> i32 {
  let exponent = magnitude >> 23u;
  if (exponent != 0u) { return i32(exponent) - 127i; }
  return i32(firstLeadingBit(magnitude)) - 149i;
}
const ZERO_PRODUCT_POWER: i32 = -512i;
fn productPower(first: f32, second: f32) -> i32 {
  let firstMagnitude = bitcast<u32>(first) & 0x7fffffffu;
  let secondMagnitude = bitcast<u32>(second) & 0x7fffffffu;
  if (firstMagnitude == 0u || secondMagnitude == 0u) { return ZERO_PRODUCT_POWER; }
  return magnitudePower(firstMagnitude) + magnitudePower(secondMagnitude);
}
fn numeratorNeedsScaling(difference: vec2f, delta: vec2f) -> bool {
  let power = max(productPower(difference.x, delta.x), productPower(difference.y, delta.y));
  // A zero numerator is valid. Otherwise a dominant product whose exponent
  // lower bound is below -126 can round to a subnormal or flush to zero.
  return power != ZERO_PRODUCT_POWER && power < -126i;
}
fn dotMayLeaveNormalRange(first: vec2f, second: vec2f) -> bool {
  if (max(differenceExponent(first), differenceExponent(second)) == 255u) { return true; }
  let firstPower = productPower(first.x, second.x);
  let secondPower = productPower(first.y, second.y);
  let dominantPower = max(firstPower, secondPower);
  // Each product is below 2^(power+2). Reserve one more bit when two
  // nonzero products can add, and leave exact zeros on the original route.
  let upperPower = select(126i, 125i, min(firstPower, secondPower) != ZERO_PRODUCT_POWER);
  return dominantPower != ZERO_PRODUCT_POWER && (dominantPower < -126i || dominantPower > upperPower);
}
fn scaledProjection(difference: vec2f, delta: vec2f) -> f32 {
  if (all(difference == delta)) { return 1.0; }
  let power = 127i - i32(clamp(differenceExponent(delta), 1u, 253u));
  let direction = scaleDifference(delta, power);
  let offset = scaleDifference(difference, power);
  if (direction.y == 0.0) { return clamp(offset.x / direction.x, 0.0, 1.0); }
  if (direction.x == 0.0) { return clamp(offset.y / direction.y, 0.0, 1.0); }
  return clamp(dot(offset, direction) / dot(direction, direction), 0.0, 1.0);
}`,
        body: `if (flags[flagsOffset + index] == 0u) { return; }
  let outputRow = scan[scanOffset + index] - 1u;
  if (outputRow >= OUTPUT_CAPACITY) { return; }
  let a = piecePoint(index);
  let b = piecePoint(index + 1u);
  let bestEdge = sourceEdgeIds[sourceEdgeIdsOffset + outputRow];
  if (bestEdge != ${GPU_TOPOLOGY_NONE}u) {
    let p = inputPoint(bestEdge);
    let q = inputPoint(bestEdge + 1u);
    var delta = q - p;
    var firstDifference = a - p;
    var secondDifference = b - p;
    // Keep the pre-stream dot products and division within their original
    // power-of-two bounds. Those bounds protect the denominator and guard
    // against overflow, but a small split can still underflow its numerator.
    let maximumDelta = max(abs(delta.x),abs(delta.y));
    if (maximumDelta >= 1.0842021724855044e-19 && maximumDelta <= 9223372036854775808.0 &&
        all(abs(firstDifference) <= vec2f(9223372036854775808.0)) &&
        all(abs(secondDifference) <= vec2f(9223372036854775808.0))) {
      let scaleFirst = numeratorNeedsScaling(firstDifference, delta);
      let scaleSecond = numeratorNeedsScaling(secondDifference, delta);
      let originalDenominator = dot(delta,delta);
      if (!scaleFirst && !scaleSecond) {
        let originalFirstNumerator = dot(firstDifference,delta);
        let originalSecondNumerator = dot(secondDifference,delta);
        startParameters[startParametersOffset + outputRow] = clamp(originalFirstNumerator / originalDenominator,0.0,1.0);
        endParameters[endParametersOffset + outputRow] = clamp(originalSecondNumerator / originalDenominator,0.0,1.0);
        return;
      }
      // A tiny split on one end does not change the arithmetic for the other.
      if (scaleFirst) {
        startParameters[startParametersOffset + outputRow] = scaledProjection(firstDifference, delta);
      } else {
        startParameters[startParametersOffset + outputRow] = clamp(dot(firstDifference,delta) / originalDenominator,0.0,1.0);
      }
      if (scaleSecond) {
        endParameters[endParametersOffset + outputRow] = scaledProjection(secondDifference, delta);
      } else {
        endParameters[endParametersOffset + outputRow] = clamp(dot(secondDifference,delta) / originalDenominator,0.0,1.0);
      }
      return;
    }
    // The pre-stream bounds can select their normalized fallback even when
    // the dot products fit. Keep that fallback's arithmetic in this overlap.
    if (!dotMayLeaveNormalRange(delta, delta) &&
        !dotMayLeaveNormalRange(firstDifference, delta) &&
        !dotMayLeaveNormalRange(secondDifference, delta)) {
      let scale = max(abs(delta.x), abs(delta.y));
      let direction = delta / scale;
      let denominator = dot(direction, direction);
      startParameters[startParametersOffset + outputRow] = clamp(dot(firstDifference / scale, direction) / denominator, 0.0, 1.0);
      endParameters[endParametersOffset + outputRow] = clamp(dot(secondDifference / scale, direction) / denominator, 0.0, 1.0);
      return;
    }
    // Opposite finite endpoints can have an unrepresentable difference.
    if (differenceExponent(delta) == 255u) {
      let scaledStart = scaleDifference(p, -2);
      delta = scaleDifference(q, -2) - scaledStart;
      firstDifference = scaleDifference(a, -2) - scaledStart;
      secondDifference = scaleDifference(b, -2) - scaledStart;
    }
    startParameters[startParametersOffset + outputRow] = scaledProjection(firstDifference, delta);
    endParameters[endParametersOffset + outputRow] = scaledProjection(secondDifference, delta);
  }`
      })
    );

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-status`,
        operation: OPERATION,
        variant: 'status',
        bindings: [
          {name: 'scan', view: segmentScan, type: 'u32', access: 'read'},
          {
            name: 'candidateOverflow',
            view: output.status.candidateOverflow,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'splitIncomplete', view: splitIncomplete, type: 'u32', access: 'read'},
          {name: 'count', view: output.status.count, type: 'u32', access: 'read_write'},
          {
            name: 'requiredCount',
            view: output.status.requiredCount,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'overflow', view: output.status.overflow, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `const OUTPUT_CAPACITY: u32 = ${outputCapacity}u;`,
        body: `let required = scan[scanOffset + ${pieceVertexCapacity - 1}u];
  candidateOverflow[candidateOverflowOffset] |= splitIncomplete[splitIncompleteOffset];
  requiredCount[requiredCountOffset] = required;
  count[countOffset] = min(required, OUTPUT_CAPACITY);
  overflow[overflowOffset] = select(0u, 1u, required > OUTPUT_CAPACITY);`
      })
    );
    if (lines.featureOffsets)
      nodes.push(
        createLineFeatureProvenanceNode(graph, {
          id: `${id}-feature-provenance`,
          geometry: lines,
          partRows: output.sourceRingIds,
          featureIds: output.sourceFeatureIds,
          count: output.status.count
        })
      );
    return nodes;
  }
}
