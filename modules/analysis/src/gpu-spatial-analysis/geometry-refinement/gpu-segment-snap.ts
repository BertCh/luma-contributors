// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  validateSinglePackedView,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getReadPointWGSL} from '../shared/spatial-wgsl-point-helpers';

export type GPUSegmentSnapProps = {
  id?: string;
  spatialContext?: GPUSpatialContext;
  positions: GraphDataView<'float32x2'>;
  lineOffsets: GraphDataView<'uint32'>;
  referencePositions: GraphDataView<'float32x2'>;
  /** Nonnegative inclusive distance in planar units. */
  tolerance: number;
  /** Fixed output slots per input line. */
  maximumVerticesPerLine: number;
  output: {
    positions: GraphDataView<'float32x2'>;
    /** Original segment's first vertex index for every emitted vertex. */
    sourceEdges: GraphDataView<'uint32'>;
    /** Per line [written vertices, required vertices, capacity overflow, nonfinite input]. */
    counts: GraphDataView<'uint32x4'>;
    /** Optional line validity bits: too few distinct vertices=1, nonfinite=2, incomplete=4. Self-crossing is valid for lines. */
    validity?: GraphDataView<'uint32'>;
  };
};

/**
 * Snap endpoints to nearest reference vertices, then insert all reference vertices
 * within tolerance of each snapped segment in projection order. Distance ties use
 * reference row order; identical references insert only once. This intentionally
 * has no GEOS safety heuristic and does not certify topology preservation.
 * Output line i occupies slots i * maximumVerticesPerLine through its written
 * count. Counts are exact even on overflow; suffix vertices are dropped.
 */
export class GPUSegmentSnap implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUSegmentSnapProps;
  readonly spatialContext: GPUSpatialContext;
  constructor(props: GPUSegmentSnapProps) {
    this.id = props.id ?? 'segment-snap';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    validateCapacity(this.id, 'maximumVerticesPerLine', props.maximumVerticesPerLine, 1);
    validateCapacity(
      this.id,
      'maximum insertion count',
      props.positions.length * (props.referencePositions.length + 1),
      0,
      0xffffffff
    );
    if (!(Number.isFinite(props.tolerance) && props.tolerance >= 0)) {
      throw new Error('Invalid snap tolerance');
    }
    for (const view of [props.positions, props.referencePositions, props.output.positions]) {
      validateSinglePackedView(view, ['float32x2'], this.id);
    }
    for (const view of [props.lineOffsets, props.output.sourceEdges]) {
      validateSinglePackedView(view, ['uint32'], this.id);
    }
    validateSinglePackedView(props.output.counts, ['uint32x4'], this.id);
    if (props.output.validity) {
      validateSinglePackedView(props.output.validity, ['uint32'], this.id);
      if (props.output.validity.length !== props.lineOffsets.length - 1) {
        throw new Error('Snap validity length mismatch');
      }
    }
    validateCapacity(this.id, 'line offsets', props.lineOffsets.length, 2);
    const lines = props.lineOffsets.length - 1;
    if (
      props.output.counts.length !== lines ||
      props.output.positions.length !== lines * props.maximumVerticesPerLine ||
      props.output.sourceEdges.length !== props.output.positions.length
    ) {
      throw new Error('Snap output length mismatch');
    }
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.positions,
      props.lineOffsets,
      props.referencePositions
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.lineOffsets,
      props.referencePositions,
      ...Object.values(props.output)
    ]);
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-insert`,
        operation: 'GPUSegmentSnap',
        variant: 'sorted-insertions',
        invocationCount: props.lineOffsets.length - 1,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'lineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'},
          {name: 'references', view: props.referencePositions, type: 'f32', access: 'read'},
          {name: 'outPositions', view: props.output.positions, type: 'f32', access: 'read_write'},
          {name: 'sourceEdges', view: props.output.sourceEdges, type: 'u32', access: 'read_write'},
          {name: 'counts', view: props.output.counts, type: 'u32', access: 'read_write'},
          ...(props.output.validity
            ? [
                {
                  name: 'validity',
                  view: props.output.validity,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        declarations: `const CAPACITY: u32 = ${props.maximumVerticesPerLine}u;
const REFERENCE_COUNT: u32 = ${props.referencePositions.length}u;
const TOLERANCE: f32 = ${getWGSLFloatLiteral(props.tolerance)};
${getReadPointWGSL('point', 'positions')}
fn reference(vertex: u32) -> vec2f { return vec2f(references[referencesOffset + vertex * 2u], references[referencesOffset + vertex * 2u + 1u]); }
fn finitePoint(value: vec2f) -> bool { return all((bitcast<vec2u>(value) & vec2u(0x7f800000u)) != vec2u(0x7f800000u)); }
fn snapPoint(value: vec2f) -> vec2f {
  var result = value; var minimum = TOLERANCE; var found = false;
  for (var candidate = 0u; candidate < REFERENCE_COUNT; candidate++) {
    let distance = length(reference(candidate) - value);
    if (distance <= TOLERANCE && (!found || distance < minimum)) { result = reference(candidate); minimum = distance; found = true; }
  }
  return result;
}
fn writeVertex(line: u32, slot: u32, value: vec2f, edge: u32) {
  if (slot >= CAPACITY) { return; }
  let destination = line * CAPACITY + slot;
  outPositions[outPositionsOffset + destination * 2u] = value.x;
  outPositions[outPositionsOffset + destination * 2u + 1u] = value.y;
  sourceEdges[sourceEdgesOffset + destination] = edge;
}`,
        body: `let first = min(lineOffsets[lineOffsetsOffset + index], ${props.positions.length}u);
let last = min(lineOffsets[lineOffsetsOffset + index + 1u], ${props.positions.length}u);
var required = 0u;
var invalid = false;
for (var candidate = 0u; candidate < REFERENCE_COUNT; candidate++) { invalid = invalid || !finitePoint(reference(candidate)); }
for (var vertex = first; vertex < last; vertex++) { invalid = invalid || !finitePoint(point(vertex)); }
if (!invalid && last > first) {
  var start = snapPoint(point(first));
  writeVertex(index, required, start, first); required++;
  for (var vertex = first; vertex + 1u < last; vertex++) {
    let end = snapPoint(point(vertex + 1u));
    let edge = end - start; let denominator = dot(edge, edge);
    var previousFraction = 0.0; var previousReference = 0xffffffffu;
    var previousPoint = start;
    if (denominator > 0.0) {
      for (var inserted = 0u; inserted < REFERENCE_COUNT; inserted++) {
        var winner = 0xffffffffu; var winnerFraction = 2.0;
        for (var candidate = 0u; candidate < REFERENCE_COUNT; candidate++) {
          let current = reference(candidate);
          var duplicate = false;
          for (var earlier = 0u; earlier < candidate; earlier++) {
            if (all(reference(earlier) == current)) { duplicate = true; break; }
          }
          if (duplicate) { continue; }
          let fraction = dot(current - start, edge) / denominator;
          if (fraction <= 0.0 || fraction >= 1.0 || length(current - (start + fraction * edge)) > TOLERANCE) { continue; }
          if (fraction < previousFraction || (fraction == previousFraction && candidate <= previousReference)) { continue; }
          if (fraction < winnerFraction || (fraction == winnerFraction && candidate < winner)) { winner = candidate; winnerFraction = fraction; }
        }
        if (winner == 0xffffffffu) { break; }
        let current = reference(winner);
        if (any(current != previousPoint)) { writeVertex(index, required, current, vertex); required++; previousPoint = current; }
        previousFraction = winnerFraction; previousReference = winner;
      }
    }
    if (any(end != previousPoint)) { writeVertex(index, required, end, vertex); required++; }
    start = end;
  }
}
counts[countsOffset + index * 4u] = min(required, CAPACITY);
counts[countsOffset + index * 4u + 1u] = required;
counts[countsOffset + index * 4u + 2u] = select(0u, 1u, required > CAPACITY);
counts[countsOffset + index * 4u + 3u] = select(0u, 1u, invalid);
${props.output.validity ? 'validity[validityOffset + index] = select(0u, 1u, required == 1u) | select(0u, 2u, invalid) | select(0u, 4u, required > CAPACITY);' : ''}`
      })
    ];
  }
}
