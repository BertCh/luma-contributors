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
import type {GPUPolygonGeometryFields} from '../../analysis-core/contracts/index';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validatePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_QUIET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getReadPointWGSL} from '../shared/spatial-wgsl-point-helpers';

/** Status of the bounded geometric search (floating point uncertainty is separate). */
export const GPU_MAXIMUM_INSCRIBED_CIRCLE_STATUS = {
  SUCCESS: 0,
  BUDGET_EXHAUSTED: 1,
  EMPTY: 2,
  NON_FINITE: 3
} as const;

export type GPUMaximumInscribedCircleProps = GPUPolygonGeometryFields & {
  id?: string;
  spatialContext?: GPUSpatialContext;
  /** Positive stopping tolerance in planar position units. */
  tolerance: number;
  /** Maximum live quadtree cells per feature, default 256. */
  cellCapacity?: number;
  /** Maximum cell splits per feature, default 1024. */
  maximumIterations?: number;
  output: {
    /** Per feature (centerX, centerY, radius, remaining geometric search error). */
    circles: GraphDataView<'float32x4'>;
    /** Closest boundary point to the reported center, per feature. */
    witnesses: GraphDataView<'float32x2'>;
    /** GPU_MAXIMUM_INSCRIBED_CIRCLE_STATUS per feature. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Best-first branch-and-bound over all rings of each feature under even-odd fill.
 * Every unexplored cell retains the Lipschitz distance upper bound, including when
 * capacity or iterations run out. No readback, CPU search, or silent cell dropping.
 * The error bounds search discretization in exact arithmetic; f32 distance and
 * coordinate rounding are not interval-certified. Use a local planar frame.
 */
export class GPUMaximumInscribedCircle implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUMaximumInscribedCircleProps;
  readonly spatialContext: GPUSpatialContext;
  readonly cellCapacity: number;
  readonly maximumIterations: number;

  constructor(props: GPUMaximumInscribedCircleProps) {
    this.id = props.id ?? 'maximum-inscribed-circle';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    this.cellCapacity = props.cellCapacity ?? 256;
    this.maximumIterations = props.maximumIterations ?? 1024;
    validateCapacity(this.id, 'cellCapacity', this.cellCapacity, 1, 4096);
    validateCapacity(this.id, 'maximumIterations', this.maximumIterations, 0, 65536);
    if (!(Number.isFinite(props.tolerance) && props.tolerance > 0)) {
      throw new Error('Invalid circle tolerance');
    }
    validatePackedView(props.positions, ['float32x2'], this.id);
    for (const offsets of [props.ringOffsets, props.polygonOffsets, props.featureOffsets]) {
      validatePackedView(offsets, ['uint32'], this.id);
      validateCapacity(this.id, 'offset length', offsets.length, 2);
    }
    for (const [name, format] of [
      ['circles', 'float32x4'],
      ['witnesses', 'float32x2'],
      ['status', 'uint32']
    ] as const) {
      validatePackedView(props.output[name], [format], this.id);
      if (props.output[name].length !== props.featureOffsets.length - 1) {
        throw new Error('Circle output length mismatch');
      }
    }
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.positions,
      props.ringOffsets,
      props.polygonOffsets,
      props.featureOffsets
    ]);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.ringOffsets,
      props.polygonOffsets,
      props.featureOffsets,
      ...Object.values(props.output)
    ]);
    const cells = createTransientView(
      graph,
      `${id}-cells`,
      'float32',
      (props.featureOffsets.length - 1) * this.cellCapacity * 4
    );
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-search`,
        operation: 'GPUMaximumInscribedCircle',
        variant: 'branch-bound',
        invocationCount: props.featureOffsets.length - 1,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
          {name: 'polygonOffsets', view: props.polygonOffsets, type: 'u32', access: 'read'},
          {name: 'featureOffsets', view: props.featureOffsets, type: 'u32', access: 'read'},
          {name: 'circles', view: props.output.circles, type: 'f32', access: 'read_write'},
          {name: 'witnesses', view: props.output.witnesses, type: 'f32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'},
          {name: 'cells', view: cells, type: 'f32', access: 'read_write'}
        ],
        declarations: `${GET_QUIET_NAN_WGSL}
fn loadCell(base: u32, cell: u32) -> vec4f {
  let address = cellsOffset + base + cell * 4u;
  return vec4f(cells[address], cells[address + 1u], cells[address + 2u], cells[address + 3u]);
}
fn storeCell(base: u32, cell: u32, value: vec4f) {
  let address = cellsOffset + base + cell * 4u;
  cells[address] = value.x; cells[address + 1u] = value.y;
  cells[address + 2u] = value.z; cells[address + 3u] = value.w;
}
const CELL_CAPACITY: u32 = ${this.cellCapacity}u;
const MAXIMUM_ITERATIONS: u32 = ${this.maximumIterations}u;
const TOLERANCE: f32 = ${getWGSLFloatLiteral(props.tolerance)};
const VERTEX_COUNT: u32 = ${props.positions.length}u;
const RING_COUNT: u32 = ${props.ringOffsets.length - 1}u;
const POLYGON_COUNT: u32 = ${props.polygonOffsets.length - 1}u;
${getReadPointWGSL('point', 'positions')}
fn signedDistance(query: vec2f, firstRing: u32, lastRing: u32) -> vec3f {
  var inside = false;
  var distanceSquared = 3.4028234663852886e38;
  var witness = query;
  for (var ring = firstRing; ring < lastRing; ring++) {
    let firstVertex = min(ringOffsets[ringOffsetsOffset + ring], VERTEX_COUNT);
    let lastVertex = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
    if (lastVertex <= firstVertex) { continue; }
    var previous = point(lastVertex - 1u);
    for (var vertex = firstVertex; vertex < lastVertex; vertex++) {
      let current = point(vertex);
      let edge = current - previous;
      let denominator = dot(edge, edge);
      if ((bitcast<u32>(denominator) & 0x7f800000u) == 0x7f800000u) { return vec3f(getQuietNaN()); }
      var fraction = 0.0;
      if (denominator > 0.0) { fraction = clamp(dot(query - previous, edge) / denominator, 0.0, 1.0); }
      let candidate = previous + fraction * edge;
      let candidateSquared = dot(query - candidate, query - candidate);
      if ((bitcast<u32>(candidateSquared) & 0x7f800000u) == 0x7f800000u) { return vec3f(getQuietNaN()); }
      if (candidateSquared < distanceSquared) { distanceSquared = candidateSquared; witness = candidate; }
      if ((current.y > query.y) != (previous.y > query.y)) {
        if (query.x < previous.x + (query.y - previous.y) * (current.x - previous.x) / (current.y - previous.y)) { inside = !inside; }
      }
      previous = current;
    }
  }
  return vec3f(witness, select(-sqrt(distanceSquared), sqrt(distanceSquared), inside));
}`,
        body: `let firstPolygon = min(featureOffsets[featureOffsetsOffset + index], POLYGON_COUNT);
let lastPolygon = min(featureOffsets[featureOffsetsOffset + index + 1u], POLYGON_COUNT);
let firstRing = min(polygonOffsets[polygonOffsetsOffset + firstPolygon], RING_COUNT);
let lastRing = min(polygonOffsets[polygonOffsetsOffset + lastPolygon], RING_COUNT);
var minimum = vec2f(3.0e38);
var maximum = vec2f(-3.0e38);
var vertexCount = 0u;
var invalid = false;
var bestPoint = vec2f(0.0);
for (var ring = firstRing; ring < lastRing; ring++) {
  for (var vertex = min(ringOffsets[ringOffsetsOffset + ring], VERTEX_COUNT); vertex < min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT); vertex++) {
    let current = point(vertex);
    if (any((bitcast<vec2u>(current) & vec2u(0x7f800000u)) == vec2u(0x7f800000u))) { invalid = true; }
    minimum = min(minimum, current); maximum = max(maximum, current);
    bestPoint = current; vertexCount++;
  }
}
let nan = getQuietNaN();
var bestRadius = 0.0;
var bestWitness = bestPoint;
var error = 0.0;
var resultStatus = 0u;
if (invalid || vertexCount == 0u) {
  bestPoint = vec2f(nan); bestRadius = nan; bestWitness = vec2f(nan); error = nan;
  resultStatus = select(2u, 3u, invalid);
} else {
  let cellBase = index * CELL_CAPACITY * 4u;
  let center = 0.5 * minimum + 0.5 * maximum;
  let halfSize = 0.5 * max(maximum.x - minimum.x, maximum.y - minimum.y);
  let seed = signedDistance(center, firstRing, lastRing);
  if (seed.z > bestRadius) { bestRadius = seed.z; bestPoint = center; bestWitness = seed.xy; }
  storeCell(cellBase, 0u, vec4f(center, halfSize, seed.z + 1.4142135623730951 * halfSize));
  var cellCount = 1u;
  var iterations = 0u;
  loop {
    if ((bitcast<u32>(seed.z) & 0x7f800000u) == 0x7f800000u) { resultStatus = 3u; break; }
    var winner = 0u;
    for (var cell = 1u; cell < cellCount; cell++) { if (loadCell(cellBase, cell).w > loadCell(cellBase, winner).w) { winner = cell; } }
    let parent = loadCell(cellBase, winner);
    error = max(0.0, parent.w - bestRadius);
    if (error <= TOLERANCE) { break; }
    if (iterations >= MAXIMUM_ITERATIONS || cellCount + 3u > CELL_CAPACITY || parent.z * 0.5 == parent.z) { resultStatus = 1u; break; }
    storeCell(cellBase, winner, loadCell(cellBase, cellCount - 1u)); cellCount--;
    let childHalfSize = parent.z * 0.5;
    for (var child = 0u; child < 4u; child++) {
      let childCenter = parent.xy + childHalfSize * vec2f(select(-1.0, 1.0, (child & 1u) != 0u), select(-1.0, 1.0, (child & 2u) != 0u));
      let distance = signedDistance(childCenter, firstRing, lastRing);
      if ((bitcast<u32>(distance.z) & 0x7f800000u) == 0x7f800000u) { resultStatus = 3u; break; }
      if (distance.z > bestRadius) { bestRadius = distance.z; bestPoint = childCenter; bestWitness = distance.xy; }
      storeCell(cellBase, cellCount, vec4f(childCenter, childHalfSize, distance.z + 1.4142135623730951 * childHalfSize)); cellCount++;
    }
    if (resultStatus == 3u) { break; }
    // Compact cells already dominated by the best lower bound, but retain one.
    var kept = 0u;
    for (var cell = 0u; cell < cellCount; cell++) {
      if (loadCell(cellBase, cell).w > bestRadius) { storeCell(cellBase, kept, loadCell(cellBase, cell)); kept++; }
    }
    if (kept == 0u) { error = 0.0; break; }
    cellCount = kept; iterations++;
  }
}
// Arithmetic overflow cannot provide a geometric bound or a usable witness.
if (resultStatus == 3u) { bestPoint = vec2f(nan); bestRadius = nan; bestWitness = vec2f(nan); error = nan; }
circles[circlesOffset + index * 4u] = bestPoint.x;
circles[circlesOffset + index * 4u + 1u] = bestPoint.y;
circles[circlesOffset + index * 4u + 2u] = bestRadius;
circles[circlesOffset + index * 4u + 3u] = error;
witnesses[witnessesOffset + index * 2u] = bestWitness.x;
witnesses[witnessesOffset + index * 2u + 1u] = bestWitness.y;
status[statusOffset + index] = resultStatus;`
      })
    ];
  }
}
