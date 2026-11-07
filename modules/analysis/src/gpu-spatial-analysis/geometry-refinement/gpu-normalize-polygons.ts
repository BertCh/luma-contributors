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
  validatePackedView,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GPUGeometryOrientation} from '../geometry-edit/gpu-geometry-orientation';
import {getReadPointWGSL} from '../shared/spatial-wgsl-point-helpers';

export type GPUNormalizePolygonsProps = GPUPolygonGeometryFields & {
  id?: string;
  output: {
    positions: GraphDataView<'float32x2'>;
    ringOffsets: GraphDataView<'uint32'>;
    polygonOffsets: GraphDataView<'uint32'>;
  };
};

/**
 * Canonical planar polygon serialization, entirely on the GPU. Rings select the
 * lexicographically smallest cyclic sequence after orienting shells clockwise
 * and holes counterclockwise; holes and polygon parts sort in descending order. Exterior rings stay first. Feature order,
 * empty parts, duplicates, closure convention and vertex counts are preserved.
 * Reuse the input featureOffsets with the outputs. Inputs must be finite, valid
 * offset tables. This follows GEOS's clockwise-exterior normalization convention.
 * Compare at identical precision; ring closure conventions remain significant.
 */
export class GPUNormalizePolygons implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUNormalizePolygonsProps;
  constructor(props: GPUNormalizePolygonsProps) {
    this.id = props.id ?? 'normalize-polygons';
    this.props = props;
    validatePackedView(props.positions, ['float32x2'], this.id);
    for (const name of ['ringOffsets', 'polygonOffsets', 'featureOffsets'] as const) {
      validatePackedView(props[name], ['uint32'], this.id);
      validateCapacity(this.id, name, props[name].length, 2);
    }
    for (const name of ['positions', 'ringOffsets', 'polygonOffsets'] as const) {
      validatePackedView(
        props.output[name],
        [name === 'positions' ? 'float32x2' : 'uint32'],
        this.id
      );
      if (props.output[name].length !== props[name].length) {
        throw new Error('Canonical output length mismatch');
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
    const canonical = createTransientView(
      graph,
      `${id}-canonical-rings`,
      'uint32x4',
      props.ringOffsets.length - 1
    );
    const orientedPositions = createTransientView(
      graph,
      `${id}-oriented`,
      'float32x2',
      props.positions.length
    );
    const parameters = createTransientView(graph, `${id}-orientation-parameters`, 'float32', 4);
    const orientationNodes = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-parameters`,
        operation: 'GPUNormalizePolygons',
        invocationCount: 4,
        bindings: [{name: 'parameters', view: parameters, type: 'f32', access: 'read_write'}],
        body: 'parameters[parametersOffset + index] = select(0.0, 1.0, index == 0u);'
      }),
      ...new GPUGeometryOrientation({
        ...props,
        id: `${id}-orient`,
        mode: 'orient-polygons',
        parameters,
        output: {positions: orientedPositions}
      }).getCommandNodes(graph)
    ];
    const access = `${getReadPointWGSL('point', 'positions')}
fn comparePoints(left: vec2f, right: vec2f) -> i32 {
  if (left.x < right.x || (left.x == right.x && left.y < right.y)) { return -1; }
  if (left.x > right.x || (left.x == right.x && left.y > right.y)) { return 1; }
  return 0;
}`;
    return [
      ...orientationNodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-rings`,
        operation: 'GPUNormalizePolygons',
        variant: 'rings',
        invocationCount: props.ringOffsets.length - 1,
        bindings: [
          {name: 'positions', view: orientedPositions, type: 'f32', access: 'read'},
          {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
          {name: 'canonical', view: canonical, type: 'u32', access: 'read_write'}
        ],
        declarations: access,
        body: `let first = ringOffsets[ringOffsetsOffset + index];
let total = ringOffsets[ringOffsetsOffset + index + 1u] - first;
var closed = 0u;
if (total > 1u && all(point(first) == point(first + total - 1u))) { closed = 1u; }
let length = total - closed;
var bestStart = 0u; var bestReverse = 0u;
for (var candidate = 0u; candidate < length; candidate++) {
  for (var reverse = 0u; reverse < 1u; reverse++) {
    for (var element = 0u; element < length; element++) {
      let left = point(first + (candidate + select(element, length - element, reverse == 1u)) % length);
      let right = point(first + (bestStart + select(element, length - element, bestReverse == 1u)) % length);
      let comparison = comparePoints(left, right);
      if (comparison < 0) { bestStart = candidate; bestReverse = reverse; }
      if (comparison != 0) { break; }
    }
  }
}
canonical[canonicalOffset + index * 4u] = bestStart;
canonical[canonicalOffset + index * 4u + 1u] = length;
canonical[canonicalOffset + index * 4u + 2u] = bestReverse;
canonical[canonicalOffset + index * 4u + 3u] = closed;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-parts`,
        operation: 'GPUNormalizePolygons',
        variant: 'parts',
        invocationCount: props.featureOffsets.length - 1,
        bindings: [
          {name: 'positions', view: orientedPositions, type: 'f32', access: 'read'},
          {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
          {name: 'polygonOffsets', view: props.polygonOffsets, type: 'u32', access: 'read'},
          {name: 'featureOffsets', view: props.featureOffsets, type: 'u32', access: 'read'},
          {name: 'canonical', view: canonical, type: 'u32', access: 'read'},
          {name: 'outPositions', view: props.output.positions, type: 'f32', access: 'read_write'},
          {
            name: 'outRingOffsets',
            view: props.output.ringOffsets,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'outPolygonOffsets',
            view: props.output.polygonOffsets,
            type: 'u32',
            access: 'read_write'
          }
        ],
        declarations: `${access}
fn canonicalPoint(ring: u32, element: u32) -> vec2f {
  let base = canonicalOffset + ring * 4u;
  let length = canonical[base + 1u];
  let offset = canonical[base] + select(element, length - element, canonical[base + 2u] == 1u);
  return point(ringOffsets[ringOffsetsOffset + ring] + offset % length);
}
fn compareRings(left: u32, right: u32) -> i32 {
  let leftLength = canonical[canonicalOffset + left * 4u + 1u];
  let rightLength = canonical[canonicalOffset + right * 4u + 1u];
  let leftCount = leftLength + canonical[canonicalOffset + left * 4u + 3u];
  let rightCount = rightLength + canonical[canonicalOffset + right * 4u + 3u];
  if (leftCount != rightCount) { return select(1, -1, leftCount < rightCount); }
  for (var element = 0u; element < min(leftLength, rightLength); element++) {
    let comparison = comparePoints(canonicalPoint(left, element), canonicalPoint(right, element));
    if (comparison != 0) { return comparison; }
  }
  if (leftLength != rightLength) { return select(1, -1, leftLength < rightLength); }
  let leftClosed = canonical[canonicalOffset + left * 4u + 3u];
  let rightClosed = canonical[canonicalOffset + right * 4u + 3u];
  return select(select(1, -1, leftClosed < rightClosed), 0, leftClosed == rightClosed);
}
fn sortedRing(polygon: u32, slot: u32) -> u32 {
  let first = polygonOffsets[polygonOffsetsOffset + polygon];
  let last = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
  if (slot == 0u) { return first; }
  for (var candidate = first + 1u; candidate < last; candidate++) {
    var rank = 1u;
    for (var other = first + 1u; other < last; other++) {
      let comparison = compareRings(other, candidate);
      if (comparison > 0 || (comparison == 0 && other < candidate)) { rank++; }
    }
    if (rank == slot) { return candidate; }
  }
  return first;
}
fn comparePolygons(left: u32, right: u32) -> i32 {
  let leftLength = polygonOffsets[polygonOffsetsOffset + left + 1u] - polygonOffsets[polygonOffsetsOffset + left];
  let rightLength = polygonOffsets[polygonOffsetsOffset + right + 1u] - polygonOffsets[polygonOffsetsOffset + right];
  for (var slot = 0u; slot < min(leftLength, rightLength); slot++) {
    let comparison = compareRings(sortedRing(left, slot), sortedRing(right, slot));
    if (comparison != 0) { return comparison; }
  }
  return select(select(1, -1, leftLength < rightLength), 0, leftLength == rightLength);
}`,
        body: `let firstPolygon = featureOffsets[featureOffsetsOffset + index];
let lastPolygon = featureOffsets[featureOffsetsOffset + index + 1u];
var destinationRing = polygonOffsets[polygonOffsetsOffset + firstPolygon];
var destinationVertex = ringOffsets[ringOffsetsOffset + destinationRing];
for (var slot = 0u; slot < lastPolygon - firstPolygon; slot++) {
  var polygon = firstPolygon;
  for (var candidate = firstPolygon; candidate < lastPolygon; candidate++) {
    var rank = 0u;
    for (var other = firstPolygon; other < lastPolygon; other++) {
      let comparison = comparePolygons(other, candidate);
      if (comparison > 0 || (comparison == 0 && other < candidate)) { rank++; }
    }
    if (rank == slot) { polygon = candidate; break; }
  }
  outPolygonOffsets[outPolygonOffsetsOffset + firstPolygon + slot] = destinationRing;
  let ringCount = polygonOffsets[polygonOffsetsOffset + polygon + 1u] - polygonOffsets[polygonOffsetsOffset + polygon];
  for (var ringSlot = 0u; ringSlot < ringCount; ringSlot++) {
    let ring = sortedRing(polygon, ringSlot);
    outRingOffsets[outRingOffsetsOffset + destinationRing] = destinationVertex;
    let length = canonical[canonicalOffset + ring * 4u + 1u];
    let closed = canonical[canonicalOffset + ring * 4u + 3u];
    for (var element = 0u; element < length + closed; element++) {
      let current = canonicalPoint(ring, element % length);
      outPositions[outPositionsOffset + destinationVertex * 2u] = current.x + 0.0;
      outPositions[outPositionsOffset + destinationVertex * 2u + 1u] = current.y + 0.0;
      destinationVertex++;
    }
    destinationRing++;
  }
}
// A unique invocation owns the two terminal offsets.
if (index == ${props.featureOffsets.length - 2}u) {
  outPolygonOffsets[outPolygonOffsetsOffset + lastPolygon] = destinationRing;
  outRingOffsets[outRingOffsetsOffset + destinationRing] = destinationVertex;
}`
      })
    ];
  }
}
