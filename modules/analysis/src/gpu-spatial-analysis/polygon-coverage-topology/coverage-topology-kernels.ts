// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {EXACT_ORIENTATION_WGSL} from '../../analysis-core/kernels/exact-orientation-wgsl';
import {IS_FINITE_FLOAT_WGSL} from '../../analysis-core/kernels/wgsl-common';

/**
 * Largest `|coordinate - origin| / snapTolerance` for which a coverage snap key is exact: the
 * float32 quotient is an integer-valued float up to here, so distinct snap cells get distinct keys.
 * @internal
 */
export const COVERAGE_SNAP_MAXIMUM_STEPS = 2 ** 24;

/**
 * WGSL for the vertex snap keys of the coverage operators (dissolve, simplification):
 *
 * - `getSnapOrigin() -> vec2f`: the first ring-owned vertex (`ringOffsets[0]`), or zero when there
 *   are no vertices or that vertex is not finite. Keys are taken relative to it, so they depend
 *   on the extent of the data and not on how far the data is from zero (projected coordinates of millions with a millimetre
 *   tolerance keep distinct keys). The snap grid is therefore anchored at the first vertex.
 * - `quantise(value, origin) -> u32`: `floor((value - origin) / snapTolerance + 0.5)` (round half
 *   up) as an order-preserving key. With a tolerance of 0 the key is the float's bits (`-0` as
 *   `0`). From `2^23` steps on the quotient is already an integer and is used as it is, because
 *   adding 0.5 there would round to even and merge neighbouring cells.
 * - `isSnapRangeExceeded(value, origin) -> bool`: the quotient reaches
 *   {@link COVERAGE_SNAP_MAXIMUM_STEPS}, where float32 can no longer tell neighbouring cells apart
 *   (the coverage operators count such vertices in `status.uncertainCount`).
 *
 * The kernel must bind `positions` (`array<f32>`, x and y interleaved) and `ringOffsets`
 * (`array<u32>`, including the terminal offset), and declare `isFiniteFloat`.
 *
 * @internal
 */
export function getCoverageSnapKeyWGSL(snapTolerance: number, ringCount: number): string {
  return /* wgsl */ `
const SNAP_TOLERANCE: f32 = ${getWGSLFloatLiteral(snapTolerance)};
const SNAP_MAXIMUM_STEPS: f32 = ${getWGSLFloatLiteral(COVERAGE_SNAP_MAXIMUM_STEPS)};

fn getSnapOrigin() -> vec2f {
  let firstVertex = ringOffsets[ringOffsetsOffset];
  if (firstVertex >= ringOffsets[ringOffsetsOffset + ${ringCount}u] ||
      firstVertex >= (arrayLength(&positions) - positionsOffset) / 2u) { return vec2f(0.0); }
  let first = vec2f(positions[positionsOffset + 2u * firstVertex], positions[positionsOffset + 2u * firstVertex + 1u]);
  return select(vec2f(0.0), first, isFiniteFloat(first.x) && isFiniteFloat(first.y));
}

fn quantise(value: f32, origin: f32) -> u32 {
  if (SNAP_TOLERANCE <= 0.0) {
    return bitcast<u32>(select(value, 0.0, value == 0.0));
  }
  let steps = (value - origin) / SNAP_TOLERANCE;
  let rounded = select(floor(steps + 0.5), steps, abs(steps) >= 8388608.0);
  return bitcast<u32>(i32(clamp(rounded, -2147483520.0, 2147483520.0))) ^ 0x80000000u;
}

fn isSnapRangeExceeded(value: f32, origin: f32) -> bool {
  return SNAP_TOLERANCE > 0.0 && abs((value - origin) / SNAP_TOLERANCE) >= SNAP_MAXIMUM_STEPS;
}
`;
}

/**
 * Builds the nodes that write `uncertainCount` (one `uint32` row): the number of finite vertices
 * that are {@link COVERAGE_SNAP_MAXIMUM_STEPS} or more snap steps from the snap origin on either
 * axis. The snap cell of such a vertex cannot be certified in float32, so its point identity comes
 * from an inexact key and it may be merged with a vertex of a neighbouring cell. With a snap
 * tolerance of 0 the count is always 0 (bit-equality keys have no range) and only the clear node
 * is built.
 *
 * @internal
 */
export function createCoverageSnapRangeNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    positions: GraphDataView<'float32x2'>;
    /** Only rows inside `[ringOffsets[0], ringOffsets[last])` are vertices; the rest is padding. */
    ringOffsets: GraphDataView<'uint32'>;
    snapTolerance: number;
    uncertainCount: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const clearNode = createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-clear`,
    operation: props.operation,
    variant: 'snap-range-clear',
    bindings: [
      {name: 'uncertainCount', view: props.uncertainCount, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 1,
    body: 'uncertainCount[uncertainCountOffset] = 0u;'
  });
  if (!(props.snapTolerance > 0)) {
    return [clearNode];
  }
  return [
    clearNode,
    createWGSLKernelNode<Parameters>(graph, {
      id: props.id,
      operation: props.operation,
      variant: 'snap-range',
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
        {
          name: 'uncertainCount',
          view: props.uncertainCount,
          type: 'atomic<u32>',
          access: 'read_write'
        }
      ],
      invocationCount: props.positions.length,
      declarations: `${IS_FINITE_FLOAT_WGSL}
${getCoverageSnapKeyWGSL(props.snapTolerance, props.ringOffsets.length - 1)}`,
      // Padding rows (outside every ring) are no vertex and hold anything: they are not counted.
      body: `if (index < ringOffsets[ringOffsetsOffset] || index >= ringOffsets[ringOffsetsOffset + ${props.ringOffsets.length - 1}u]) { return; }
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  if (!isFiniteFloat(x) || !isFiniteFloat(y)) { return; }
  let origin = getSnapOrigin();
  if (isSnapRangeExceeded(x, origin.x) || isSnapRangeExceeded(y, origin.y)) {
    atomicAdd(&uncertainCount[uncertainCountOffset], 1u);
  }`
    })
  ];
}

/**
 * Builds the node that writes `ringFlip`: one `uint32` per ring, `1` when the ring has its polygon
 * interior on the right of the vertex order (a counter-clockwise hole or a clockwise shell) and
 * `0` otherwise. The first ring of a polygon is its shell, the others are holes. Rings with fewer
 * than three vertices, rings outside every polygon and zero-area rings get `0`.
 *
 * The winding is the exact turn (`orientSign`) at the ring's lexicographically smallest vertex (x,
 * then y, first row on ties), between its nearest distinct neighbours: an extreme vertex of a
 * simple ring turns the way the ring winds, and the sign does not depend on float32 cancellation
 * (a float32 shoelace sum returns 0 for a clockwise sliver whose exact doubled area is -1). A zero
 * or uncertain turn (a spike at the extreme vertex, non-finite coordinates) falls back to the sign
 * of the float32 shoelace sum relative to the ring's first vertex. One thread walks each ring, so
 * a single very long ring costs a serial walk of its length.
 *
 * @internal
 */
export function createRingOrientationNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    positions: GraphDataView<'float32x2'>;
    ringOffsets: GraphDataView<'uint32'>;
    polygonOffsets: GraphDataView<'uint32'>;
    ringFlip: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {positions, ringOffsets, polygonOffsets, ringFlip} = props;
  const ringCount = ringOffsets.length - 1;
  const polygonCount = polygonOffsets.length - 1;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'ring-orientation',
    bindings: [
      {name: 'positions', view: positions, type: 'f32', access: 'read'},
      {name: 'ringOffsets', view: ringOffsets, type: 'u32', access: 'read'},
      {name: 'polygonOffsets', view: polygonOffsets, type: 'u32', access: 'read'},
      {name: 'ringFlip', view: ringFlip, type: 'u32', access: 'read_write'}
    ],
    invocationCount: ringCount,
    declarations: `const VERTEX_COUNT: u32 = ${positions.length}u;
const POLYGON_COUNT: u32 = ${polygonCount}u;
${EXACT_ORIENTATION_WGSL}
fn readRingPoint(row: u32) -> vec2f {
  return vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
}`,
    body: `let begin = ringOffsets[ringOffsetsOffset + index];
  let end = min(ringOffsets[ringOffsetsOffset + index + 1u], VERTEX_COUNT);
  var flip = 0u;
  if (end > begin + 2u && index < polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT]) {
    var polygon = 0u;
    var high = POLYGON_COUNT;
    while (polygon + 1u < high) {
      let middle = (polygon + high) / 2u;
      if (polygonOffsets[polygonOffsetsOffset + middle] <= index) {
        polygon = middle;
      } else {
        high = middle;
      }
    }
    let isShell = polygonOffsets[polygonOffsetsOffset + polygon] == index;
    // Exact turn at the lexicographically smallest vertex, between its nearest distinct neighbours.
    var extreme = begin;
    var center = readRingPoint(begin);
    for (var row = begin + 1u; row < end; row++) {
      let candidate = readRingPoint(row);
      if (candidate.x < center.x || (candidate.x == center.x && candidate.y < center.y)) {
        extreme = row;
        center = candidate;
      }
    }
    let count = end - begin;
    var before = center;
    var after = center;
    var foundBefore = false;
    var foundAfter = false;
    for (var step = 1u; step < count; step++) {
      if (!foundBefore) {
        before = readRingPoint(begin + (extreme - begin + count - step) % count);
        foundBefore = any(before != center);
      }
      if (!foundAfter) {
        after = readRingPoint(begin + (extreme - begin + step) % count);
        foundAfter = any(after != center);
      }
      if (foundBefore && foundAfter) { break; }
    }
    var winding = 0;
    if (foundBefore && foundAfter) { winding = orientSign(before, center, after); }
    if (winding == 0 || winding == 2) {
      // No certified turn: the sign of the float32 shoelace sum about the first vertex.
      let origin = readRingPoint(begin);
      var previous = vec2f(0.0);
      var twiceArea = 0.0;
      for (var vertex = begin + 1u; vertex < end; vertex++) {
        let current = readRingPoint(vertex) - origin;
        twiceArea += previous.x * current.y - previous.y * current.x;
        previous = current;
      }
      winding = i32(sign(twiceArea));
    }
    flip = select(select(0u, 1u, winding > 0), select(0u, 1u, winding < 0), isShell);
  }
  ringFlip[ringFlipOffset + index] = flip;`
  });
}
