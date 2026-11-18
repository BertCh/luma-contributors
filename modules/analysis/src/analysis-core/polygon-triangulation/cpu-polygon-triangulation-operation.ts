// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {resolveGPUSpatialContext, type GPUSpatialContext} from '../contracts/spatial-context';
import {validateCapacity, validateRequirement} from '../kernels/gpu-contributor-utils';
import {GPU_POLYGON_TRIANGULATION_DEFAULT_MAXIMUM_WORK} from './gpu-polygon-triangulation';
import {computeTriangulationOrientation as computePolygonOrientation} from './cpu-triangulation-predicates';
import {triangulateMonotoneOnCPU} from './cpu-monotone-triangulation';
import {GPU_POLYGON_TRIANGULATION_DEFAULT_MONOTONE_MINIMUM_VERTICES} from './polygon-triangulation-monotone';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../operations/cpu-operation-rows';
import {
  GPUPolygonTriangulationOperation,
  snapshotPolygonTriangulationOperationProps,
  type PolygonTriangulationOperationProps
} from './gpu-polygon-triangulation-operation';

/** Four-level polygon geometry, in the contributor's packed row formats. */
export type CPUPolygonTriangulationInput = {
  positions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
};

/** CPU output allocation, coordinate context, Earcut budget and monotone routing options. */
export type CPUPolygonTriangulationOptions = {
  indexCapacity: number;
  spatialContext?: GPUSpatialContext;
  /** Positive elementary-step budget per polygon; defaults to the contributor's 4M steps. */
  maximumWork?: number;
  /** Routing method, as on the contributor; defaults to 'monotone'. */
  largePolygonMethod?: 'monotone' | 'none';
  /** Vertex threshold for the first monotone route; defaults to 4096. */
  monotoneMinimumVertices?: number;
  /** Total raw vertex capacity for routed polygons; defaults to all positions. */
  monotoneMaximumVertices?: number;
};

/** Polygon slices, not a compact triangle stream; offsets count triangles, not index words. */
export type CPUPolygonTriangulationResult = {
  indices: Uint32Array;
  valid: Uint32Array;
  triangleCount: Uint32Array;
  triangleOffsets: Uint32Array;
};

type PolygonPoint = readonly [number, number];
type PolygonNode = {
  vertex: number;
  point: PolygonPoint;
  previous: PolygonNode;
  next: PolygonNode;
};

function comparePoints(first: PolygonPoint, second: PolygonPoint): boolean {
  return first[0] === second[0] && first[1] === second[1];
}

function containsPointInBox(first: PolygonPoint, point: PolygonPoint, last: PolygonPoint): boolean {
  return (
    point[0] >= Math.min(first[0], last[0]) &&
    point[0] <= Math.max(first[0], last[0]) &&
    point[1] >= Math.min(first[1], last[1]) &&
    point[1] <= Math.max(first[1], last[1])
  );
}

/** Domain implementation of the contributor's corrected Earcut walk with a per-polygon budget. */
class PolygonEarWalk {
  failed = false;
  filtered = false;
  readonly triangles: number[] = [];
  private workDone = 0;

  constructor(
    private readonly positions: Float32Array,
    private readonly maximumWork: number
  ) {}

  spendWork(): boolean {
    this.workDone++;
    if (this.workDone > this.maximumWork) this.failed = true;
    return !this.failed;
  }

  readPoint(vertex: number): PolygonPoint {
    return [this.positions[2 * vertex], this.positions[2 * vertex + 1]];
  }

  computeOrientation(first: PolygonPoint, second: PolygonPoint, third: PolygonPoint): number {
    const orientation = computePolygonOrientation(first, second, third);
    if (orientation === 2) {
      this.failed = true;
      return 0;
    }
    return orientation;
  }

  computeArea(first: PolygonNode, second: PolygonNode, third: PolygonNode): number {
    return -this.computeOrientation(first.point, second.point, third.point);
  }

  insertNode(vertex: number, last?: PolygonNode): PolygonNode {
    const node = {vertex, point: this.readPoint(vertex)} as PolygonNode;
    node.previous = last ?? node;
    node.next = last?.next ?? node;
    node.next.previous = node;
    if (last) last.next = node;
    return node;
  }

  removeNode(node: PolygonNode): void {
    node.next.previous = node.previous;
    node.previous.next = node.next;
  }

  createRing(start: number, end: number, counterClockwise: boolean): PolygonNode | undefined {
    if (start === end) return undefined;
    let extreme = start;
    for (let vertex = start + 1; vertex < end; vertex++) {
      const candidate = this.readPoint(vertex);
      const best = this.readPoint(extreme);
      if (candidate[0] < best[0] || (candidate[0] === best[0] && candidate[1] < best[1])) {
        extreme = vertex;
      }
    }
    const count = end - start;
    const center = this.readPoint(extreme);
    let previous = extreme;
    let next = extreme;
    for (let step = 1; step < count; step++) {
      if (comparePoints(this.readPoint(previous), center)) {
        previous = start + ((extreme - start + count - step) % count);
      }
      if (comparePoints(this.readPoint(next), center)) {
        next = start + ((extreme - start + step) % count);
      }
    }
    const winding = computePolygonOrientation(
      this.readPoint(previous),
      center,
      this.readPoint(next)
    );
    const forward = winding === 1 || winding === -1 ? (winding === 1) === counterClockwise : true;
    let last: PolygonNode | undefined;
    for (let step = 0; step < count; step++) {
      last = this.insertNode(forward ? start + step : end - 1 - step, last);
    }
    if (last && comparePoints(last.point, last.next.point)) {
      this.removeNode(last);
      last = last.next;
    }
    return last;
  }

  filterPoints(start: PolygonNode, end: PolygonNode = start): PolygonNode {
    const full = end === start;
    let node = start;
    do {
      if (!this.spendWork()) break;
      let again = false;
      const following = node.next;
      if (
        node !== following &&
        (comparePoints(node.point, following.point) ||
          this.computeArea(node.previous, node, following) === 0)
      ) {
        if (full || node === end) end = node.previous;
        this.filtered = true;
        this.removeNode(node);
        node = node.previous;
        again = true;
      } else if (full || node !== end) {
        node = following;
        again = !full;
      }
      if (!(again || node !== end)) break;
    } while (true);
    return end;
  }

  isEar(ear: PolygonNode): boolean {
    const first = ear.previous;
    const last = ear.next;
    let node = last.next;
    const low: PolygonPoint = [
      Math.min(first.point[0], ear.point[0], last.point[0]),
      Math.min(first.point[1], ear.point[1], last.point[1])
    ];
    const high: PolygonPoint = [
      Math.max(first.point[0], ear.point[0], last.point[0]),
      Math.max(first.point[1], ear.point[1], last.point[1])
    ];
    while (node !== first) {
      if (!this.spendWork()) return false;
      const point = node.point;
      if (
        containsPointInBox(low, point, high) &&
        !comparePoints(first.point, point) &&
        this.computeOrientation(point, last.point, first.point) >= 0 &&
        this.computeOrientation(point, first.point, ear.point) >= 0 &&
        this.computeOrientation(point, ear.point, last.point) >= 0 &&
        this.computeArea(node.previous, node, node.next) >= 0
      ) {
        return false;
      }
      node = node.next;
    }
    return true;
  }

  intersectSegments(
    first: PolygonPoint,
    second: PolygonPoint,
    third: PolygonPoint,
    fourth: PolygonPoint,
    includeBoundary: boolean
  ): boolean {
    const firstSide = this.computeOrientation(first, second, third);
    const secondSide = this.computeOrientation(first, second, fourth);
    const thirdSide = this.computeOrientation(third, fourth, first);
    const fourthSide = this.computeOrientation(third, fourth, second);
    if (firstSide * secondSide < 0 && thirdSide * fourthSide < 0) return true;
    return (
      includeBoundary &&
      ((firstSide === 0 && containsPointInBox(first, third, second)) ||
        (secondSide === 0 && containsPointInBox(first, fourth, second)) ||
        (thirdSide === 0 && containsPointInBox(third, first, fourth)) ||
        (fourthSide === 0 && containsPointInBox(third, second, fourth)))
    );
  }

  isLocallyInside(first: PolygonNode, second: PolygonNode): boolean {
    if (this.computeArea(first.previous, first, first.next) < 0) {
      return (
        this.computeArea(first, second, first.next) >= 0 &&
        this.computeArea(first, first.previous, second) >= 0
      );
    }
    return (
      this.computeArea(first, second, first.previous) < 0 ||
      this.computeArea(first, first.next, second) < 0
    );
  }

  isMiddleInside(first: PolygonNode, second: PolygonNode): boolean {
    const middle: PolygonPoint = [
      (first.point[0] + second.point[0]) / 2,
      (first.point[1] + second.point[1]) / 2
    ];
    let node = first;
    let inside = false;
    do {
      if (!this.spendWork()) return false;
      const start = node.point;
      const end = node.next.point;
      if (start[1] >= middle[1] !== end[1] >= middle[1]) {
        let side = this.computeOrientation(start, end, middle);
        if (side === 0) {
          side = end[0] !== start[0] ? -Math.sign(end[0] - start[0]) : Math.sign(end[1] - start[1]);
        }
        if ((end[1] > start[1] && side > 0) || (end[1] <= start[1] && side < 0)) inside = !inside;
      }
      node = node.next;
    } while (node !== first);
    return inside;
  }

  intersectsPolygon(first: PolygonNode, second: PolygonNode): boolean {
    let node = first;
    do {
      if (!this.spendWork()) return true;
      const following = node.next;
      if (
        node.vertex !== first.vertex &&
        following.vertex !== first.vertex &&
        node.vertex !== second.vertex &&
        following.vertex !== second.vertex &&
        this.intersectSegments(node.point, following.point, first.point, second.point, true)
      )
        return true;
      node = following;
    } while (node !== first);
    return false;
  }

  isValidDiagonal(first: PolygonNode, second: PolygonNode): boolean {
    const equal = comparePoints(first.point, second.point);
    const zeroLength =
      equal &&
      this.computeArea(first.previous, first, first.next) > 0 &&
      this.computeArea(second.previous, second, second.next) > 0;
    if (first.next.vertex === second.vertex || (equal && !zeroLength)) return false;
    if (
      !(
        zeroLength ||
        (this.isLocallyInside(first, second) &&
          this.isLocallyInside(second, first) &&
          (this.computeArea(first.previous, first, second.previous) !== 0 ||
            this.computeArea(first, second.previous, second) !== 0))
      )
    )
      return false;
    return (
      !this.intersectsPolygon(first, second) && (zeroLength || this.isMiddleInside(first, second))
    );
  }

  splitPolygon(first: PolygonNode, second: PolygonNode): PolygonNode {
    const firstCopy = this.insertNode(first.vertex);
    const secondCopy = this.insertNode(second.vertex);
    const after = first.next;
    const before = second.previous;
    first.next = second;
    second.previous = first;
    firstCopy.next = after;
    after.previous = firstCopy;
    secondCopy.next = firstCopy;
    firstCopy.previous = secondCopy;
    before.next = secondCopy;
    secondCopy.previous = before;
    return secondCopy;
  }

  hasLocalIntersection(start: PolygonNode): boolean {
    let node = start;
    do {
      if (!this.spendWork()) return false;
      const before = node.previous;
      const after = node.next.next;
      if (
        this.intersectSegments(before.point, node.point, node.next.point, after.point, false) &&
        this.isLocallyInside(before, after) &&
        this.isLocallyInside(after, before)
      )
        return true;
      node = node.next;
    } while (node !== start);
    return false;
  }

  findSplit(start: PolygonNode): [PolygonNode, PolygonNode] | undefined {
    let first = start;
    do {
      let second = first.next.next;
      while (second !== first.previous) {
        if (!this.spendWork()) return undefined;
        if (first.vertex !== second.vertex && this.isValidDiagonal(first, second))
          return [first, second];
        second = second.next;
      }
      first = first.next;
    } while (first !== start);
    return undefined;
  }

  clipEars(first: PolygonNode): void {
    let ear = first;
    const pending: PolygonNode[] = [];
    do {
      let stop = ear;
      let cured = false;
      while (!this.failed) {
        if (!this.spendWork()) break;
        const before = ear.previous;
        const after = ear.next;
        if (before === after) break;
        if (this.computeArea(before, ear, after) < 0 && this.isEar(ear)) {
          this.triangles.push(before.vertex, ear.vertex, after.vertex);
          this.removeNode(ear);
          ear = after;
          stop = after;
          continue;
        }
        ear = after;
        if (ear !== stop) continue;
        this.filtered = false;
        ear = this.filterPoints(ear);
        if (this.filtered) {
          stop = ear;
          continue;
        }
        if (!cured) {
          if (this.hasLocalIntersection(ear)) this.failed = true;
          stop = ear;
          cured = true;
          continue;
        }
        const diagonal = this.findSplit(ear);
        if (!diagonal) {
          this.failed = true;
          break;
        }
        const other = this.splitPolygon(diagonal[0], diagonal[1]);
        const kept = this.filterPoints(diagonal[0], diagonal[0].next);
        pending.push(this.filterPoints(other, other.next));
        ear = kept;
        stop = kept;
        cured = false;
      }
      if (this.failed || pending.length === 0) break;
      ear = pending.pop()!;
    } while (true);
  }

  findNextDistinct(node: PolygonNode): PolygonNode {
    let next = node.next;
    while (next !== node && comparePoints(next.point, node.point)) {
      if (!this.spendWork()) break;
      next = next.next;
    }
    return next;
  }

  isHoleBefore(first: PolygonNode, second: PolygonNode): boolean {
    if (first.point[0] !== second.point[0]) return first.point[0] < second.point[0];
    if (first.point[1] !== second.point[1]) return first.point[1] < second.point[1];
    return (
      this.computeOrientation(
        first.point,
        this.findNextDistinct(first).point,
        this.findNextDistinct(second).point
      ) > 0
    );
  }

  containsSector(first: PolygonNode, second: PolygonNode): boolean {
    return (
      this.computeArea(first.previous, first, second.previous) < 0 &&
      this.computeArea(second.next, first, first.next) < 0
    );
  }

  crossesRowRightOf(
    start: PolygonPoint,
    end: PolygonPoint,
    bestStart: PolygonPoint,
    bestEnd: PolygonPoint,
    height: number,
    crossingIsRight: boolean
  ): boolean {
    const startSide = this.computeOrientation(bestStart, bestEnd, start);
    const endSide = this.computeOrientation(bestStart, bestEnd, end);
    if (startSide >= 0 && endSide >= 0) {
      return (
        (startSide > 0 && endSide > 0) ||
        (startSide > 0 && end[1] !== height) ||
        (endSide > 0 && start[1] !== height)
      );
    }
    if (startSide <= 0 && endSide <= 0) return false;
    const bestStartSide = this.computeOrientation(start, end, bestStart);
    const bestEndSide = this.computeOrientation(start, end, bestEnd);
    if (bestStartSide <= 0 && bestEndSide <= 0) {
      return (
        (bestStartSide < 0 && bestEndSide < 0) ||
        (bestStartSide < 0 && bestEnd[1] !== height) ||
        (bestEndSide < 0 && bestStart[1] !== height)
      );
    }
    if (bestStartSide >= 0 && bestEndSide >= 0) return false;
    return crossingIsRight;
  }

  findEndOnRow(node: PolygonNode, height: number): PolygonNode | undefined {
    if (node.point[1] === height) return node;
    if (node.next.point[1] === height) return node.next;
    return undefined;
  }

  findHoleBridge(hole: PolygonNode, outer: PolygonNode): PolygonNode | undefined {
    const point = hole.point;
    let node = outer;
    let crossing = -Infinity;
    let candidate: PolygonNode | undefined;
    let hitNode: PolygonNode | undefined;
    let hitStart = point;
    let hitEnd = point;
    if (comparePoints(point, node.point)) return node;
    do {
      if (!this.spendWork()) return undefined;
      const following = node.next;
      const start = node.point;
      const end = following.point;
      if (comparePoints(point, end)) return following;
      if (
        containsPointInBox(start, point, end) &&
        this.computeOrientation(start, end, point) === 0
      ) {
        return this.insertNode(hole.vertex, node);
      }
      if (
        point[1] <= start[1] &&
        point[1] >= end[1] &&
        end[1] !== start[1] &&
        this.computeOrientation(start, end, point) > 0
      ) {
        const horizontal =
          start[0] + ((point[1] - start[1]) * (end[0] - start[0])) / (end[1] - start[1]);
        let nearer =
          !candidate ||
          this.crossesRowRightOf(start, end, hitStart, hitEnd, point[1], horizontal > crossing);
        if (!nearer && hitNode) {
          const hitRowEnd = this.findEndOnRow(hitNode, point[1]);
          if (
            hitRowEnd &&
            !this.isLocallyInside(hitRowEnd, hole) &&
            this.computeOrientation(start, end, hitRowEnd.point) === 0
          ) {
            const rowEnd = this.findEndOnRow(node, point[1]);
            nearer = !rowEnd || this.isLocallyInside(rowEnd, hole);
          }
        }
        if (nearer) {
          crossing = horizontal;
          hitStart = start;
          hitEnd = end;
          hitNode = node;
          candidate = start[0] < end[0] ? node : following;
        }
      }
      node = following;
    } while (node !== outer);
    if (!candidate) return undefined;
    const initialPoint = candidate.point;
    const stop = candidate;
    const above = point[1] < initialPoint[1];
    let hasCandidate = false;
    node = candidate;
    do {
      if (!this.spendWork()) return undefined;
      const current = node.point;
      if (point[0] >= current[0] && current[0] >= initialPoint[0] && point[0] !== current[0]) {
        const onRaySide = above ? current[1] >= point[1] : current[1] <= point[1];
        const candidateSide = above
          ? this.computeOrientation(point, initialPoint, current)
          : this.computeOrientation(initialPoint, point, current);
        if (
          onRaySide &&
          candidateSide >= 0 &&
          this.computeOrientation(hitStart, hitEnd, current) >= 0 &&
          this.isLocallyInside(node, hole)
        ) {
          const best = candidate.point;
          const turn = above
            ? this.computeOrientation(point, best, current)
            : this.computeOrientation(point, current, best);
          if (
            !hasCandidate ||
            turn > 0 ||
            (turn === 0 &&
              (current[0] > best[0] ||
                (current[0] === best[0] && this.containsSector(candidate, node))))
          ) {
            candidate = node;
            hasCandidate = true;
          }
        }
      }
      node = node.next;
    } while (node !== stop);
    return candidate;
  }

  eliminateHoles(outer: PolygonNode, holes: PolygonNode[]): PolygonNode {
    const queue: (PolygonNode | undefined)[] = holes.map(start => {
      let leftmost = start;
      let node = start;
      do {
        if (!this.spendWork()) return leftmost;
        if (
          node.point[0] < leftmost.point[0] ||
          (node.point[0] === leftmost.point[0] && node.point[1] < leftmost.point[1])
        )
          leftmost = node;
        node = node.next;
      } while (node !== start);
      return leftmost;
    });
    // Stable selection order matches the shader even for coincident holes with equal slopes.
    for (let round = 0; round < queue.length && !this.failed; round++) {
      let best = -1;
      for (let slot = 0; slot < queue.length; slot++) {
        if (!this.spendWork()) break;
        const candidate = queue[slot];
        if (candidate && (best < 0 || this.isHoleBefore(candidate, queue[best]!))) best = slot;
      }
      if (this.failed || best < 0) break;
      const hole = queue[best]!;
      // The shader scans the original queue length on every round, including vacated slots.
      queue[best] = undefined;
      const bridge = this.findHoleBridge(hole, outer);
      if (!bridge) {
        this.failed = true;
        break;
      }
      const reverse = this.splitPolygon(bridge, hole);
      this.filterPoints(reverse, reverse.next);
      outer = this.filterPoints(bridge, bridge.next);
    }
    return this.filterPoints(outer);
  }
}

function validatePolygonOffsets(name: string, offsets: Uint32Array, limit: number): void {
  if (offsets.length === 0) throw new Error(`${name} requires a terminal entry`);
  let previous = 0;
  for (const offset of offsets) {
    if (offset < previous || offset > limit) {
      throw new Error(`${name} must be nondecreasing and in [0, ${limit}]`);
    }
    previous = offset;
  }
}

/**
 * Triangulates each live shell and its holes with the contributor's Earcut/monotone routing.
 * Ring classification, winding, hole order, bridge ties, ear order, failure padding and polygon
 * slice offsets follow the current kernels, including uint32 layout prefixes and signed int32
 * slot arithmetic for clamped ring ends. Degenerate shells are empty, including their holes.
 *
 * What differs from the GPU contributor:
 * - Earcut honors maximumWork (default 4M steps) per polygon, with no GPU scratch limit.
 *   This reference uses un-hashed Earcut; its work count can differ from GPU hashing.
 *   The selected monotone route and retry honor the same raw-vertex capacity and threshold.
 * - Midpoint placement and crossing arithmetic use float64. Exact orientation has an arbitrary
 *   precision fallback, so finite exponent spans that the GPU cannot certify are accepted.
 *   Diagonal choices at float32 midpoint rounding boundaries can differ.
 * - Non-finite coordinates in live rings fail the polygon; subnormal coordinates are treated
 *   as their exact values. Invalid polygons are not repaired; no tiling is promised for them.
 * - Decreasing offsets and out-of-range feature/polygon offsets throw with the argument name;
 *   the GPU does not validate them. Ring ends beyond positions are clamped for reads, while
 *   slice capacities use the original offsets, as on the GPU. Output allocation limits remain.
 *
 * @throws Error naming positions, featureOffsets, polygonOffsets, ringOffsets or indexCapacity
 * for inconsistent arrays or an invalid allocation.
 */
export function computePolygonTriangulationOnCPU(
  input: CPUPolygonTriangulationInput,
  options: CPUPolygonTriangulationOptions
): CPUPolygonTriangulationResult {
  const {positions, featureOffsets, polygonOffsets, ringOffsets} = input;
  if (positions.length % 2 !== 0) throw new Error('positions must hold float32x2 rows');
  const vertexCount = positions.length / 2;
  validatePolygonOffsets('ringOffsets', ringOffsets, 0xffffffff);
  validatePolygonOffsets('polygonOffsets', polygonOffsets, ringOffsets.length - 1);
  validatePolygonOffsets('featureOffsets', featureOffsets, polygonOffsets.length - 1);
  const polygonCount = polygonOffsets.length - 1;
  const maximumWork = options.maximumWork ?? GPU_POLYGON_TRIANGULATION_DEFAULT_MAXIMUM_WORK;
  validateCapacity('polygon-triangulation', 'maximumWork', maximumWork);
  const monotone = (options.largePolygonMethod ?? 'monotone') === 'monotone';
  validateRequirement(
    'polygon-triangulation',
    options.largePolygonMethod === undefined ||
      ['none', 'monotone'].includes(options.largePolygonMethod),
    "largePolygonMethod must be 'monotone' or 'none'"
  );
  const monotoneMinimumVertices =
    options.monotoneMinimumVertices ?? GPU_POLYGON_TRIANGULATION_DEFAULT_MONOTONE_MINIMUM_VERTICES;
  const monotoneCapacity = options.monotoneMaximumVertices ?? Math.max(1, vertexCount);
  validateCapacity('polygon-triangulation', 'monotoneMinimumVertices', monotoneMinimumVertices);
  validateCapacity('polygon-triangulation', 'monotoneMaximumVertices', monotoneCapacity);
  validateCapacity('polygon-triangulation', 'indexCapacity', options.indexCapacity, 0);
  const minimumIndexCount = Math.max(
    0,
    3 * (vertexCount + 2 * (ringOffsets.length - 1) - 4 * polygonCount)
  );
  validateRequirement(
    'polygon-triangulation',
    options.indexCapacity >= minimumIndexCount,
    `indexCapacity must be at least ${minimumIndexCount}`
  );
  resolveGPUSpatialContext('polygon-triangulation', options.spatialContext);
  const result: CPUPolygonTriangulationResult = {
    indices: new Uint32Array(options.indexCapacity),
    valid: new Uint32Array(polygonCount),
    triangleCount: new Uint32Array(polygonCount),
    triangleOffsets: new Uint32Array(polygonCount + 1)
  };
  const liveRings = new Uint32Array(ringOffsets.length - 1);
  const readPoint = (vertex: number): PolygonPoint => [
    positions[2 * vertex],
    positions[2 * vertex + 1]
  ];
  for (let ring = 0; ring < liveRings.length; ring++) {
    const start = ringOffsets[ring];
    const end = Math.min(ringOffsets[ring + 1], vertexCount);
    for (let vertex = start + 1; vertex + 1 < end; vertex++) {
      if (
        computePolygonOrientation(readPoint(start), readPoint(vertex), readPoint(vertex + 1)) !== 0
      ) {
        liveRings[ring] = 1;
        break;
      }
    }
  }
  let routedVertexCount = 0;
  for (let polygon = 0; polygon < polygonCount; polygon++) {
    const firstRing = polygonOffsets[polygon];
    const lastRing = polygonOffsets[polygon + 1];
    const live = firstRing < lastRing && liveRings[firstRing] !== 0;
    let triangleCapacity = 0;
    let liveVertexCount = 0;
    if (live) {
      for (let ring = firstRing; ring < lastRing; ring++) {
        if (liveRings[ring]) {
          const count = ringOffsets[ring + 1] - ringOffsets[ring];
          liveVertexCount += count;
          triangleCapacity += count + 2;
        }
      }
      triangleCapacity -= 4;
    }
    const triangleStart = result.triangleOffsets[polygon];
    // The capacity kernel and its uint32 exclusive scan both wrap modulo 2^32.
    const triangleEnd = (triangleStart + (triangleCapacity >>> 0)) >>> 0;
    result.triangleOffsets[polygon + 1] = triangleEnd;
    if (!live) {
      result.valid[polygon] = 1;
      continue;
    }
    const firstVertex = ringOffsets[firstRing];
    // Match baseSlotOf, slotEnd and slotCount in the Earcut kernel. A clamped ring end can
    // overflow the signed index product even with a tiny positions/output buffer. In that
    // case the failure padding must use the wrapped slot count, not the output capacity.
    const slotStart = Math.imul(3, triangleStart);
    const slotLimit = Math.imul(3, triangleEnd);
    const slotEnd = Math.min(slotLimit, options.indexCapacity | 0);
    const slotCount = Math.max((slotEnd - slotStart) | 0, 0);
    const indexStart = Math.max(slotStart, 0);
    const availableTriangles = Math.floor(slotCount / 3);
    let triangles: number[] = [];
    if (slotLimit <= (options.indexCapacity | 0) && availableTriangles > 0) {
      const routeFirst = monotone && liveVertexCount >= monotoneMinimumVertices;
      const attemptEarcut = () => {
        const walk = new PolygonEarWalk(positions, maximumWork);
        // Non-finite predicates cannot support a topology decision.
        for (let ring = firstRing; ring < lastRing; ring++) {
          if (!liveRings[ring]) continue;
          for (
            let vertex = ringOffsets[ring];
            vertex < Math.min(ringOffsets[ring + 1], vertexCount);
            vertex++
          ) {
            if (!readPoint(vertex).every(Number.isFinite)) walk.failed = true;
          }
        }
        let outer = walk.createRing(
          firstVertex,
          Math.min(ringOffsets[firstRing + 1], vertexCount),
          true
        );
        if (!outer || outer.next === outer.previous) walk.failed = true;
        if (outer && !walk.failed) {
          const holes: PolygonNode[] = [];
          for (let ring = firstRing + 1; ring < lastRing; ring++) {
            if (!liveRings[ring]) continue;
            const hole = walk.createRing(
              ringOffsets[ring],
              Math.min(ringOffsets[ring + 1], vertexCount),
              false
            );
            if (hole) holes.push(hole);
          }
          if (holes.length) outer = walk.eliminateHoles(outer, holes);
          if (!walk.failed) walk.clipEars(outer);
        }
        if (!walk.failed && walk.triangles.length <= availableTriangles * 3) {
          triangles = walk.triangles;
          result.valid[polygon] = 1;
          result.triangleCount[polygon] = triangles.length / 3;
        }
        return result.valid[polygon] === 0;
      };
      // Large polygons go straight to monotone; smaller polygons retry only after Earcut fails.
      const needsMonotone = routeFirst || attemptEarcut();
      const rawVertexCount = Math.max(
        0,
        Math.min(ringOffsets[lastRing], vertexCount) - Math.min(firstVertex, vertexCount)
      );
      if (monotone && needsMonotone) {
        // The GPU scans all routed raw counts before rejecting over-capacity polygons. A
        // rejected polygon still occupies its prefix, so later polygons cannot reuse it.
        routedVertexCount += rawVertexCount;
        if (routedVertexCount <= monotoneCapacity) {
          const rings: PolygonPoint[][] = [];
          const rows: number[] = [];
          for (let ring = firstRing; ring < lastRing; ring++) {
            if (!liveRings[ring]) continue;
            const points: PolygonPoint[] = [];
            for (
              let vertex = ringOffsets[ring];
              vertex < Math.min(ringOffsets[ring + 1], vertexCount);
              vertex++
            ) {
              points.push(readPoint(vertex));
              rows.push(vertex);
            }
            rings.push(points);
          }
          if (rings.every(ring => ring.every(point => point.every(Number.isFinite)))) {
            const rescued = triangulateMonotoneOnCPU(rings);
            if (rescued.valid && rescued.triangles.length <= availableTriangles) {
              triangles = rescued.triangles.flatMap(triangle => triangle.map(row => rows[row]));
              result.valid[polygon] = 1;
              result.triangleCount[polygon] = triangles.length / 3;
            }
          }
        }
      }
      if (routeFirst && result.valid[polygon] === 0) attemptEarcut();
    }
    // Keep the signed slot origin; typed-array indexed writes clip to this output allocation.
    for (let word = 0; word < triangles.length; word++) {
      result.indices[indexStart + word] = triangles[word];
    }
    // The GPU only writes whole triangle slots; any trailing one/two words remain zero.
    result.indices.fill(
      firstVertex,
      indexStart + triangles.length,
      indexStart + availableTriangles * 3
    );
  }
  return result;
}

/** CPU twin with budgeted un-hashed Earcut, monotone rescue and the same polygon slices. */
export class CPUPolygonTriangulationOperation extends GPUCPUOperation {
  constructor(props: PolygonTriangulationOperationProps) {
    props = snapshotPolygonTriangulationOperationProps(props);
    const operation = new GPUPolygonTriangulationOperation(props);
    const {id} = operation;
    validateCapacity(id, 'polygonCount', props.polygonCount, 1);
    validateCapacity(id, 'indexCapacity', props.indexCapacity, 1);
    resolveGPUSpatialContext(id, props.spatialContext);
    super({
      id,
      inputs: operation.inputs,
      outputs: Object.fromEntries(
        Object.values(operation.outputs).map(output => [
          output.value,
          gpuGraph.vector(output.format, {length: output.length})
        ])
      ),
      execute({inputs}) {
        const polygonOffsets = getCPUOperationUint32Rows(id, inputs, 'polygonOffsets');
        if (polygonOffsets.length !== props.polygonCount + 1) {
          throw new Error(`${id} polygonOffsets must hold polygonCount + 1 rows`);
        }
        const result = computePolygonTriangulationOnCPU(
          {
            positions: getCPUOperationFloat32Rows(id, inputs, 'positions', 'float32x2'),
            featureOffsets: getCPUOperationUint32Rows(id, inputs, 'featureOffsets'),
            polygonOffsets,
            ringOffsets: getCPUOperationUint32Rows(id, inputs, 'ringOffsets')
          },
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUPolygonTriangulationResult]
          ])
        );
      }
    });
  }
}
