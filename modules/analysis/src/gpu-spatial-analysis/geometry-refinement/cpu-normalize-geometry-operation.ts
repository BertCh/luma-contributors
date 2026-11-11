// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  compareCanonicalPoints,
  computeCanonicalWinding,
  type CanonicalPoint
} from './cpu-canonical-geometry';
import {GPU_CANONICAL_GEOMETRY_TYPE} from './gpu-normalize-geometry';
import {
  GPUNormalizeGeometryOperation,
  type NormalizeGeometryOperationProps
} from './gpu-normalize-geometry-operation';

/** The contributor's packed postorder tree: [kind, first, count, stableSourceId] per node. */
export type CPUNormalizeGeometryInput = {
  positions: Float32Array;
  nodes: Uint32Array;
  children: Uint32Array;
};
/** Same positions/children shapes, and one status word per node. */
export type CPUNormalizeGeometryResult = {
  positions: Float32Array;
  children: Uint32Array;
  status: Uint32Array;
};

function isLeaf(kind: number): boolean {
  return (
    kind === GPU_CANONICAL_GEOMETRY_TYPE.point ||
    kind === GPU_CANONICAL_GEOMETRY_TYPE.lineString ||
    kind === GPU_CANONICAL_GEOMETRY_TYPE.linearRing
  );
}

function readPoint(positions: Float32Array, vertex: number): CanonicalPoint {
  return [positions[vertex * 2], positions[vertex * 2 + 1]];
}

function compareNodes(
  input: CPUNormalizeGeometryInput,
  result: CPUNormalizeGeometryResult,
  firstRoot: number,
  secondRoot: number
): number {
  // Explicit comparison stack removes the GPU's 64-entry subtree-comparison stack limit.
  const pending: ({first: number; second: number} | {comparison: number})[] = [
    {first: firstRoot, second: secondRoot}
  ];
  while (pending.length > 0) {
    const item = pending.pop()!;
    if ('comparison' in item) {
      if (item.comparison !== 0) return item.comparison;
      continue;
    }
    const firstKind = input.nodes[item.first * 4];
    const secondKind = input.nodes[item.second * 4];
    if (firstKind !== secondKind) return firstKind - secondKind;
    const firstStart = input.nodes[item.first * 4 + 1];
    const secondStart = input.nodes[item.second * 4 + 1];
    const firstCount = input.nodes[item.first * 4 + 2];
    const secondCount = input.nodes[item.second * 4 + 2];
    if (isLeaf(firstKind)) {
      if (firstCount !== secondCount) return firstCount - secondCount;
      for (let vertex = 0; vertex < firstCount; vertex++) {
        const comparison = compareCanonicalPoints(
          readPoint(result.positions, firstStart + vertex),
          readPoint(result.positions, secondStart + vertex)
        );
        if (comparison !== 0) return comparison;
      }
    } else {
      pending.push({comparison: firstCount - secondCount});
      for (let child = Math.min(firstCount, secondCount) - 1; child >= 0; child--)
        pending.push({
          first: result.children[firstStart + child],
          second: result.children[secondStart + child]
        });
    }
  }
  return 0;
}

/**
 * Normalizes finite leaves and stably orders collections in descending (kind, canonical contents)
 * order. Polygon exteriors stay first; holes sort after them. Leaf comparison is vertex-count
 * first; collection comparison is child-sequence first and count last. stableSourceId is copied
 * only through the unchanged nodes input and does not break ties, matching the kernel.
 *
 * What differs from the GPU contributor:
 * - Postorder evaluation and an explicit comparison stack replace dependency rounds. There is
 *   no nesting cap, so CPU status 2 never means depth exhaustion. The operation accepts but
 *   ignores maximumNestingDepth, including values over 64.
 * - Winding is an unbounded exact BigInt predicate on float32 coordinates, so uncertain GPU
 *   orientations succeed on the CPU. Finite subnormals are supported.
 * - Overlapping leaf vertex ranges or collection child-slot ranges throw an Error naming nodes;
 *   the GPU has conflicting writes and does not define their result. Nonfinite leaves retain
 *   their input coordinates with status 1; bad kind/range/child order and invalid parents get 2.
 * - All other output is copied exactly (no arithmetic rounding). Signed zero is preserved.
 *   Closed lines use the shell winding rule; zero-winding rings retain direction; duplicate
 *   minimum vertices retain the first minimum, as in the current kernel.
 *
 * @throws If arrays are not packed or node-owned ranges overlap. Individual invalid node ranges
 * and child references are reported with status 2 instead of throwing, as on the GPU.
 */
export function computeNormalizeGeometryOnCPU(
  input: CPUNormalizeGeometryInput
): CPUNormalizeGeometryResult {
  if (input.positions.length % 2 !== 0) throw new Error('positions must contain float32x2 rows');
  if (input.nodes.length % 4 !== 0) throw new Error('nodes must contain uint32x4 rows');
  const rowCount = input.positions.length / 2;
  const nodeCount = input.nodes.length / 4;
  const result: CPUNormalizeGeometryResult = {
    positions: input.positions.slice(),
    children: input.children.slice(),
    status: new Uint32Array(nodeCount)
  };
  const holeFlags = new Uint8Array(nodeCount);
  const leafRanges: {first: number; last: number}[] = [];
  const childRanges: {first: number; last: number}[] = [];
  for (let node = 0; node < nodeCount; node++) {
    const kind = input.nodes[node * 4];
    const first = input.nodes[node * 4 + 1];
    const count = input.nodes[node * 4 + 2];
    if (kind > GPU_CANONICAL_GEOMETRY_TYPE.collection) continue;
    const limit = isLeaf(kind) ? rowCount : input.children.length;
    if (
      first <= limit &&
      count <= limit - first &&
      count > 0 &&
      (kind !== GPU_CANONICAL_GEOMETRY_TYPE.point || count <= 1)
    )
      (isLeaf(kind) ? leafRanges : childRanges).push({first, last: first + count});
    if (kind === GPU_CANONICAL_GEOMETRY_TYPE.polygon) {
      const length = Math.min(
        count,
        input.children.length - Math.min(first, input.children.length)
      );
      for (let child = 1; child < length; child++) {
        const hole = input.children[first + child];
        if (hole < node) holeFlags[hole] = 1;
      }
    }
  }
  for (const ranges of [leafRanges, childRanges]) {
    ranges.sort((first, second) => first.first - second.first);
    for (let index = 1; index < ranges.length; index++)
      if (ranges[index].first < ranges[index - 1].last)
        throw new Error('nodes must own disjoint leaf vertex and collection child-slot ranges');
  }
  for (let node = 0; node < nodeCount; node++) {
    const kind = input.nodes[node * 4];
    const first = input.nodes[node * 4 + 1];
    const count = input.nodes[node * 4 + 2];
    if (kind > GPU_CANONICAL_GEOMETRY_TYPE.collection) {
      result.status[node] = 2;
      continue;
    }
    if (isLeaf(kind)) {
      if (
        first > rowCount ||
        count > rowCount - first ||
        (kind === GPU_CANONICAL_GEOMETRY_TYPE.point && count > 1)
      ) {
        result.status[node] = 2;
        continue;
      }
      const points = Array.from({length: count}, (_, index) =>
        readPoint(input.positions, first + index)
      );
      if (points.some(point => !point.every(Number.isFinite))) {
        result.status[node] = 1;
        continue;
      }
      if (count < 2) continue;
      const closed = compareCanonicalPoints(points[0], points[count - 1]) === 0;
      if (closed || kind === GPU_CANONICAL_GEOMETRY_TYPE.linearRing) {
        const length = count - Number(closed);
        const unique = points.slice(0, length);
        let rotation = 0;
        for (let vertex = 1; vertex < length; vertex++)
          if (compareCanonicalPoints(unique[vertex], unique[rotation]) < 0) rotation = vertex;
        const winding = computeCanonicalWinding(unique);
        const hole = kind === GPU_CANONICAL_GEOMETRY_TYPE.linearRing && holeFlags[node] !== 0;
        const reverse = hole ? winding < 0 : winding > 0;
        for (let vertex = 0; vertex < count; vertex++)
          result.positions.set(
            unique[(rotation + (reverse ? length - (vertex % length) : vertex % length)) % length],
            (first + vertex) * 2
          );
      } else {
        let reverse = false;
        for (let vertex = 0; vertex < count; vertex++) {
          const comparison = compareCanonicalPoints(points[vertex], points[count - 1 - vertex]);
          if (comparison !== 0) {
            reverse = comparison > 0;
            break;
          }
        }
        if (reverse)
          for (let vertex = 0; vertex < count; vertex++)
            result.positions.set(points[count - 1 - vertex], (first + vertex) * 2);
      }
    } else {
      if (first > input.children.length || count > input.children.length - first) {
        result.status[node] = 2;
        continue;
      }
      const children = Array.from(input.children.subarray(first, first + count));
      if (children.some(child => child >= node || result.status[child] !== 0)) {
        result.status[node] = 2;
        continue;
      }
      const start = kind === GPU_CANONICAL_GEOMETRY_TYPE.polygon ? 1 : 0;
      const ordered = children
        .slice(start)
        .sort((firstChild, secondChild) => -compareNodes(input, result, firstChild, secondChild));
      result.children.set(ordered, first + Math.min(start, count));
    }
  }
  return result;
}

/** CPU tree normalization without GPU dependency-round or comparison-stack limits. */
export class CPUNormalizeGeometryOperation extends GPUCPUOperation {
  constructor(props: NormalizeGeometryOperationProps) {
    const id = props.id ?? 'normalize-geometry';
    for (const name of ['rowCount', 'nodeCount'] as const)
      validateCapacity(id, name, props[name], 0);
    validateCapacity(id, 'childCount', props.childCount, 0);
    validateCapacity(id, 'maximumNestingDepth', props.maximumNestingDepth ?? 64);
    const operation = new GPUNormalizeGeometryOperation(props);
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
        const input = {
          positions: getCPUOperationFloat32Rows(id, inputs, 'positions', 'float32x2'),
          nodes: getCPUOperationUint32Rows(id, inputs, 'nodes', 'uint32x4'),
          children: getCPUOperationUint32Rows(id, inputs, 'children')
        };
        for (const [name, expected] of [
          ['positions', props.rowCount * 2],
          ['nodes', props.nodeCount * 4],
          ['children', props.childCount]
        ] as const)
          if (input[name].length !== expected)
            throw new Error(`${id} ${name} length must equal ${expected}`);
        const result = computeNormalizeGeometryOnCPU(input);
        return {
          [props.output.positions]: result.positions,
          [props.output.children]: result.children,
          [props.output.status]: result.status
        };
      }
    });
  }
}
