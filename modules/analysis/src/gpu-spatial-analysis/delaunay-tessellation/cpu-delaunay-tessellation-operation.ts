// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotDelaunayTessellationOperationProps} from './delaunay-construction-snapshot';
import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  computeTriangulationOrientation,
  computeTriangulationIncircle
} from '../../analysis-core/polygon-triangulation/cpu-triangulation-predicates';
import {
  GPUDelaunayTessellationOperation,
  type DelaunayTessellationOperationProps
} from './gpu-delaunay-tessellation-operation';

export type CPUDelaunayTessellationInput = {positions: Float32Array; groupIds?: Uint32Array};
export type CPUDelaunayTessellationOptions = {
  capacity: number;
  groupCount?: number;
  spatialContext?: GPUSpatialContext;
};
export type CPUDelaunayTessellationResult = {
  triangles: Uint32Array;
  count: Uint32Array;
  requiredCount: Uint32Array;
  overflow: Uint32Array;
  invalidCount: Uint32Array;
  duplicateCount: Uint32Array;
  converged: Uint32Array;
  iterationCount: Uint32Array;
  iterationLimitReached: Uint32Array;
  candidateOverflow: Uint32Array;
  uncertainCount: Uint32Array;
  approximation: Uint32Array;
  groupTriangleCounts: Uint32Array;
  groupTriangleOffsets: Uint32Array;
  groupConverged: Uint32Array;
};

type Triangle = [number, number, number];

function computeGroupTriangles(positions: Float32Array, sites: number[]): Triangle[] {
  const pointCount = positions.length / 2;
  const readPoint = (row: number) => [positions[2 * row], positions[2 * row + 1]];
  const orientRealSuper = (real: number, superRow: number, site: number): number => {
    const horizontal = Math.sign(positions[2 * site] - positions[2 * real]);
    const vertical = Math.sign(positions[2 * site + 1] - positions[2 * real + 1]);
    if (superRow === pointCount) return vertical || horizontal;
    if (superRow === pointCount + 1) return -horizontal || vertical;
    return -vertical || horizontal;
  };
  const orientSymbolic = (first: number, second: number, site: number): number => {
    if (first < pointCount && second < pointCount)
      return computeTriangulationOrientation(readPoint(first), readPoint(second), readPoint(site));
    if (first < pointCount) return orientRealSuper(first, second, site);
    if (second < pointCount) return -orientRealSuper(second, first, site);
    return second - pointCount === (first - pointCount + 1) % 3 ? 1 : -1;
  };
  const isInside = (triangle: Triangle, site: number): boolean => {
    let farthest = -1;
    for (let slot = 0; slot < 3; slot++) {
      if (triangle[slot] >= pointCount && (farthest < 0 || triangle[slot] > triangle[farthest]))
        farthest = slot;
    }
    if (farthest < 0) {
      const circle = computeTriangulationIncircle(
        readPoint(triangle[0]),
        readPoint(triangle[1]),
        readPoint(triangle[2]),
        readPoint(site)
      );
      if (circle !== 0) return circle > 0;
      // The smallest row has the dominant positive lift. Its determinant cofactor decides.
      const rows = [...triangle, site];
      let dominant = 0;
      for (let slot = 1; slot < 4; slot++) if (rows[slot] < rows[dominant]) dominant = slot;
      const others = rows.filter((_, slot) => slot !== dominant);
      return (dominant % 2 === 0 ? 1 : -1) * orientSymbolic(others[0], others[1], others[2]) > 0;
    }
    const first = triangle[(farthest + 1) % 3];
    const second = triangle[(farthest + 2) % 3];
    const side = orientSymbolic(first, second, site);
    if (side !== 0) return side > 0;
    const start = readPoint(first);
    const end = readPoint(second);
    const point = readPoint(site);
    const axis = start[0] !== end[0] ? 0 : 1;
    return (
      point[axis] > Math.min(start[axis], end[axis]) &&
      point[axis] < Math.max(start[axis], end[axis])
    );
  };
  let triangles: Triangle[] = [[pointCount, pointCount + 1, pointCount + 2]];
  for (const site of sites) {
    const survivors: Triangle[] = [];
    const boundary = new Map<string, [number, number]>();
    for (const triangle of triangles) {
      if (!isInside(triangle, site)) {
        survivors.push(triangle);
        continue;
      }
      for (let edge = 0; edge < 3; edge++) {
        const first = triangle[edge];
        const second = triangle[(edge + 1) % 3];
        const key = first < second ? `${first},${second}` : `${second},${first}`;
        if (boundary.has(key)) boundary.delete(key);
        else boundary.set(key, [first, second]);
      }
    }
    for (const [first, second] of boundary.values()) {
      const side = orientSymbolic(first, second, site);
      if (side !== 0) survivors.push(side > 0 ? [first, second, site] : [second, first, site]);
    }
    triangles = survivors;
  }
  return triangles
    .filter(triangle => triangle.every(vertex => vertex < pointCount))
    .map(triangle => {
      const minimum = triangle.indexOf(Math.min(...triangle));
      return [
        triangle[minimum],
        triangle[(minimum + 1) % 3],
        triangle[(minimum + 2) % 3]
      ] as Triangle;
    })
    .sort((first, second) => first[0] - second[0] || first[1] - second[1]);
}

/**
 * Exact Delaunay construction by sequential Bowyer-Watson cavities with symbolic vertices at
 * infinity and the contributor's row-order lifted perturbation. No finite super triangle loses
 * thin hull triangles. Duplicates belong to the first row; triangles are canonical and grouped.
 *
 * What differs from the GPU contributor:
 * - Sequential cavities replace parallel insertion/Lawson rounds. No GPU step, scratch or
 *   command-node limit applies; iterationLimitReached is always zero. iterationCount counts
 *   distinct finite sites inserted into valid groups, rather than GPU insertion/flip rounds.
 * - Exact arbitrary-precision fallbacks accept every finite float32 exponent span; the GPU
 *   rejects spans above 55. Subnormals retain their exact values.
 * - Nonfinite sites cancel their group (or the whole ungrouped result), as on the GPU. Global
 *   invalidCount counts nonfinite rows only for ungrouped input; grouped validity is per group.
 * - Unused output capacity is zeroed. The GPU leaves unused triangle rows unwritten.
 *
 * @throws Error naming positions, groupIds, groupCount or capacity for inconsistent inputs.
 */
export function computeDelaunayTessellationOnCPU(
  input: CPUDelaunayTessellationInput,
  options: CPUDelaunayTessellationOptions
): CPUDelaunayTessellationResult {
  const id = 'delaunay-tessellation';
  const {positions, groupIds} = input;
  if (positions.length % 2 !== 0) throw new Error('positions must hold float32x2 rows');
  validateCapacity(id, 'capacity', options.capacity, 0);
  if (options.groupCount !== undefined) validateCapacity(id, 'groupCount', options.groupCount);
  if ((groupIds !== undefined) !== (options.groupCount !== undefined))
    throw new Error('groupIds and groupCount must be supplied together');
  if (groupIds && groupIds.length !== positions.length / 2)
    throw new Error('groupIds must have one row per position');
  resolveGPUSpatialContext(id, options.spatialContext);
  const groupCount = options.groupCount ?? 1;
  const groups = Array.from({length: groupCount}, () => ({
    sites: [] as number[],
    owners: new Set<string>(),
    invalid: false
  }));
  let duplicateCount = 0;
  let invalidCount = 0;
  for (let row = 0; row < positions.length / 2; row++) {
    const group = groupIds ? groupIds[row] : 0;
    if (group >= groupCount) continue;
    const horizontal = positions[2 * row];
    const vertical = positions[2 * row + 1];
    if (!Number.isFinite(horizontal) || !Number.isFinite(vertical)) {
      groups[group].invalid = true;
      if (!groupIds) invalidCount++;
      continue;
    }
    const key = `${horizontal},${vertical}`;
    if (groups[group].owners.has(key)) duplicateCount++;
    else {
      groups[group].owners.add(key);
      groups[group].sites.push(row);
    }
  }
  const result: CPUDelaunayTessellationResult = {
    triangles: new Uint32Array(3 * options.capacity),
    count: new Uint32Array(1),
    requiredCount: new Uint32Array(1),
    overflow: new Uint32Array(1),
    duplicateCount: Uint32Array.of(duplicateCount),
    invalidCount: Uint32Array.of(invalidCount),
    converged: Uint32Array.of(invalidCount === 0 ? 1 : 0),
    iterationCount: new Uint32Array(1),
    iterationLimitReached: new Uint32Array(1),
    candidateOverflow: new Uint32Array(1),
    uncertainCount: new Uint32Array(1),
    approximation: new Uint32Array(1),
    groupTriangleCounts: new Uint32Array(options.groupCount ?? 0),
    groupTriangleOffsets: new Uint32Array(options.groupCount ?? 0),
    groupConverged: new Uint32Array(options.groupCount ?? 0)
  };
  let requiredCount = 0;
  groups.forEach((group, groupIndex) => {
    const triangles = group.invalid ? [] : computeGroupTriangles(positions, group.sites);
    if (options.groupCount !== undefined) {
      result.groupTriangleCounts[groupIndex] = triangles.length;
      result.groupTriangleOffsets[groupIndex] = requiredCount;
      result.groupConverged[groupIndex] = Number(!group.invalid);
    }
    if (!group.invalid) result.iterationCount[0] += group.sites.length;
    for (const triangle of triangles) {
      if (requiredCount < options.capacity) result.triangles.set(triangle, requiredCount * 3);
      requiredCount++;
    }
  });
  validateCapacity(id, 'requiredCount', requiredCount, 0);
  result.requiredCount[0] = requiredCount;
  result.count[0] = Math.min(requiredCount, options.capacity);
  result.overflow[0] = Number(requiredCount > options.capacity);
  return result;
}

/** CPU Delaunay twin, without GPU step/command budgets or exponent-span restrictions. */
export class CPUDelaunayTessellationOperation extends GPUCPUOperation {
  constructor(props: DelaunayTessellationOperationProps) {
    props = snapshotDelaunayTessellationOperationProps(props);
    const operation = new GPUDelaunayTessellationOperation(props);
    const {id} = operation;
    validateCapacity(id, 'capacity', props.capacity, 0);
    if (props.groups) validateCapacity(id, 'groups.count', props.groups.count);
    validateRequirement(
      id,
      Boolean(props.groups) === Boolean(props.output.groups),
      'groups and output.groups must be supplied together'
    );
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
        const result = computeDelaunayTessellationOnCPU(
          {
            positions: getCPUOperationFloat32Rows(id, inputs, 'positions', 'float32x2'),
            groupIds: props.groups ? getCPUOperationUint32Rows(id, inputs, 'groupIds') : undefined
          },
          {
            capacity: props.capacity,
            groupCount: props.groups?.count,
            spatialContext: props.spatialContext
          }
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUDelaunayTessellationResult]
          ])
        );
      }
    });
  }
}
