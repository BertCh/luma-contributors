// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotRefinementOperationProps} from './refinement-construction-snapshot';

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCPUOperationFloat32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {computeDelaunayTessellationOnCPU} from '../delaunay-tessellation/cpu-delaunay-tessellation-operation';
import {getDelaunayMaximumTriangleCount} from '../delaunay-tessellation/gpu-delaunay-tessellation';
import {
  GPURatioConcaveHullOperation,
  type RatioConcaveHullOperationProps
} from './gpu-ratio-concave-hull-operation';

export type CPURatioConcaveHullOptions = {
  ratio: number;
  allowHoles?: boolean;
  maximumIterations?: number;
  spatialContext?: GPUSpatialContext;
};
export type CPURatioConcaveHullResult = {
  triangles: Uint32Array;
  triangleValidity: Uint32Array;
  boundaryEdges: Uint32Array;
  edgeValidity: Uint32Array;
  converged: Uint32Array;
};

/**
 * Erodes the exact Delaunay mesh in simultaneous local-priority rounds. Border candidates have
 * one exposed edge and an interior opposite vertex; optional hole candidates have three interior
 * vertices and are considered only when no border candidate remains. Eligible candidates maximize
 * length at every incident vertex; lowest triangle row wins ties. This is the GPU's definition.
 *
 * What differs from the GPU contributor:
 * - An explicit maximumIterations limits the number of triangle removals, including an entire
 *   simultaneous round or none of it. Omitted budgets allow erosion to finish.
 * - Edge lengths, extrema and thresholds use float64. The ratio is first rounded to float32,
 *   matching the GPU override. Near threshold/tie boundaries, float32 GPU lengths can select
 *   different triangles. Distances that overflow float32 remain finite on the CPU.
 * - Initial triangles come from computeDelaunayTessellationOnCPU, with the same row-order
 *   cocircular rule, no exponent-span cap and exact subnormals. Nonfinite sites cancel the hull.
 * - Unused triangle capacity is zeroed; GPU Delaunay leaves unused triangle rows unwritten.
 *
 * @throws Error naming positions or ratio for inconsistent rows or an invalid ratio.
 */
export function computeRatioConcaveHullOnCPU(
  positions: Float32Array,
  options: CPURatioConcaveHullOptions
): CPURatioConcaveHullResult {
  if (positions.length % 2 !== 0) throw new Error('positions must hold float32x2 rows');
  validateFiniteNumber('ratio-concave-hull', 'ratio', options.ratio, 0, 1);
  if (options.maximumIterations !== undefined)
    validateCapacity('ratio-concave-hull', 'maximumIterations', options.maximumIterations, 0);
  validateRequirement(
    'ratio-concave-hull',
    options.ratio >= 0 && options.ratio <= 1,
    'ratio must be in [0, 1]'
  );
  const pointCount = positions.length / 2;
  const capacity = getDelaunayMaximumTriangleCount(pointCount);
  const delaunay = computeDelaunayTessellationOnCPU(
    {positions},
    {capacity, spatialContext: options.spatialContext}
  );
  const {triangles, converged} = delaunay;
  const triangleCount = delaunay.count[0];
  const result: CPURatioConcaveHullResult = {
    triangles,
    converged,
    triangleValidity: new Uint32Array(capacity),
    boundaryEdges: new Uint32Array(capacity * 6),
    edgeValidity: new Uint32Array(capacity * 3)
  };
  result.triangleValidity.fill(1, 0, triangleCount);
  const adjacency = new Int32Array(capacity * 3).fill(-1);
  const edges = new Map<string, number>();
  const lengths = new Float64Array(triangleCount * 3);
  let minimumLength = Infinity;
  let maximumLength = 0;
  for (let row = 0; row < triangleCount; row++) {
    for (let side = 0; side < 3; side++) {
      const slot = 3 * row + side;
      const first = triangles[slot];
      const second = triangles[3 * row + ((side + 1) % 3)];
      const key = first < second ? `${first},${second}` : `${second},${first}`;
      const other = edges.get(key);
      if (other !== undefined) {
        adjacency[slot] = Math.floor(other / 3);
        adjacency[other] = row;
      } else edges.set(key, slot);
      lengths[slot] = Math.hypot(
        positions[2 * first] - positions[2 * second],
        positions[2 * first + 1] - positions[2 * second + 1]
      );
      minimumLength = Math.min(minimumLength, lengths[slot]);
      maximumLength = Math.max(maximumLength, lengths[slot]);
    }
  }
  const isExposed = (slot: number): boolean =>
    adjacency[slot] === -1 || result.triangleValidity[adjacency[slot]] === 0;
  const threshold = minimumLength + Math.fround(options.ratio) * (maximumLength - minimumLength);
  if (converged[0] !== 0 && triangleCount > 0) {
    let removedCount = 0;
    do {
      const boundary = new Uint8Array(pointCount);
      for (let row = 0; row < triangleCount; row++) {
        if (!result.triangleValidity[row]) continue;
        for (let side = 0; side < 3; side++) {
          if (!isExposed(3 * row + side)) continue;
          boundary[triangles[3 * row + side]] = 1;
          boundary[triangles[3 * row + ((side + 1) % 3)]] = 1;
        }
      }
      const candidates = new Float64Array(triangleCount);
      let candidateCount = 0;
      for (let row = 0; row < triangleCount; row++) {
        if (!result.triangleValidity[row]) continue;
        let exposedCount = 0;
        let exposedSide = 0;
        for (let side = 0; side < 3; side++)
          if (isExposed(3 * row + side)) {
            exposedCount++;
            exposedSide = side;
          }
        if (
          exposedCount === 1 &&
          !boundary[triangles[3 * row + ((exposedSide + 2) % 3)]] &&
          lengths[3 * row + exposedSide] > threshold
        ) {
          candidates[row] = lengths[3 * row + exposedSide];
          candidateCount++;
        }
      }
      if (candidateCount === 0 && options.allowHoles) {
        for (let row = 0; row < triangleCount; row++) {
          if (!result.triangleValidity[row]) continue;
          if ([0, 1, 2].some(side => boundary[triangles[3 * row + side]])) continue;
          const size = Math.max(lengths[3 * row], lengths[3 * row + 1], lengths[3 * row + 2]);
          if (size > threshold) candidates[row] = size;
        }
      }
      const maxima = new Float64Array(pointCount);
      for (let row = 0; row < triangleCount; row++) {
        for (let corner = 0; corner < 3; corner++) {
          const vertex = triangles[3 * row + corner];
          maxima[vertex] = Math.max(maxima[vertex], candidates[row]);
        }
      }
      const winners = new Uint32Array(pointCount).fill(0xffffffff);
      for (let row = 0; row < triangleCount; row++) {
        if (
          candidates[row] === 0 ||
          ![0, 1, 2].every(corner => maxima[triangles[3 * row + corner]] === candidates[row])
        )
          continue;
        for (let corner = 0; corner < 3; corner++) {
          const vertex = triangles[3 * row + corner];
          winners[vertex] = Math.min(winners[vertex], row);
        }
      }
      let removalCount = 0;
      for (let row = 0; row < triangleCount; row++) {
        if (
          candidates[row] > 0 &&
          [0, 1, 2].every(corner => winners[triangles[3 * row + corner]] === row)
        ) {
          removalCount++;
        }
      }
      if (removalCount === 0) break;
      if (removedCount + removalCount > (options.maximumIterations ?? Infinity)) {
        result.converged[0] = 0;
        break;
      }
      for (let row = 0; row < triangleCount; row++) {
        if (
          candidates[row] > 0 &&
          [0, 1, 2].every(corner => winners[triangles[3 * row + corner]] === row)
        )
          result.triangleValidity[row] = 0;
      }
      removedCount += removalCount;
    } while (true);
  }
  for (let row = 0; row < capacity; row++) {
    for (let side = 0; side < 3; side++) {
      const slot = row * 3 + side;
      result.boundaryEdges[2 * slot] = triangles[slot];
      result.boundaryEdges[2 * slot + 1] = triangles[3 * row + ((side + 1) % 3)];
      result.edgeValidity[slot] = Number(result.triangleValidity[row] !== 0 && isExposed(slot));
    }
  }
  return result;
}

/** CPU ratio hull twin, preserving local-priority rounds and explicit removal budgets. */
export class CPURatioConcaveHullOperation extends GPUCPUOperation {
  constructor(props: RatioConcaveHullOperationProps) {
    props = snapshotRefinementOperationProps(props, 'hull');
    const id = props.id ?? 'ratio-concave-hull';
    validateCapacity(id, 'pointCount', props.pointCount, 3);
    if (props.maximumIterations !== undefined)
      validateCapacity(id, 'maximumIterations', props.maximumIterations, 0);
    validateFiniteNumber(id, 'ratio', props.ratio, 0, 1);
    validateRequirement(id, props.ratio >= 0 && props.ratio <= 1, 'ratio must be in [0, 1]');
    resolveGPUSpatialContext(id, props.spatialContext);
    const operation = new GPURatioConcaveHullOperation(props);
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
        const positions = getCPUOperationFloat32Rows(id, inputs, 'positions', 'float32x2');
        if (positions.length !== props.pointCount * 2)
          throw new Error(`${id} positions must hold pointCount rows`);
        const result = computeRatioConcaveHullOnCPU(positions, props);
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPURatioConcaveHullResult]
          ])
        );
      }
    });
  }
}
