// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {resolveGPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  computePolygonTriangulationOnCPU,
  type CPUPolygonTriangulationInput,
  type CPUPolygonTriangulationOptions,
  type CPUPolygonTriangulationResult
} from '../../analysis-core/polygon-triangulation/cpu-polygon-triangulation-operation';
import {
  computeTriangulationIncircle,
  computeTriangulationOrientation
} from '../../analysis-core/polygon-triangulation/cpu-triangulation-predicates';
import {
  GPUConstrainedDelaunayTriangulationOperation,
  type ConstrainedDelaunayTriangulationOperationProps
} from './gpu-constrained-delaunay-triangulation-operation';

export type CPUConstrainedDelaunayTriangulationInput = CPUPolygonTriangulationInput;
export type CPUConstrainedDelaunayTriangulationOptions = CPUPolygonTriangulationOptions;
export type CPUConstrainedDelaunayTriangulationResult = CPUPolygonTriangulationResult & {
  converged: Uint32Array;
};

/**
 * Triangulates polygon slices and flips strictly illegal interior edges to completion. The first
 * illegal pair/side in slice order is flipped, as in the GPU kernel; cocircular edges stay put.
 *
 * What differs from the GPU contributor:
 * - maximumFlips is ignored: Lawson flipping continues until no strictly illegal convex
 *   quadrilateral remains. Every valid polygon then has converged=1; failed polygons have 0.
 * - The initial triangulation uses budgeted un-hashed Earcut and the selected monotone route,
 *   as documented by computePolygonTriangulationOnCPU. GPU hashing can change the initial slices.
 * - Arbitrary-precision predicates accept all finite float32 exponent spans, without the GPU
 *   incircle fallback above span 55. Midpoints/crossings in the initial Earcut use float64.
 * - Inconsistent arrays throw with the argument name. Nonfinite live rings fail; subnormals
 *   retain their exact values. Output slices, padding and allocation limits are retained.
 *
 * @throws Error naming the inconsistent geometry array or indexCapacity.
 */
export function computeConstrainedDelaunayTriangulationOnCPU(
  input: CPUConstrainedDelaunayTriangulationInput,
  options: CPUConstrainedDelaunayTriangulationOptions
): CPUConstrainedDelaunayTriangulationResult {
  const initial = computePolygonTriangulationOnCPU(input, options);
  const result = {...initial, converged: new Uint32Array(initial.valid.length)};
  const {indices, triangleOffsets} = result;
  const readPoint = (vertex: number) => [
    input.positions[2 * vertex],
    input.positions[2 * vertex + 1]
  ];
  const readTriangle = (row: number) => [
    indices[3 * row],
    indices[3 * row + 1],
    indices[3 * row + 2]
  ];
  for (let polygon = 0; polygon < result.valid.length; polygon++) {
    if (!result.valid[polygon]) continue;
    const start = triangleOffsets[polygon];
    const end = triangleOffsets[polygon + 1];
    let changed: boolean;
    do {
      changed = false;
      for (let firstRow = start; firstRow < end && !changed; firstRow++) {
        const first = readTriangle(firstRow);
        if (new Set(first).size !== 3) continue;
        for (let secondRow = firstRow + 1; secondRow < end && !changed; secondRow++) {
          const second = readTriangle(secondRow);
          if (new Set(second).size !== 3) continue;
          for (let side = 0; side < 3 && !changed; side++) {
            const firstVertex = first[side];
            const secondVertex = first[(side + 1) % 3];
            const firstOpposite = first[(side + 2) % 3];
            for (let otherSide = 0; otherSide < 3; otherSide++) {
              if (second[otherSide] !== secondVertex || second[(otherSide + 1) % 3] !== firstVertex)
                continue;
              const secondOpposite = second[(otherSide + 2) % 3];
              if (
                computeTriangulationOrientation(
                  readPoint(firstOpposite),
                  readPoint(secondOpposite),
                  readPoint(secondVertex)
                ) !== 1 ||
                computeTriangulationOrientation(
                  readPoint(secondOpposite),
                  readPoint(firstOpposite),
                  readPoint(firstVertex)
                ) !== 1
              )
                continue;
              if (
                computeTriangulationIncircle(
                  readPoint(firstVertex),
                  readPoint(secondVertex),
                  readPoint(firstOpposite),
                  readPoint(secondOpposite)
                ) <= 0
              )
                continue;
              indices.set([firstOpposite, secondOpposite, secondVertex], 3 * firstRow);
              indices.set([secondOpposite, firstOpposite, firstVertex], 3 * secondRow);
              changed = true;
              break;
            }
          }
        }
      }
    } while (changed);
    result.converged[polygon] = 1;
  }
  return result;
}

/** CPU constrained triangulation twin; no GPU work, flip, scratch or exponent-span budgets. */
export class CPUConstrainedDelaunayTriangulationOperation extends GPUCPUOperation {
  constructor(props: ConstrainedDelaunayTriangulationOperationProps) {
    const operation = new GPUConstrainedDelaunayTriangulationOperation(props);
    const {id} = operation;
    validateCapacity(id, 'polygonCount', props.polygonCount);
    validateCapacity(id, 'indexCapacity', props.indexCapacity);
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
        if (polygonOffsets.length !== props.polygonCount + 1)
          throw new Error(`${id} polygonOffsets must hold polygonCount + 1 rows`);
        const result = computeConstrainedDelaunayTriangulationOnCPU(
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
            result[port as keyof CPUConstrainedDelaunayTriangulationResult]
          ])
        );
      }
    });
  }
}
