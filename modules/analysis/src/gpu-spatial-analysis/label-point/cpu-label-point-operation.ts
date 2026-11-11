// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  resolveGPUSpatialContext,
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT
} from '../../analysis-core/contracts/spatial-context';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows,
  type CPUOperationInputs
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  getCPUPolygonDistance,
  getCPUPolygonFeatureRows,
  getCPUPolygonRings,
  validateCPUPolygonRows,
  type CPUPolygonRows,
  type CPUPolygonRing,
  type CPUPolygonFeatureRows
} from '../shared/cpu-polygon-rows';
import {GPULabelPointOperation, type LabelPointOperationProps} from './gpu-label-point-operation';

/** Reads the four polygon ports shared by the geometry CPU operations. @internal */
export function readCPUPolygonRows(id: string, inputs: CPUOperationInputs): CPUPolygonRows {
  return {
    positions: getCPUOperationFloat32Rows(id, inputs, 'positions', 'float32x2'),
    ringOffsets: getCPUOperationUint32Rows(id, inputs, 'ringOffsets'),
    polygonOffsets: getCPUOperationUint32Rows(id, inputs, 'polygonOffsets'),
    featureOffsets: getCPUOperationUint32Rows(id, inputs, 'featureOffsets')
  };
}

/** Unbounded quadtree search result, before output rounding. @internal */
export type CPUPolygonPole = {
  centerX: number;
  centerY: number;
  radius: number;
  witnessX: number;
  witnessY: number;
  error: number;
  status: number;
};

function findScanlineSeed(
  positions: Float32Array,
  rings: readonly CPUPolygonRing[],
  lineY: number
): number | undefined {
  let firstX = Infinity;
  let secondX = Infinity;
  let firstCount = 0;
  for (const ring of rings) {
    if (ring.end <= ring.start) continue;
    let previousX = positions[(ring.end - 1) * 2];
    let previousY = positions[(ring.end - 1) * 2 + 1];
    for (let row = ring.start; row < ring.end; row++) {
      const currentX = positions[row * 2];
      const currentY = positions[row * 2 + 1];
      if (currentY > lineY !== previousY > lineY) {
        const crossingX =
          previousX + ((lineY - previousY) * (currentX - previousX)) / (currentY - previousY);
        if (crossingX < firstX) {
          secondX = firstX;
          firstX = crossingX;
          firstCount = 1;
        } else if (crossingX === firstX) {
          firstCount++;
        } else {
          secondX = Math.min(secondX, crossingX);
        }
      }
      previousX = currentX;
      previousY = currentY;
    }
  }
  return firstCount % 2 === 1 && secondX < Infinity ? firstX * 0.5 + secondX * 0.5 : undefined;
}

/**
 * Best-first quadtree, with the GPU circle search's live-slot and segment tie order.
 * Dynamic storage removes capacity and iteration budgets. Label mode first evaluates the GPU
 * label scanlines in the documented order, retaining the earlier seed on equal distances.
 * @internal
 */
export function findCPUPolygonPole(
  positions: Float32Array,
  rings: readonly CPUPolygonRing[],
  tolerance: number,
  firstVertexFallback = false,
  boundsRows?: CPUPolygonFeatureRows
): CPUPolygonPole {
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  let fallbackX = NaN;
  let fallbackY = NaN;
  let invalid = false;
  for (const ring of boundsRows ? [boundsRows] : rings) {
    for (let row = ring.start; row < ring.end; row++) {
      const positionX = positions[row * 2];
      const positionY = positions[row * 2 + 1];
      invalid ||= !Number.isFinite(positionX) || !Number.isFinite(positionY);
      minimumX = Math.min(minimumX, positionX);
      minimumY = Math.min(minimumY, positionY);
      maximumX = Math.max(maximumX, positionX);
      maximumY = Math.max(maximumY, positionY);
      if (!firstVertexFallback || Number.isNaN(fallbackX)) {
        fallbackX = positionX;
        fallbackY = positionY;
      }
    }
  }
  if (invalid || minimumX === Infinity) {
    return {
      centerX: NaN,
      centerY: NaN,
      radius: NaN,
      witnessX: NaN,
      witnessY: NaN,
      error: NaN,
      status: invalid ? 3 : 2
    };
  }
  let best = {
    centerX: fallbackX,
    centerY: fallbackY,
    radius: 0,
    witnessX: fallbackX,
    witnessY: fallbackY,
    error: 0,
    status: 0
  };
  const extent = Math.max(maximumX - minimumX, maximumY - minimumY);
  // A collapsed axis has no interior. The boundary fallback is an exact solution.
  if (minimumX === maximumX || minimumY === maximumY) return best;
  const precision = tolerance || Math.max(extent * 1e-6, extent * Number.EPSILON);
  type Cell = {centerX: number; centerY: number; halfSize: number; bound: number};
  const makeCell = (centerX: number, centerY: number, halfSize: number): Cell => {
    const result = getCPUPolygonDistance(positions, rings, centerX, centerY);
    if (result.distance > best.radius) {
      best = {
        ...best,
        centerX,
        centerY,
        radius: result.distance,
        witnessX: result.witnessX,
        witnessY: result.witnessY
      };
    }
    return {centerX, centerY, halfSize, bound: result.distance + Math.SQRT2 * halfSize};
  };
  if (firstVertexFallback) {
    const seedScanlines = (centerY: number, height: number): boolean => {
      let found = false;
      for (const fraction of [0, -0.2, 0.2]) {
        const lineY = centerY + fraction * height;
        const seedX = findScanlineSeed(positions, rings, lineY);
        if (seedX !== undefined) {
          makeCell(seedX, lineY, 0);
          found = true;
        }
      }
      return found;
    };
    if (!seedScanlines(minimumY * 0.5 + maximumY * 0.5, maximumY - minimumY)) {
      let largestExtent = -Infinity;
      let largestMinimumY = 0;
      let largestMaximumY = 0;
      for (const ring of rings) {
        let ringMinimumX = Infinity;
        let ringMinimumY = Infinity;
        let ringMaximumX = -Infinity;
        let ringMaximumY = -Infinity;
        for (let row = ring.start; row < ring.end; row++) {
          ringMinimumX = Math.min(ringMinimumX, positions[row * 2]);
          ringMaximumX = Math.max(ringMaximumX, positions[row * 2]);
          ringMinimumY = Math.min(ringMinimumY, positions[row * 2 + 1]);
          ringMaximumY = Math.max(ringMaximumY, positions[row * 2 + 1]);
        }
        const ringExtent = Math.max(ringMaximumX - ringMinimumX, ringMaximumY - ringMinimumY);
        if (ringExtent > largestExtent) {
          largestExtent = ringExtent;
          largestMinimumY = ringMinimumY;
          largestMaximumY = ringMaximumY;
        }
      }
      seedScanlines(
        largestMinimumY * 0.5 + largestMaximumY * 0.5,
        largestMaximumY - largestMinimumY
      );
    }
  }
  const cells = [
    makeCell(minimumX * 0.5 + maximumX * 0.5, minimumY * 0.5 + maximumY * 0.5, extent * 0.5)
  ];
  while (cells.length > 0) {
    let winner = 0;
    for (let cell = 1; cell < cells.length; cell++) {
      if (cells[cell].bound > cells[winner].bound) winner = cell;
    }
    const parent = cells[winner];
    best.error = Math.max(0, parent.bound - best.radius);
    if (best.error <= precision) break;
    const halfSize = parent.halfSize * 0.5;
    if (
      halfSize === 0 ||
      (parent.centerX + halfSize === parent.centerX && parent.centerY + halfSize === parent.centerY)
    ) {
      best.status = 1;
      break;
    }
    cells[winner] = cells[cells.length - 1];
    cells.pop();
    for (let child = 0; child < 4; child++) {
      cells.push(
        makeCell(
          parent.centerX + (child % 2 === 0 ? -halfSize : halfSize),
          parent.centerY + (child < 2 ? -halfSize : halfSize),
          halfSize
        )
      );
    }
    let kept = 0;
    for (const cell of cells) {
      if (cell.bound > best.radius) cells[kept++] = cell;
    }
    cells.length = kept;
    if (kept === 0) best.error = 0;
  }
  return best;
}

/** Result rows of GPULabelPoint, including all optional columns. */
export type CPULabelPointResult = {
  points: Float32Array;
  distances: Float32Array;
  degenerate: Uint32Array;
};

/**
 * Finds polygon poles with unbounded best-first subdivision in binary64.
 * Emptiness and bounds use the GPU's contiguous feature interval, while signed distances and
 * scanline seeds walk individual rings. This distinction also applies to descending offsets.
 *
 * What differs from the GPU contributor:
 * - Searches every cell whose distance bound can improve the answer, rather than a capped set
 *   of coarse grid lobes. Stops at 1e-6 of the feature's maximum extent (binary64 resolution
 *   is the lower limit), independent of all GPU grid and refinement settings.
 * - Equal distances retain the existing winner. The first vertex and GPU scanline seeds keep
 *   their order; subsequent candidates use bounds-center/quadtree order instead of grid order.
 * - Non-finite features return NaN points/distances and degenerate = 1; GPU label arithmetic
 *   does not define a non-finite status. Empty features have that same output.
 * - Collapsed-axis features return the first vertex and zero distance without subdivision.
 * - Empty position storage is accepted for empty features; the GPU requires at least one row.
 * - Computes in binary64 and rounds each output once; no float32 intermediate rounding.
 *
 * @throws If positions or any offsets array is inconsistent.
 */
export function computeLabelPointOnCPU(input: CPUPolygonRows): CPULabelPointResult {
  validateCPUPolygonRows(input);
  const featureCount = input.featureOffsets.length - 1;
  const points = new Float32Array(featureCount * 2);
  const distances = new Float32Array(featureCount);
  const degenerate = new Uint32Array(featureCount);
  for (let feature = 0; feature < featureCount; feature++) {
    const pole = findCPUPolygonPole(
      input.positions,
      getCPUPolygonRings(input, feature),
      0,
      true,
      getCPUPolygonFeatureRows(input, feature)
    );
    points.set([pole.centerX, pole.centerY], feature * 2);
    distances[feature] = pole.radius;
    degenerate[feature] = pole.radius > 0 ? 0 : 1;
  }
  return {points, distances, degenerate};
}

/** CPU twin; GPU grid sizes, candidate limits and refinement budgets are unused. */
export class CPULabelPointOperation extends GPUCPUOperation {
  constructor(props: LabelPointOperationProps) {
    const operation = new GPULabelPointOperation(props);
    const {id} = operation;
    validateCapacity(id, 'featureCount', props.featureCount, 1);
    resolveGPUSpatialContext(id, props.spatialContext, {
      ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
      measure: 'distance'
    });
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
        const input = readCPUPolygonRows(id, inputs);
        validateRequirement(
          id,
          input.featureOffsets.length === props.featureCount + 1,
          'featureOffsets length must equal featureCount + 1'
        );
        const result = computeLabelPointOnCPU(input);
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPULabelPointResult]
          ])
        );
      }
    });
  }
}
