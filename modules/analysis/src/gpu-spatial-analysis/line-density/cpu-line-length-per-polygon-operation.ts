// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getCPUOperationFloat32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {CPUSpatialJoinGeometry} from '../spatial-join/cpu-feature-join';
import {
  readCPUPointJoinPolygons,
  getCPUPointJoinOrientation,
  classifyCPUPointJoinPolygon
} from '../spatial-join/cpu-point-join-geometry';
import {
  createPointJoinCPUOperation,
  readPointJoinGeometry,
  snapshotPointJoinGeometry,
  validatePointJoinOffsets
} from '../spatial-join/point-join-operation-utils';
import {
  createLineLengthPerPolygonOperation,
  LINE_LENGTH_MAXIMUM_CROSSING_COUNT,
  type LineLengthPerPolygonOperationProps
} from './gpu-line-length-per-polygon-operation';

/** Explicit pair/crossing budgets bind both CPU and GPU operations. */
export type CPULineLengthPerPolygonOptions = Pick<
  LineLengthPerPolygonOperationProps,
  'pairCapacity' | 'maximumCrossingCount'
>;
/**
 * Sums all clipped segment lengths of every line part, weighted once by the owning feature.
 * What differs from the GPU contributor:
 * - Planar float64 clipping and accumulation, rounded once at output. Near-edge cut fractions
 *   and totals can differ from float32. Both forms include boundary-coincident pieces.
 * - Offset hierarchies are checked on CPU; malformed layouts throw. Nonfinite line edges skip.
 * - Empty coordinate arrays with valid offsets are accepted, without GPU storage-binding limits.
 * - Pair and crossing budgets are preserved; overflow reports partial results, never completeness.
 */
export function computeLineLengthPerPolygonOnCPU(
  input: {
    lines: CPUSpatialJoinGeometry;
    polygons: CPUSpatialJoinGeometry;
    lineWeights?: Float32Array;
  },
  options: CPULineLengthPerPolygonOptions
) {
  const id = 'CPULineLengthPerPolygon';
  const {lines, lineWeights} = input;
  validateRequirement(
    id,
    lines.kind === 'lines' && lines.positions.length % 2 === 0 && Boolean(lines.lineOffsets),
    'lines needs XY positions and lineOffsets'
  );
  const partOffsets = lines.lineOffsets!;
  validatePointJoinOffsets(id, 'lines.lineOffsets', partOffsets, lines.positions.length / 2);
  const featureOffsets =
    lines.featureOffsets ?? Uint32Array.from({length: partOffsets.length}, (_, row) => row);
  validatePointJoinOffsets(id, 'lines.featureOffsets', featureOffsets, partOffsets.length - 1);
  validateRequirement(
    id,
    !lineWeights || lineWeights.length === featureOffsets.length - 1,
    'lineWeights needs one row per line feature'
  );
  const polygons = readCPUPointJoinPolygons(id, input.polygons);
  const maximumCrossingCount = options.maximumCrossingCount ?? 64;
  validateCapacity(id, 'pairCapacity', options.pairCapacity);
  validateCapacity(
    id,
    'maximumCrossingCount',
    maximumCrossingCount,
    1,
    LINE_LENGTH_MAXIMUM_CROSSING_COUNT
  );
  const lengths = new Float64Array(polygons.length),
    weightedLengths = new Float64Array(polygons.length),
    segmentCounts = new Uint32Array(polygons.length);
  let pairCount = 0,
    crossingOverflow = false;
  for (let feature = 0; feature + 1 < featureOffsets.length; feature++)
    for (let part = featureOffsets[feature]; part < featureOffsets[feature + 1]; part++)
      for (let vertex = partOffsets[part]; vertex + 1 < partOffsets[part + 1]; vertex++) {
        const startX = lines.positions[2 * vertex],
          startY = lines.positions[2 * vertex + 1],
          endX = lines.positions[2 * vertex + 2],
          endY = lines.positions[2 * vertex + 3];
        if (![startX, startY, endX, endY].every(Number.isFinite)) continue;
        const deltaX = endX - startX,
          deltaY = endY - startY;
        for (let polygon = 0; polygon < polygons.length; polygon++) {
          const bounds = polygons[polygon].bounds;
          if (
            Math.min(startX, endX) > bounds[2] ||
            Math.max(startX, endX) < bounds[0] ||
            Math.min(startY, endY) > bounds[3] ||
            Math.max(startY, endY) < bounds[1]
          )
            continue;
          if (pairCount++ >= options.pairCapacity) continue;
          if (deltaX === 0 && deltaY === 0) continue;
          // Anchor the intersection calculation at the endpoint nearest this polygon. Store
          // actual crossing coordinates: two distinct cuts can have the same rounded global t.
          const reverse =
            Math.hypot(endX - bounds[0], endY - bounds[1]) <
            Math.hypot(startX - bounds[0], startY - bounds[1]);
          const originX = reverse ? endX : startX,
            originY = reverse ? endY : startY;
          const directionX = reverse ? -deltaX : deltaX,
            directionY = reverse ? -deltaY : deltaY;
          const axis = Math.abs(deltaX) >= Math.abs(deltaY) ? 0 : 1;
          const crossings: [number, number][] = [];
          for (const rings of polygons[polygon].parts)
            for (const ring of rings) {
              if (ring.length < 6) continue;
              let previousX = ring[ring.length - 2],
                previousY = ring[ring.length - 1];
              for (let coordinate = 0; coordinate < ring.length; coordinate += 2) {
                const currentX = ring[coordinate],
                  currentY = ring[coordinate + 1];
                const edgeX = currentX - previousX,
                  edgeY = currentY - previousY;
                const denominator = directionX * edgeY - directionY * edgeX;
                if (denominator !== 0) {
                  const offsetX = previousX - originX,
                    offsetY = previousY - originY;
                  const edgeFraction = (offsetX * directionY - offsetY * directionX) / denominator;
                  // Reuse a vertex exactly when it is on the segment's supporting line.
                  // Independently reconstructing its two incident-edge cuts invents a sliver.
                  const previousOnLine =
                    getCPUPointJoinOrientation(startX, startY, endX, endY, previousX, previousY) ===
                    0;
                  const currentOnLine =
                    getCPUPointJoinOrientation(startX, startY, endX, endY, currentX, currentY) ===
                    0;
                  const crossing: [number, number] = previousOnLine
                    ? [previousX, previousY]
                    : currentOnLine
                      ? [currentX, currentY]
                      : [previousX + edgeFraction * edgeX, previousY + edgeFraction * edgeY];
                  const coordinate = crossing[axis];
                  const first = axis === 0 ? startX : startY,
                    last = axis === 0 ? endX : endY;
                  if (
                    coordinate > Math.min(first, last) &&
                    coordinate < Math.max(first, last) &&
                    edgeFraction >= -1e-6 &&
                    edgeFraction <= 1 + 1e-6
                  ) {
                    if (crossings.length < maximumCrossingCount) crossings.push(crossing);
                    else crossingOverflow = true;
                  }
                }
                previousX = currentX;
                previousY = currentY;
              }
            }
          crossings.push([startX, startY], [endX, endY]);
          crossings.sort((first, second) => first[axis] - second[axis]);
          let inside = 0;
          for (let piece = 1; piece < crossings.length; piece++) {
            const first = crossings[piece - 1],
              last = crossings[piece];
            if (
              last[axis] > first[axis] &&
              classifyCPUPointJoinPolygon(
                first[0] + (last[0] - first[0]) / 2,
                first[1] + (last[1] - first[1]) / 2,
                polygons[polygon],
                'closed'
              )
            )
              inside += Math.hypot(last[0] - first[0], last[1] - first[1]);
          }
          if (inside > 0) {
            lengths[polygon] += inside;
            weightedLengths[polygon] += inside * (lineWeights?.[feature] ?? 1);
            segmentCounts[polygon]++;
          }
        }
      }
  const candidateOverflow = Number(pairCount > options.pairCapacity);
  const clippedLengths = Float32Array.from(lengths);
  return {
    clippedLengths,
    weightedLengths: Float32Array.from(weightedLengths),
    segmentCounts,
    overflow: Uint32Array.of(Number(Boolean(candidateOverflow || crossingOverflow))),
    candidateOverflow: Uint32Array.of(candidateOverflow)
  };
}
/** CPU twin of the packed planar line-length operation. @internal */
export function createCPULineLengthPerPolygonOperation(input: LineLengthPerPolygonOperationProps) {
  const id = input.id ?? 'line-length-per-polygon-operation';
  const lines = snapshotPointJoinGeometry(id, input.lines),
    polygons = snapshotPointJoinGeometry(id, input.polygons);
  const weightInput = input.lineWeights;
  const lineWeights = weightInput ? Object.freeze({...weightInput}) : undefined;
  const hasWeights = Boolean(lineWeights);
  const options = Object.freeze({
    pairCapacity: input.pairCapacity,
    maximumCrossingCount: input.maximumCrossingCount
  });
  const operation = createLineLengthPerPolygonOperation({
    id,
    lines,
    polygons,
    lineWeights,
    ...options,
    output: input.output
  });
  return createPointJoinCPUOperation(operation, inputs =>
    computeLineLengthPerPolygonOnCPU(
      {
        lines: readPointJoinGeometry(operation.id, 'lines', lines, inputs),
        polygons: readPointJoinGeometry(operation.id, 'polygons', polygons, inputs),
        lineWeights: hasWeights
          ? getCPUOperationFloat32Rows(operation.id, inputs, 'lineWeights')
          : undefined
      },
      options
    )
  );
}
