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
  validateCanonicalOffsets,
  type CanonicalPoint
} from './cpu-canonical-geometry';
import {
  GPUNormalizePolygonsOperation,
  type NormalizePolygonsOperationProps
} from './gpu-normalize-polygons-operation';

/** Packed planar polygon hierarchy. Feature order is preserved. */
export type CPUNormalizePolygonsInput = {
  positions: Float32Array;
  ringOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  featureOffsets: Uint32Array;
};
/** Normalized coordinates and hierarchy; featureOffsets are unchanged and caller-owned. */
export type CPUNormalizePolygonsResult = {
  positions: Float32Array;
  ringOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
};
type Ring = {points: CanonicalPoint[]; closed: boolean; total: number};

function compareRings(first: Ring, second: Ring): number {
  if (first.total !== second.total) return first.total - second.total;
  for (let index = 0; index < Math.min(first.points.length, second.points.length); index++) {
    const comparison = compareCanonicalPoints(first.points[index], second.points[index]);
    if (comparison !== 0) return comparison;
  }
  return first.points.length - second.points.length || Number(first.closed) - Number(second.closed);
}

function comparePolygons(first: Ring[], second: Ring[]): number {
  for (let index = 0; index < Math.min(first.length, second.length); index++) {
    const comparison = compareRings(first[index], second[index]);
    if (comparison !== 0) return comparison;
  }
  return first.length - second.length;
}

/**
 * Orients exterior rings clockwise and holes counterclockwise, selects the smallest complete
 * cyclic sequence, then stably sorts holes and polygon parts in descending canonical order.
 * Ring comparison starts with total vertex count, including a closing coordinate. Zero winding
 * retains its original direction; closure, duplicates, empty rings/parts and feature order remain.
 *
 * What differs from the GPU contributor:
 * - Winding is an unbounded exact BigInt predicate on float32 coordinates. The GPU's uncertain
 *   orientation (200 binary orders of product exponents) leaves the original direction alone;
 *   the CPU still orients such a ring. Finite subnormals are supported on the CPU.
 * - Sorting replaces repeated exhaustive ranking, retaining source order on equal comparisons.
 * - Slots not owned by a covered feature are zero; the GPU leaves these slots unwritten.
 * - Nonfinite coordinates throw an Error naming positions; the GPU requires finite inputs and
 *   does not define a canonical result for them. Negative zero becomes positive zero on output.
 *
 * @throws If positions are malformed/nonfinite or an offset table is inconsistent.
 */
export function computeNormalizePolygonsOnCPU(
  input: CPUNormalizePolygonsInput
): CPUNormalizePolygonsResult {
  if (input.positions.length % 2 !== 0 || !input.positions.every(Number.isFinite))
    throw new Error('positions must contain finite float32x2 rows');
  validateCanonicalOffsets('ringOffsets', input.ringOffsets, input.positions.length / 2);
  validateCanonicalOffsets('polygonOffsets', input.polygonOffsets, input.ringOffsets.length - 1);
  validateCanonicalOffsets('featureOffsets', input.featureOffsets, input.polygonOffsets.length - 1);
  const rings: Ring[] = [];
  const exteriors = new Set(input.polygonOffsets);
  for (let ring = 0; ring + 1 < input.ringOffsets.length; ring++) {
    const points: CanonicalPoint[] = [];
    for (let vertex = input.ringOffsets[ring]; vertex < input.ringOffsets[ring + 1]; vertex++)
      points.push([input.positions[vertex * 2], input.positions[vertex * 2 + 1]]);
    const total = points.length;
    const winding = computeCanonicalWinding(points);
    const exterior =
      exteriors.has(ring) ||
      ring < input.polygonOffsets[0] ||
      ring >= input.polygonOffsets[input.polygonOffsets.length - 1];
    if ((exterior && winding > 0) || (!exterior && winding < 0)) points.reverse();
    const closed = total > 1 && compareCanonicalPoints(points[0], points[total - 1]) === 0;
    if (closed) points.pop();
    let rotation = 0;
    for (let candidate = 1; candidate < points.length; candidate++)
      for (let element = 0; element < points.length; element++) {
        const comparison = compareCanonicalPoints(
          points[(candidate + element) % points.length],
          points[(rotation + element) % points.length]
        );
        if (comparison < 0) rotation = candidate;
        if (comparison !== 0) break;
      }
    rings.push({points: points.slice(rotation).concat(points.slice(0, rotation)), closed, total});
  }
  const polygons: Ring[][] = [];
  for (let polygon = 0; polygon + 1 < input.polygonOffsets.length; polygon++) {
    const parts = rings.slice(input.polygonOffsets[polygon], input.polygonOffsets[polygon + 1]);
    polygons.push(
      parts.length === 0
        ? []
        : [parts[0], ...parts.slice(1).sort((first, second) => -compareRings(first, second))]
    );
  }
  const result: CPUNormalizePolygonsResult = {
    positions: new Float32Array(input.positions.length),
    ringOffsets: new Uint32Array(input.ringOffsets.length),
    polygonOffsets: new Uint32Array(input.polygonOffsets.length)
  };
  for (let feature = 0; feature + 1 < input.featureOffsets.length; feature++) {
    const first = input.featureOffsets[feature];
    const last = input.featureOffsets[feature + 1];
    const ordered = polygons
      .slice(first, last)
      .sort((firstPolygon, secondPolygon) => -comparePolygons(firstPolygon, secondPolygon));
    let destinationRing = input.polygonOffsets[first];
    let destinationVertex = input.ringOffsets[destinationRing];
    for (let slot = 0; slot < ordered.length; slot++) {
      result.polygonOffsets[first + slot] = destinationRing;
      for (const ring of ordered[slot]) {
        result.ringOffsets[destinationRing++] = destinationVertex;
        for (let element = 0; element < ring.total; element++) {
          const point = ring.points[element % ring.points.length];
          result.positions.set([point[0] + 0, point[1] + 0], destinationVertex++ * 2);
        }
      }
    }
    if (feature === input.featureOffsets.length - 2) {
      result.polygonOffsets[last] = destinationRing;
      result.ringOffsets[destinationRing] = destinationVertex;
    }
  }
  return result;
}

/** CPU canonical polygons with exact winding and stable sorting. */
export class CPUNormalizePolygonsOperation extends GPUCPUOperation {
  constructor(props: NormalizePolygonsOperationProps) {
    const id = props.id ?? 'normalize-polygons';
    for (const name of ['rowCount', 'ringCount', 'polygonCount', 'featureCount'] as const)
      validateCapacity(id, name, props[name], 0);
    const operation = new GPUNormalizePolygonsOperation(props);
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
          ringOffsets: getCPUOperationUint32Rows(id, inputs, 'ringOffsets'),
          polygonOffsets: getCPUOperationUint32Rows(id, inputs, 'polygonOffsets'),
          featureOffsets: getCPUOperationUint32Rows(id, inputs, 'featureOffsets')
        };
        for (const [name, expected] of [
          ['positions', props.rowCount * 2],
          ['ringOffsets', props.ringCount + 1],
          ['polygonOffsets', props.polygonCount + 1],
          ['featureOffsets', props.featureCount + 1]
        ] as const)
          if (input[name].length !== expected)
            throw new Error(`${id} ${name} length must equal ${expected}`);
        const result = computeNormalizePolygonsOnCPU(input);
        return {
          [props.output.positions]: result.positions,
          [props.output.ringOffsets]: result.ringOffsets,
          [props.output.polygonOffsets]: result.polygonOffsets
        };
      }
    });
  }
}
