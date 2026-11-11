// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {resolveGPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateFiniteNumber
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPUSegmentSnapOperation,
  type SegmentSnapOperationProps
} from './gpu-segment-snap-operation';

/** Input line vertices and planar reference vertices. */
export type CPUSegmentSnapInput = {
  positions: Float32Array;
  lineOffsets: Uint32Array;
  referencePositions: Float32Array;
};
/** Fixed-slot output shape and inclusive distance threshold. */
export type CPUSegmentSnapOptions = {tolerance: number; maximumVerticesPerLine: number};
/** Per-line slots and [written, required, overflow, nonfinite] status rows. */
export type CPUSegmentSnapResult = {
  positions: Float32Array;
  sourceEdges: Uint32Array;
  counts: Uint32Array;
  validity: Uint32Array;
};
type Point = {x: number; y: number; row: number};

function readPoint(positions: Float32Array, row: number): Point {
  return {x: positions[row * 2], y: positions[row * 2 + 1], row};
}

function snapPoint(point: Point, references: Point[], tolerance: number): Point {
  let result = point;
  let minimum = Infinity;
  for (const reference of references) {
    const distance = Math.hypot(reference.x - point.x, reference.y - point.y);
    if (distance <= tolerance && distance < minimum) {
      result = reference;
      minimum = distance;
    }
  }
  return result;
}

/**
 * Snaps endpoints and inserts unique nearby references in (projection fraction,reference row)
 * order. Consecutive equal coordinates are suppressed. Counts cover the complete result even
 * when only a prefix fits the fixed slots for a line.
 *
 * What differs from the GPU contributor:
 * - Distances and projection fractions use float64; near tolerance boundaries or projection ties
 *   can differ. The numeric tolerance is rounded to the GPU's float32 literal before use.
 * - Candidates are sorted once per segment instead of repeatedly scanning for the next winner.
 *   Reference duplicates retain their first row, exactly as on the GPU.
 * - Unused positions/sourceEdges slots are zero; the GPU leaves them unwritten. Only each line's
 *   written prefix is meaningful. Out-of-range offsets are clamped and reversed intervals are
 *   empty, as explicitly defined by the current kernel.
 * - Nothing else: nonfinite references invalidate every line, including empty lines; nonfinite
 *   input vertices invalidate their line. An empty line has validity=0; a collapsed line has bit 1.
 *
 * @throws If arrays are not packed, lineOffsets is too short, options are invalid, or the
 * uint32 required-count/output-indexing bounds would be exceeded.
 */
export function computeSegmentSnapOnCPU(
  input: CPUSegmentSnapInput,
  options: CPUSegmentSnapOptions
): CPUSegmentSnapResult {
  const id = 'computeSegmentSnapOnCPU';
  validateCapacity(id, 'maximumVerticesPerLine', options.maximumVerticesPerLine);
  validateFiniteNumber(id, 'tolerance', options.tolerance, 0);
  if (input.positions.length % 2 !== 0) throw new Error('positions must contain float32x2 rows');
  if (input.referencePositions.length % 2 !== 0)
    throw new Error('referencePositions must contain float32x2 rows');
  if (input.lineOffsets.length < 2)
    throw new Error('lineOffsets must contain at least two entries');
  const rowCount = input.positions.length / 2;
  const referenceCount = input.referencePositions.length / 2;
  validateCapacity(
    id,
    'positions.length * (referencePositions.length + 1)',
    rowCount * (referenceCount + 1),
    0
  );
  const lineCount = input.lineOffsets.length - 1;
  const capacity = options.maximumVerticesPerLine;
  validateCapacity(id, 'lineCount * maximumVerticesPerLine', lineCount * capacity, 0);
  const result: CPUSegmentSnapResult = {
    positions: new Float32Array(lineCount * capacity * 2),
    sourceEdges: new Uint32Array(lineCount * capacity),
    counts: new Uint32Array(lineCount * 4),
    validity: new Uint32Array(lineCount)
  };
  const references = Array.from({length: referenceCount}, (_, row) =>
    readPoint(input.referencePositions, row)
  );
  const invalidReferences = references.some(
    point => !Number.isFinite(point.x) || !Number.isFinite(point.y)
  );
  const unique: Point[] = [];
  const coordinates = new Map<number, Set<number>>();
  for (const reference of references) {
    let ordinates = coordinates.get(reference.x);
    if (!ordinates) {
      ordinates = new Set();
      coordinates.set(reference.x, ordinates);
    }
    if (!ordinates.has(reference.y)) {
      unique.push(reference);
      ordinates.add(reference.y);
    }
  }
  const tolerance = Math.fround(options.tolerance);
  for (let line = 0; line < lineCount; line++) {
    const first = Math.min(rowCount, input.lineOffsets[line]);
    const last = Math.min(rowCount, input.lineOffsets[line + 1]);
    let invalid = invalidReferences;
    for (let row = first; row < last; row++)
      invalid ||=
        !Number.isFinite(input.positions[row * 2]) ||
        !Number.isFinite(input.positions[row * 2 + 1]);
    let required = 0;
    const writeVertex = (point: Point, source: number) => {
      if (required < capacity) {
        const slot = line * capacity + required;
        result.positions.set([point.x, point.y], slot * 2);
        result.sourceEdges[slot] = source;
      }
      required++;
    };
    if (!invalid && last > first) {
      let start = snapPoint(readPoint(input.positions, first), references, tolerance);
      writeVertex(start, first);
      for (let row = first; row + 1 < last; row++) {
        const end = snapPoint(readPoint(input.positions, row + 1), references, tolerance);
        const deltaX = end.x - start.x;
        const deltaY = end.y - start.y;
        const denominator = deltaX * deltaX + deltaY * deltaY;
        let previous = start;
        if (denominator > 0) {
          const insertions: {point: Point; fraction: number}[] = [];
          for (const reference of unique) {
            const fraction =
              ((reference.x - start.x) * deltaX + (reference.y - start.y) * deltaY) / denominator;
            if (
              fraction <= 0 ||
              fraction >= 1 ||
              Math.hypot(
                reference.x - (start.x + fraction * deltaX),
                reference.y - (start.y + fraction * deltaY)
              ) > tolerance
            )
              continue;
            insertions.push({point: reference, fraction});
          }
          insertions.sort(
            (firstInsertion, secondInsertion) =>
              firstInsertion.fraction - secondInsertion.fraction ||
              firstInsertion.point.row - secondInsertion.point.row
          );
          for (const insertion of insertions)
            if (insertion.point.x !== previous.x || insertion.point.y !== previous.y) {
              writeVertex(insertion.point, row);
              previous = insertion.point;
            }
        }
        if (end.x !== previous.x || end.y !== previous.y) writeVertex(end, row);
        start = end;
      }
    }
    result.counts.set(
      [Math.min(required, capacity), required, Number(required > capacity), Number(invalid)],
      line * 4
    );
    result.validity[line] =
      Number(required === 1) | (invalid ? 2 : 0) | (required > capacity ? 4 : 0);
  }
  return result;
}

/** CPU segment snap twin; fixed slots and uint32 required-count limits are retained. */
export class CPUSegmentSnapOperation extends GPUCPUOperation {
  constructor(props: SegmentSnapOperationProps) {
    const id = props.id ?? 'segment-snap';
    resolveGPUSpatialContext(id, props.spatialContext);
    validateCapacity(id, 'rowCount', props.rowCount, 0);
    validateCapacity(id, 'referenceCount', props.referenceCount, 0);
    validateCapacity(id, 'lineCount', props.lineCount);
    validateCapacity(id, 'maximumVerticesPerLine', props.maximumVerticesPerLine);
    validateCapacity(
      id,
      'rowCount * (referenceCount + 1)',
      props.rowCount * (props.referenceCount + 1),
      0
    );
    validateCapacity(
      id,
      'lineCount * maximumVerticesPerLine',
      props.lineCount * props.maximumVerticesPerLine
    );
    validateFiniteNumber(id, 'tolerance', props.tolerance, 0);
    const operation = new GPUSegmentSnapOperation(props);
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
        const referencePositions = getCPUOperationFloat32Rows(
          id,
          inputs,
          'referencePositions',
          'float32x2'
        );
        const lineOffsets = getCPUOperationUint32Rows(id, inputs, 'lineOffsets');
        if (positions.length !== props.rowCount * 2)
          throw new Error(`${id} positions length must equal rowCount`);
        if (referencePositions.length !== props.referenceCount * 2)
          throw new Error(`${id} referencePositions length must equal referenceCount`);
        if (lineOffsets.length !== props.lineCount + 1)
          throw new Error(`${id} lineOffsets length must equal lineCount + 1`);
        const result = computeSegmentSnapOnCPU({positions, referencePositions, lineOffsets}, props);
        return {
          [props.output.positions]: result.positions,
          [props.output.sourceEdges]: result.sourceEdges,
          [props.output.counts]: result.counts,
          ...(props.output.validity !== undefined ? {[props.output.validity]: result.validity} : {})
        };
      }
    });
  }
}
