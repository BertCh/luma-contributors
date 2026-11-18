// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {canExhaustivelySnap, MAXIMUM_SEGMENT_SNAP_EXHAUSTIVE_WORK} from './gpu-segment-snap';
import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {resolveGPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPUSegmentSnapOperation,
  snapshotSegmentSnapOperationProps,
  type SegmentSnapOperationProps
} from './gpu-segment-snap-operation';

/** Inclusive thresholds must remain finite in the GPU twin's f32 comparisons. */
const MAXIMUM_TOLERANCE = 3.4028234663852886e38;
/** Two scalar coordinates per row are indexed by uint32 in the GPU twin. */
const MAXIMUM_POSITION_COUNT = 0x7fffffff;
/** Four status words per line are indexed by uint32 in the GPU twin. */
const MAXIMUM_LINE_COUNT = 0x3fffffff;
/** Each GPU candidate record needs two headers, after three shared batch-header words. */
const MAXIMUM_CANDIDATE_CAPACITY = 0xfffffffa;
/** Default bounded grid list, matching the GPU contributor. */
const DEFAULT_CANDIDATE_CAPACITY = 1024;

/** Input line vertices and planar reference vertices. */
export type CPUSegmentSnapInput = {
  positions: Float32Array;
  lineOffsets: Uint32Array;
  referencePositions: Float32Array;
};
/** Fixed-slot output shape and inclusive distance threshold. */
export type CPUSegmentSnapOptions = {
  tolerance: number;
  maximumVerticesPerLine: number;
  searchStructure?: 'grid' | 'brute-force';
  candidateCapacity?: number;
  candidateOverflowPolicy?: 'bounded-brute-force' | 'incomplete';
};
/** Per-line slots and [written, required, overflow, nonfinite] status rows. */
export type CPUSegmentSnapResult = {
  positions: Float32Array;
  sourceEdges: Uint32Array;
  counts: Uint32Array;
  validity: Uint32Array;
  issueCode: Uint32Array;
  candidateOverflow: Uint32Array;
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
 *   input vertices invalidate their line. An empty line has validity=1; a collapsed line has
 *   issueCode bit 1 and validity=0.
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
  validateCPUSegmentSnapOptions(id, options);
  if (input.positions.length % 2 !== 0) throw new Error('positions must contain float32x2 rows');
  if (input.referencePositions.length % 2 !== 0)
    throw new Error('referencePositions must contain float32x2 rows');
  if (input.lineOffsets.length < 2)
    throw new Error('lineOffsets must contain at least two entries');
  const rowCount = input.positions.length / 2;
  const referenceCount = input.referencePositions.length / 2;
  if (options.searchStructure === 'brute-force')
    validateCapacity(
      id,
      'positions.length * (referencePositions.length + 1)',
      rowCount * (referenceCount + 1),
      0
    );
  validateCapacity(id, 'positions.length', rowCount, 0, MAXIMUM_POSITION_COUNT);
  validateCapacity(id, 'referencePositions.length', referenceCount, 0, MAXIMUM_POSITION_COUNT);
  const lineCount = input.lineOffsets.length - 1;
  validateCapacity(id, 'lineCount', lineCount, 1, MAXIMUM_LINE_COUNT);
  const capacity = options.maximumVerticesPerLine;
  validateCapacity(
    id,
    'lineCount * maximumVerticesPerLine',
    lineCount * capacity,
    0,
    MAXIMUM_POSITION_COUNT
  );
  if (options.searchStructure === 'brute-force')
    validateRequirement(
      id,
      canExhaustivelySnap(rowCount, referenceCount, lineCount),
      `(positions.length + lineCount) * (referencePositions.length + 1)^3 must not exceed MAXIMUM_SEGMENT_SNAP_EXHAUSTIVE_WORK (${MAXIMUM_SEGMENT_SNAP_EXHAUSTIVE_WORK})`
    );
  const exhaustiveAllowed =
    options.candidateOverflowPolicy !== 'incomplete' &&
    canExhaustivelySnap(rowCount, referenceCount, lineCount);
  const result: CPUSegmentSnapResult = {
    positions: new Float32Array(lineCount * capacity * 2),
    sourceEdges: new Uint32Array(lineCount * capacity),
    counts: new Uint32Array(lineCount * 4),
    candidateOverflow: computeCandidateOverflow(input, options),
    validity: new Uint32Array(lineCount),
    issueCode: new Uint32Array(lineCount)
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
    const incomplete = !exhaustiveAllowed && result.candidateOverflow[line] !== 0;
    let required = 0;
    const writeVertex = (point: Point, source: number) => {
      if (required < capacity) {
        const slot = line * capacity + required;
        result.positions.set([point.x, point.y], slot * 2);
        result.sourceEdges[slot] = source;
      }
      required = Math.min(0xffffffff, required + 1);
    };
    if (!invalid && !incomplete && last > first) {
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
              Math.abs((reference.x - start.x) * deltaY - (reference.y - start.y) * deltaX) /
                Math.sqrt(denominator) >
                tolerance
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
    result.issueCode[line] =
      Number(required === 1) | (invalid ? 2 : 0) | (required > capacity || incomplete ? 4 : 0);
    result.validity[line] = Number(result.issueCode[line] === 0);
  }
  return result;
}

/** CPU segment snap twin; fixed slots and uint32 required-count limits are retained. */
export class CPUSegmentSnapOperation extends GPUCPUOperation {
  constructor(props: SegmentSnapOperationProps) {
    props = snapshotSegmentSnapOperationProps(props);
    const id = props.id ?? 'segment-snap';
    resolveGPUSpatialContext(id, props.spatialContext);
    validateCapacity(id, 'rowCount', props.rowCount, 0);
    validateCapacity(id, 'referenceCount', props.referenceCount, 0);
    validateCapacity(id, 'lineCount', props.lineCount);
    validateCapacity(id, 'maximumVerticesPerLine', props.maximumVerticesPerLine);
    if (props.searchStructure === 'brute-force')
      validateCapacity(
        id,
        'rowCount * (referenceCount + 1)',
        props.rowCount * (props.referenceCount + 1),
        0
      );
    validateCapacity(
      id,
      'lineCount * maximumVerticesPerLine',
      props.lineCount * props.maximumVerticesPerLine,
      1,
      MAXIMUM_POSITION_COUNT
    );
    if (props.searchStructure === 'brute-force')
      validateRequirement(
        id,
        canExhaustivelySnap(props.rowCount, props.referenceCount, props.lineCount),
        `(rowCount + lineCount) * (referenceCount + 1)^3 must not exceed MAXIMUM_SEGMENT_SNAP_EXHAUSTIVE_WORK (${MAXIMUM_SEGMENT_SNAP_EXHAUSTIVE_WORK})`
      );
    validateCPUSegmentSnapOptions(id, props);
    validateCapacity(id, 'rowCount', props.rowCount, 0, MAXIMUM_POSITION_COUNT);
    validateCapacity(id, 'referenceCount', props.referenceCount, 0, MAXIMUM_POSITION_COUNT);
    validateCapacity(id, 'lineCount', props.lineCount, 1, MAXIMUM_LINE_COUNT);
    const operation = new GPUSegmentSnapOperation(props);
    const candidateOverflow =
      props.output.status?.candidateOverflow ?? props.output.candidateOverflow;
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
          ...(candidateOverflow !== undefined
            ? {[candidateOverflow]: result.candidateOverflow}
            : {}),
          ...(props.output.validity !== undefined
            ? {[props.output.validity]: result.validity}
            : {}),
          ...(props.output.issueCode !== undefined
            ? {[props.output.issueCode]: result.issueCode}
            : {})
        };
      }
    });
  }
}

/** CPU oracle for the index's conservative per-line candidate diagnostic. */
function computeCandidateOverflow(
  input: CPUSegmentSnapInput,
  options: CPUSegmentSnapOptions
): Uint32Array {
  const output = new Uint32Array(input.lineOffsets.length - 1);
  if (options.searchStructure === 'brute-force' || input.referencePositions.length === 0)
    return output;
  const width = 256;
  const minimum = [Infinity, Infinity];
  const maximum = [-Infinity, -Infinity];
  for (let row = 0; row < input.referencePositions.length; row += 2) {
    if (
      !Number.isFinite(input.referencePositions[row]) ||
      !Number.isFinite(input.referencePositions[row + 1])
    )
      continue;
    for (let axis = 0; axis < 2; axis++) {
      minimum[axis] = Math.min(minimum[axis], input.referencePositions[row + axis]);
      maximum[axis] = Math.max(maximum[axis], input.referencePositions[row + axis]);
    }
  }
  const normalize = (value: number, axis: number) => {
    const lower = Math.fround(minimum[axis] * 0.5);
    const extent = Math.fround(Math.fround(maximum[axis] * 0.5) - lower);
    return extent >= 2 ** -126
      ? Math.min(
          1,
          Math.max(0, Math.fround(Math.fround(Math.fround(value * 0.5) - lower) / extent))
        )
      : 0.5;
  };
  const cells = new Uint32Array(width * width);
  for (let row = 0; row < input.referencePositions.length; row += 2) {
    const horizontal = input.referencePositions[row];
    const vertical = input.referencePositions[row + 1];
    if (Number.isFinite(horizontal) && Number.isFinite(vertical))
      cells[
        Math.min(width - 1, Math.floor(normalize(vertical, 1) * width)) * width +
          Math.min(width - 1, Math.floor(normalize(horizontal, 0) * width))
      ]++;
  }
  const tolerance = Math.fround(options.tolerance);
  for (let line = 0; line < output.length; line++) {
    const lower = [1, 1];
    const upper = [0, 0];
    const first = Math.min(input.lineOffsets[line], input.positions.length / 2);
    const last = Math.min(input.lineOffsets[line + 1], input.positions.length / 2);
    let invalid = false;
    for (let row = first; row < last; row++)
      for (let axis = 0; axis < 2; axis++) {
        const value = input.positions[row * 2 + axis];
        if (!Number.isFinite(value)) {
          invalid = true;
          continue;
        }
        const roundingMargin = Math.fround(
          Math.fround(Math.max(Math.abs(value), tolerance) * 2 ** -20) + 2 ** -126
        );
        const low = normalize(
          Math.fround(Math.fround(Math.fround(value - tolerance) - tolerance) - roundingMargin),
          axis
        );
        const high = normalize(
          Math.fround(Math.fround(Math.fround(value + tolerance) + tolerance) + roundingMargin),
          axis
        );
        lower[axis] = Math.min(lower[axis], Number.isFinite(low) ? low : 0);
        upper[axis] = Math.max(upper[axis], Number.isFinite(high) ? high : 1);
      }
    if (invalid) continue;
    let required = 0;
    if (last > first)
      for (
        let vertical = Math.max(0, Math.floor(lower[1] * width) - 1);
        vertical <= Math.min(width - 1, Math.floor(upper[1] * width) + 1);
        vertical++
      )
        for (
          let horizontal = Math.max(0, Math.floor(lower[0] * width) - 1);
          horizontal <= Math.min(width - 1, Math.floor(upper[0] * width) + 1);
          horizontal++
        )
          required += cells[vertical * width + horizontal];
    output[line] = Number(required > (options.candidateCapacity ?? 1024));
  }
  return output;
}

function validateCPUSegmentSnapOptions(id: string, options: CPUSegmentSnapOptions): void {
  validateRequirement(
    id,
    options.candidateOverflowPolicy === undefined ||
      options.candidateOverflowPolicy === 'bounded-brute-force' ||
      options.candidateOverflowPolicy === 'incomplete',
    'candidateOverflowPolicy must be bounded-brute-force or incomplete'
  );

  validateFiniteNumber(id, 'tolerance', options.tolerance, 0, MAXIMUM_TOLERANCE);
  validateCapacity(
    id,
    'candidateCapacity',
    options.candidateCapacity ?? DEFAULT_CANDIDATE_CAPACITY,
    1,
    MAXIMUM_CANDIDATE_CAPACITY
  );
  validateRequirement(
    id,
    options.searchStructure === undefined ||
      options.searchStructure === 'grid' ||
      options.searchStructure === 'brute-force',
    'searchStructure must be grid or brute-force'
  );
}
