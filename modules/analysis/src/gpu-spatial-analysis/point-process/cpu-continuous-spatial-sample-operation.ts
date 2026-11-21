// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateSamplingSeed} from './sampling-seed';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotContinuousSampleFrame} from './continuous-sample-frame-construction';
import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {resolveGPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getPhilox4x32,
  getPhiloxUnitFloat,
  PHILOX_PURPOSES
} from '../../analysis-core/kernels/philox-random';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPUContinuousSpatialSampleOperation,
  snapshotContinuousSampleOperation,
  type ContinuousSpatialSampleOperationProps
} from './gpu-continuous-spatial-sample-operation';

const NONE = 0xffffffff;

/** Packed frame vertices and optional legacy sites (arc lengths for lines, xy pairs for areas). */
export type CPUContinuousSpatialSampleInput = {
  positions: Float32Array;
  legacySites?: Float32Array;
  parameters?: Uint32Array;
};
/** Complete-wave design properties. maximumCandidates is a GPU-only budget and is ignored. */
export type CPUContinuousSpatialSampleOptions = Pick<
  ContinuousSpatialSampleOperationProps,
  | 'id'
  | 'spatialContext'
  | 'spacing'
  | 'seed'
  | 'replicate'
  | 'waves'
  | 'capacity'
  | 'maximumCandidates'
> & {frame: {type: 'line'} | {type: 'area'; bounds: readonly [number, number, number, number]}};
/** Wave-major slots and [written, required, randomAccepted, flags] per wave. */
export type CPUContinuousSpatialSampleResult = {
  positions: Float32Array;
  designWeights: Float32Array;
  provenance: Uint32Array;
  status: Uint32Array;
};

function validateOptions(
  options: CPUContinuousSpatialSampleOptions,
  frameRowCount: number,
  legacySiteCount: number,
  resident = false
): void {
  const id = options.id ?? 'continuous-spatial-sample';
  resolveGPUSpatialContext(id, options.spatialContext);
  validateRequirement(
    id,
    ['line', 'area'].includes(options.frame.type),
    `frame.type must be line or area; received ${options.frame.type}`
  );
  validateCapacity(id, 'frameRowCount', frameRowCount, options.frame.type === 'area' ? 3 : 2);
  validateRequirement(
    id,
    options.frame.type !== 'area' || frameRowCount % 3 === 0,
    'frame.positions must hold whole triangles (a multiple of 3 rows)'
  );
  validateRequirement(
    id,
    Number.isFinite(options.spacing) &&
      options.spacing > 0 &&
      Number.isFinite(options.spacing ** 2),
    'spacing must be positive and have a finite float64 square'
  );
  validateCapacity(id, 'waves', options.waves ?? 1);
  validateCapacity(id, 'capacity', options.capacity);
  validateMaximumCount(id, 'waves times capacity', (options.waves ?? 1) * options.capacity, NONE);
  validateCapacity(id, 'legacySiteCount', legacySiteCount, 0);
  validateSamplingSeed(id, options.seed, resident);
  validateCapacity(id, 'replicate', options.replicate ?? 0, 0, NONE - (options.waves ?? 1) + 1);
  if (options.frame.type === 'area') {
    const bounds = options.frame.bounds;
    validateRequirement(
      id,
      bounds.length === 4 && bounds.every(Number.isFinite),
      'frame.bounds must contain four finite numbers'
    );
    validateRequirement(
      id,
      bounds[2] > bounds[0] && bounds[3] > bounds[1],
      'frame.bounds must have a positive extent on both axes'
    );
    const columns = Math.ceil((bounds[2] - bounds[0]) / options.spacing) + 1;
    const rows = Math.ceil((bounds[3] - bounds[1]) / options.spacing) + 1;
    // Candidate ordinals and required counts occupy uint32 fields, also on CPU.
    validateMaximumCount(
      id,
      'frame.bounds lattice cells plus legacySiteCount',
      columns * rows + legacySiteCount,
      NONE - 1
    );
  }
}

/**
 * Enumerates complete random-shift line or square lattices in float64. Triangle edges are
 * inclusive, first triangle wins ties, upper area bounds and the line's final arc length are
 * excluded. Legacy sites appear first, carry zero weight and exclude candidates strictly closer
 * than spacing in the frame's metric. Invalid geometry or legacy sites set flag 1 and emit no
 * rows; capacity truncation sets flag 2 while preserving the full requirement. Unused slots are
 * zero with source/ordinal/role 0xffffffff. A resident replicate overflow invalidates that wave.
 *
 * What differs from the GPU contributor:
 * - Float64 geometry, phases and interpolation round once at output; compare coordinates with
 *   absolute tolerance 2e-5 on unit-scale frames. Design weights match the GPU's float32 literals
 *   exactly for corresponding slots.
 *   Near triangle edges, segment boundaries or legacy exclusion radii, classifications may differ.
 * - All lattice candidates are examined: maximumCandidates is ignored and flag 4 is never set.
 * - GPU-only frame, wave, capacity and legacy-site caps and the coordinate/spacing precision
 *   guard are removed. uint32 slot, candidate-ordinal and required-count layout limits remain.
 * - Finite positive spacing with a finite float64 square and finite bounds are accepted even
 *   outside float32 arithmetic limits; output values may round to zero or infinity. Non-finite
 *   frame or legacy coordinates still yield invalid flag 1, as on GPU.
 *
 * @throws If positions, legacySites or parameters lengths are inconsistent, or a lattice cannot
 * be represented by the uint32 provenance/count layout.
 */
export function computeContinuousSpatialSampleOnCPU(
  input: CPUContinuousSpatialSampleInput,
  options: CPUContinuousSpatialSampleOptions
): CPUContinuousSpatialSampleResult {
  const capturedOptions = snapshotConstructionFields(options, [
    'id',
    'spatialContext',
    'spacing',
    'seed',
    'replicate',
    'waves',
    'capacity',
    'maximumCandidates',
    'frame'
  ]);
  options = {
    ...capturedOptions,
    frame: snapshotContinuousSampleFrame(
      capturedOptions.id ?? 'continuous-spatial-sample',
      capturedOptions.frame
    )
  };
  const {positions, legacySites, parameters} = input;
  const area = options.frame.type === 'area';
  if (positions.length % 2 !== 0) throw new Error('positions must contain xy pairs');
  if (legacySites && area && legacySites.length % 2 !== 0)
    throw new Error('legacySites must contain xy pairs for an area');
  if (parameters && parameters.length !== 2)
    throw new Error('parameters must contain [seed, replicate]');
  const frameRowCount = positions.length / 2;
  const legacySiteCount = (legacySites?.length ?? 0) / (area ? 2 : 1);
  validateOptions(options, frameRowCount, legacySiteCount, parameters !== undefined);
  const waves = options.waves ?? 1;
  const {spacing, capacity} = options;
  const [seed, initialReplicate] = parameters ?? [options.seed!, options.replicate ?? 0];
  const bounds = options.frame.type === 'area' ? options.frame.bounds : undefined;
  const columns = bounds ? Math.ceil((bounds[2] - bounds[0]) / spacing) + 1 : 0;
  const rows = bounds ? Math.ceil((bounds[3] - bounds[1]) / spacing) + 1 : 0;
  const result: CPUContinuousSpatialSampleResult = {
    positions: new Float32Array(waves * capacity * 2),
    designWeights: new Float32Array(waves * capacity),
    provenance: new Uint32Array(waves * capacity * 4).fill(NONE),
    status: new Uint32Array(waves * 4)
  };
  const getPosition = (row: number): [number, number] => [
    positions[row * 2],
    positions[row * 2 + 1]
  ];
  const getCross = (first: readonly number[], second: readonly number[]) =>
    first[0] * second[1] - first[1] * second[0];
  const subtractPoints = (first: readonly number[], second: readonly number[]) => [
    first[0] - second[0],
    first[1] - second[1]
  ];
  const findTriangle = (point: readonly number[]): number => {
    for (let triangle = 0; triangle < frameRowCount / 3; triangle++) {
      const first = getPosition(triangle * 3);
      const second = getPosition(triangle * 3 + 1);
      const third = getPosition(triangle * 3 + 2);
      const orientation = Math.sign(
        getCross(subtractPoints(second, first), subtractPoints(third, first))
      );
      if (
        getCross(subtractPoints(second, first), subtractPoints(point, first)) * orientation >= 0 &&
        getCross(subtractPoints(third, second), subtractPoints(point, second)) * orientation >= 0 &&
        getCross(subtractPoints(first, third), subtractPoints(point, third)) * orientation >= 0
      )
        return triangle;
    }
    return NONE;
  };
  const segmentLengths = new Float64Array(frameRowCount - 1);
  let measure = 0;
  let invalidFrame = false;
  for (let vertex = 0; vertex < frameRowCount; vertex++) {
    const point = getPosition(vertex);
    if (
      !point.every(Number.isFinite) ||
      (bounds &&
        (point[0] < bounds[0] ||
          point[0] > bounds[2] ||
          point[1] < bounds[1] ||
          point[1] > bounds[3]))
    )
      invalidFrame = true;
    if (!area && vertex + 1 < frameRowCount) {
      const difference = subtractPoints(point, getPosition(vertex + 1));
      segmentLengths[vertex] = Math.sqrt(difference[0] ** 2 + difference[1] ** 2);
      measure += segmentLengths[vertex];
    }
  }
  if (area) {
    for (let triangle = 0; triangle < frameRowCount / 3; triangle++) {
      const first = getPosition(triangle * 3);
      const determinant = getCross(
        subtractPoints(getPosition(triangle * 3 + 1), first),
        subtractPoints(getPosition(triangle * 3 + 2), first)
      );
      if (!Number.isFinite(determinant) || determinant === 0) invalidFrame = true;
    }
  } else if (!Number.isFinite(measure) || measure <= 0) invalidFrame = true;
  const locateMeasure = (distanceAlong: number): {point: [number, number]; source: number} => {
    let consumed = 0;
    for (let segment = 0; segment < segmentLengths.length; segment++) {
      const extent = segmentLengths[segment];
      if (extent > 0 && distanceAlong <= consumed + extent) {
        const fraction = Math.min(1, Math.max(0, (distanceAlong - consumed) / extent));
        const first = getPosition(segment);
        const second = getPosition(segment + 1);
        return {
          point: [
            first[0] + (second[0] - first[0]) * fraction,
            first[1] + (second[1] - first[1]) * fraction
          ],
          source: segment
        };
      }
      consumed += extent;
    }
    return {point: getPosition(frameRowCount - 1), source: frameRowCount - 2};
  };
  const legacyPoints = Array.from({length: legacySiteCount}, (_, index) =>
    area
      ? [legacySites![index * 2], legacySites![index * 2 + 1]]
      : locateMeasure(legacySites![index]).point
  );
  for (const [index, point] of legacyPoints.entries()) {
    const distanceAlong = area ? 0 : legacySites![index];
    if (
      !point.every(Number.isFinite) ||
      !Number.isFinite(distanceAlong) ||
      (area ? findTriangle(point) === NONE : distanceAlong < 0 || distanceAlong > measure)
    )
      invalidFrame = true;
    for (let previous = 0; previous < index; previous++) {
      const separation = area
        ? Math.hypot(point[0] - legacyPoints[previous][0], point[1] - legacyPoints[previous][1])
        : Math.abs(distanceAlong - legacySites![previous]);
      if (separation < spacing) invalidFrame = true;
    }
  }
  if (!area && !invalidFrame)
    validateMaximumCount(
      options.id ?? 'continuous-spatial-sample',
      'positions arc-length lattice plus legacySites',
      Math.ceil(measure / spacing) + legacySiteCount,
      NONE - 1
    );
  for (let wave = 0; wave < waves; wave++) {
    const replicate = (initialReplicate + wave) >>> 0;
    let failure = invalidFrame || initialReplicate > NONE - wave ? 1 : 0;
    let required = 0;
    let randomAccepted = 0;
    const writeSample = (
      point: readonly number[],
      weight: number,
      source: number,
      ordinal: number,
      role: number
    ) => {
      if (required < capacity) {
        const slot = wave * capacity + required;
        result.positions.set(point, slot * 2);
        result.designWeights[slot] = weight;
        result.provenance.set([source, replicate, ordinal, role], slot * 4);
      }
      required++;
    };
    // Replicate is defined even for invalid/unused slots.
    for (let slot = 0; slot < capacity; slot++)
      result.provenance[(wave * capacity + slot) * 4 + 1] = replicate;
    if (failure === 0) {
      for (const [index, point] of legacyPoints.entries()) writeSample(point, 0, index, index, 2);
      const words = getPhilox4x32(
        [replicate, 0, 0, 0],
        [seed, PHILOX_PURPOSES.continuousSpatialSampling.id]
      );
      const horizontalPhase = getPhiloxUnitFloat(words[0]) * spacing;
      const verticalPhase = getPhiloxUnitFloat(words[1]) * spacing;
      const candidateCount = area
        ? columns * rows
        : Math.max(0, Math.ceil((measure - horizontalPhase) / spacing));
      for (let candidate = 0; candidate < candidateCount; candidate++) {
        const distanceAlong = horizontalPhase + candidate * spacing;
        const location = bounds
          ? {
              point: [
                bounds[0] + horizontalPhase + (candidate % columns) * spacing,
                bounds[1] + verticalPhase + Math.floor(candidate / columns) * spacing
              ],
              source: NONE
            }
          : locateMeasure(distanceAlong);
        if (bounds) location.source = findTriangle(location.point);
        let accepted = bounds
          ? location.source !== NONE &&
            location.point[0] < bounds[2] &&
            location.point[1] < bounds[3]
          : distanceAlong < measure;
        for (const [index, point] of legacyPoints.entries()) {
          const separation = area
            ? Math.hypot(location.point[0] - point[0], location.point[1] - point[1])
            : Math.abs(distanceAlong - legacySites![index]);
          if (separation < spacing) accepted = false;
        }
        if (accepted) {
          writeSample(
            location.point,
            area ? spacing ** 2 : spacing,
            location.source,
            candidate,
            wave === 0 ? 0 : 1
          );
          randomAccepted++;
        }
      }
    }
    if (required > capacity) failure |= 2;
    result.status.set([Math.min(required, capacity), required, randomAccepted, failure], wave * 4);
  }
  return result;
}

/** CPU twin; enumerate complete waves without the GPU's candidate or reference-tier work caps. */
export class CPUContinuousSpatialSampleOperation extends GPUCPUOperation {
  constructor(props: ContinuousSpatialSampleOperationProps) {
    props = snapshotContinuousSampleOperation(props);
    const id = props.id ?? 'continuous-spatial-sample';
    const legacySiteCount = props.legacySiteCount ?? 0;
    validateOptions(props, props.frameRowCount, legacySiteCount, props.parameters !== undefined);
    validateRequirement(
      id,
      props.frame.legacySites !== undefined || legacySiteCount === 0,
      'legacySiteCount must be zero without frame.legacySites'
    );
    const operation = new GPUContinuousSpatialSampleOperation(props);
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
        const legacySites =
          props.frame.legacySites !== undefined
            ? getCPUOperationFloat32Rows(
                id,
                inputs,
                'legacySites',
                props.frame.type === 'line' ? 'float32' : 'float32x2'
              )
            : undefined;
        if (positions.length !== props.frameRowCount * 2)
          throw new Error(`${id} positions length must equal frameRowCount * 2`);
        if ((legacySites?.length ?? 0) !== legacySiteCount * (props.frame.type === 'line' ? 1 : 2))
          throw new Error(`${id} legacySites length must match legacySiteCount`);
        const result = computeContinuousSpatialSampleOnCPU(
          {
            positions,
            legacySites,
            parameters:
              props.parameters !== undefined
                ? getCPUOperationUint32Rows(id, inputs, 'parameters')
                : undefined
          },
          props
        );
        return Object.fromEntries(
          Object.entries(props.output).map(([port, value]) => [
            value,
            result[port as keyof CPUContinuousSpatialSampleResult]
          ])
        );
      }
    });
  }
}
