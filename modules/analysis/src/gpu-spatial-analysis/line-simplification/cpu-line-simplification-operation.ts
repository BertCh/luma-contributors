// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  resolveGPUSpatialContext,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  getGPUGreatCircleRadius,
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
  computeLongitudeDelta,
  computeCosLatitude,
  computeCentralAngle,
  wrapLongitude
} from '../../analysis-core/kernels/cpu-geodesic';
import {GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH} from './line-simplification-parameters';
import {
  GPULineSimplificationOperation,
  type LineSimplificationOperationProps
} from './gpu-line-simplification-operation';

/** Packed line rows; importance is required in selection-only mode. */
export type CPULineSimplificationInput = {
  positions: Float32Array;
  lineOffsets: Uint32Array;
  timestamps?: Float32Array;
  parameters?: Float32Array;
  importance?: Float32Array;
};
/** Settings for the pure computation; GPU scheduling budgets are accepted but not imposed. */
export type CPULineSimplificationOptions = {
  spatialContext?: GPUSpatialContext;
  method?: 'douglas-peucker' | 'visvalingam';
  metric?: 'segment' | 'time-ratio';
  computeImportance?: boolean;
  finishSpanLimit?: number;
  capacity?: number;
};
/** Same fixed-size and compact columns as GPULineSimplification. */
export type CPULineSimplificationResult = {
  importance: Float32Array;
  ids: Uint32Array;
  outputMask: Uint32Array;
  lineCounts: Uint32Array;
  lineStarts: Uint32Array;
  count: Uint32Array;
  requiredCount: Uint32Array;
  overflow: Uint32Array;
  converged: Uint32Array;
  iterationCount: Uint32Array;
};

function resolveOptions(id: string, options: CPULineSimplificationOptions): number | undefined {
  validateRequirement(
    id,
    options.method === undefined ||
      options.method === 'douglas-peucker' ||
      options.method === 'visvalingam',
    'method must be douglas-peucker or visvalingam'
  );
  validateRequirement(
    id,
    options.metric === undefined || options.metric === 'segment' || options.metric === 'time-ratio',
    'metric must be segment or time-ratio'
  );
  const context = resolveGPUSpatialContext(id, options.spatialContext, {
    ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
    measure: options.method === 'visvalingam' ? 'area' : 'distance'
  });
  const radius = getGPUGreatCircleRadius(context);
  validateRequirement(
    id,
    options.method !== 'visvalingam' ||
      (radius === undefined && (options.metric ?? 'segment') === 'segment'),
    'method visvalingam requires planar positions and metric segment'
  );
  validateCapacity(id, 'finishSpanLimit', options.finishSpanLimit ?? 32, 0);
  return radius;
}

function computeDistance(
  input: CPULineSimplificationInput,
  options: CPULineSimplificationOptions,
  radius: number | undefined,
  row: number,
  first: number,
  last: number
): number {
  const positions = input.positions;
  const pointX = positions[row * 2];
  const pointY = positions[row * 2 + 1];
  const firstX = positions[first * 2];
  const firstY = positions[first * 2 + 1];
  const lastX = positions[last * 2];
  const lastY = positions[last * 2 + 1];
  let fraction = 0;
  if (options.metric === 'time-ratio') {
    const timestamps = input.timestamps!;
    const duration = timestamps[last] - timestamps[first];
    if (duration > 0) fraction = (timestamps[row] - timestamps[first]) / duration;
    const expectedX =
      firstX + fraction * (radius === undefined ? lastX - firstX : wrapLongitude(lastX - firstX));
    const expectedY = firstY + fraction * (lastY - firstY);
    return radius === undefined
      ? Math.hypot(pointX - expectedX, pointY - expectedY)
      : radius *
          computeCentralAngle(
            (computeLongitudeDelta(pointX, expectedX) * Math.PI) / 180,
            pointY,
            expectedY
          );
  }
  if (radius === undefined) {
    const segmentX = lastX - firstX;
    const segmentY = lastY - firstY;
    const offsetX = pointX - firstX;
    const offsetY = pointY - firstY;
    const squared = segmentX * segmentX + segmentY * segmentY;
    const projection = offsetX * segmentX + offsetY * segmentY;
    if (squared <= 0 || projection <= 0) return Math.hypot(offsetX, offsetY);
    if (projection >= squared) return Math.hypot(pointX - lastX, pointY - lastY);
    return Math.abs(segmentX * offsetY - segmentY * offsetX) / Math.sqrt(squared);
  }
  // geodesicSegmentFraction and geodesicSegmentDistance, including their separate seam deltas.
  const cosine = computeCosLatitude(pointY);
  const startX = computeLongitudeDelta(pointX, firstX) * cosine;
  const startY = firstY - pointY;
  const segmentX = computeLongitudeDelta(pointX, lastX) * cosine - startX;
  const segmentY = lastY - pointY - startY;
  const squared = segmentX * segmentX + segmentY * segmentY;
  if ((firstX !== lastX || firstY !== lastY) && squared > 0)
    fraction = Math.max(0, Math.min(1, -(startX * segmentX + startY * segmentY) / squared));
  const longitudeOffset =
    computeLongitudeDelta(pointX, firstX) + fraction * computeLongitudeDelta(firstX, lastX);
  const latitudeOffset = firstY - pointY + fraction * (lastY - firstY);
  const term =
    Math.sin((latitudeOffset * Math.PI) / 360) ** 2 +
    computeCosLatitude(pointY) *
      computeCosLatitude(pointY + latitudeOffset) *
      Math.sin((longitudeOffset * Math.PI) / 360) ** 2;
  return 2 * radius * Math.asin(Math.sqrt(Math.max(0, Math.min(1, term))));
}

/**
 * Computes a complete Douglas-Peucker split tree or sequential Visvalingam-Whyatt effective
 * areas, then selects rows using the packed tolerance. Intermediates stay in float64, including
 * comparisons to tolerance; the returned importance column is rounded only after selection.
 *
 * What differs from the GPU contributor:
 * - No iteration, finish-span, or neighborhood budget limits correctness. Douglas-Peucker is
 *   complete; Visvalingam uses global smallest-(area,row) sequential removal per line, with the
 *   effective-area floor propagated to surviving neighbors, rather than a parallel approximation.
 * - Distances/areas use float64 (ordinary division and square root), not the GPU's defined f32
 *   floor operations. Near ties and tolerance boundaries can therefore select different rows.
 * - converged is always 1. iterationCount is the unbounded logical split-round count for
 *   Douglas-Peucker (honoring finishSpanLimit), or the maximum sequential removals in one line
 *   for Visvalingam (at least 1). The latter is not the GPU's parallel round count.
 * - Unused ids slots are zero; the GPU leaves them unwritten. Only the first count slots are
 *   meaningful. Per-line counts/starts are unclamped, as on the GPU.
 * - Nonfinite positions or timestamps used by the metric throw an Error naming the argument. GPU results for them
 *   and subnormal arithmetic are unspecified; finite subnormals are evaluated in float64 here.
 * - Finite inputs whose GPU intermediates overflow are evaluated in float64. If the final
 *   importance exceeds the float32 range, it rounds to positive Infinity.
 *
 * @throws For inconsistent arrays, nonmonotone/out-of-range offsets, or unsupported options.
 */
export function computeLineSimplificationOnCPU(
  input: CPULineSimplificationInput,
  options: CPULineSimplificationOptions = {}
): CPULineSimplificationResult {
  const radius = resolveOptions('computeLineSimplificationOnCPU', options);
  const rowCount = input.positions.length / 2;
  if (!Number.isInteger(rowCount)) throw new Error('positions must contain float32x2 rows');
  if (!input.positions.every(Number.isFinite)) throw new Error('positions must be finite');
  if (input.lineOffsets.length < 2)
    throw new Error('lineOffsets must contain at least two entries');
  for (let index = 0; index < input.lineOffsets.length; index++)
    if (
      input.lineOffsets[index] > rowCount ||
      (index > 0 && input.lineOffsets[index] < input.lineOffsets[index - 1])
    )
      throw new Error('lineOffsets must be nondecreasing and within positions');
  if (input.timestamps && input.timestamps.length !== rowCount)
    throw new Error('timestamps length must equal positions row count');
  if (
    options.metric === 'time-ratio' &&
    options.computeImportance !== false &&
    (!input.timestamps || !input.timestamps.every(Number.isFinite))
  )
    throw new Error('timestamps must contain one finite value per position');
  if (
    options.computeImportance === false &&
    (!input.importance || input.importance.length !== rowCount)
  )
    throw new Error('importance length must equal positions row count');
  if (input.parameters && input.parameters.length < GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH)
    throw new Error(`parameters must contain ${GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH} values`);
  const capacity = options.capacity ?? rowCount;
  validateCapacity('computeLineSimplificationOnCPU', 'capacity', capacity, 0);
  const lineCount = input.lineOffsets.length - 1;
  const importance =
    options.computeImportance === false
      ? Array.from(input.importance!)
      : Array<number>(rowCount).fill(0);
  let iterationCount = 1;
  if (options.computeImportance !== false)
    for (let line = 0; line < lineCount; line++) {
      const first = input.lineOffsets[line];
      const last = input.lineOffsets[line + 1] - 1;
      if (last < first) continue;
      importance[first] = Infinity;
      importance[last] = Infinity;
      if (options.method === 'visvalingam') {
        const survivors = Array.from({length: last - first + 1}, (_, index) => first + index);
        const floors = new Map<number, number>();
        const interiorCount = survivors.length - 2;
        iterationCount = Math.max(iterationCount, interiorCount);
        while (survivors.length > 2) {
          let bestSlot = 1;
          let bestArea = Infinity;
          for (let slot = 1; slot < survivors.length - 1; slot++) {
            const previous = survivors[slot - 1];
            const row = survivors[slot];
            const next = survivors[slot + 1];
            const area =
              Math.abs(
                (input.positions[row * 2] - input.positions[previous * 2]) *
                  (input.positions[next * 2 + 1] - input.positions[previous * 2 + 1]) -
                  (input.positions[row * 2 + 1] - input.positions[previous * 2 + 1]) *
                    (input.positions[next * 2] - input.positions[previous * 2])
              ) / 2;
            if (area < bestArea) {
              bestArea = area;
              bestSlot = slot;
            }
          }
          const row = survivors[bestSlot];
          importance[row] = Math.max(bestArea, floors.get(row) ?? 0);
          for (const neighbor of [survivors[bestSlot - 1], survivors[bestSlot + 1]])
            floors.set(neighbor, Math.max(floors.get(neighbor) ?? 0, importance[row]));
          survivors.splice(bestSlot, 1);
        }
      } else {
        const pending = [{first, last, parent: Infinity, round: 1}];
        while (pending.length > 0) {
          const interval = pending.pop()!;
          if (interval.last - interval.first < 2) continue;
          iterationCount = Math.max(iterationCount, interval.round);
          let bestDistance = 0;
          let bestRow = -1;
          for (let row = interval.first + 1; row < interval.last; row++) {
            const distance = computeDistance(
              input,
              options,
              radius,
              row,
              interval.first,
              interval.last
            );
            if (distance > bestDistance) {
              bestDistance = distance;
              bestRow = row;
            }
          }
          if (bestRow < 0) continue;
          const value = Math.min(bestDistance, interval.parent);
          importance[bestRow] = value;
          const round =
            interval.last - interval.first - 1 <= (options.finishSpanLimit ?? 32)
              ? interval.round
              : interval.round + 1;
          pending.push(
            {first: interval.first, last: bestRow, parent: value, round},
            {first: bestRow, last: interval.last, parent: value, round}
          );
        }
      }
    }
  const result: CPULineSimplificationResult = {
    importance: new Float32Array(importance),
    ids: new Uint32Array(capacity),
    outputMask: new Uint32Array(rowCount),
    lineCounts: new Uint32Array(lineCount),
    lineStarts: new Uint32Array(lineCount),
    count: new Uint32Array(1),
    requiredCount: new Uint32Array(1),
    overflow: new Uint32Array(1),
    converged: new Uint32Array([1]),
    iterationCount: new Uint32Array([iterationCount])
  };
  // The packer defines [tolerance, 0, 0, 0]; it exposes a length constant, no offset constants.
  const [tolerance] = input.parameters ?? [Infinity];
  let total = 0;
  for (let line = 0; line < lineCount; line++) {
    result.lineStarts[line] = total;
    for (let row = input.lineOffsets[line]; row < input.lineOffsets[line + 1]; row++)
      if (importance[row] > tolerance) {
        result.outputMask[row] = 1;
        if (total < capacity) result.ids[total] = row;
        total++;
        result.lineCounts[line]++;
      }
  }
  result.count[0] = Math.min(total, capacity);
  result.requiredCount[0] = total;
  result.overflow[0] = total > capacity ? 1 : 0;
  return result;
}

/** Complete CPU simplification; GPU iteration and neighborhood caps are dropped. */
export class CPULineSimplificationOperation extends GPUCPUOperation {
  constructor(props: LineSimplificationOperationProps) {
    const id = props.id ?? 'line-simplification';
    resolveOptions(id, props);
    validateCapacity(id, 'rowCount', props.rowCount);
    validateCapacity(id, 'lineCount', props.lineCount);
    validateCapacity(id, 'capacity', props.capacity ?? props.rowCount, 0);
    validateCapacity(id, 'maximumIterations', props.maximumIterations ?? 64);
    if (props.method === 'visvalingam')
      validateCapacity(id, 'neighborhoodRadius', props.neighborhoodRadius ?? 3);
    validateRequirement(
      id,
      props.computeImportance !== false || props.output?.ids !== undefined,
      'computeImportance=false requires output.ids'
    );
    validateRequirement(
      id,
      props.computeImportance !== false ||
        (props.output?.status?.converged === undefined &&
          props.output?.status?.iterationCount === undefined),
      'output convergence diagnostics require computeImportance'
    );
    validateRequirement(
      id,
      props.output?.ids === undefined ||
        (props.parameters !== undefined &&
          props.output.status?.count !== undefined &&
          props.output.status.overflow !== undefined),
      'output.ids requires parameters and status.count/overflow'
    );
    validateRequirement(
      id,
      props.metric !== 'time-ratio' ||
        props.computeImportance === false ||
        props.timestamps !== undefined,
      'metric time-ratio requires timestamps'
    );
    validateRequirement(
      id,
      props.output?.ids !== undefined ||
        [
          props.output?.outputMask,
          props.output?.lineCounts,
          props.output?.lineStarts,
          props.output?.status?.count,
          props.output?.status?.overflow,
          props.output?.status?.requiredCount
        ].every(value => value === undefined),
      'selection columns require output.ids'
    );
    const operation = new GPULineSimplificationOperation(props);
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
        const lineOffsets = getCPUOperationUint32Rows(id, inputs, 'lineOffsets');
        if (positions.length !== props.rowCount * 2)
          throw new Error(`${id} positions length must equal rowCount`);
        if (lineOffsets.length !== props.lineCount + 1)
          throw new Error(`${id} lineOffsets length must equal lineCount + 1`);
        const result = computeLineSimplificationOnCPU(
          {
            positions,
            lineOffsets,
            timestamps:
              props.timestamps !== undefined
                ? getCPUOperationFloat32Rows(id, inputs, 'timestamps')
                : undefined,
            parameters:
              props.parameters !== undefined
                ? getCPUOperationFloat32Rows(id, inputs, 'parameters')
                : undefined,
            importance:
              props.computeImportance === false
                ? getCPUOperationFloat32Rows(id, inputs, 'importance')
                : undefined
          },
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPULineSimplificationResult]
          ])
        );
      }
    });
  }
}
