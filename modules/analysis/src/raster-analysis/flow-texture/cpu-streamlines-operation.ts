// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateMaximumCount
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {getPhilox4x32, getPhiloxUnitFloat} from '../../analysis-core/kernels/philox-random';
import {
  GPU_STREAMLINES_PARAMETER_LENGTH,
  GPU_STREAMLINES_UINT32_PARAMETER_LENGTH,
  STREAMLINES_SEED_PURPOSE
} from './streamlines-parameters';
import {GPUStreamlinesOperation, type StreamlinesOperationProps} from './gpu-streamlines-operation';

/** Priority keys reserve the low 16 bits for 0xffff minus the seed index. */
export const CPU_STREAMLINES_MAXIMUM_SEED_COUNT = 65535;

export type CPUStreamlinesInput = {
  velocities: Float32Array;
  parameters: Float32Array;
  uint32Parameters: Uint32Array;
};
export type CPUStreamlinesOptions = Pick<
  StreamlinesOperationProps,
  | 'fieldWidth'
  | 'fieldHeight'
  | 'gridWidth'
  | 'gridHeight'
  | 'seedColumns'
  | 'seedRows'
  | 'stepsPerDirection'
  | 'lineCapacity'
  | 'pointCapacity'
>;
export type CPUStreamlinesResult = {
  ids: Uint32Array;
  lineOffsets: Uint32Array;
  positions: Float32Array;
  positionCount: Uint32Array;
  count: Uint32Array;
  requiredCount: Uint32Array;
  overflow: Uint32Array;
  iterationLimitReached: Uint32Array;
  candidatePositions: Float32Array;
  candidateSpans: Uint32Array;
};

/**
 * Traces jittered seeds using normalized RK2 and bilinear cell-centre sampling, then resolves
 * occupancy greedily in descending Philox priority. Publishes complete lines in ascending seed
 * order, stopping at the first line that exceeds either capacity. Only active rows are defined.
 *
 * What differs from the GPU contributor:
 * - No pruning round limit: iterationLimitReached is zero and overflow never contains bit 2.
 *   Candidate spans contain traced backward/forward counts, then final kept counts for accepted
 *   seeds and zeros for rejected seeds; GPU rejected seeds can retain intermediate kept counts.
 * - Seed placement, interpolation, normalization and RK2 arithmetic use float64,
 *   rounded once to float32 positions. Compare positions with absolute tolerance 3e-5 times
 *   placement magnitude; near field/grid boundaries integer selections can differ.
 *   Seeds and steps whose final float32 position is outside the grid are rejected/truncated.
 *   Occupancy uses published float32 coordinates with the GPU's correctly rounded subtraction
 *   and reciprocal multiplication, so the exclusive bound is checked in that representation.
 * - The 65535 seed limit remains because priorities reserve 16 bits for seed index. GPU dimension
 *   caps are dropped; field rows still fit uint32 and occupancy indices must be safe integers.
 *   No maximumIterations limit is imposed by the CPU. Unused output rows are zero-initialized.
 * - NaN field corners invalidate a sample even at zero interpolation weight. Nonfinite
 *   directions/positions terminate tracing; subnormal finite velocities remain usable.
 *
 * @throws Error naming inconsistent arrays or invalid dimensions/capacities.
 */
export function computeStreamlinesOnCPU(
  input: CPUStreamlinesInput,
  options: CPUStreamlinesOptions
): CPUStreamlinesResult {
  validateOptions('computeStreamlinesOnCPU', options);
  if (input.velocities.length !== options.fieldWidth * options.fieldHeight * 2)
    throw new Error('velocities must contain fieldWidth * fieldHeight pairs');
  if (input.parameters.length < GPU_STREAMLINES_PARAMETER_LENGTH)
    throw new Error('parameters must hold the streamlines layout');
  if (input.uint32Parameters.length < GPU_STREAMLINES_UINT32_PARAMETER_LENGTH)
    throw new Error('uint32Parameters must hold the streamlines layout');
  // getGPUStreamlinesParameterValues and getGPUStreamlinesUint32ParameterValues layouts.
  const [
    fieldOriginX,
    fieldOriginY,
    fieldCellWidth,
    fieldCellHeight,
    gridOriginX,
    gridOriginY,
    gridCellWidth,
    gridCellHeight,
    inverseGridWidth,
    inverseGridHeight,
    stepLength,
    minimumSpeed
  ] = input.parameters;
  const [randomSeed, minimumPoints] = input.uint32Parameters;
  const seedCount = options.seedColumns * options.seedRows;
  const stride = 2 * options.stepsPerDirection + 1;
  const points = new Float64Array(seedCount * stride * 2).fill(NaN);
  const spans = new Uint32Array(seedCount * 2);
  const keys = new Uint32Array(seedCount);
  const getCell = (horizontal: number, vertical: number): number => {
    const localX = (horizontal - gridOriginX) * inverseGridWidth;
    const localY = (vertical - gridOriginY) * inverseGridHeight;
    if (!(localX >= 0 && localY >= 0 && localX < options.gridWidth && localY < options.gridHeight))
      return -1;
    return Math.floor(localY) * options.gridWidth + Math.floor(localX);
  };
  const getPublishedCell = (horizontal: number, vertical: number): number => {
    const localX = Math.fround(
      Math.fround(Math.fround(horizontal) - gridOriginX) * inverseGridWidth
    );
    const localY = Math.fround(
      Math.fround(Math.fround(vertical) - gridOriginY) * inverseGridHeight
    );
    if (!(localX >= 0 && localY >= 0 && localX < options.gridWidth && localY < options.gridHeight))
      return -1;
    return Math.floor(localY) * options.gridWidth + Math.floor(localX);
  };
  // Float64 counterpart of vector-field-sampling.ts; its CPU helper intentionally rounds every
  // intermediate to float32 and therefore cannot implement this twin's arithmetic contract.
  const getDirection = (
    horizontal: number,
    vertical: number,
    sign: number
  ): [number, number] | undefined => {
    const localX = (horizontal - fieldOriginX) / fieldCellWidth;
    const localY = (vertical - fieldOriginY) / fieldCellHeight;
    if (
      !(localX >= 0 && localY >= 0 && localX < options.fieldWidth && localY < options.fieldHeight)
    )
      return undefined;
    const gridX = Math.min(Math.max(localX - 0.5, 0), options.fieldWidth - 1);
    const gridY = Math.min(Math.max(localY - 0.5, 0), options.fieldHeight - 1);
    const column = Math.floor(gridX);
    const row = Math.floor(gridY);
    const nextColumn = Math.min(column + 1, options.fieldWidth - 1);
    const nextRow = Math.min(row + 1, options.fieldHeight - 1);
    const fractionX = gridX - column;
    const fractionY = gridY - row;
    const corners = [
      row * options.fieldWidth + column,
      row * options.fieldWidth + nextColumn,
      nextRow * options.fieldWidth + column,
      nextRow * options.fieldWidth + nextColumn
    ];
    if (
      corners.some(
        cell =>
          Number.isNaN(input.velocities[cell * 2]) || Number.isNaN(input.velocities[cell * 2 + 1])
      )
    )
      return undefined;
    const velocity = [0, 0];
    for (let component = 0; component < 2; component++) {
      const bottom =
        input.velocities[corners[0] * 2 + component] * (1 - fractionX) +
        input.velocities[corners[1] * 2 + component] * fractionX;
      const top =
        input.velocities[corners[2] * 2 + component] * (1 - fractionX) +
        input.velocities[corners[3] * 2 + component] * fractionX;
      velocity[component] = bottom * (1 - fractionY) + top * fractionY;
    }
    const speed = Math.hypot(...velocity);
    if (!(speed > 0) || speed < minimumSpeed || !Number.isFinite(speed)) return undefined;
    return [(sign * velocity[0]) / speed, (sign * velocity[1]) / speed];
  };
  const seedCellWidth = (gridCellWidth * options.gridWidth) / options.seedColumns;
  const seedCellHeight = (gridCellHeight * options.gridHeight) / options.seedRows;
  for (let seed = 0; seed < seedCount; seed++) {
    const random = getPhilox4x32([seed, 0, 0, 0], [randomSeed, STREAMLINES_SEED_PURPOSE]);
    keys[seed] = (((random[2] >>> 16) << 16) | (0xffff - seed)) >>> 0;
    const startX =
      gridOriginX + ((seed % options.seedColumns) + getPhiloxUnitFloat(random[0])) * seedCellWidth;
    const startY =
      gridOriginY +
      (Math.floor(seed / options.seedColumns) + getPhiloxUnitFloat(random[1])) * seedCellHeight;
    if (
      !getDirection(startX, startY, 1) ||
      getCell(startX, startY) < 0 ||
      getPublishedCell(startX, startY) < 0
    )
      continue;
    const base = seed * stride + options.stepsPerDirection;
    points.set([startX, startY], base * 2);
    for (const sign of [-1, 1]) {
      let horizontal = startX;
      let vertical = startY;
      for (let step = 1; step <= options.stepsPerDirection; step++) {
        const first = getDirection(horizontal, vertical, sign);
        if (!first) break;
        const middle = getDirection(
          horizontal + 0.5 * stepLength * first[0],
          vertical + 0.5 * stepLength * first[1],
          sign
        );
        if (!middle) break;
        const nextX = horizontal + stepLength * middle[0];
        const nextY = vertical + stepLength * middle[1];
        if (getCell(nextX, nextY) < 0 || getPublishedCell(nextX, nextY) < 0) break;
        horizontal = nextX;
        vertical = nextY;
        points.set([horizontal, vertical], (base + sign * step) * 2);
        spans[seed * 2 + Number(sign > 0)] = step;
      }
    }
  }
  const candidatePositions = Float32Array.from(points);
  const order = Array.from({length: seedCount}, (_, seed) => seed).sort(
    (first, second) => keys[second] - keys[first]
  );
  const occupied = new Set<number>();
  const kept = new Uint32Array(seedCount * 2);
  const accepted = new Uint8Array(seedCount);
  const getCandidateCell = (seed: number, step: number) => {
    const position = 2 * (seed * stride + options.stepsPerDirection + step);
    return getPublishedCell(candidatePositions[position], candidatePositions[position + 1]);
  };
  for (const seed of order) {
    const cell = getCandidateCell(seed, 0);
    if (cell < 0 || occupied.has(cell)) continue;
    const counts = [0, 0];
    for (const sign of [-1, 1]) {
      const direction = Number(sign > 0);
      while (
        counts[direction] < spans[seed * 2 + direction] &&
        !occupied.has(getCandidateCell(seed, sign * (counts[direction] + 1)))
      )
        counts[direction]++;
    }
    if (1 + counts[0] + counts[1] < minimumPoints) continue;
    accepted[seed] = 1;
    kept.set(counts, seed * 2);
    for (let step = -counts[0]; step <= counts[1]; step++)
      occupied.add(getCandidateCell(seed, step));
  }
  const result: CPUStreamlinesResult = {
    ids: new Uint32Array(options.lineCapacity),
    lineOffsets: new Uint32Array(options.lineCapacity + 1),
    positions: new Float32Array(options.pointCapacity * 2),
    positionCount: new Uint32Array(1),
    count: new Uint32Array(1),
    requiredCount: new Uint32Array(1),
    overflow: new Uint32Array(1),
    iterationLimitReached: new Uint32Array(1),
    candidatePositions,
    candidateSpans: new Uint32Array(seedCount * 4)
  };
  let requiredCount = 0;
  let count = 0;
  let positionCount = 0;
  let full = false;
  for (let seed = 0; seed < seedCount; seed++) {
    if (!accepted[seed]) continue;
    requiredCount++;
    const length = 1 + kept[seed * 2] + kept[seed * 2 + 1];
    if (count >= options.lineCapacity || positionCount + length > options.pointCapacity)
      full = true;
    if (full) continue;
    result.ids[count] = seed;
    result.lineOffsets[count] = positionCount;
    for (let step = -kept[seed * 2]; step <= kept[seed * 2 + 1]; step++) {
      const source = 2 * (seed * stride + options.stepsPerDirection + step);
      result.positions.set(candidatePositions.subarray(source, source + 2), positionCount++ * 2);
    }
    result.lineOffsets[++count] = positionCount;
  }
  result.count[0] = count;
  result.positionCount[0] = positionCount;
  result.requiredCount[0] = requiredCount;
  result.overflow[0] = Number(count < requiredCount);
  for (let seed = 0; seed < seedCount; seed++)
    result.candidateSpans.set(
      [spans[seed * 2], spans[seed * 2 + 1], kept[seed * 2], kept[seed * 2 + 1]],
      seed * 4
    );
  return result;
}

function validateOptions(id: string, options: CPUStreamlinesOptions): void {
  for (const port of [
    'fieldWidth',
    'fieldHeight',
    'gridWidth',
    'gridHeight',
    'seedColumns',
    'seedRows'
  ] as const) {
    validateCapacity(id, port, options[port]);
  }
  validateMaximumCount(
    id,
    'seedColumns * seedRows',
    options.seedColumns * options.seedRows,
    CPU_STREAMLINES_MAXIMUM_SEED_COUNT
  );
  validateCapacity(id, 'fieldWidth * fieldHeight', options.fieldWidth * options.fieldHeight);
  validateMaximumCount(
    id,
    'gridWidth * gridHeight',
    options.gridWidth * options.gridHeight,
    Number.MAX_SAFE_INTEGER
  );
  for (const port of ['stepsPerDirection', 'lineCapacity', 'pointCapacity'] as const)
    validateCapacity(id, port, options[port]);
}

/** Exact greedy streamline pruning without the GPU round budget. */
export class CPUStreamlinesOperation extends GPUCPUOperation {
  constructor(props: StreamlinesOperationProps) {
    const operation = new GPUStreamlinesOperation(props);
    const {id} = operation;
    validateOptions(id, props);
    if (props.maximumIterations !== undefined)
      validateCapacity(id, 'maximumIterations', props.maximumIterations);
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
        const result = computeStreamlinesOnCPU(
          {
            velocities: getCPUOperationFloat32Rows(id, inputs, 'velocities', 'float32x2'),
            parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters'),
            uint32Parameters: getCPUOperationUint32Rows(id, inputs, 'uint32Parameters')
          },
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUStreamlinesResult]
          ])
        );
      }
    });
  }
}
