// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement,
  validateFiniteNumber
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  ISOLINES_MAXIMUM_SAMPLE_COUNT,
  validateIsolinesTile,
  validateIsolinesMaximumSegmentCount
} from './isolines-tile';
import {GPU_ISOLINES_PARAMETER_LENGTH} from './isolines-parameters';
import {GPUIsolinesOperation, type IsolinesOperationProps} from './gpu-isolines-operation';

/** Two raster edge families must fit in uint32 global edge identifiers. */
export const CPU_ISOLINES_MAXIMUM_CELL_COUNT = ISOLINES_MAXIMUM_SAMPLE_COUNT;

export type CPUIsolinesInput = {
  values: Float32Array;
  mask?: Uint32Array;
  levels: Float32Array;
  parameters: Float32Array;
};
export type CPUIsolinesOptions = Pick<
  IsolinesOperationProps,
  'grid' | 'noDataValue' | 'segmentCapacity' | 'tile'
> & {vertexCapacity?: number};
export type CPUIsolinesResult = {
  segments: Float32Array;
  segmentLevels: Uint32Array;
  segmentEdges: Uint32Array;
  count: Uint32Array;
  requiredCount: Uint32Array;
  overflow: Uint32Array;
  polylineOverflow: Uint32Array;
  vertices: Float32Array;
  polylineOffsets: Uint32Array;
  polylineLevels: Uint32Array;
  polylineClosed: Uint32Array;
  polylineCount: Uint32Array;
  vertexCount: Uint32Array;
};

/**
 * Marching squares in cell/level/exit-edge order, followed by directed edge-chain walks.
 * Keeps high values on the left, uses the inclusive asymptotic saddle decider, retains raw
 * zero-length bridges, and checks complete-chain capacity before collapsing duplicate vertices.
 * Only rows below count/vertexCount and offsets through polylineCount are defined outputs.
 *
 * What differs from the GPU contributor:
 * - Interpolation and the saddle determinant use float64, with one rounding per output coordinate.
 *   Compare coordinates with absolute tolerance 2e-6 times the extent magnitude. Saddle decisions
 *   near a float32 determinant tie and duplicate collapse near rounding boundaries can differ.
 * - Chains are walked directly rather than pointer-jumped; no GPU round or command budget applies.
 * - Finite input differences that overflow float32 are evaluated in float64. In particular,
 *   opposing maximum-float32 samples at level zero cross at the midpoint (x = 1 on a unit
 *   2-by-2 placement). GPU interpolation after denominator overflow is outside the comparison
 *   domain; its device/compiler-dependent result is not assumed here.
 * - Nonfinite active levels emit no segments; nonfinite placement parameters propagate to vertices.
 *   Subnormal values are retained. Unused output capacity is zero-initialized on the CPU.
 *
 * @throws Error naming arrays that are too short or invalid dimensions/capacities.
 */
export function computeIsolinesOnCPU(
  input: CPUIsolinesInput,
  options: CPUIsolinesOptions
): CPUIsolinesResult {
  validateOptions('computeIsolinesOnCPU', options);
  validateIsolinesMaximumSegmentCount(
    'computeIsolinesOnCPU',
    options.grid,
    options.tile,
    input.levels.length
  );
  const {width, height} = options.grid;
  if (input.values.length < width * height)
    throw new Error('values must contain at least width * height rows');
  if (input.mask && input.mask.length < width * height)
    throw new Error('mask must contain at least width * height rows');
  if (input.levels.length < 1) throw new Error('levels must contain at least one row');
  if (input.parameters.length < GPU_ISOLINES_PARAMETER_LENGTH)
    throw new Error('parameters must hold the isolines layout');
  // getGPUIsolinesParameterValues layout.
  const [requestedLevels, anchorX, anchorY, cellWidth, cellHeight, columnAnchor, rowAnchor] =
    input.parameters;
  const levelCount = Math.min(input.levels.length, Math.max(0, Math.trunc(requestedLevels)));
  const capacity = options.segmentCapacity;
  const result: CPUIsolinesResult = {
    segments: new Float32Array(capacity * 4),
    segmentLevels: new Uint32Array(capacity),
    segmentEdges: new Uint32Array(capacity * 2),
    count: new Uint32Array(1),
    requiredCount: new Uint32Array(1),
    overflow: new Uint32Array(1),
    polylineOverflow: new Uint32Array(1),
    vertices: new Float32Array((options.vertexCapacity ?? 0) * 2),
    polylineOffsets: new Uint32Array(capacity + 1),
    polylineLevels: new Uint32Array(capacity),
    polylineClosed: new Uint32Array(capacity),
    polylineCount: new Uint32Array(1),
    vertexCount: new Uint32Array(1)
  };
  const sentinel = options.noDataValue === undefined ? undefined : Math.fround(options.noDataValue);
  const tile = options.tile;
  const domainWidth = tile?.domainWidth ?? width;
  const domainHeight = tile?.domainHeight ?? height;
  const columnOffset = tile?.columnOffset ?? 0;
  const rowOffset = tile?.rowOffset ?? 0;
  const [minColumn, minRow, maxColumn, maxRow] = tile?.ownedCells ?? [0, 0, width - 1, height - 1];
  let requiredCount = 0;
  for (let row = minRow; row < maxRow; row++)
    for (let column = minColumn; column < maxColumn; column++) {
      const samples = [
        row * width + column,
        row * width + column + 1,
        (row + 1) * width + column + 1,
        (row + 1) * width + column
      ];
      if (
        samples.some(
          sample =>
            !Number.isFinite(input.values[sample]) ||
            input.values[sample] === sentinel ||
            input.mask?.[sample] === 0
        )
      )
        continue;
      const corners = samples.map(sample => input.values[sample]);
      const domainColumn = column + columnOffset;
      const domainRow = row + rowOffset;
      const edgeIds = [
        domainRow * (domainWidth - 1) + domainColumn,
        domainHeight * (domainWidth - 1) + domainRow * domainWidth + domainColumn + 1,
        (domainRow + 1) * (domainWidth - 1) + domainColumn,
        domainHeight * (domainWidth - 1) + domainRow * domainWidth + domainColumn
      ];
      for (let levelIndex = 0; levelIndex < levelCount; levelIndex++) {
        const level = input.levels[levelIndex];
        if (!Number.isFinite(level)) continue;
        const high = corners.map(value => value >= level);
        const delta = corners.map(value => value - level);
        const determinant = delta[0] * delta[2] - delta[1] * delta[3];
        const joined = high[0] ? determinant >= 0 : determinant <= 0;
        const getCrossing = (edge: number): [number, number] => {
          const first = [0, 1, 3, 0][edge];
          const second = [1, 2, 2, 3][edge];
          const fraction = (level - corners[first]) / (corners[second] - corners[first]);
          return [
            anchorX +
              (domainColumn -
                columnAnchor +
                (edge === 0 || edge === 2 ? fraction : Number(edge === 1)) +
                0.5) *
                cellWidth,
            anchorY +
              (domainRow -
                rowAnchor +
                (edge === 1 || edge === 3 ? fraction : Number(edge === 2)) +
                0.5) *
                cellHeight
          ];
        };
        for (let exit = 0; exit < 4; exit++) {
          if (!high[exit] || high[(exit + 1) % 4]) continue;
          let entry = exit;
          for (let step = 1; step < 4; step++) {
            const candidate = (exit + (joined ? step : 4 - step)) % 4;
            if (!high[candidate] && high[(candidate + 1) % 4]) {
              entry = candidate;
              break;
            }
          }
          const destination = requiredCount++;
          if (destination >= capacity) continue;
          result.segments.set([...getCrossing(exit), ...getCrossing(entry)], destination * 4);
          result.segmentLevels[destination] = levelIndex;
          result.segmentEdges.set([edgeIds[exit], edgeIds[entry]], destination * 2);
        }
      }
    }
  result.requiredCount[0] = requiredCount;
  result.count[0] = Math.min(requiredCount, capacity);
  result.overflow[0] = Number(requiredCount > capacity);
  if (options.vertexCapacity === undefined) return result;
  if (result.overflow[0]) {
    result.polylineOverflow[0] = 1;
    return result;
  }
  const byStart = new Map<string, number>();
  for (let segment = 0; segment < requiredCount; segment++)
    byStart.set(`${result.segmentLevels[segment]}:${result.segmentEdges[segment * 2]}`, segment);
  const next = new Int32Array(requiredCount).fill(-1);
  const previous = new Int32Array(requiredCount).fill(-1);
  for (let segment = 0; segment < requiredCount; segment++) {
    const target = byStart.get(
      `${result.segmentLevels[segment]}:${result.segmentEdges[segment * 2 + 1]}`
    );
    if (target !== undefined) {
      next[segment] = target;
      previous[target] = segment;
    }
  }
  const visited = new Uint8Array(requiredCount);
  const chains: {head: number; closed: boolean; segments: number[]}[] = [];
  const walk = (head: number, closed: boolean) => {
    const segments: number[] = [];
    for (let segment = head; segment >= 0 && !visited[segment]; segment = next[segment]) {
      visited[segment] = 1;
      segments.push(segment);
    }
    chains.push({head, closed, segments});
  };
  for (let segment = 0; segment < requiredCount; segment++)
    if (previous[segment] < 0) walk(segment, false);
  for (let segment = 0; segment < requiredCount; segment++)
    if (!visited[segment]) walk(segment, true);
  chains.sort((first, second) => first.head - second.head);
  let uncollapsedCount = 0;
  let vertexCount = 0;
  let polylineCount = 0;
  for (const chain of chains) {
    uncollapsedCount += chain.segments.length + 1;
    if (uncollapsedCount > options.vertexCapacity) {
      result.polylineOverflow[0] = 1;
      break;
    }
    result.polylineOffsets[polylineCount] = vertexCount;
    result.polylineLevels[polylineCount] = result.segmentLevels[chain.head];
    result.polylineClosed[polylineCount] = Number(chain.closed);
    const start = vertexCount;
    const append = (offset: number) => {
      const horizontal = result.segments[offset];
      const vertical = result.segments[offset + 1];
      if (
        vertexCount > start &&
        result.vertices[vertexCount * 2 - 2] === horizontal &&
        result.vertices[vertexCount * 2 - 1] === vertical
      )
        return;
      result.vertices.set([horizontal, vertical], vertexCount++ * 2);
    };
    for (const segment of chain.segments) append(segment * 4);
    append(chain.closed ? chain.head * 4 : chain.segments[chain.segments.length - 1] * 4 + 2);
    polylineCount++;
    result.polylineOffsets[polylineCount] = vertexCount;
  }
  result.polylineCount[0] = polylineCount;
  result.vertexCount[0] = vertexCount;
  return result;
}

function validateOptions(id: string, options: CPUIsolinesOptions): void {
  validateIsolinesTile(id, options.grid, options.tile);
  validateRequirement(
    id,
    options.grid.affine.length === 6 && options.grid.affine.every(Number.isFinite),
    'grid.affine must contain six finite values'
  );
  validateMaximumCount(
    id,
    'grid cells',
    options.grid.width * options.grid.height,
    CPU_ISOLINES_MAXIMUM_CELL_COUNT
  );
  validateCapacity(id, 'segmentCapacity', options.segmentCapacity);
  if (options.vertexCapacity !== undefined)
    validateCapacity(id, 'vertexCapacity', options.vertexCapacity);
  if (options.noDataValue !== undefined)
    validateFiniteNumber(id, 'noDataValue', options.noDataValue);
}

/** CPU isolines and exact chain walks, with the same bounded output layouts. */
export class CPUIsolinesOperation extends GPUCPUOperation {
  constructor(props: IsolinesOperationProps) {
    const operation = new GPUIsolinesOperation(props);
    const {id} = operation;
    const options = {...props, vertexCapacity: props.output.polylines?.vertexCapacity};
    validateOptions(id, options);
    validateRequirement(
      id,
      !props.output.polylines || props.output.status.polylineOverflow !== undefined,
      'polylines require status.polylineOverflow'
    );
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
        const levels = getCPUOperationFloat32Rows(id, inputs, 'levels');
        const result = computeIsolinesOnCPU(
          {
            values: getCPUOperationFloat32Rows(id, inputs, 'values'),
            levels,
            parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters'),
            mask:
              props.mask !== undefined ? getCPUOperationUint32Rows(id, inputs, 'mask') : undefined
          },
          options
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUIsolinesResult]
          ])
        );
      }
    });
  }
}
