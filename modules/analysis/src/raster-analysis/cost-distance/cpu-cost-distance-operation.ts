// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  calibrateRasterOperationBand,
  readRasterOperationBand,
  validateRasterOperationBand,
  type CPURasterOperationBand
} from '../raster-grid/raster-operation-band';
import {validateRasterGridAxisAligned, validateRasterGridOrthogonal} from '../raster-band-utils';
import {validateRasterCellSizeMode} from '../../analysis-core/contracts/raster-grid';
import {
  GPU_COST_DISTANCE_NONE,
  GPU_COST_DISTANCE_PARAMETER_LENGTH,
  GPU_COST_DISTANCE_UNRESOLVED
} from './gpu-cost-distance';
import {validateRasterIterations} from './raster-relaxation';
import {relaxCostDistanceRasterOnCPU} from './cpu-raster-relaxation';
import {GPU_RASTER_D8_DIRECTIONS, getRasterD8Distance} from './raster-grid-utils';
import {
  GPUCostDistanceOperation,
  type CostDistanceOperationProps
} from './gpu-cost-distance-operation';

export type CPUCostDistanceInput = {
  friction: CPURasterOperationBand;
  parameters: Float32Array;
  frictionParameters?: Float32Array;
  sources?: Uint32Array;
  sourceCosts?: Float32Array;
  sourceCount?: Uint32Array;
  sourceMask?: Uint32Array;
  costBreaks?: Float32Array;
};
export type CPUCostDistanceOptions = Pick<
  CostDistanceOperationProps,
  'grid' | 'cellSizeMode' | 'maximumIterations' | 'maximumTieIterations' | 'sweepAfterIteration'
>;
export type CPUCostDistanceResult = {
  costs: Float32Array;
  settledAt: Uint32Array;
  iterationCount: Uint32Array;
  backLinks: Uint32Array;
  bands: Uint32Array;
  bandCounts: Uint32Array;
  converged: Uint32Array;
  iterationLimitReached: Uint32Array;
};

/**
 * Computes least costs with float64 Dijkstra when unbounded, or bounded tiled relaxation,
 * and cycle-safe plateau levels. The pure function returns back-links, so it applies the tie budget;
 * the operation omits that phase when back-links are not requested.
 * Strict predecessors minimize candidate cost, with smallest D8 direction on ties; plateau
 * links use the smallest direction one level nearer a strict entry or surviving source.
 *
 * What differs from the GPU contributor:
 * - Explicit cost/tie budgets use 16x16 tiled rounds, with round-start snapshots between
 *   workgroups and ordered writes within sweep lines. The final unchanged round consumes budget.
 *   Without an explicit budget the CPU computes the full fixed point. Stateful continuation is omitted.
 * - settledAt is the minimum hop count of a float64 least-cost path from a surviving seed,
 *   or zero for unreached cells. iterationCount is the largest such count plus one (the first
 *   unchanged synchronous round). With an explicit cost budget these become tiled last-decrease
 *   stamps and the executed tiled round count. GPU scheduling can give different diagnostics.
 * - Calibration, distances and path sums use float64, rounded once at output. Compare costs
 *   with relative tolerance 3e-5. Near ties, threshold bands and back-links can differ.
 * - Strict tightness uses the GPU's 4-ulp relative slack in float64. Equal-cost edges use its
 *   addition test; subnormal and float64-finite costs that exceed float32 are retained internally.
 *   A cost that rounds to infinity has no published back-link or band and is not counted.
 * - Nonfinite or negative transition arithmetic produces no usable transitions; invalid seeds
 *   are ignored. Negative packed cell sizes are outside the documented GPU parameter layout.
 *
 * @throws Error naming an inconsistent input array or invalid grid.
 */
export function computeCostDistanceOnCPU(
  input: CPUCostDistanceInput,
  options: CPUCostDistanceOptions
): CPUCostDistanceResult {
  return computeCostDistance(input, options, true);
}

function computeCostDistance(
  input: CPUCostDistanceInput,
  options: CPUCostDistanceOptions,
  includeBackLinks: boolean
): CPUCostDistanceResult {
  const id = 'computeCostDistanceOnCPU';
  validateRasterOperationBand(id, input.friction, options.grid);
  validateOptions(id, options);
  const {width, height} = options.grid;
  const length = width * height;
  if (input.parameters.length < GPU_COST_DISTANCE_PARAMETER_LENGTH)
    throw new Error('parameters must hold the cost-distance layout');
  if (input.frictionParameters && input.frictionParameters.length < 2)
    throw new Error('frictionParameters must hold scale and offset');
  if (input.sourceCosts && input.sourceCosts.length !== input.sources?.length)
    throw new Error('sourceCosts must match sources');
  if (input.sourceCount && (!input.sources || input.sourceCount.length !== 1))
    throw new Error('sourceCount must have one row and requires sources');
  if (input.sourceMask && input.sourceMask.length !== length)
    throw new Error('sourceMask must match grid cells');
  if (!input.sources && !input.sourceMask) throw new Error('sources or sourceMask is required');
  if (input.costBreaks && input.costBreaks.length === 0)
    throw new Error('costBreaks must contain at least one break');
  const friction = calibrateRasterOperationBand(input.friction, length);
  for (let cell = 0; cell < length; cell++) {
    const value =
      friction[cell] * (input.frictionParameters?.[0] ?? 1) + (input.frictionParameters?.[1] ?? 0);
    friction[cell] = Number.isFinite(value) && value >= 0 ? value : NaN;
  }
  // getGPUCostDistanceParameterValues layout, including the shared grid prefix.
  const [cellSizeX, cellSizeY, northEdge, southEdge, costLimit] = input.parameters;
  const georeference = {cellSize: [cellSizeX, cellSizeY] as [number, number], northEdge, southEdge};
  const getNeighbor = (cell: number, direction: number): number => {
    const {columnOffset, rowOffset} = GPU_RASTER_D8_DIRECTIONS[direction];
    const column = (cell % width) + columnOffset;
    const row = Math.floor(cell / width) + rowOffset;
    return column < 0 || column >= width || row < 0 || row >= height ? -1 : row * width + column;
  };
  const getEdge = (cell: number, neighbor: number, direction: number): number => {
    const cost =
      getRasterD8Distance(
        options.cellSizeMode ?? 'uniform',
        direction,
        Math.floor(cell / width),
        height,
        georeference
      ) *
      (0.5 * (friction[cell] + friction[neighbor]));
    return cost >= 0 && Number.isFinite(cost) ? cost : Infinity;
  };
  const costs = new Float64Array(length).fill(Infinity);
  const seeds = new Float64Array(length).fill(Infinity);
  const heap: {cell: number; cost: number}[] = [];
  const push = (cell: number, cost: number) => {
    heap.push({cell, cost});
    let position = heap.length - 1;
    while (position > 0) {
      const parent = Math.floor((position - 1) / 2);
      if (heap[parent].cost <= heap[position].cost) break;
      [heap[parent], heap[position]] = [heap[position], heap[parent]];
      position = parent;
    }
  };
  const seedCell = (cell: number, cost: number) => {
    if (
      cell >= length ||
      !Number.isFinite(friction[cell]) ||
      !Number.isFinite(cost) ||
      cost < 0 ||
      cost > costLimit
    )
      return;
    seeds[cell] = Math.min(seeds[cell], cost);
    if (cost < costs[cell]) {
      costs[cell] = cost;
      if (options.maximumIterations === undefined) push(cell, cost);
    }
  };
  for (
    let source = 0;
    source < Math.min(input.sources?.length ?? 0, input.sourceCount?.[0] ?? Infinity);
    source++
  )
    seedCell(input.sources![source], input.sourceCosts?.[source] ?? 0);
  for (let cell = 0; cell < length; cell++) if (input.sourceMask?.[cell]) seedCell(cell, 0);
  const settledAt = new Uint32Array(length);
  const costState =
    options.maximumIterations === undefined
      ? undefined
      : relaxCostDistanceRasterOnCPU({
          width,
          height,
          values: costs,
          maximumIterations: options.maximumIterations,
          sweepAfterIteration: options.sweepAfterIteration ?? 96,
          costLimit,
          participates: cell => Number.isFinite(friction[cell]),
          getEdgeCost: getEdge,
          settledAt
        });
  if (costState && !costState.converged) {
    // Stop here: neither unbounded Dijkstra nor the dependent tie phase may consume more rounds.
    return {
      costs: Float32Array.from(friction, value => (Number.isFinite(value) ? NaN : Infinity)),
      settledAt,
      iterationCount: Uint32Array.of(costState.iterationCount),
      backLinks: Uint32Array.from(friction, value =>
        Number.isFinite(value) ? GPU_COST_DISTANCE_UNRESOLVED : GPU_COST_DISTANCE_NONE
      ),
      bands: Uint32Array.from(friction, value =>
        Number.isFinite(value) ? GPU_COST_DISTANCE_UNRESOLVED : GPU_COST_DISTANCE_NONE
      ),
      bandCounts: new Uint32Array(input.costBreaks?.length ?? 0).fill(GPU_COST_DISTANCE_UNRESOLVED),
      converged: Uint32Array.of(0),
      iterationLimitReached: Uint32Array.of(1)
    };
  }
  while (heap.length) {
    const current = heap[0];
    const last = heap.pop()!;
    if (heap.length) {
      heap[0] = last;
      let position = 0;
      for (;;) {
        const left = 2 * position + 1;
        const right = left + 1;
        let smallest = position;
        if (left < heap.length && heap[left].cost < heap[smallest].cost) smallest = left;
        if (right < heap.length && heap[right].cost < heap[smallest].cost) smallest = right;
        if (smallest === position) break;
        [heap[position], heap[smallest]] = [heap[smallest], heap[position]];
        position = smallest;
      }
    }
    if (current.cost !== costs[current.cell]) continue;
    for (let direction = 0; direction < 8; direction++) {
      const neighbor = getNeighbor(current.cell, direction);
      if (neighbor < 0 || !Number.isFinite(friction[neighbor])) continue;
      const candidate = current.cost + getEdge(current.cell, neighbor, direction);
      if (candidate <= costLimit && candidate < costs[neighbor]) {
        costs[neighbor] = candidate;
        push(neighbor, candidate);
      }
    }
  }
  const publishedCosts = Float32Array.from(costs, cost => (cost === 0 ? 0 : cost));
  const hopCounts = new Float64Array(length).fill(Infinity);
  const hopQueue: number[] = [];
  if (!costState)
    for (let cell = 0; cell < length; cell++) {
      if (Number.isFinite(publishedCosts[cell]) && seeds[cell] === costs[cell]) {
        hopCounts[cell] = 0;
        hopQueue.push(cell);
      }
    }
  let maximumHopCount = 0;
  for (let position = 0; position < hopQueue.length; position++) {
    const cell = hopQueue[position];
    for (let direction = 0; direction < 8; direction++) {
      const neighbor = getNeighbor(cell, direction);
      if (
        neighbor < 0 ||
        !Number.isFinite(publishedCosts[neighbor]) ||
        hopCounts[neighbor] !== Infinity
      )
        continue;
      if (costs[cell] + getEdge(cell, neighbor, direction) !== costs[neighbor]) continue;
      const hopCount = hopCounts[cell] + 1;
      hopCounts[neighbor] = hopCount;
      settledAt[neighbor] = hopCount;
      maximumHopCount = Math.max(maximumHopCount, hopCount);
      hopQueue.push(neighbor);
    }
  }
  const iterationCount = costState?.iterationCount ?? maximumHopCount + 1;
  if (costState) {
    for (let cell = 0; cell < length; cell++)
      if (!Number.isFinite(publishedCosts[cell])) settledAt[cell] = 0;
  }
  const backLinks = new Uint32Array(length).fill(GPU_COST_DISTANCE_NONE);
  const levels = new Float64Array(length).fill(Infinity);
  const queue: number[] = [];
  const isTie = (cell: number, neighbor: number, direction: number) =>
    neighbor >= 0 &&
    Number.isFinite(publishedCosts[cell]) &&
    Number.isFinite(publishedCosts[neighbor]) &&
    Number.isFinite(friction[neighbor]) &&
    costs[neighbor] === costs[cell] &&
    costs[neighbor] + getEdge(cell, neighbor, direction) <= costs[cell];
  if (includeBackLinks)
    for (let cell = 0; cell < length; cell++) {
      if (!Number.isFinite(publishedCosts[cell])) continue;
      let best = Infinity;
      let bestDirection = -1;
      for (let direction = 0; direction < 8; direction++) {
        const neighbor = getNeighbor(cell, direction);
        if (neighbor < 0 || !(costs[neighbor] < costs[cell])) continue;
        const candidate = costs[neighbor] + getEdge(cell, neighbor, direction);
        if (candidate < best) {
          best = candidate;
          bestDirection = direction;
        }
      }
      if (bestDirection >= 0 && best <= costs[cell] * (1 + 4.76837158e-7))
        backLinks[cell] = 1 << bestDirection;
      if (backLinks[cell] !== GPU_COST_DISTANCE_NONE || seeds[cell] === costs[cell]) {
        levels[cell] = 0;
        queue.push(cell);
        if (backLinks[cell] === GPU_COST_DISTANCE_NONE) backLinks[cell] = 0;
      }
    }
  const maximumTieIterations = includeBackLinks
    ? (options.maximumTieIterations ?? options.maximumIterations)
    : undefined;
  let tieConverged = true;
  if (maximumTieIterations !== undefined) {
    tieConverged = relaxCostDistanceRasterOnCPU({
      width,
      height,
      values: levels,
      maximumIterations: maximumTieIterations,
      sweepAfterIteration: options.sweepAfterIteration ?? 96,
      costLimit: Infinity,
      participates: cell => Number.isFinite(publishedCosts[cell]),
      getEdgeCost: (cell, neighbor, direction) => (isTie(cell, neighbor, direction) ? 1 : Infinity)
    }).converged;
  } else
    for (let position = 0; position < queue.length; position++) {
      const cell = queue[position];
      for (let direction = 0; direction < 8; direction++) {
        const neighbor = getNeighbor(cell, direction);
        if (
          neighbor >= 0 &&
          levels[neighbor] === Infinity &&
          isTie(neighbor, cell, (direction + 4) % 8)
        ) {
          levels[neighbor] = levels[cell] + 1;
          queue.push(neighbor);
        }
      }
    }
  if (includeBackLinks)
    for (let cell = 0; cell < length; cell++) {
      if (Number.isFinite(publishedCosts[cell]) && levels[cell] === Infinity) {
        backLinks[cell] = GPU_COST_DISTANCE_UNRESOLVED;
        continue;
      }
      if (!(levels[cell] > 0) || !Number.isFinite(levels[cell])) continue;
      for (let direction = 0; direction < 8; direction++) {
        const neighbor = getNeighbor(cell, direction);
        if (isTie(cell, neighbor, direction) && levels[neighbor] === levels[cell] - 1) {
          backLinks[cell] = 1 << direction;
          break;
        }
      }
    }
  const bands = new Uint32Array(length).fill(GPU_COST_DISTANCE_NONE);
  const bandCounts = new Uint32Array(input.costBreaks?.length ?? 0);
  for (let cell = 0; cell < length; cell++) {
    if (!Number.isFinite(publishedCosts[cell])) continue;
    let passed = 0;
    for (const threshold of input.costBreaks ?? []) if (threshold < costs[cell]) passed++;
    if (passed < bandCounts.length) {
      bands[cell] = passed;
      bandCounts[passed]++;
    }
  }
  return {
    costs: publishedCosts,
    settledAt,
    iterationCount: Uint32Array.of(iterationCount),
    backLinks,
    bands,
    bandCounts,
    converged: Uint32Array.of(Number(tieConverged)),
    iterationLimitReached: Uint32Array.of(Number(!tieConverged))
  };
}

function validateOptions(id: string, options: CPUCostDistanceOptions): void {
  validateRasterCellSizeMode(id, options.cellSizeMode);
  validateRasterGridAxisAligned(id, options.grid, options.cellSizeMode);
  validateRasterGridOrthogonal(id, options.grid);
  for (const name of ['maximumIterations', 'maximumTieIterations'] as const)
    if (options[name] !== undefined) validateRasterIterations(id, name, options[name]);
  if (options.sweepAfterIteration !== undefined)
    validateCapacity(id, 'sweepAfterIteration', options.sweepAfterIteration, 0, 0xffffffff);
}

/** CPU fixed point with explicit tiled cost/tie budgets and unresolved output propagation. */
export class CPUCostDistanceOperation extends GPUCPUOperation {
  constructor(props: CostDistanceOperationProps) {
    props = {
      ...props,
      grid: {...props.grid, affine: [...props.grid.affine]},
      friction: {...props.friction, storage: {...props.friction.storage}},
      output: {...props.output, status: {...props.output.status}}
    };
    const operation = new GPUCostDistanceOperation(props);
    const {id} = operation;
    validateRasterOperationBand(id, props.friction, props.grid);
    validateOptions(id, props);
    validateRequirement(
      id,
      props.sources !== undefined || props.sourceMask !== undefined,
      'sources or sourceMask is required'
    );
    validateRequirement(
      id,
      !(props.sourceCosts || props.sourceCount) || props.sources !== undefined,
      'sourceCosts and sourceCount require sources'
    );
    validateRequirement(
      id,
      Boolean(props.costBreaks) === Boolean(props.output.bands || props.output.bandCounts),
      'costBreaks is required exactly with bands or bandCounts'
    );
    if (props.output.bandCounts !== undefined)
      validateCapacity(id, 'costBreakCount', props.costBreakCount!);
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
        const input: CPUCostDistanceInput = {
          friction: readRasterOperationBand(id, inputs, props.friction),
          parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters')
        };
        for (const port of ['frictionParameters', 'sourceCosts', 'costBreaks'] as const)
          if (props[port] !== undefined) input[port] = getCPUOperationFloat32Rows(id, inputs, port);
        for (const port of ['sources', 'sourceCount', 'sourceMask'] as const)
          if (props[port] !== undefined) input[port] = getCPUOperationUint32Rows(id, inputs, port);
        if (
          props.output.bandCounts !== undefined &&
          input.costBreaks?.length !== props.costBreakCount
        )
          throw new Error(`${id} costBreaks must match costBreakCount`);
        const result = computeCostDistance(input, props, props.output.backLinks !== undefined);
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUCostDistanceResult]
          ])
        );
      }
    });
  }
}
