// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {assertCompatibleGPUSpatialContexts} from '../../analysis-core/contracts/spatial-context';
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
  GPUSpatialWeightsAlgebraOperation,
  validateSpatialWeightsOperationInformation,
  type SpatialWeightsAlgebraOperationProps
} from './gpu-spatial-weights-algebra-operation';
import type {GPUSpatialWeightsCombineRule} from './gpu-spatial-weights-algebra';
import type {
  GPUSpaceTimeWeightsLayout,
  GPUSpatialWeightsSpaceTimeLag
} from './spatial-weights-space-time-stages';

/** Packed CSR input. Slots beyond the last offset are unused capacity. */
export type CPUSpatialWeightsAlgebraRows = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
  distances?: Float32Array;
};
/** Arrays read by the chosen operation. */
export type CPUSpatialWeightsAlgebraInput = {
  left?: CPUSpatialWeightsAlgebraRows;
  right?: CPUSpatialWeightsAlgebraRows;
  weights?: CPUSpatialWeightsAlgebraRows;
  mask?: Uint32Array;
  groupIds?: Uint32Array;
  selfWeights?: Float32Array;
};
/** CPU options; output row count is inferred from the input CSR (or groups). */
export type CPUSpatialWeightsAlgebraOptions = {capacity: number; distances?: boolean} & (
  | {
      operation: 'union' | 'intersection' | 'difference' | 'symmetricDifference';
      weightRule?: GPUSpatialWeightsCombineRule;
    }
  | {operation: 'higherOrder'; order: number; cumulative?: boolean}
  | {operation: 'selfWeight'; selfWeight?: number}
  | {operation: 'subgraph'}
  | {operation: 'block'; groupCount: number}
  | {
      operation: 'spaceTime';
      periodCount: number;
      layout: GPUSpaceTimeWeightsLayout;
      lags: readonly GPUSpatialWeightsSpaceTimeLag[];
    }
);
/** Output CSR and status scalars; unused capacity is zero. */
export type CPUSpatialWeightsAlgebraResult = CPUSpatialWeightsAlgebraRows & {
  overflow: Uint32Array;
  requiredCount: Uint32Array;
};

function validateOptions(id: string, options: CPUSpatialWeightsAlgebraOptions): void {
  validateCapacity(id, 'capacity', options.capacity);
  validateRequirement(
    id,
    [
      'union',
      'intersection',
      'difference',
      'symmetricDifference',
      'higherOrder',
      'selfWeight',
      'subgraph',
      'block',
      'spaceTime'
    ].includes(options.operation),
    'operation must name a supported algebra operation'
  );
  if ('weightRule' in options)
    validateRequirement(
      id,
      ['left', 'right', 'sum', 'min', 'max', 'product', 'binary'].includes(
        options.weightRule ?? 'left'
      ),
      'weightRule must name a supported combination'
    );
  if (options.operation === 'higherOrder') validateCapacity(id, 'order', options.order);
  if (options.operation === 'selfWeight' && options.selfWeight !== undefined) {
    validateFiniteNumber(id, 'selfWeight', options.selfWeight);
    validateRequirement(id, options.selfWeight >= 0, 'selfWeight must be >= 0');
  }
  if (options.operation === 'block') validateCapacity(id, 'groupCount', options.groupCount);
  if (options.operation === 'spaceTime') {
    validateCapacity(id, 'periodCount', options.periodCount);
    validateCapacity(id, 'lags.length', options.lags.length);
    validateRequirement(
      id,
      ['cell-major', 'period-major'].includes(options.layout),
      'layout must be cell-major or period-major'
    );
    validateRequirement(
      id,
      new Set(options.lags.map(entry => entry.lag)).size === options.lags.length,
      'lags must be distinct'
    );
    for (const entry of options.lags) {
      validateCapacity(id, 'lag', entry.lag, -0x7fffffff, 0x7fffffff);
      validateFiniteNumber(id, 'lag weight', entry.weight ?? 1);
      validateRequirement(id, (entry.weight ?? 1) > 0, 'lag weight must be > 0');
      validateRequirement(
        id,
        !entry.includeSelf || entry.lag !== 0,
        'includeSelf is not allowed at lag 0'
      );
    }
  }
  validateRequirement(
    id,
    !options.distances || !['higherOrder', 'block', 'spaceTime'].includes(options.operation),
    'distances are unsupported for this operation'
  );
}

function validateRows(
  name: string,
  rows: CPUSpatialWeightsAlgebraRows | undefined
): CPUSpatialWeightsAlgebraRows {
  if (!rows) throw new Error(`${name} is required`);
  const {offsets, neighbors, weights, distances} = rows;
  if (
    offsets.length < 2 ||
    offsets[0] !== 0 ||
    offsets.some(
      (offset, index) => offset > neighbors.length || (index > 0 && offset < offsets[index - 1])
    )
  )
    throw new Error(
      `${name}.offsets must be nondecreasing CSR offsets within neighbors length, starting at zero`
    );
  if (weights.length !== neighbors.length)
    throw new Error(`${name}.weights length must equal neighbors length`);
  if (distances && distances.length !== neighbors.length)
    throw new Error(`${name}.distances length must equal neighbors length`);
  for (let row = 0; row + 1 < offsets.length; row++) {
    for (let slot = offsets[row] + 1; slot < offsets[row + 1]; slot++) {
      if (neighbors[slot] <= neighbors[slot - 1])
        throw new Error(`${name}.neighbors must be strictly ascending within each row`);
    }
  }
  return rows;
}

type Entry = {neighbor: number; weight: number; distance: number};
function getRow(rows: CPUSpatialWeightsAlgebraRows, row: number): Entry[] {
  const entries: Entry[] = [];
  for (let slot = rows.offsets[row]; slot < rows.offsets[row + 1]; slot++)
    entries.push({
      neighbor: rows.neighbors[slot],
      weight: rows.weights[slot],
      distance: rows.distances?.[slot] ?? 0
    });
  return entries;
}

/**
 * Computes sparse set algebra, shortest-path neighborhoods and space-time products in float64,
 * rounding weights and distances once on output. Higher-order expansion follows ascending
 * frontier order; a capacity-limited final frontier is truncated before sorting, as on GPU.
 *
 * What differs from the GPU contributor:
 * - Higher-order searches use dynamic frontiers: workCapacity and the order-32 GPU cap are
 *   dropped. overflow reports only final output truncation; requiredCount is always exact
 *   (saturated at uint32 maximum), including when GPU intermediate sets would overflow.
 * - The GPU limit of 16 temporal lags is dropped. Index and output sizes remain uint32.
 *   Empty input slot arrays are accepted; output capacity remains at least one.
 * - Arithmetic is float64 with one output rounding; near rounding boundaries products/sums
 *   may differ. Nonfinite per-row self weights are copied, as the GPU does for that view.
 * - Malformed CSR offsets, unsorted/duplicate neighbors and inconsistent array lengths throw.
 *   Unused output capacity is zero; GPU unused capacity is unspecified.
 *
 * @throws Error naming an inconsistent input array or invalid option.
 */
export function computeSpatialWeightsAlgebraOnCPU(
  input: CPUSpatialWeightsAlgebraInput,
  options: CPUSpatialWeightsAlgebraOptions
): CPUSpatialWeightsAlgebraResult {
  validateOptions('spatial-weights-algebra', options);
  const left =
    'weightRule' in options ||
    ['union', 'intersection', 'difference', 'symmetricDifference'].includes(options.operation)
      ? validateRows('left', input.left)
      : undefined;
  const right = left ? validateRows('right', input.right) : undefined;
  const source =
    !left && options.operation !== 'block' ? validateRows('weights', input.weights) : undefined;
  const baseRowCount = left
    ? left.offsets.length - 1
    : source
      ? source.offsets.length - 1
      : (input.groupIds?.length ?? 0);
  if (right && right.offsets.length !== left!.offsets.length)
    throw new Error('right.offsets length must equal left.offsets length');
  if (options.operation === 'block' && !input.groupIds) throw new Error('groupIds is required');
  if (options.operation === 'subgraph' && input.mask?.length !== baseRowCount)
    throw new Error('mask length must equal row count');
  if (
    options.operation === 'selfWeight' &&
    options.selfWeight === undefined &&
    input.selfWeights?.length !== baseRowCount
  )
    throw new Error('selfWeights length must equal row count');
  if (options.distances) {
    if (!(left ?? source)?.distances)
      throw new Error(`${left ? 'left' : 'weights'}.distances is required`);
    if (
      (options.operation === 'union' || options.operation === 'symmetricDifference') &&
      !right?.distances
    )
      throw new Error('right.distances is required');
  }
  const rowCount = baseRowCount * (options.operation === 'spaceTime' ? options.periodCount : 1);
  validateCapacity('spatial-weights-algebra', 'rowCount', rowCount);
  const result: CPUSpatialWeightsAlgebraResult = {
    offsets: new Uint32Array(rowCount + 1),
    neighbors: new Uint32Array(options.capacity),
    weights: new Float32Array(options.capacity),
    distances: options.distances ? new Float32Array(options.capacity) : undefined,
    overflow: new Uint32Array(1),
    requiredCount: new Uint32Array(1)
  };
  let requiredCount = 0;
  for (let row = 0; row < rowCount; row++) {
    let entries: Entry[] = [];
    if (left && right) {
      const rightEntries = new Map(getRow(right, row).map(entry => [entry.neighbor, entry]));
      const leftEntries = new Map(getRow(left, row).map(entry => [entry.neighbor, entry]));
      for (const neighbor of new Set([...leftEntries.keys(), ...rightEntries.keys()])) {
        const first = leftEntries.get(neighbor);
        const second = rightEntries.get(neighbor);
        if (
          (options.operation === 'intersection' && !(first && second)) ||
          (options.operation === 'difference' && (!first || second)) ||
          (options.operation === 'symmetricDifference' && Boolean(first) === Boolean(second))
        )
          continue;
        const rule = 'weightRule' in options ? (options.weightRule ?? 'left') : 'left';
        let weight = (first ?? second)!.weight;
        if (first && second) {
          switch (rule) {
            case 'right':
              weight = second.weight;
              break;
            case 'sum':
              weight += second.weight;
              break;
            case 'min':
              weight = Math.min(weight, second.weight);
              break;
            case 'max':
              weight = Math.max(weight, second.weight);
              break;
            case 'product':
              weight *= second.weight;
              break;
          }
        }
        entries.push({
          neighbor,
          weight: rule === 'binary' ? 1 : weight,
          distance: (first ?? second)!.distance
        });
      }
    } else if (options.operation === 'block') {
      const group = input.groupIds![row];
      if (group < options.groupCount)
        for (let neighbor = 0; neighbor < rowCount; neighbor++)
          if (neighbor !== row && input.groupIds![neighbor] === group)
            entries.push({neighbor, weight: 1, distance: 0});
    } else if (options.operation === 'higherOrder') {
      const visited = new Set([row]);
      let frontier = [row];
      for (let level = 1; level <= options.order && frontier.length; level++) {
        const next: number[] = [];
        for (const middle of frontier)
          for (const entry of getRow(source!, middle))
            if (entry.neighbor < rowCount && !visited.has(entry.neighbor)) {
              visited.add(entry.neighbor);
              next.push(entry.neighbor);
            }
        if (level === options.order)
          entries = next.map(neighbor => ({neighbor, weight: 1, distance: 0}));
        frontier = next.sort((first, second) => first - second);
      }
      if (options.cumulative)
        entries = [...visited]
          .filter(neighbor => neighbor !== row)
          .map(neighbor => ({neighbor, weight: 1, distance: 0}));
      // The GPU's exact-order emit pass sorts only its retained generation-order prefix.
      if (!options.cumulative) {
        const available = Math.max(0, options.capacity - requiredCount);
        const count = entries.length;
        entries = entries
          .slice(0, available)
          .sort((first, second) => first.neighbor - second.neighbor);
        for (const [index, entry] of entries.entries()) {
          const slot = Math.min(requiredCount, options.capacity) + index;
          result.neighbors[slot] = entry.neighbor;
          result.weights[slot] = 1;
        }
        requiredCount += count;
        result.offsets[row + 1] = Math.min(requiredCount, options.capacity);
        continue;
      }
    } else if (options.operation === 'selfWeight') {
      entries = getRow(source!, row).filter(entry => entry.neighbor !== row);
      entries.push({
        neighbor: row,
        weight: options.selfWeight ?? input.selfWeights![row],
        distance: 0
      });
    } else if (options.operation === 'subgraph') {
      if (input.mask![row] !== 0)
        entries = getRow(source!, row).filter(
          entry => entry.neighbor < rowCount && input.mask![entry.neighbor] !== 0
        );
    } else if (options.operation === 'spaceTime') {
      const cell =
        options.layout === 'period-major'
          ? row % baseRowCount
          : Math.floor(row / options.periodCount);
      const period =
        options.layout === 'period-major'
          ? Math.floor(row / baseRowCount)
          : row % options.periodCount;
      for (const lag of options.lags) {
        const sourcePeriod = period - lag.lag;
        if (sourcePeriod < 0 || sourcePeriod >= options.periodCount) continue;
        const spatial = getRow(source!, cell).filter(
          entry => entry.neighbor < baseRowCount && (!lag.includeSelf || entry.neighbor !== cell)
        );
        if (lag.includeSelf) spatial.push({neighbor: cell, weight: 1, distance: 0});
        for (const entry of spatial)
          entries.push({
            neighbor:
              options.layout === 'period-major'
                ? sourcePeriod * baseRowCount + entry.neighbor
                : entry.neighbor * options.periodCount + sourcePeriod,
            weight: entry.weight * (lag.weight ?? 1),
            distance: 0
          });
      }
    }
    entries.sort((first, second) => first.neighbor - second.neighbor);
    for (const entry of entries) {
      if (requiredCount < options.capacity) {
        result.neighbors[requiredCount] = entry.neighbor;
        result.weights[requiredCount] = entry.weight;
        if (result.distances) result.distances[requiredCount] = entry.distance;
      }
      requiredCount++;
    }
    result.offsets[row + 1] = Math.min(requiredCount, options.capacity);
  }
  result.requiredCount[0] = Math.min(requiredCount, 0xffffffff);
  result.overflow[0] = Number(requiredCount > options.capacity);
  return result;
}

/** CPU twin without GPU frontier, lag-count or order limits. */
export class CPUSpatialWeightsAlgebraOperation extends GPUCPUOperation {
  constructor(props: SpatialWeightsAlgebraOperationProps) {
    const id = props.id ?? 'spatial-weights-algebra';
    const options: CPUSpatialWeightsAlgebraOptions = {
      ...props,
      distances: props.output.distances !== undefined,
      ...(props.operation === 'selfWeight'
        ? {selfWeight: typeof props.selfWeight === 'number' ? props.selfWeight : undefined}
        : {})
    } as CPUSpatialWeightsAlgebraOptions;
    validateOptions(id, options);
    validateCapacity(id, 'rowCount', props.rowCount);
    if (props.output.distances !== undefined) {
      if ('left' in props) {
        validateRequirement(
          id,
          props.left.distances !== undefined,
          'output.distances requires left.distances'
        );
        if (props.operation === 'union' || props.operation === 'symmetricDifference')
          validateRequirement(
            id,
            props.right.distances !== undefined,
            'output.distances requires right.distances'
          );
      } else if ('weights' in props)
        validateRequirement(
          id,
          props.weights.distances !== undefined,
          'output.distances requires weights.distances'
        );
    }
    if (props.operation === 'spaceTime')
      validateRequirement(
        id,
        props.rowCount % props.periodCount === 0,
        'rowCount must be a multiple of periodCount'
      );
    if ('left' in props) {
      if (props.left.spatialContext && props.right.spatialContext) {
        assertCompatibleGPUSpatialContexts(
          id,
          props.left.spatialContext,
          props.right.spatialContext
        );
      }
      validateSpatialWeightsOperationInformation(
        id,
        'left',
        props.left.information,
        props.rowCount
      );
      validateSpatialWeightsOperationInformation(
        id,
        'right',
        props.right.information,
        props.rowCount
      );
    } else if ('weights' in props) {
      validateSpatialWeightsOperationInformation(
        id,
        'weights',
        props.weights.information,
        props.operation === 'spaceTime' ? props.rowCount / props.periodCount : props.rowCount
      );
    }
    const operation = new GPUSpatialWeightsAlgebraOperation(props);
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
        const readWeights = (prefix: string): CPUSpatialWeightsAlgebraRows => ({
          offsets: getCPUOperationUint32Rows(id, inputs, `${prefix}Offsets`),
          neighbors: getCPUOperationUint32Rows(id, inputs, `${prefix}Neighbors`),
          weights: getCPUOperationFloat32Rows(id, inputs, `${prefix}Weights`),
          distances:
            operation.inputs[`${prefix}Distances`] === undefined
              ? undefined
              : getCPUOperationFloat32Rows(id, inputs, `${prefix}Distances`)
        });
        const result = computeSpatialWeightsAlgebraOnCPU(
          {
            ...('left' in props
              ? {left: readWeights('left'), right: readWeights('right')}
              : 'weights' in props
                ? {weights: readWeights('source')}
                : {}),
            mask:
              props.operation === 'subgraph'
                ? getCPUOperationUint32Rows(id, inputs, 'mask')
                : undefined,
            groupIds:
              props.operation === 'block'
                ? getCPUOperationUint32Rows(id, inputs, 'groupIds')
                : undefined,
            selfWeights:
              props.operation === 'selfWeight' && typeof props.selfWeight === 'string'
                ? getCPUOperationFloat32Rows(id, inputs, 'selfWeight')
                : undefined
          },
          options
        );
        if (result.offsets.length !== props.rowCount + 1)
          throw new Error(`${id} offsets/groupIds row count must equal rowCount`);
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUSpatialWeightsAlgebraResult]!
          ])
        );
      }
    });
  }
}
