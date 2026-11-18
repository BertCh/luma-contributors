// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getPhilox4x32, PHILOX_PURPOSES} from '../../analysis-core/kernels/philox-random';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {validateSpatialWeightsOperationInformation} from '../spatial-weights/gpu-spatial-weights-algebra-operation';
import type {GPUSpatialWeightsInformation} from '../spatial-weights/spatial-weights';
import {
  GPU_MAX_P_DEFAULT_START_COUNT,
  GPU_MAX_P_DEFAULT_MAXIMUM_ROUNDS,
  GPU_MAX_P_REGIONS_PARAMETER_LENGTH,
  getGPUMaxPRoundCounts,
  GPU_MAX_P_STATUS
} from './gpu-max-p-regions';
import {
  GPUMaxPRegionsOperation,
  type MaxPRegionsOperationProps
} from './gpu-max-p-regions-operation';
import {computeAZPRegionsTableOnCPU} from './cpu-azp-regions-operation';
import {
  countCPURegionInvalid,
  getCPURegionAdjacency,
  getCPURegionObjective,
  getCPURegionTable,
  validateCPURegionInput,
  type CPURegionInput
} from './cpu-region-inputs';

/** Packed inputs; the threshold view contributes its first word only. */
export type CPUMaxPRegionsInput = CPURegionInput & {
  extensiveValues: Float32Array;
  threshold: Float32Array;
};
/** Construction and improvement round budgets match the contributor. */
export type CPUMaxPRegionsOptions = Pick<
  MaxPRegionsOperationProps,
  'columnCount' | 'standardize' | 'startCount' | 'maximumGrowthSteps' | 'maximumIterations'
> & {information?: GPUSpatialWeightsInformation};
/** Exact graph port layouts, indexed by GPU_MAX_P_STATUS. */
export type CPUMaxPRegionsResult = {
  labels: Uint32Array;
  objective: Float32Array;
  status: Uint32Array;
};

const NONE = 0xffffffff;
const GROWING = 1;
const COMPLETE = 2;
const FAILED = 3;
const TIE_MULTIPLIER = 0x9e3779b1;
type GrowthOffer = {row: number; region: number; priority: number; distance: number; tie: number};
type StartResult = {
  labels: Uint32Array;
  regionCount: number;
  infeasibleRegionCount: number;
  squares: number;
};

function validateOptions(id: string, options: CPUMaxPRegionsOptions): void {
  validateCapacity(id, 'columnCount', options.columnCount ?? 1);
  validateCapacity(id, 'startCount', options.startCount ?? GPU_MAX_P_DEFAULT_START_COUNT);
  validateCapacity(
    id,
    'maximumGrowthSteps',
    options.maximumGrowthSteps ?? GPU_MAX_P_DEFAULT_MAXIMUM_ROUNDS
  );
  validateCapacity(id, 'maximumIterations', options.maximumIterations ?? 0, 0);
}

function constructStart(
  input: CPUMaxPRegionsInput,
  options: CPUMaxPRegionsOptions,
  adjacency: number[][],
  table: Float64Array,
  start: number
): StartResult {
  const rowCount = adjacency.length;
  const columnCount = options.columnCount ?? 1;
  const [seed] = input.parameters; // getGPUMaxPRegionsParameterValues: [seed].
  const {growthRounds, enclaveRounds} = getGPUMaxPRoundCounts(
    rowCount,
    options.maximumGrowthSteps ?? GPU_MAX_P_DEFAULT_MAXIMUM_ROUNDS
  );
  const threshold = input.threshold[0];
  const purpose = PHILOX_PURPOSES.maxPRegionsSeedOrder.id;
  const salt = getPhilox4x32([NONE, start, 0, 0], [seed, purpose])[1];
  const keys = Uint32Array.from(
    {length: rowCount},
    (_, row) => getPhilox4x32([row, start, 0, 0], [seed, purpose])[0]
  );
  const getTie = (row: number) => (Math.imul(row, TIE_MULTIPLIER) ^ salt) >>> 0;
  const labels = new Uint32Array(rowCount).fill(NONE);
  const phases = new Uint8Array(rowCount);
  const sizes = new Float64Array(rowCount);
  const extensiveSums = new Float64Array(rowCount);
  const squares = new Float64Array(rowCount);
  const sums = new Float64Array(rowCount * columnCount);
  const getDistance = (row: number, region: number) => {
    let distance = 0;
    for (let column = 0; column < columnCount; column++) {
      const difference =
        table[row * columnCount + column] - sums[region * columnCount + column] / sizes[region];
      distance += difference * difference;
    }
    return distance;
  };
  const mean = input.extensiveValues.reduce((sum, value) => sum + value, 0) / rowCount;
  const expectedSize = mean > 0 ? Math.max(1, Math.ceil(threshold / mean)) : 1;
  const period = Math.ceil(expectedSize / 2);
  let fraction = 0.4 / expectedSize;
  let nextWave = 0;
  let round = 0;
  let unassigned = rowCount;
  let growing = 0;
  const compareOffers = (first: GrowthOffer, second: GrowthOffer) =>
    first.priority - second.priority || first.distance - second.distance || first.tie - second.tie;
  while (round < growthRounds && (growing > 0 || unassigned > 0)) {
    if (unassigned > 0 && (growing === 0 || round >= nextWave)) {
      const cut = fraction >= 1 ? NONE : Math.floor(fraction * 4294967040);
      fraction = Math.min(1, fraction * 1.3);
      nextWave = round + period;
      const eligible = (row: number) => labels[row] === NONE && keys[row] <= cut;
      const seeds: number[] = [];
      for (let row = 0; row < rowCount; row++) {
        if (!eligible(row)) continue;
        const blocked = adjacency[row].some(
          neighbor =>
            neighbor !== row &&
            ((labels[neighbor] !== NONE && phases[labels[neighbor]] === GROWING) ||
              (eligible(neighbor) &&
                (keys[neighbor] < keys[row] || (keys[neighbor] === keys[row] && neighbor < row))))
        );
        if (!blocked) seeds.push(row);
      }
      for (const row of seeds) {
        labels[row] = row;
        sizes[row] = 1;
        extensiveSums[row] = input.extensiveValues[row];
        for (let column = 0; column < columnCount; column++)
          sums[row * columnCount + column] = table[row * columnCount + column];
        phases[row] = extensiveSums[row] >= threshold ? COMPLETE : GROWING;
        if (phases[row] === GROWING) growing++;
        unassigned--;
      }
    }
    round++;
    const offers = new Map<number, GrowthOffer>();
    for (let row = 0; row < rowCount; row++) {
      if (labels[row] !== NONE) continue;
      const priority = adjacency[row].filter(
        neighbor => neighbor !== row && labels[neighbor] === NONE
      ).length;
      for (const neighbor of adjacency[row]) {
        const region = labels[neighbor];
        if (region === NONE || phases[region] !== GROWING) continue;
        const offer = {row, region, priority, distance: getDistance(row, region), tie: getTie(row)};
        const previous = offers.get(region);
        if (!previous || compareOffers(offer, previous) < 0) offers.set(region, offer);
      }
    }
    const claims = new Map<number, GrowthOffer>();
    for (const offer of offers.values()) {
      const previous = claims.get(offer.row);
      // Row ties are equal here; the region's seeded order resolves conflicting claims.
      if (
        !previous ||
        compareOffers(offer, previous) < 0 ||
        (compareOffers(offer, previous) === 0 && getTie(offer.region) < getTie(previous.region))
      )
        claims.set(offer.row, offer);
    }
    for (let region = 0; region < rowCount; region++) {
      if (phases[region] !== GROWING) continue;
      const offer = offers.get(region);
      if (!offer) {
        phases[region] = FAILED;
        growing--;
        continue;
      }
      if (claims.get(offer.row) !== offer) continue;
      const count = sizes[region];
      squares[region] += (count / (count + 1)) * offer.distance;
      sizes[region]++;
      extensiveSums[region] += input.extensiveValues[offer.row];
      for (let column = 0; column < columnCount; column++)
        sums[region * columnCount + column] += table[offer.row * columnCount + column];
      labels[offer.row] = region;
      unassigned--;
      if (extensiveSums[region] >= threshold) {
        phases[region] = COMPLETE;
        growing--;
      }
    }
  }
  // Enclave distances deliberately use the frozen growth-phase means, as in the GPU.
  for (let enclaveRound = 0; enclaveRound < enclaveRounds; enclaveRound++) {
    const proposals = new Map<number, number>();
    for (let row = 0; row < rowCount; row++) {
      if (labels[row] !== NONE && phases[labels[row]] === COMPLETE) continue;
      let bestRegion = NONE;
      let bestDistance = Infinity;
      for (const neighbor of adjacency[row]) {
        const region = labels[neighbor];
        if (region === NONE || phases[region] !== COMPLETE) continue;
        const distance = getDistance(row, region);
        if (
          distance < bestDistance ||
          (distance === bestDistance && getTie(region) < getTie(bestRegion))
        ) {
          bestRegion = region;
          bestDistance = distance;
        }
      }
      if (bestRegion !== NONE) proposals.set(row, bestRegion);
    }
    if (proposals.size === 0) break;
    for (const [row, region] of proposals) labels[row] = region;
  }
  for (let row = 0; row < rowCount; row++)
    if (labels[row] === NONE) {
      labels[row] = row;
      phases[row] = FAILED;
    }
  const used = [...new Set(labels)].sort((first, second) => first - second);
  return {
    labels,
    regionCount: used.length,
    infeasibleRegionCount: used.filter(region => phases[region] !== COMPLETE).length,
    squares: used.reduce((sum, region) => sum + squares[region], 0)
  };
}

/**
 * Runs budgeted seeded parallel Max-P growth and enclave assignment, selects by infeasible
 * regions, region count, grown-region SSD and start index, and optionally refines with two-hop AZP.
 * This is the documented multistart heuristic, not a global maximum-P optimizer. Seed rows and
 * ties use the contributor's registered Philox stream and bijective order. Enclave means stay
 * frozen at the end of growth; the published objective is recomputed from the final labels.
 *
 * What differs from the GPU contributor:
 * - Float64 standardization, means, distances, growth scores and objective round only at output.
 *   Compare objectives with relative tolerance 1e-4; near ties or extensive-sum threshold
 *   boundaries can change labels and selectedStart.
 * - Growth and enclaves honor maximumGrowthSteps independently, as on GPU. The expected-size cap,
 *   63-neighbor priority cap and truncated float32 distance keys are removed; priorities then
 *   exact distances then the same seeded ties order offers. Wave arithmetic also uses float64.
 * - Drops GPU column, start-count, storage and command-node limits; uint32 layouts remain.
 *   Empty neighbor-slot arrays are accepted.
 * - Positive maximumIterations bounds AZP refinement rounds with two-hop admissibility.
 *   improvementIterationCount is the number of successful batched rounds, as on GPU.
 *   Its neighborhood storage/degree caps are dropped. Refinement extensive sums and departing-
 *   value subtraction use float64, so near-threshold feasibility decisions can differ as in AZP.
 * - Finite distances above float32 range remain eligible. Final objective may round to Infinity.
 *   Nonfinite features, negative/nonfinite extensive values or threshold, and bad neighbor IDs
 *   follow the GPU invalid status path (singleton labels, zero objective and regionCount).
 *
 * @throws Error naming inconsistent arrays, including malformed CSR offsets.
 */
export function computeMaxPRegionsOnCPU(
  input: CPUMaxPRegionsInput,
  options: CPUMaxPRegionsOptions = {}
): CPUMaxPRegionsResult {
  validateOptions('computeMaxPRegionsOnCPU', options);
  const columnCount = options.columnCount ?? 1;
  const rowCount = validateCPURegionInput(input, columnCount, GPU_MAX_P_REGIONS_PARAMETER_LENGTH);
  validateRequirement(
    'computeMaxPRegionsOnCPU',
    (options.maximumIterations ?? 0) === 0 || rowCount >= 2,
    'maximumIterations needs at least 2 rows'
  );
  validateSpatialWeightsOperationInformation(
    'computeMaxPRegionsOnCPU',
    'weights',
    options.information,
    rowCount
  );
  if (input.extensiveValues.length !== rowCount)
    throw new Error('extensiveValues length must equal rowCount');
  if (input.threshold.length < 1) throw new Error('threshold must hold one value');
  const status = new Uint32Array(GPU_MAX_P_STATUS.length);
  let invalidCount = countCPURegionInvalid(input);
  for (const value of [...input.extensiveValues, input.threshold[0]])
    if (!Number.isFinite(value) || value < 0) invalidCount++;
  if (invalidCount > 0) {
    status[GPU_MAX_P_STATUS.invalidCount] = invalidCount;
    status[GPU_MAX_P_STATUS.infeasibleRegionCount] = rowCount;
    return {
      labels: Uint32Array.from({length: rowCount}, (_, row) => row),
      objective: new Float32Array(1),
      status
    };
  }
  const adjacency = getCPURegionAdjacency(input, options.information);
  const table = getCPURegionTable(input.values, columnCount, options.standardize);
  const [seed] = input.parameters; // Same seed layout passed unchanged to AZP refinement.
  let selected: StartResult | undefined;
  for (let start = 0; start < (options.startCount ?? GPU_MAX_P_DEFAULT_START_COUNT); start++) {
    const candidate = constructStart(input, options, adjacency, table, start);
    if (
      !selected ||
      candidate.infeasibleRegionCount < selected.infeasibleRegionCount ||
      (candidate.infeasibleRegionCount === selected.infeasibleRegionCount &&
        (candidate.regionCount > selected.regionCount ||
          (candidate.regionCount === selected.regionCount && candidate.squares < selected.squares)))
    ) {
      selected = candidate;
      status[GPU_MAX_P_STATUS.selectedStart] = start;
    }
  }
  const construction = selected!;
  const ranks = new Map(
    [...new Set(construction.labels)]
      .sort((first, second) => first - second)
      .map((region, rank) => [region, rank])
  );
  let labels: Uint32Array = Uint32Array.from(construction.labels, region => ranks.get(region)!);
  if ((options.maximumIterations ?? 0) > 0) {
    const improvement = computeAZPRegionsTableOnCPU(
      adjacency,
      table,
      labels,
      rowCount,
      columnCount,
      seed,
      1,
      input.extensiveValues,
      input.threshold[0],
      0,
      options.maximumIterations
    );
    labels = improvement.labels;
    invalidCount = improvement.invalidCount;
    status[GPU_MAX_P_STATUS.improvementIterationCount] = improvement.iterationCount;
  }
  status[GPU_MAX_P_STATUS.regionCount] = invalidCount > 0 ? 0 : construction.regionCount;
  status[GPU_MAX_P_STATUS.feasible] = Number(
    invalidCount === 0 && construction.infeasibleRegionCount === 0
  );
  status[GPU_MAX_P_STATUS.infeasibleRegionCount] = construction.infeasibleRegionCount;
  status[GPU_MAX_P_STATUS.invalidCount] = invalidCount;
  return {
    labels,
    objective: Float32Array.of(
      invalidCount > 0 ? 0 : getCPURegionObjective(table, labels, columnCount)
    ),
    status
  };
}

/** CPU twin with budgeted construction and optional budgeted two-hop AZP. */
export class CPUMaxPRegionsOperation extends GPUCPUOperation {
  constructor(props: MaxPRegionsOperationProps) {
    props = Object.freeze({...props});
    const id = props.id ?? 'max-p-regions';
    validateOptions(id, props);
    validateCapacity(id, 'rowCount', props.rowCount);
    validateRequirement(
      id,
      (props.maximumIterations ?? 0) === 0 || props.rowCount >= 2,
      'maximumIterations needs at least 2 rows'
    );
    validateSpatialWeightsOperationInformation(
      id,
      'weights',
      props.weights.information,
      props.rowCount
    );
    const operation = new GPUMaxPRegionsOperation(props);
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
        const offsets = getCPUOperationUint32Rows(id, inputs, 'offsets');
        if (offsets.length !== props.rowCount + 1)
          throw new Error(`${id} offsets length must equal rowCount + 1`);
        const result = computeMaxPRegionsOnCPU(
          {
            offsets,
            neighbors: getCPUOperationUint32Rows(id, inputs, 'neighbors'),
            weights: getCPUOperationFloat32Rows(id, inputs, 'weights'),
            distances:
              props.weights.distances === undefined
                ? undefined
                : getCPUOperationFloat32Rows(id, inputs, 'distances'),
            values: getCPUOperationFloat32Rows(id, inputs, 'values'),
            extensiveValues: getCPUOperationFloat32Rows(id, inputs, 'extensiveValues'),
            threshold: getCPUOperationFloat32Rows(id, inputs, 'threshold'),
            parameters: getCPUOperationUint32Rows(id, inputs, 'parameters')
          },
          {...props, information: props.weights.information}
        );
        return {
          [props.output.labels]: result.labels,
          [props.output.objective]: result.objective,
          [props.output.status]: result.status
        };
      }
    });
  }
}
