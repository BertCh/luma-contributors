// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCPUOperationFloat32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {
  getRasterGroundCellSize,
  validateRasterCellSizeMode,
  type RasterGridGeoreference
} from '../../analysis-core/contracts/raster-grid';
import {
  validateRasterGridAxisAligned,
  validateRasterGridOrthogonal
} from '../../raster-analysis/raster-band-utils';
import {
  calibrateRasterOperationBand,
  readRasterOperationBand,
  validateRasterOperationBand,
  type CPURasterOperationBand
} from '../../raster-analysis/raster-grid/raster-operation-band';
import {
  getRasterD8Distance,
  GPU_RASTER_D8_DIRECTIONS
} from '../../raster-analysis/cost-distance/raster-grid-utils';
import {getTerrainDrainageReceiver} from './cpu-terrain-drainage-utils';
import {RASTER_RELAXATION_TILE_SIZE} from '../../raster-analysis/cost-distance/raster-relaxation';
import {TERRAIN_FLOW_SWEEP_AFTER_ITERATION} from './terrain-flow-passes';
import {
  GPU_TERRAIN_FLOW_CELL_CLASS as CELL_CLASS,
  GPU_TERRAIN_FLOW_NONE,
  GPU_TERRAIN_FLOW_UNRESOLVED,
  GPU_TERRAIN_FLOW_PARAMETER_LENGTH,
  GPU_TERRAIN_FLOW_STREAM_UNRESOLVED
} from './gpu-terrain-flow';
import {
  GPUTerrainFlowOperation,
  snapshotTerrainFlowOperationProps,
  type TerrainFlowOperationProps
} from './gpu-terrain-flow-operation';

/** Borrowed packed input rows and the elevation band's calibration metadata. */
export type CPUTerrainFlowInput = {
  elevation: CPURasterOperationBand;
  parameters: Float32Array;
  runoff?: Float32Array;
};

/** Explicit positive-epsilon fill and flat budgets bind tiled relaxations; accumulation is exact. */
export type CPUTerrainFlowOptions = Pick<
  TerrainFlowOperationProps,
  | 'grid'
  | 'cellSizeMode'
  | 'fillDepressions'
  | 'resolveFlats'
  | 'flowRouting'
  | 'accumulationUnits'
  | 'maximumFillIterations'
  | 'maximumFlatIterations'
  | 'maximumAccumulationIterations'
>;

/** Same per-cell layouts and scalar status words as GPUTerrainFlowOutput. */
export type CPUTerrainFlowResult = {
  filledElevation: Float32Array;
  flowDirections: Uint32Array;
  cellClasses: Uint32Array;
  accumulation: Float32Array;
  streams: Uint32Array;
  status: {
    fillConverged: Uint32Array;
    flatsConverged: Uint32Array;
    accumulationConverged: Uint32Array;
    fillIterationCount: Uint32Array;
    flatsIterationCount: Uint32Array;
    accumulationIterationCount: Uint32Array;
  };
};

type GetNeighbor = (cell: number, direction: number) => number;
type FlowGeometry = {
  getNeighbor: GetNeighbor;
  getDistance: (cell: number, direction: number) => number;
  getGroundSize: (rowCoordinate: number) => [number, number];
  width: number;
};

/** Integer no-data metadata must be representable in the band's declared 32-bit scalar format. */
const TERRAIN_FLOW_NO_DATA_LIMITS = {
  uint32: {minimum: 0, maximum: 0xffffffff},
  sint32: {minimum: -0x80000000, maximum: 0x7fffffff}
} as const;

/**
 * Fills by a multi-source minimax priority flood, resolves Barnes flats by breadth-first hop
 * distances, and accumulates the complete receiver DAG in topological order. Boundary means the
 * tile edge or any of eight neighbors missing. Strict D8/facet/mask comparisons retain the
 * contributor's first-direction tie rule. Nonfinite/negative runoff contributes zero.
 *
 * What differs from the GPU contributor:
 * - Calibration, fill increments, slopes, fractions, areas and accumulation use float64, with
 *   one float32 rounding at output. Use relative tolerance 3e-5 on well-conditioned float outputs;
 *   near slope/facet ties, rounded flat heights and stream thresholds can change integer outputs.
 *   GPU float32 intermediate overflow can invalidate additional elevations or routings; the twin
 *   keeps finite float64 intermediates even when their published elevation overflows float32.
 * - Zero-epsilon spill filling and DAG accumulation complete without relaxation budgets.
 *   Without explicit budgets, positive-epsilon filling and flat resolution run to completion.
 * - An explicit maximumFillIterations bounds positive-epsilon tiled filling, including unchanged
 *   confirmation. Exhaustion clears fillConverged/flatsConverged and suppresses dependent outputs.
 * - An explicit maximumFlatIterations bounds each of the three tiled flat stages, including
 *   unchanged confirmation. Plain tile halos read a round snapshot; after 96 rounds both bounded
 *   fields cycle tiled/right/left/down/up, publishing each swept tile to the next on its line.
 *   GPU workgroup scheduling can advance other lines sooner. Exhaustion clears flatsConverged and suppresses
 *   dependent directions, classes, accumulation and streams with unresolved values.
 * - Iteration-count words are 0 except explicit fill/flat budgets, which count executed stage rounds.
 *   Accumulation convergence describes DAG completion, not the GPU's conservative last-round flag.
 * - Explicit flat budgets must be in [1,1024]. Other GPU-only upper iteration limits are dropped.
 *   Optional persistent flatContinuation and texture bands have no operation ports.
 * - Nonfinite calibrated elevations are missing; subnormals are retained. NaN slopes do not
 *   route, and non-positive/nonfinite epsilon and exponent use the contributor's defaults.
 *   A float64 accumulation that overflows float32 is published as infinity and its stream word
 *   is unresolved, as for any nonfinite GPU accumulation.
 * - Both forms write requested flat statuses even on filledElevation-only requests.
 *
 * No-data metadata follows the scalar format: uint32 and sint32 require an integer in their
 * respective ranges; float32 accepts NaN or a value that stays finite after float32 rounding.
 * Invalid metadata (including uint32 -1 or 0.5, sint32 NaN and float32 Infinity) throws an Error
 * naming elevation.noDataValue, both at operation construction and in this function.
 *
 * @throws Error naming inconsistent elevation.values, elevation.validity, parameters or runoff,
 * or invalid elevation.noDataValue metadata.
 */
export function computeTerrainFlowOnCPU(
  input: CPUTerrainFlowInput,
  options: CPUTerrainFlowOptions
): CPUTerrainFlowResult {
  const id = 'computeTerrainFlowOnCPU';
  validateRasterOperationBand(id, input.elevation, options.grid, 'elevation');
  validateFlowNoDataValue(id, input.elevation);
  validateFlowOptions(id, options);
  const {width, height} = options.grid;
  const count = width * height;
  if (input.elevation.values.length !== count)
    throw new Error('elevation.values must contain exactly one sample per grid cell');
  if (input.elevation.validity && input.elevation.validity.length !== count)
    throw new Error('elevation.validity must contain exactly one flag per grid cell');
  if (input.parameters.length < GPU_TERRAIN_FLOW_PARAMETER_LENGTH)
    throw new Error('parameters must hold the getGPUTerrainFlowParameterValues layout');
  if (input.runoff && input.runoff.length !== count)
    throw new Error('runoff must contain exactly one weight per grid cell');
  // getGPUTerrainFlowParameterValues has one published length, not individual slot constants.
  // Destructure its documented layout instead of repeating numeric offsets.
  const [cellSizeX, cellSizeY, northEdge, southEdge, fillEpsilon, streamThreshold, flowExponent] =
    input.parameters;
  const georeference: RasterGridGeoreference = {
    cellSize: [cellSizeX, cellSizeY],
    northEdge,
    southEdge
  };
  const cellSizeMode = options.cellSizeMode ?? 'uniform';
  const geometry: FlowGeometry = {
    width,
    getNeighbor: (cell, direction) =>
      getTerrainDrainageReceiver(cell, GPU_RASTER_D8_DIRECTIONS[direction].code, width, height),
    getDistance: (cell, direction) =>
      getRasterD8Distance(cellSizeMode, direction, Math.floor(cell / width), height, georeference),
    getGroundSize: rowCoordinate =>
      getRasterGroundCellSize(cellSizeMode, georeference, rowCoordinate, height)
  };
  const elevation = calibrateRasterOperationBand(input.elevation, count);
  const epsilon = Number.isFinite(fillEpsilon) && fillEpsilon >= 0 ? fillEpsilon : 0;
  const fill =
    options.fillDepressions && epsilon > 0 && options.maximumFillIterations !== undefined
      ? fillSurfaceWithBudget(
          elevation,
          epsilon,
          geometry.getNeighbor,
          width,
          options.maximumFillIterations
        )
      : {
          surface: options.fillDepressions
            ? fillSurface(elevation, epsilon, geometry.getNeighbor)
            : elevation,
          converged: true,
          iterationCount: 0
        };
  const surface = fill.surface;
  const flowDirections = new Uint32Array(count);
  const cellClasses = new Uint32Array(count);
  const receivers = new Uint32Array(count).fill(GPU_TERRAIN_FLOW_NONE);
  for (let cell = 0; cell < count; cell++) {
    if (!Number.isFinite(elevation[cell])) {
      flowDirections[cell] = GPU_TERRAIN_FLOW_NONE;
      cellClasses[cell] = CELL_CLASS.invalid;
      continue;
    }
    let bestDrop = 0;
    let boundary = false;
    let hasEqualNeighbor = false;
    for (let direction = 0; direction < 8; direction++) {
      const neighbor = geometry.getNeighbor(cell, direction);
      if (neighbor === GPU_TERRAIN_FLOW_NONE || !Number.isFinite(elevation[neighbor])) {
        boundary = true;
        continue;
      }
      hasEqualNeighbor ||= surface[neighbor] === surface[cell];
      const drop = (surface[cell] - surface[neighbor]) / geometry.getDistance(cell, direction);
      if (drop > bestDrop) {
        bestDrop = drop;
        flowDirections[cell] = GPU_RASTER_D8_DIRECTIONS[direction].code;
        receivers[cell] = neighbor;
      }
    }
    cellClasses[cell] =
      bestDrop > 0
        ? CELL_CLASS.draining
        : boundary
          ? CELL_CLASS.outlet
          : hasEqualNeighbor
            ? CELL_CLASS.flat
            : CELL_CLASS.pit;
  }
  const flatStatus = options.resolveFlats
    ? resolveFlatReceivers(
        surface,
        flowDirections,
        cellClasses,
        receivers,
        geometry.getNeighbor,
        width,
        options.maximumFlatIterations
      )
    : {converged: 1, iterationCount: 0};

  const fractions = new Float64Array(count * 8);
  for (let cell = 0; cell < count; cell++) {
    if (!Number.isFinite(elevation[cell])) continue;
    computeFlowFractions(
      cell,
      surface,
      receivers,
      options.flowRouting ?? 'd8',
      flowExponent,
      geometry,
      fractions
    );
  }
  const pendingDonors = new Uint8Array(count);
  const totals = new Float64Array(count).fill(NaN);
  const queue: number[] = [];
  let validCount = 0;
  for (let cell = 0; cell < count; cell++) {
    if (!Number.isFinite(elevation[cell])) continue;
    validCount++;
    for (let direction = 0; direction < 8; direction++) {
      if (fractions[cell * 8 + direction] > 0)
        pendingDonors[geometry.getNeighbor(cell, direction)]++;
    }
  }
  for (let cell = 0; cell < count; cell++) {
    if (Number.isFinite(elevation[cell]) && pendingDonors[cell] === 0) queue.push(cell);
  }
  for (let head = 0; head < queue.length; head++) {
    const cell = queue[head];
    const runoff = input.runoff?.[cell] ?? 1;
    let total = Number.isFinite(runoff) && runoff >= 0 ? runoff : 0;
    if (options.accumulationUnits === 'area') {
      const [groundX, groundY] = geometry.getGroundSize(Math.floor(cell / width) + 0.5);
      total *= groundX * groundY;
    }
    // Pull finalized donors in the same fixed direction order as the contributor.
    for (let direction = 0; direction < 8; direction++) {
      const neighbor = geometry.getNeighbor(cell, direction);
      if (neighbor === GPU_TERRAIN_FLOW_NONE) continue;
      const fraction = fractions[neighbor * 8 + ((direction + 4) % 8)];
      if (fraction > 0) total += fraction * totals[neighbor];
    }
    totals[cell] = total;
    for (let direction = 0; direction < 8; direction++) {
      if (!(fractions[cell * 8 + direction] > 0)) continue;
      const receiver = geometry.getNeighbor(cell, direction);
      if (--pendingDonors[receiver] === 0) queue.push(receiver);
    }
  }
  const accumulation = Float32Array.from(totals);
  // Threshold the published rounded value, including its nonfinite rule.
  const streams = Uint32Array.from(accumulation, (value, cell) =>
    !Number.isFinite(elevation[cell])
      ? 0
      : !Number.isFinite(value)
        ? GPU_TERRAIN_FLOW_STREAM_UNRESOLVED
        : Number(value >= streamThreshold)
  );
  const filledElevation = Float32Array.from(surface);
  if (!fill.converged || !flatStatus.converged) {
    for (let cell = 0; cell < count; cell++) {
      if (!Number.isFinite(elevation[cell])) continue;
      if (!fill.converged) filledElevation[cell] = NaN;
      flowDirections[cell] = GPU_TERRAIN_FLOW_UNRESOLVED;
      cellClasses[cell] = CELL_CLASS.unresolved;
      accumulation[cell] = NaN;
      streams[cell] = GPU_TERRAIN_FLOW_STREAM_UNRESOLVED;
    }
  }
  return {
    filledElevation,
    flowDirections,
    cellClasses,
    accumulation,
    streams,
    status: {
      fillConverged: Uint32Array.of(Number(fill.converged)),
      flatsConverged: Uint32Array.of(Number(fill.converged && flatStatus.converged)),
      accumulationConverged: Uint32Array.of(
        Number(fill.converged && flatStatus.converged && queue.length === validCount)
      ),
      fillIterationCount: Uint32Array.of(fill.iterationCount),
      flatsIterationCount: Uint32Array.of(flatStatus.iterationCount),
      accumulationIterationCount: Uint32Array.of(0)
    }
  };
}

/** Supplements the shared band validator with the canonicalization path's scalar constraints. */
function validateFlowNoDataValue(
  id: string,
  elevation: Pick<CPURasterOperationBand, 'format' | 'noDataValue'>
): void {
  const {format, noDataValue} = elevation;
  if (noDataValue === undefined) return;
  if (format === 'float32') {
    validateRequirement(
      id,
      Number.isNaN(noDataValue) || Number.isFinite(Math.fround(noDataValue)),
      'elevation.noDataValue must be a finite float32 value or NaN'
    );
    return;
  }
  const {minimum, maximum} = TERRAIN_FLOW_NO_DATA_LIMITS[format];
  validateRequirement(
    id,
    Number.isSafeInteger(noDataValue) && noDataValue >= minimum && noDataValue <= maximum,
    `elevation.noDataValue must be an integer in ${format} range [${minimum}, ${maximum}]`
  );
}

function validateFlowOptions(id: string, options: CPUTerrainFlowOptions): void {
  validateCapacity(id, 'grid.width', options.grid.width, 1);
  validateCapacity(id, 'grid.height', options.grid.height, 1);
  validateMaximumCount(
    id,
    'grid cells',
    options.grid.width * options.grid.height,
    GPU_TERRAIN_FLOW_NONE
  );
  validateRasterCellSizeMode(id, options.cellSizeMode);
  validateRasterGridAxisAligned(id, options.grid, options.cellSizeMode);
  validateRasterGridOrthogonal(id, options.grid);
  validateRequirement(
    id,
    ['d8', 'd-infinity', 'mfd-freeman', 'mfd-quinn'].includes(options.flowRouting ?? 'd8'),
    'flowRouting must be d8, d-infinity, mfd-freeman or mfd-quinn'
  );
  validateRequirement(
    id,
    ['cells', 'area'].includes(options.accumulationUnits ?? 'cells'),
    'accumulationUnits must be cells or area'
  );
  for (const port of [
    'maximumFillIterations',
    'maximumFlatIterations',
    'maximumAccumulationIterations'
  ] as const)
    if (options[port] !== undefined)
      validateCapacity(
        id,
        port,
        options[port]!,
        1,
        port === 'maximumFlatIterations' ? 1024 : Number.MAX_SAFE_INTEGER
      );
}

/** Multi-source Dijkstra for the monotone transition max(elevation, previous + epsilon). */
function fillSurface(
  elevation: Float64Array,
  epsilon: number,
  getNeighbor: GetNeighbor
): Float64Array {
  const filled = new Float64Array(elevation.length).fill(Infinity);
  const heap: {cell: number; level: number}[] = [];
  const pushCell = (cell: number, level: number) => {
    heap.push({cell, level});
    let position = heap.length - 1;
    while (position > 0) {
      const parent = Math.floor((position - 1) / 2);
      if (heap[parent].level <= heap[position].level) break;
      [heap[parent], heap[position]] = [heap[position], heap[parent]];
      position = parent;
    }
  };
  for (let cell = 0; cell < elevation.length; cell++) {
    if (!Number.isFinite(elevation[cell])) {
      filled[cell] = NaN;
      continue;
    }
    for (let direction = 0; direction < 8; direction++) {
      const neighbor = getNeighbor(cell, direction);
      if (neighbor === GPU_TERRAIN_FLOW_NONE || !Number.isFinite(elevation[neighbor])) {
        filled[cell] = elevation[cell];
        pushCell(cell, filled[cell]);
        break;
      }
    }
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
        if (left < heap.length && heap[left].level < heap[smallest].level) smallest = left;
        if (right < heap.length && heap[right].level < heap[smallest].level) smallest = right;
        if (smallest === position) break;
        [heap[position], heap[smallest]] = [heap[smallest], heap[position]];
        position = smallest;
      }
    }
    if (current.level !== filled[current.cell]) continue;
    for (let direction = 0; direction < 8; direction++) {
      const neighbor = getNeighbor(current.cell, direction);
      if (neighbor === GPU_TERRAIN_FLOW_NONE || !Number.isFinite(elevation[neighbor])) continue;
      const candidate = Math.max(elevation[neighbor], current.level + epsilon);
      if (candidate < filled[neighbor]) {
        filled[neighbor] = candidate;
        pushCell(neighbor, candidate);
      }
    }
  }
  return filled;
}

/** Positive-epsilon fill from boundary seeds with an explicit unchanged-confirmation budget. */
function fillSurfaceWithBudget(
  elevation: Float64Array,
  epsilon: number,
  getNeighbor: GetNeighbor,
  width: number,
  maximumIterations: number
): {surface: Float64Array; converged: boolean; iterationCount: number} {
  const surface = new Float64Array(elevation.length).fill(Infinity);
  const eligible = Uint8Array.from(elevation, value => Number(Number.isFinite(value)));
  for (let cell = 0; cell < elevation.length; cell++) {
    if (!eligible[cell]) {
      surface[cell] = NaN;
      continue;
    }
    for (let direction = 0; direction < 8; direction++) {
      const neighbor = getNeighbor(cell, direction);
      if (neighbor === GPU_TERRAIN_FLOW_NONE || !eligible[neighbor]) {
        surface[cell] = elevation[cell];
        break;
      }
    }
  }
  const status = relaxTerrainField(
    surface,
    elevation,
    eligible,
    width,
    maximumIterations,
    epsilon,
    true
  );
  return {surface, ...status};
}

/** Exact traversals by default, or bounded tiled hop relaxations for explicit flat budgets. */
function resolveFlatReceivers(
  surface: Float64Array,
  directions: Uint32Array,
  classes: Uint32Array,
  receivers: Uint32Array,
  getNeighbor: GetNeighbor,
  width: number,
  maximumIterations?: number
): {converged: number; iterationCount: number} {
  const count = surface.length;
  const lower = new Float64Array(count).fill(Infinity);
  const higher = new Float64Array(count);
  const lowEdges: number[] = [];
  const highEdges: number[] = [];
  for (let cell = 0; cell < count; cell++) {
    const isFlat = classes[cell] === CELL_CLASS.flat;
    const hasFlow = classes[cell] === CELL_CLASS.draining || classes[cell] === CELL_CLASS.outlet;
    for (let direction = 0; direction < 8; direction++) {
      const neighbor = getNeighbor(cell, direction);
      if (neighbor === GPU_TERRAIN_FLOW_NONE || !Number.isFinite(surface[neighbor])) continue;
      if (
        hasFlow &&
        classes[neighbor] === CELL_CLASS.flat &&
        surface[neighbor] === surface[cell] &&
        lower[cell] !== 1
      ) {
        lower[cell] = 1;
        lowEdges.push(cell);
      }
      if (isFlat && surface[neighbor] > surface[cell] && higher[cell] !== 1) {
        higher[cell] = 1;
        highEdges.push(cell);
      }
    }
  }
  const propagateHops = (queue: number[], distances: Float64Array, unreached: number) => {
    for (let head = 0; head < queue.length; head++) {
      const cell = queue[head];
      for (let direction = 0; direction < 8; direction++) {
        const neighbor = getNeighbor(cell, direction);
        if (
          neighbor !== GPU_TERRAIN_FLOW_NONE &&
          classes[neighbor] === CELL_CLASS.flat &&
          surface[neighbor] === surface[cell] &&
          distances[neighbor] === unreached
        ) {
          distances[neighbor] = distances[cell] + 1;
          queue.push(neighbor);
        }
      }
    }
  };
  const componentMaximum = new Float64Array(count);
  let converged = 1;
  let iterationCount = 0;
  if (maximumIterations !== undefined) {
    const lowerEligible = Uint8Array.from(classes, (value, cell) =>
      Number(value === CELL_CLASS.flat || lower[cell] === 1)
    );
    const higherEligible = Uint8Array.from(classes, value => Number(value === CELL_CLASS.flat));
    const higherValues = Float64Array.from(higher, value => (value === 1 ? 1 : Infinity));
    const lowerStatus = relaxTerrainField(
      lower,
      surface,
      lowerEligible,
      width,
      maximumIterations,
      1
    );
    const higherStatus = relaxTerrainField(
      higherValues,
      surface,
      higherEligible,
      width,
      maximumIterations,
      1
    );
    const maximumValues = Float64Array.from(higherValues, value =>
      Number.isFinite(value) ? -value : 0
    );
    const maximumStatus = relaxTerrainField(
      maximumValues,
      surface,
      Uint8Array.from(surface, value => Number(Number.isFinite(value))),
      width,
      maximumIterations,
      0
    );
    higher.set(Float64Array.from(higherValues, value => (Number.isFinite(value) ? value : 0)));
    componentMaximum.set(Float64Array.from(maximumValues, value => -value));
    converged = Number(lowerStatus.converged && higherStatus.converged && maximumStatus.converged);
    iterationCount =
      lowerStatus.iterationCount + higherStatus.iterationCount + maximumStatus.iterationCount;
  } else {
    propagateHops(lowEdges, lower, Infinity);
    propagateHops(highEdges, higher, 0);
    const visited = new Uint8Array(count);
    for (let seed = 0; seed < count; seed++) {
      if (!Number.isFinite(surface[seed]) || visited[seed]) continue;
      const component = [seed];
      visited[seed] = 1;
      let maximum = 0;
      for (let head = 0; head < component.length; head++) {
        const cell = component[head];
        maximum = Math.max(maximum, higher[cell]);
        for (let direction = 0; direction < 8; direction++) {
          const neighbor = getNeighbor(cell, direction);
          if (
            neighbor !== GPU_TERRAIN_FLOW_NONE &&
            !visited[neighbor] &&
            surface[neighbor] === surface[cell]
          ) {
            visited[neighbor] = 1;
            component.push(neighbor);
          }
        }
      }
      for (const cell of component) componentMaximum[cell] = maximum;
    }
  }
  const mask = Float64Array.from(
    lower,
    (value, cell) => 2 * value + (higher[cell] > 0 ? componentMaximum[cell] - higher[cell] : 0)
  );
  for (let cell = 0; cell < count; cell++) {
    if (classes[cell] !== CELL_CLASS.flat || !Number.isFinite(mask[cell])) continue;
    let bestMask = mask[cell];
    for (let direction = 0; direction < 8; direction++) {
      const neighbor = getNeighbor(cell, direction);
      if (
        neighbor !== GPU_TERRAIN_FLOW_NONE &&
        surface[neighbor] === surface[cell] &&
        mask[neighbor] < bestMask
      ) {
        bestMask = mask[neighbor];
        directions[cell] = GPU_RASTER_D8_DIRECTIONS[direction].code;
        receivers[cell] = neighbor;
        classes[cell] = CELL_CLASS.draining;
      }
    }
  }
  return {converged, iterationCount};
}

/** Bounded tiled relaxation, with directional tile sweeps on the GPU's logical round schedule. */
function relaxTerrainField(
  values: Float64Array,
  surface: Float64Array,
  eligible: Uint8Array,
  width: number,
  maximumIterations: number,
  increment: number,
  fillDepressions = false
): {converged: boolean; iterationCount: number} {
  for (let iteration = 0; iteration < maximumIterations; iteration++) {
    if (
      !relaxTerrainFieldRound(
        values,
        surface,
        eligible,
        width,
        increment,
        iteration,
        fillDepressions
      )
    )
      return {converged: true, iterationCount: iteration + 1};
  }
  return {converged: false, iterationCount: maximumIterations};
}

/** One production tiled/swept round; exposed internally for controlled visibility fixtures. @internal */
export function relaxTerrainFieldRound(
  values: Float64Array,
  surface: Float64Array,
  eligible: Uint8Array,
  width: number,
  increment: number,
  iteration: number,
  fillDepressions = false
): boolean {
  const tileSize = RASTER_RELAXATION_TILE_SIZE;
  const haloSize = tileSize + 2;
  const height = values.length / width;
  const tilesX = Math.ceil(width / tileSize);
  const tilesY = Math.ceil(height / tileSize);
  const updated = values.slice();
  let changed = false;
  // Zero-based 96 is another tiled pass; 97/98/99/100 sweep right/left/down/up.
  const phase =
    iteration < TERRAIN_FLOW_SWEEP_AFTER_ITERATION
      ? 0
      : (iteration - TERRAIN_FLOW_SWEEP_AFTER_ITERATION) % 5;
  const vertical = phase >= 3;
  const reverse = phase === 2 || phase === 4;
  const lineCount = vertical ? tilesX : tilesY;
  const stepCount = vertical ? tilesY : tilesX;
  for (let line = 0; line < lineCount; line++) {
    for (let step = 0; step < stepCount; step++) {
      const position = reverse ? stepCount - 1 - step : step;
      const originRow = (vertical ? position : line) * tileSize;
      const originColumn = (vertical ? line : position) * tileSize;
      const cells = new Float64Array(haloSize * haloSize).fill(-1);
      let tile = new Float64Array(cells.length).fill(Infinity);
      for (let row = 0; row < haloSize; row++)
        for (let column = 0; column < haloSize; column++) {
          const globalRow = originRow + row - 1,
            globalColumn = originColumn + column - 1;
          if (globalRow < 0 || globalColumn < 0 || globalRow >= height || globalColumn >= width)
            continue;
          const local = row * haloSize + column;
          cells[local] = globalRow * width + globalColumn;
          // Sweeps publish each tile before the next visit, including across lines.
          // Visiting complete lines sequentially is one legal GPU workgroup schedule.
          tile[local] = phase !== 0 ? updated[cells[local]] : values[cells[local]];
        }
      for (let inner = 0; inner < 2 * tileSize; inner++) {
        const next = tile.slice();
        let tileChanged = false;
        for (let row = 1; row <= tileSize; row++)
          for (let column = 1; column <= tileSize; column++) {
            const local = row * haloSize + column,
              cell = cells[local];
            if (cell < 0 || !eligible[cell]) continue;
            for (const direction of GPU_RASTER_D8_DIRECTIONS) {
              const neighborLocal = local + direction.rowOffset * haloSize + direction.columnOffset;
              const neighbor = cells[neighborLocal];
              if (
                neighbor < 0 ||
                !eligible[neighbor] ||
                (!fillDepressions && surface[neighbor] !== surface[cell])
              )
                continue;
              const candidate = fillDepressions
                ? Math.max(surface[cell], tile[neighborLocal] + increment)
                : tile[neighborLocal] + increment;
              if (candidate < next[local]) {
                next[local] = candidate;
                tileChanged = true;
              }
            }
          }
        tile = next;
        if (!tileChanged) break;
      }
      for (let row = 1; row <= tileSize; row++)
        for (let column = 1; column <= tileSize; column++) {
          const local = row * haloSize + column,
            cell = cells[local];
          if (cell >= 0 && tile[local] < values[cell]) {
            updated[cell] = tile[local];
            changed = true;
          }
        }
    }
  }
  values.set(updated);
  return changed;
}

/** Writes eight fractions; only positive fractions form DAG edges. */
function computeFlowFractions(
  cell: number,
  surface: Float64Array,
  receivers: Uint32Array,
  routing: NonNullable<CPUTerrainFlowOptions['flowRouting']>,
  exponent: number,
  geometry: FlowGeometry,
  fractions: Float64Array
): void {
  const center = surface[cell];
  const offset = cell * 8;
  const fallback = () => {
    for (let direction = 0; direction < 8; direction++) {
      if (
        receivers[cell] !== GPU_TERRAIN_FLOW_NONE &&
        geometry.getNeighbor(cell, direction) === receivers[cell]
      )
        fractions[offset + direction] = 1;
    }
  };
  if (routing === 'd8') {
    fallback();
    return;
  }
  const row = Math.floor(cell / geometry.width);
  if (routing === 'd-infinity') {
    let bestSlope = 0;
    let bestCardinal = -1;
    let bestDiagonal = -1;
    let bestFraction = 0;
    // Same facet order as getFacetCardinal/getFacetDiagonal, including E/NE last for E.
    for (let facet = 0; facet < 8; facet++) {
      const cardinal = Math.floor(facet / 2) * 2;
      const diagonal =
        facet % 2 === 0 ? (cardinal === 0 ? 1 : cardinal - 1) : cardinal === 0 ? 7 : cardinal + 1;
      const cardinalNeighbor = geometry.getNeighbor(cell, cardinal);
      const diagonalNeighbor = geometry.getNeighbor(cell, diagonal);
      const cardinalUsable =
        cardinalNeighbor !== GPU_TERRAIN_FLOW_NONE && Number.isFinite(surface[cardinalNeighbor]);
      const diagonalUsable =
        diagonalNeighbor !== GPU_TERRAIN_FLOW_NONE && Number.isFinite(surface[diagonalNeighbor]);
      if (!cardinalUsable && !diagonalUsable) continue;
      const cardinalDistance = geometry.getDistance(cell, cardinal);
      const diagonalDistance = geometry.getDistance(cell, diagonal);
      const ground = geometry.getGroundSize(
        row + (GPU_RASTER_D8_DIRECTIONS[diagonal].rowOffset + 1) * 0.5
      );
      const otherDistance = cardinal === 0 || cardinal === 4 ? ground[1] : ground[0];
      let slope = 0;
      let fraction = 0;
      if (cardinalUsable && diagonalUsable) {
        const firstSlope = (center - surface[cardinalNeighbor]) / cardinalDistance;
        const secondSlope = (surface[cardinalNeighbor] - surface[diagonalNeighbor]) / otherDistance;
        if (firstSlope > 0 && secondSlope > 0) {
          const angle = Math.atan2(secondSlope, firstSlope);
          const maximumAngle = Math.atan2(otherDistance, cardinalDistance);
          if (angle <= maximumAngle) {
            slope = Math.hypot(firstSlope, secondSlope);
            fraction = angle / maximumAngle;
          } else {
            slope = (center - surface[diagonalNeighbor]) / diagonalDistance;
            fraction = 1;
          }
        } else if (firstSlope > 0) slope = firstSlope;
        else if (secondSlope > 0) {
          slope = (center - surface[diagonalNeighbor]) / diagonalDistance;
          fraction = 1;
        }
      } else if (cardinalUsable) slope = (center - surface[cardinalNeighbor]) / cardinalDistance;
      else {
        slope = (center - surface[diagonalNeighbor]) / diagonalDistance;
        fraction = 1;
      }
      if (slope > bestSlope) {
        bestSlope = slope;
        bestCardinal = cardinal;
        bestDiagonal = diagonal;
        bestFraction = fraction;
      }
    }
    if (bestCardinal < 0) fallback();
    else {
      fractions[offset + bestCardinal] = 1 - bestFraction;
      fractions[offset + bestDiagonal] = bestFraction;
    }
    return;
  }
  const quinn = routing === 'mfd-quinn';
  const power = Number.isFinite(exponent) && exponent > 0 ? exponent : quinn ? 1 : 1.1;
  const ground = geometry.getGroundSize(row + 0.5);
  let maximumScore = 0;
  const lowerDirections: number[] = [];
  for (let direction = 0; direction < 8; direction++) {
    const neighbor = geometry.getNeighbor(cell, direction);
    if (neighbor === GPU_TERRAIN_FLOW_NONE) continue;
    const drop = center - surface[neighbor];
    if (!Number.isFinite(drop) || !(drop > 0)) continue;
    lowerDirections.push(direction);
    const distance = geometry.getDistance(cell, direction);
    const contour =
      direction === 0 || direction === 4
        ? 0.5 * ground[1]
        : direction === 2 || direction === 6
          ? 0.5 * ground[0]
          : 0.25 * distance;
    const score = (drop / distance) * (quinn ? contour : 1);
    fractions[offset + direction] = score;
    maximumScore = Math.max(maximumScore, score);
  }
  if (!lowerDirections.length) {
    fallback();
    return;
  }
  let total = 0;
  for (const direction of lowerDirections) {
    const score = fractions[offset + direction];
    // Match getMultipleFlowWeight's fallback when all scores are zero (e.g. infinite distances).
    const ratio = maximumScore > 0 ? score / maximumScore : 1;
    const weight = ratio > 0 ? ratio ** power : 0;
    fractions[offset + direction] = weight;
    total += weight;
  }
  for (let direction = 0; direction < 8; direction++)
    fractions[offset + direction] = total > 0 ? fractions[offset + direction] / total : 0;
}

/** Buffer-band CPU twin with bounded explicit fill/flat stages and exact DAG accumulation. */
export class CPUTerrainFlowOperation extends GPUCPUOperation {
  constructor(props: TerrainFlowOperationProps) {
    props = snapshotTerrainFlowOperationProps(props);
    const operation = new GPUTerrainFlowOperation(props);
    const {id} = operation;
    validateRasterOperationBand(id, props.elevation, props.grid, 'elevation');
    validateFlowNoDataValue(id, props.elevation);
    validateFlowOptions(id, props);
    const {output} = props;
    const status = output.status;
    validateRequirement(
      id,
      ['filledElevation', 'flowDirections', 'cellClasses', 'accumulation', 'streams'].some(
        port => output[port as keyof typeof output] !== undefined
      ),
      'output requires at least one per-cell output'
    );
    validateRequirement(
      id,
      !(output.filledElevation || status?.fillConverged || status?.fillIterationCount) ||
        Boolean(props.fillDepressions),
      'output.filledElevation and fill status require fillDepressions'
    );
    validateRequirement(
      id,
      !(status?.flatsConverged || status?.flatsIterationCount) || Boolean(props.resolveFlats),
      'output.status flat words require resolveFlats'
    );
    validateRequirement(
      id,
      !(status?.accumulationConverged || status?.accumulationIterationCount) ||
        Boolean(output.accumulation || output.streams),
      'output.status accumulation words require output.accumulation or output.streams'
    );
    super({
      id,
      inputs: operation.inputs,
      outputs: Object.fromEntries(
        Object.values(operation.outputs).map(outputPort => [
          outputPort.value,
          gpuGraph.vector(outputPort.format, {length: outputPort.length})
        ])
      ),
      execute({inputs}) {
        const result = computeTerrainFlowOnCPU(
          {
            elevation: readRasterOperationBand(id, inputs, props.elevation),
            parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters'),
            runoff:
              props.runoff !== undefined
                ? getCPUOperationFloat32Rows(id, inputs, 'runoff')
                : undefined
          },
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, outputPort]) => [
            outputPort.value,
            port in result.status
              ? result.status[port as keyof CPUTerrainFlowResult['status']]
              : result[port as Exclude<keyof CPUTerrainFlowResult, 'status'>]
          ])
        );
      }
    });
  }
}
