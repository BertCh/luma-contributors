// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  validateCapacity,
  validateFiniteNumber
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  calibrateRasterOperationBand,
  readRasterOperationBand,
  validateRasterOperationBand,
  type CPURasterOperationBand
} from '../../raster-analysis/raster-grid/raster-operation-band';
import {
  GPUTerrainPeakProminenceOperation,
  type TerrainPeakProminenceOperationProps
} from './gpu-terrain-peak-prominence-operation';

const NO_PEAK = 0xffffffff;
/** Native scalar integer ranges define valid nodata metadata; these are not GPU resource caps. */
const NATIVE_NODATA_RANGES = {
  uint32: [0, 4294967295],
  sint32: [-2147483648, 2147483647]
} as const;
/** Stable unordered edge order of the contributor's spill graph. */
const EDGE_OFFSETS = [
  [1, 0],
  [1, 1],
  [0, 1],
  [-1, 1]
] as const;

export type CPUTerrainPeakProminenceResult = {
  prominence: Float32Array;
  keySaddle: Float32Array;
  parentPeak: Uint32Array;
};

/**
 * Builds the upper-excursion elder join tree by stable descending-saddle edge unions. Higher
 * peaks survive; equal heights favor the lowest cell index. Validity, raw nodata and calibration
 * use the shared raster-band helper. Disconnected elders have infinite prominence.
 *
 * What differs from the GPU contributor:
 * - Buffer bands only; texture storage is excluded from both operation forms.
 * - Calibration, saddle ordering and height differences use float64, rounding only final float
 *   outputs. GPU float32 calibration can collapse distinct heights, changing elder and parent ties,
 *   or overflow samples that float64 retains. Native float32 nodata is rounded to its input format.
 * - Serial union-find replaces atomic parent splicing and ancestor doubling, retaining stable
 *   cell/E/SE/S/SW edge order and exact integer parent ties when calibrated heights agree.
 * - The GPU's five-tree-nodes-per-cell storage-binding limit is dropped; the CPU needs no GPU scratch.
 * - Zero saddles are always +0 in the twin; GPU source normalizes them too, but the test GPU (Apple, measured 2026-11-11) publishes -0.
 */
export function computeTerrainPeakProminenceOnCPU(
  elevation: CPURasterOperationBand,
  grid: GPURasterMetadata
): CPUTerrainPeakProminenceResult {
  validateRasterOperationBand('computeTerrainPeakProminenceOnCPU', elevation, grid);
  validateTerrainPeakProminenceNoDataValue('computeTerrainPeakProminenceOnCPU', elevation);
  const count = grid.width * grid.height;
  const values = calibrateRasterOperationBand(
    elevation.format === 'float32' && elevation.noDataValue !== undefined
      ? {...elevation, noDataValue: Math.fround(elevation.noDataValue)}
      : elevation,
    count
  );
  const parents = Uint32Array.from({length: count}, (_, cell) => cell);
  const sizes = new Uint32Array(count).fill(1);
  const peaks = parents.slice();
  const saddles = Float64Array.from(values, value => (Number.isFinite(value) ? -Infinity : NaN));
  const parentPeak = new Uint32Array(count).fill(NO_PEAK);
  const edges: {first: number; second: number; saddle: number; order: number}[] = [];
  for (let cell = 0; cell < count; cell++) {
    if (!Number.isFinite(values[cell])) continue;
    const column = cell % grid.width;
    const row = Math.floor(cell / grid.width);
    for (const [direction, [columnOffset, rowOffset]] of EDGE_OFFSETS.entries()) {
      const nextColumn = column + columnOffset;
      const nextRow = row + rowOffset;
      if (nextColumn < 0 || nextColumn >= grid.width || nextRow >= grid.height) continue;
      const neighbor = nextRow * grid.width + nextColumn;
      if (Number.isFinite(values[neighbor])) {
        const saddle = Math.min(values[cell], values[neighbor]);
        edges.push({
          first: cell,
          second: neighbor,
          saddle: saddle === 0 ? 0 : saddle,
          order: cell * EDGE_OFFSETS.length + direction
        });
      }
    }
  }
  edges.sort((first, second) => second.saddle - first.saddle || first.order - second.order);
  const findRoot = (cell: number): number => {
    let root = cell;
    while (parents[root] !== root) root = parents[root];
    while (parents[cell] !== cell) {
      const next = parents[cell];
      parents[cell] = root;
      cell = next;
    }
    return root;
  };
  for (const edge of edges) {
    let first = findRoot(edge.first);
    let second = findRoot(edge.second);
    if (first === second) continue;
    const firstPeak = peaks[first];
    const secondPeak = peaks[second];
    const firstWins =
      values[firstPeak] > values[secondPeak] ||
      (values[firstPeak] === values[secondPeak] && firstPeak < secondPeak);
    const winner = firstWins ? firstPeak : secondPeak;
    const loser = firstWins ? secondPeak : firstPeak;
    saddles[loser] = edge.saddle;
    parentPeak[loser] = winner;
    if (sizes[first] < sizes[second]) [first, second] = [second, first];
    parents[second] = first;
    sizes[first] += sizes[second];
    peaks[first] = winner;
  }
  return {
    prominence: Float32Array.from(values, (value, cell) => value - saddles[cell]),
    keySaddle: Float32Array.from(saddles),
    parentPeak
  };
}

/** Matches native scalar nodata rules before any band calibration or graph execution. */
function validateTerrainPeakProminenceNoDataValue(
  id: string,
  elevation: Pick<CPURasterOperationBand, 'format' | 'noDataValue'>
): void {
  const {format, noDataValue} = elevation;
  if (noDataValue === undefined) return;
  const name = `elevation.noDataValue (${format})`;
  if (format === 'float32') {
    if (!Number.isNaN(noDataValue)) validateFiniteNumber(id, name, noDataValue);
  } else {
    const [minimum, maximum] = NATIVE_NODATA_RANGES[format];
    validateCapacity(id, name, noDataValue, minimum, maximum);
  }
}

/** CPU twin for buffer bands; no GPU tree scratch or storage-binding cell cap is required. */
export class CPUTerrainPeakProminenceOperation extends GPUCPUOperation {
  constructor(props: TerrainPeakProminenceOperationProps) {
    const operation = new GPUTerrainPeakProminenceOperation(props);
    const {id} = operation;
    validateRasterOperationBand(id, props.elevation, props.grid);
    validateTerrainPeakProminenceNoDataValue(id, props.elevation);
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
        const result = computeTerrainPeakProminenceOnCPU(
          readRasterOperationBand(id, inputs, props.elevation),
          props.grid
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUTerrainPeakProminenceResult]
          ])
        );
      }
    });
  }
}
