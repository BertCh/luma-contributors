// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {KeplerAllocationError} from '../runtime/dataset-registry';
import {
  GPUInverseDistanceWeighting,
  GPUKernelDensity,
  GPUKriging,
  GPULineDensity,
  GPUPointDensity,
  getGPUInverseDistanceWeightingParameterValues,
  getGPUKernelDensityParameterValues,
  getGPUKrigingParameterValues,
  getGPULineDensityParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import type {CatalogGpuContext, CatalogRunOutput} from './catalog-binding-types';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  getFeatureIdentity,
  getLineParts,
  getPolygonParts,
  makeCatalogTable,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic,
  type CatalogFeatureCollection,
  type CatalogPosition
} from './catalog-geojson';
import {CatalogGpuUnsupportedError, withCatalogGpuSession} from './catalog-gpu-session';
import {
  addDroppedDiagnostic,
  collectAllPositions,
  getSampleExtent,
  makeCellPolygonText,
  makeGridSpec,
  readPointSamples,
  type Values,
  type GridSpec,
  type Sample
} from './catalog-platform3-b-common';
import {addLocalFrameDiagnostic, makeLocalFrame, type LocalFrame} from './catalog-local-frame';
import {solveDenseLinearSystem} from './dense-linear-system';

/** Largest sample-times-cell work the CPU references accept before throwing. */
const MAXIMUM_CPU_WORK = 6e7;

export type GridField = {
  values: ArrayLike<number>;
  counts?: ArrayLike<number>;
  variances?: ArrayLike<number>;
};

type GridOutputOptions = {
  tableId: string;
  label: string;
  valueColumn: string;
  includeEmptyCells: boolean;
  /** Cells to publish when empty cells are excluded. */
  isEmpty: (index: number) => boolean;
  integerValues?: boolean;
};

/** Cell table shared by every grid operation: identity is (row, column) and `cell_id`. */
export function makeGridTable(
  frame: LocalFrame,
  grid: GridSpec,
  field: GridField,
  options: GridOutputOptions
) {
  const table = makeCatalogTable(
    options.tableId,
    options.label,
    {
      cell_id: 'string',
      column: 'int32',
      row: 'int32',
      [options.valueColumn]: options.integerValues ? 'int32' : 'float64',
      sample_count: 'int32',
      ...(field.variances ? {variance: 'float64' as const} : {}),
      geometry: 'string'
    },
    'geometry'
  );
  for (let row = 0; row < grid.rows; row++)
    for (let column = 0; column < grid.columns; column++) {
      const index = row * grid.columns + column;
      if (!options.includeEmptyCells && options.isEmpty(index)) continue;
      const value = field.values[index];
      appendCatalogRow(table, {
        cell_id: `${row}:${column}`,
        column,
        row,
        [options.valueColumn]: Number.isFinite(value) ? value : null,
        sample_count: field.counts ? field.counts[index] : null,
        ...(field.variances
          ? {variance: Number.isFinite(field.variances[index]) ? field.variances[index] : null}
          : {}),
        geometry: makeCellPolygonText(frame, grid, column, row)
      });
    }
  return table;
}

export function getCellCenter(grid: GridSpec, column: number, row: number): [number, number] {
  return [grid.minX + (column + 0.5) * grid.cellSize, grid.minY + (row + 0.5) * grid.cellSize];
}

export type PointGridContext = {
  frame: LocalFrame;
  sample: Sample;
  grid: GridSpec;
  diagnostics: CatalogDiagnostic[];
};

/** Reads point samples and sizes the grid over their extent plus `padding`. */
export function preparePointGrid(
  points: CatalogFeatureCollection,
  valueProperty: string,
  coordinateSystem: CatalogCoordinateSystem,
  cellSize: number,
  padding = 0
): PointGridContext {
  const diagnostics: CatalogDiagnostic[] = [];
  // Kernel density, IDW and kriging stay on the equidistant frame: their bandwidths are distances.
  const frame = makeLocalFrame(coordinateSystem, collectAllPositions(points), {shift: 'minimum'});
  const sample = readPointSamples(points, valueProperty, frame);
  addDroppedDiagnostic(diagnostics, sample.droppedCount);
  addLocalFrameDiagnostic(diagnostics, frame);
  // No usable point samples; nothing to grid.
  if (!sample.count) throw new Error('No usable samples');
  const {width, height} = getSampleExtent(sample);
  return {frame, sample, grid: makeGridSpec(width, height, cellSize, padding), diagnostics};
}

function assertCpuWork(cells: number, samples: number): void {
  // Sample-times-cell work exceeds the CPU reference capacity; use a larger cell size or the GPU engine.
  if (cells * samples > MAXIMUM_CPU_WORK)
    throw new KeplerAllocationError('capacity', {message: 'Work capacity exceeded'});
}

// ---------------------------------------------------------------------------------------------
// Inverse distance weighting
// ---------------------------------------------------------------------------------------------

export type IdwSettings = {
  searchRadius: number;
  power: number;
  neighborCount: number;
  minimumNeighborCount: number;
};

export function computeIdwField(sample: Sample, grid: GridSpec, settings: IdwSettings): GridField {
  const cells = grid.columns * grid.rows;
  assertCpuWork(cells, sample.count);
  const values = new Float64Array(cells).fill(Number.NaN);
  const counts = new Uint32Array(cells);
  const radiusSquared = settings.searchRadius ** 2;
  const minimum = Math.max(settings.minimumNeighborCount, 1);
  for (let row = 0; row < grid.rows; row++)
    for (let column = 0; column < grid.columns; column++) {
      const [cx, cy] = getCellCenter(grid, column, row);
      const candidates: {distanceSquared: number; index: number}[] = [];
      let exact = -1;
      for (let index = 0; index < sample.count; index++) {
        const distanceSquared = (sample.x[index] - cx) ** 2 + (sample.y[index] - cy) ** 2;
        if (distanceSquared > radiusSquared) continue;
        if (distanceSquared === 0) {
          exact = index;
          break;
        }
        candidates.push({distanceSquared, index});
      }
      const cell = row * grid.columns + column;
      if (exact >= 0) {
        values[cell] = sample.value[exact];
        counts[cell] = 1;
        continue;
      }
      candidates.sort((a, b) => a.distanceSquared - b.distanceSquared || a.index - b.index);
      const used =
        settings.neighborCount > 0 ? candidates.slice(0, settings.neighborCount) : candidates;
      counts[cell] = used.length;
      if (used.length < minimum) continue;
      let weightSum = 0;
      let weighted = 0;
      for (const {distanceSquared, index} of used) {
        const weight = distanceSquared ** (-settings.power / 2);
        weightSum += weight;
        weighted += weight * sample.value[index];
      }
      values[cell] = weighted / weightSum;
    }
  return {values, counts};
}

function toFloat32Positions(sample: Sample): Float32Array {
  const positions = new Float32Array(sample.count * 2);
  for (let index = 0; index < sample.count; index++) {
    positions[2 * index] = sample.x[index];
    positions[2 * index + 1] = sample.y[index];
  }
  return positions;
}

function makeRasterMetadata(grid: GridSpec): GPURasterMetadata {
  return {
    width: grid.columns,
    height: grid.rows,
    affine: [grid.cellSize, 0, grid.minX, 0, grid.cellSize, grid.minY],
    pixelInterpretation: 'area'
  };
}

function getIndexGridSize(sampleCount: number, targetPerCell = 4): [number, number] {
  const side = Math.max(1, Math.min(256, Math.ceil(Math.sqrt(sampleCount / targetPerCell))));
  return [side, side];
}

function getGridBounds(grid: GridSpec): [number, number, number, number] {
  return [
    grid.minX,
    grid.minY,
    grid.minX + grid.columns * grid.cellSize,
    grid.minY + grid.rows * grid.cellSize
  ];
}

export async function computeIdwFieldGpu(
  context: CatalogGpuContext,
  sample: Sample,
  grid: GridSpec,
  settings: IdwSettings
): Promise<GridField> {
  // The index domain is float32; an unbounded radius needs the CPU engine's exact arithmetic.
  if (!Number.isFinite(settings.searchRadius))
    throw new CatalogGpuUnsupportedError('Search radius must be finite');
  const cells = grid.columns * grid.rows;
  const raster = makeRasterMetadata(grid);
  return withCatalogGpuSession(context.device, 'catalog-idw-grid', async session => {
    const sampleValues = new Float32Array(sample.count);
    sample.value.forEach((value, index) => (sampleValues[index] = value));
    const values = session.output('values', 'float32', cells);
    const counts = session.output('counts', 'uint32', cells);
    session.graph.add(
      new GPUInverseDistanceWeighting({
        id: 'catalog-idw-grid',
        positions: session.input('positions', toFloat32Positions(sample), 'float32x2'),
        values: session.input('sample-values', sampleValues, 'float32'),
        parameters: session.input(
          'parameters',
          getGPUInverseDistanceWeightingParameterValues(
            {
              searchRadius: settings.searchRadius,
              power: settings.power,
              neighborCount: settings.neighborCount,
              minimumNeighborCount: settings.minimumNeighborCount
            },
            raster
          ),
          'float32'
        ),
        grid: raster,
        indexGridSize: getIndexGridSize(sample.count),
        indexBounds: getGridBounds(grid),
        maximumNeighborCount: settings.neighborCount,
        output: {values: values.view, counts: counts.view}
      })
    );
    session.run();
    return {
      values: await session.readFloat32(values),
      counts: await session.readUint32(counts)
    };
  });
}

// ---------------------------------------------------------------------------------------------
// Kernel density
// ---------------------------------------------------------------------------------------------

export type KernelDensitySettings = {
  bandwidth: number;
  kernel: 'gaussian' | 'quartic' | 'epanechnikov';
};

function getKernelValue(
  kernel: KernelDensitySettings['kernel'],
  distanceSquared: number,
  h: number
) {
  const t = distanceSquared / (h * h);
  if (kernel === 'gaussian') {
    // Truncated at five bandwidths, as GPUKernelDensity does.
    if (distanceSquared > (5 * h) ** 2) return 0;
    return Math.exp(-t / 2) / (2 * Math.PI * h * h);
  }
  if (t >= 1) return 0;
  if (kernel === 'quartic') return ((3 / Math.PI) * (1 - t) ** 2) / (h * h);
  return ((2 / Math.PI) * (1 - t)) / (h * h);
}

export function computeKernelDensityField(
  sample: Sample,
  grid: GridSpec,
  settings: KernelDensitySettings
): GridField {
  const cells = grid.columns * grid.rows;
  assertCpuWork(cells, sample.count);
  const values = new Float64Array(cells);
  for (let row = 0; row < grid.rows; row++)
    for (let column = 0; column < grid.columns; column++) {
      const [cx, cy] = getCellCenter(grid, column, row);
      let sum = 0;
      for (let index = 0; index < sample.count; index++) {
        const weight = sample.value[index];
        if (weight < 0) continue;
        sum +=
          weight *
          getKernelValue(
            settings.kernel,
            (sample.x[index] - cx) ** 2 + (sample.y[index] - cy) ** 2,
            settings.bandwidth
          );
      }
      values[row * grid.columns + column] = sum;
    }
  return {values};
}

export async function computeKernelDensityFieldGpu(
  context: CatalogGpuContext,
  sample: Sample,
  grid: GridSpec,
  settings: KernelDensitySettings,
  weighted: boolean
): Promise<GridField> {
  const cells = grid.columns * grid.rows;
  const raster = makeRasterMetadata(grid);
  return withCatalogGpuSession(context.device, 'catalog-kernel-density', async session => {
    const weights = new Float32Array(sample.count);
    sample.value.forEach((value, index) => (weights[index] = value));
    const values = session.output('values', 'float32', cells);
    const bounds = getGridBounds(grid);
    const extent = Math.max(bounds[2] - bounds[0], bounds[3] - bounds[1]);
    const side = Math.max(1, Math.min(128, Math.ceil(extent / settings.bandwidth)));
    session.graph.add(
      new GPUKernelDensity({
        id: 'catalog-kernel-density',
        positions: session.input('positions', toFloat32Positions(sample), 'float32x2'),
        ...(weighted ? {weights: session.input('weights', weights, 'float32')} : {}),
        parameters: session.input(
          'parameters',
          getGPUKernelDensityParameterValues({bandwidth: settings.bandwidth}, raster),
          'float32'
        ),
        grid: raster,
        indexGridSize: [side, side],
        indexBounds: bounds,
        kernel: settings.kernel,
        edgeCorrection: 'none',
        normalization: 'intensity',
        method: 'exact',
        output: {values: values.view}
      })
    );
    session.run();
    return {values: await session.readFloat32(values)};
  });
}

// ---------------------------------------------------------------------------------------------
// Point density (binned statistic per cell)
// ---------------------------------------------------------------------------------------------

export type PointDensityStatistic = 'count' | 'sum' | 'mean';

function getBinnedCell(grid: GridSpec, x: number, y: number): number {
  const column = Math.min(
    grid.columns - 1,
    Math.max(0, Math.floor((x - grid.minX) / grid.cellSize))
  );
  const row = Math.min(grid.rows - 1, Math.max(0, Math.floor((y - grid.minY) / grid.cellSize)));
  return row * grid.columns + column;
}

export function computePointDensityField(
  sample: Sample,
  grid: GridSpec,
  statistic: PointDensityStatistic
): GridField {
  const cells = grid.columns * grid.rows;
  const counts = new Uint32Array(cells);
  const sums = new Float64Array(cells);
  for (let index = 0; index < sample.count; index++) {
    const cell = getBinnedCell(grid, sample.x[index], sample.y[index]);
    counts[cell]++;
    sums[cell] += sample.value[index];
  }
  const values = new Float64Array(cells);
  for (let cell = 0; cell < cells; cell++)
    values[cell] =
      statistic === 'count'
        ? counts[cell]
        : statistic === 'sum'
          ? sums[cell]
          : counts[cell]
            ? sums[cell] / counts[cell]
            : 0;
  return {values, counts};
}

export async function computePointDensityFieldGpu(
  context: CatalogGpuContext,
  sample: Sample,
  grid: GridSpec,
  statistic: PointDensityStatistic,
  weighted: boolean
): Promise<GridField> {
  const cells = grid.columns * grid.rows;
  return withCatalogGpuSession(context.device, 'catalog-point-density', async session => {
    const weights = new Float32Array(sample.count);
    sample.value.forEach((value, index) => (weights[index] = value));
    const values = session.output('values', 'float32', cells);
    const counts = session.output('counts', 'uint32', cells);
    session.graph.add(
      new GPUPointDensity({
        id: 'catalog-point-density',
        positions: session.input('positions', toFloat32Positions(sample), 'float32x2'),
        ...(weighted ? {weights: session.input('weights', weights, 'float32')} : {}),
        bounds: getGridBounds(grid),
        gridSize: [grid.columns, grid.rows],
        statistic,
        output: {values: values.view, counts: counts.view}
      })
    );
    session.run();
    return {
      values: await session.readFloat32(values),
      counts: await session.readUint32(counts)
    };
  });
}

// ---------------------------------------------------------------------------------------------
// Ordinary kriging with a stated variogram
// ---------------------------------------------------------------------------------------------

/** Canonical coordinates and stable feature keys also fix GPU distance ties. */
export function prepareKrigingGrid(
  points: CatalogFeatureCollection,
  valueProperty: string,
  coordinateSystem: CatalogCoordinateSystem,
  cellSize: number
): PointGridContext {
  const diagnostics: CatalogDiagnostic[] = [];
  const frame = makeLocalFrame(coordinateSystem, collectAllPositions(points), {shift: 'minimum'});
  const sample = readPointSamples(points, valueProperty, frame);
  validateRequirement(
    'kriging-grid',
    sample.count > 0 && sample.count <= 2000,
    `points must contain 1 to 2,000 usable samples (received ${sample.count}); reduce or aggregate the points`
  );
  const {width, height} = getSampleExtent(sample);
  const cells = (Math.floor(width / cellSize) + 1) * (Math.floor(height / cellSize) + 1);
  validateRequirement(
    'kriging-grid',
    cells <= 40000,
    `cellSize produces ${cells} cells, exceeding the 40,000 cell limit; increase cellSize`
  );
  const key = (index: number) => {
    const feature = points.features[sample.sourceIndex[index]];
    const properties = feature.properties ?? {};
    return JSON.stringify([
      getFeatureIdentity(feature),
      Object.keys(properties)
        .sort()
        .map(name => [name, properties[name]]),
      sample.partIndex[index]
    ]);
  };
  const keys = Array.from({length: sample.count}, (_, index) => key(index));
  const order = keys
    .map((_, index) => index)
    .sort(
      (first, second) =>
        sample.x[first] - sample.x[second] ||
        sample.y[first] - sample.y[second] ||
        keys[first].localeCompare(keys[second])
    );
  const canonical = {
    ...sample,
    x: Float64Array.from(order, index => sample.x[index]),
    y: Float64Array.from(order, index => sample.y[index]),
    translationErrorX: Float64Array.from(order, index => sample.translationErrorX?.[index] ?? 0),
    translationErrorY: Float64Array.from(order, index => sample.translationErrorY?.[index] ?? 0),
    value: Float64Array.from(order, index => sample.value[index]),
    sourceIndex: Int32Array.from(order, index => sample.sourceIndex[index]),
    partIndex: Int32Array.from(order, index => sample.partIndex[index])
  };
  addDroppedDiagnostic(diagnostics, sample.droppedCount);
  addLocalFrameDiagnostic(diagnostics, frame);
  return {frame, sample: canonical, grid: makeGridSpec(width, height, cellSize), diagnostics};
}

/** Moment estimate: total sill is sample variance; nearest-neighbor semivariance estimates the nugget. */
export function resolveKrigingSettings(sample: Sample, values: Values): KrigingSettings {
  const mean = sample.value.reduce((sum, value) => sum + value, 0) / sample.count;
  const variance =
    sample.value.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
    Math.max(1, sample.count - 1);
  let nugget = values['nugget'] as number | undefined;
  validateRequirement(
    'kriging-grid',
    nugget === undefined || nugget >= 0,
    'nugget must be at least zero; omit nugget for estimation'
  );
  if (nugget === undefined) {
    let sum = 0;
    let count = 0;
    for (let first = 0; first < sample.count; first++) {
      let nearest = Infinity;
      let semivariance = 0;
      let ties = 0;
      for (let second = 0; second < sample.count; second++) {
        if (first === second) continue;
        const distance =
          (sample.x[first] - sample.x[second]) ** 2 + (sample.y[first] - sample.y[second]) ** 2;
        if (distance < nearest) {
          nearest = distance;
          semivariance = 0;
          ties = 0;
        }
        if (distance === nearest) {
          semivariance += (sample.value[first] - sample.value[second]) ** 2 / 2;
          ties++;
        }
      }
      if (ties) {
        sum += semivariance / ties;
        count++;
      }
    }
    nugget = Math.min(variance, count ? sum / count : variance);
  }
  const sill = (values['sill'] as number | undefined) ?? Math.max(0, variance - nugget);
  validateRequirement(
    'kriging-grid',
    sill >= 0 && nugget + sill > 0,
    'sill must be at least zero and sill plus nugget must exceed zero; provide a positive sill for constant samples'
  );
  const {width, height} = getSampleExtent(sample);
  const range =
    (values['range'] as number | undefined) ?? Math.max(Math.hypot(width, height) / 3, 1e-6);
  validateRequirement(
    'kriging-grid',
    range > 0,
    'range must exceed zero; omit range for estimation'
  );
  return {
    model: values['model'] as KrigingSettings['model'],
    nugget,
    sill,
    range,
    neighborCount: values['neighborCount'] as number,
    searchRadius: values['searchRadius'] as number
  };
}

export type KrigingSettings = {
  model: 'spherical' | 'exponential' | 'gaussian';
  nugget: number;
  sill: number;
  range: number;
  neighborCount: number;
  searchRadius: number;
};

function getSemivariance(settings: KrigingSettings, distance: number): number {
  if (distance <= 0) return 0;
  const ratio = distance / settings.range;
  let shape: number;
  if (settings.model === 'spherical') shape = ratio >= 1 ? 1 : 1.5 * ratio - 0.5 * ratio ** 3;
  else if (settings.model === 'exponential') shape = 1 - Math.exp(-3 * ratio);
  else shape = 1 - Math.exp(-3 * ratio * ratio);
  return settings.nugget + settings.sill * shape;
}

/**
 * GPUKriging over the same cell centers and explicit variogram as the CPU reference. The
 * contributor keeps at most 16 neighbors, rejects ill-conditioned systems as nodata (the CPU
 * reference only rejects singular ones) and solves in float32.
 */
export async function computeKrigingFieldGpu(
  context: CatalogGpuContext,
  sample: Sample,
  grid: GridSpec,
  settings: KrigingSettings
): Promise<GridField & {singularCount: number}> {
  // GPUKriging holds at most 16 neighbors per cell.
  if (settings.neighborCount > 16)
    throw new CatalogGpuUnsupportedError('GPU kriging keeps at most 16 neighbors');
  const cells = grid.columns * grid.rows;
  const raster = makeRasterMetadata(grid);
  const varianceScale = settings.nugget + settings.sill;
  const valueScale = Math.sqrt(varianceScale);
  return withCatalogGpuSession(context.device, 'catalog-kriging-grid', async session => {
    const sampleValues = new Float32Array(sample.count);
    sample.value.forEach((value, index) => (sampleValues[index] = value / valueScale));
    const values = session.output('values', 'float32', cells);
    const variances = session.output('variances', 'float32', cells);
    const uncertain = session.output('uncertain', 'uint32', 1);
    const invalid = session.output('invalid', 'uint32', 1);
    session.graph.add(
      new GPUKriging({
        id: 'catalog-kriging-grid',
        positions: session.input('positions', toFloat32Positions(sample), 'float32x2'),
        values: session.input('sample-values', sampleValues, 'float32'),
        parameters: session.input(
          'parameters',
          getGPUKrigingParameterValues(
            {
              searchRadius: settings.searchRadius > 0 ? settings.searchRadius : Infinity,
              neighborCount: settings.neighborCount,
              minimumNeighborCount: 1,
              variogram: {
                model: settings.model,
                nugget: settings.nugget / varianceScale,
                sill: settings.sill / varianceScale,
                range: settings.range
              }
            },
            raster
          ),
          'float32'
        ),
        grid: raster,
        indexGridSize: getIndexGridSize(sample.count),
        indexBounds: getGridBounds(grid),
        maximumNeighborCount: settings.neighborCount,
        output: {
          values: values.view,
          variance: variances.view,
          status: {uncertainCount: uncertain.view, invalidCount: invalid.view}
        }
      })
    );
    session.run();
    return {
      values: Float64Array.from(await session.readFloat32(values), value => value * valueScale),
      variances: Float64Array.from(
        await session.readFloat32(variances),
        value => value * varianceScale
      ),
      singularCount: await session.readScalar(uncertain)
    };
  });
}

/** Solves A x = b with the shared scaled-tolerance solver; null when singular. */
function solveLinearSystem(matrix: number[][], rhs: number[]): number[] | null {
  const solution = rhs.map(value => [value]);
  return solveDenseLinearSystem(matrix, solution) ? solution.map(([value]) => value) : null;
}

export function computeKrigingField(
  sample: Sample,
  grid: GridSpec,
  settings: KrigingSettings
): GridField & {singularCount: number} {
  const cells = grid.columns * grid.rows;
  validateRequirement(
    'kriging-grid',
    cells <= 40000 && sample.count <= 2000,
    `cell/point limits are 40,000 cells and 2,000 points (received ${cells} cells and ${sample.count} points); increase cellSize or reduce the points`
  );
  if (cells * sample.count > MAXIMUM_CPU_WORK)
    throw new KeplerAllocationError('capacity', {
      message: `kriging-grid cell/point work ${cells} * ${sample.count} exceeds the ${MAXIMUM_CPU_WORK.toLocaleString('en-US')} cell-point limit; increase cellSize or reduce the points`
    });
  const values = new Float64Array(cells).fill(Number.NaN);
  const variances = new Float64Array(cells).fill(Number.NaN);
  const counts = new Uint32Array(cells);
  let singularCount = 0;
  // Keep variogram entries commensurate with the unit constraint regardless of value units.
  const varianceScale = settings.nugget + settings.sill;
  const normalizedSettings = {
    ...settings,
    nugget: settings.nugget / varianceScale,
    sill: settings.sill / varianceScale
  };
  const radiusSquared = settings.searchRadius > 0 ? settings.searchRadius ** 2 : Infinity;
  for (let row = 0; row < grid.rows; row++)
    for (let column = 0; column < grid.columns; column++) {
      const [cx, cy] = getCellCenter(grid, column, row);
      const candidates: {distanceSquared: number; index: number}[] = [];
      for (let index = 0; index < sample.count; index++) {
        const distanceSquared = (sample.x[index] - cx) ** 2 + (sample.y[index] - cy) ** 2;
        if (distanceSquared <= radiusSquared) candidates.push({distanceSquared, index});
      }
      candidates.sort(
        (a, b) =>
          a.distanceSquared - b.distanceSquared ||
          sample.x[a.index] - sample.x[b.index] ||
          sample.y[a.index] - sample.y[b.index] ||
          a.index - b.index
      );
      const used = candidates.slice(0, settings.neighborCount);
      const cell = row * grid.columns + column;
      counts[cell] = used.length;
      if (!used.length) continue;
      const size = used.length + 1;
      const matrix: number[][] = [];
      const rhs: number[] = [];
      for (let first = 0; first < used.length; first++) {
        const line = new Array<number>(size).fill(1);
        for (let second = 0; second < used.length; second++)
          line[second] = getSemivariance(
            normalizedSettings,
            Math.hypot(
              sample.x[used[first].index] - sample.x[used[second].index],
              sample.y[used[first].index] - sample.y[used[second].index]
            )
          );
        line[used.length] = 1;
        matrix.push(line);
        rhs.push(getSemivariance(normalizedSettings, Math.sqrt(used[first].distanceSquared)));
      }
      const last = new Array<number>(size).fill(1);
      last[used.length] = 0;
      matrix.push(last);
      rhs.push(1);
      const right = rhs.slice();
      const solution = solveLinearSystem(matrix, rhs);
      if (!solution) {
        singularCount++;
        continue;
      }
      let estimate = 0;
      let variance = solution[used.length];
      for (let neighbor = 0; neighbor < used.length; neighbor++) {
        estimate += solution[neighbor] * sample.value[used[neighbor].index];
        variance += solution[neighbor] * right[neighbor];
      }
      values[cell] = estimate;
      variances[cell] = Math.max(0, variance) * varianceScale;
    }
  return {values, counts, variances, singularCount};
}

// ---------------------------------------------------------------------------------------------
// Line density and line length per polygon
// ---------------------------------------------------------------------------------------------

type FrameSegment = [number, number, number, number];

function collectLineSegments(lines: CatalogFeatureCollection, frame: LocalFrame) {
  const segments: FrameSegment[] = [];
  const owners: number[] = [];
  let droppedCount = 0;
  lines.features.forEach((feature, featureIndex) => {
    const parts = getLineParts(feature.geometry).filter(part => part.length >= 2);
    if (!parts.length) droppedCount++;
    for (const part of parts)
      for (let index = 1; index < part.length; index++) {
        const [ax, ay] = frame.forward(part[index - 1]);
        const [bx, by] = frame.forward(part[index]);
        if (ax === bx && ay === by) continue;
        segments.push([ax, ay, bx, by]);
        owners.push(featureIndex);
      }
  });
  return {segments, owners, droppedCount};
}

export function prepareLineDensity(
  lines: CatalogFeatureCollection,
  coordinateSystem: CatalogCoordinateSystem,
  cellSize: number
) {
  const diagnostics: CatalogDiagnostic[] = [];
  const frame = makeLocalFrame(coordinateSystem, collectAllPositions(lines), {shift: 'minimum'});
  const {segments, droppedCount} = collectLineSegments(lines, frame);
  addDroppedDiagnostic(
    diagnostics,
    droppedCount,
    'Non-line features or degenerate lines are skipped.'
  );
  // No usable line segments; nothing to accumulate.
  if (!segments.length) throw new Error('No usable samples');
  let width = 0;
  let height = 0;
  for (const [ax, ay, bx, by] of segments) {
    width = Math.max(width, ax, bx);
    height = Math.max(height, ay, by);
  }
  const grid = makeGridSpec(width, height, cellSize);
  return {frame, grid, segments, diagnostics};
}

/**
 * GPULineDensity over the same frame, grid and segments as the CPU reference. Every segment is
 * its own two-vertex line, so the contributor's segment walk equals the reference's exact cell
 * splitting; lengths come back as float32.
 */
export async function computeLineDensityGpu(
  context: CatalogGpuContext,
  prepared: ReturnType<typeof prepareLineDensity>,
  cellSize: number
) {
  const {frame, grid, segments, diagnostics} = prepared;
  const cells = grid.columns * grid.rows;
  const positions = new Float32Array(segments.length * 4);
  const lineOffsets = new Uint32Array(segments.length + 1);
  segments.forEach((segment, index) => {
    positions.set(segment, index * 4);
    lineOffsets[index + 1] = 2 * (index + 1);
  });
  const lengths = await withCatalogGpuSession(
    context.device,
    'catalog-line-density',
    async session => {
      const lengthsOutput = session.output('lengths', 'float32', cells);
      const overflow = session.output('overflow', 'uint32', 1);
      session.graph.add(
        new GPULineDensity({
          id: 'catalog-line-density',
          positions: session.input('positions', positions, 'float32x2'),
          lineOffsets: session.input('line-offsets', lineOffsets, 'uint32'),
          columns: grid.columns,
          rows: grid.rows,
          // Each segment crosses grid lines at most columns + rows times.
          recordCapacity: Math.max(1024, segments.length * 4),
          parameters: session.input(
            'parameters',
            getGPULineDensityParameterValues({
              minX: grid.minX,
              minY: grid.minY,
              cellWidth: cellSize,
              cellHeight: cellSize
            }),
            'float32'
          ),
          output: {lengths: lengthsOutput.view, status: {overflow: overflow.view}}
        })
      );
      session.run();
      // A dropped piece would make lengths low; refuse so the CPU reference answers instead.
      const result = await session.readFloat32(lengthsOutput);
      if ((await session.readScalar(overflow)) !== 0)
        throw new CatalogGpuUnsupportedError('Line density piece capacity overflowed');
      return result;
    }
  );
  const density = new Float64Array(cells);
  const lengthsDouble = new Float64Array(cells);
  for (let cell = 0; cell < cells; cell++) {
    lengthsDouble[cell] = lengths[cell];
    density[cell] = lengths[cell] / cellSize ** 2;
  }
  return {
    frame,
    grid,
    field: {values: density, counts: undefined} as GridField,
    lengths: lengthsDouble,
    diagnostics
  };
}

export function computeLineDensity(
  lines: CatalogFeatureCollection,
  coordinateSystem: CatalogCoordinateSystem,
  cellSize: number
) {
  const {frame, grid, segments, diagnostics} = prepareLineDensity(
    lines,
    coordinateSystem,
    cellSize
  );
  const lengths = new Float64Array(grid.columns * grid.rows);
  for (const [ax, ay, bx, by] of segments) {
    const cuts = [0, 1];
    const dx = bx - ax;
    const dy = by - ay;
    if (dx !== 0)
      for (let column = 0; column <= grid.columns; column++) {
        const t = (grid.minX + column * cellSize - ax) / dx;
        if (t > 0 && t < 1) cuts.push(t);
      }
    if (dy !== 0)
      for (let row = 0; row <= grid.rows; row++) {
        const t = (grid.minY + row * cellSize - ay) / dy;
        if (t > 0 && t < 1) cuts.push(t);
      }
    cuts.sort((a, b) => a - b);
    const total = Math.hypot(dx, dy);
    for (let index = 1; index < cuts.length; index++) {
      const middle = (cuts[index - 1] + cuts[index]) / 2;
      const cell = getBinnedCell(grid, ax + dx * middle, ay + dy * middle);
      lengths[cell] += (cuts[index] - cuts[index - 1]) * total;
    }
  }
  const density = new Float64Array(lengths.length);
  for (let cell = 0; cell < lengths.length; cell++) density[cell] = lengths[cell] / cellSize ** 2;
  return {
    frame,
    grid,
    field: {values: density, counts: undefined} as GridField,
    lengths,
    diagnostics
  };
}

function isInsideFramePolygon(
  x: number,
  y: number,
  rings: readonly (readonly [number, number][])[]
): boolean {
  const insideRing = (ring: readonly [number, number][]) => {
    let inside = false;
    for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index++) {
      const [ax, ay] = ring[previous];
      const [bx, by] = ring[index];
      if (ay > y !== by > y && x < ((bx - ax) * (y - ay)) / (by - ay) + ax) inside = !inside;
    }
    return inside;
  };
  return insideRing(rings[0]) && !rings.slice(1).some(insideRing);
}

export function getFrameRingArea(ring: readonly [number, number][]): number {
  let sum = 0;
  for (let index = 0; index < ring.length - 1; index++)
    sum += ring[index][0] * ring[index + 1][1] - ring[index + 1][0] * ring[index][1];
  return sum / 2;
}

export function readFramePolygons(
  polygons: CatalogFeatureCollection,
  frame: LocalFrame
): {rings: [number, number][][][]; area: number; featureIndex: number}[] {
  const result: {rings: [number, number][][][]; area: number; featureIndex: number}[] = [];
  polygons.features.forEach((feature, featureIndex) => {
    const parts = getPolygonParts(feature.geometry).map(polygon =>
      polygon.map(ring => ring.map(position => frame.forward(position)))
    );
    let area = 0;
    for (const polygon of parts) {
      area += Math.abs(getFrameRingArea(polygon[0]));
      for (const hole of polygon.slice(1)) area -= Math.abs(getFrameRingArea(hole));
    }
    result.push({rings: parts as [number, number][][][], area, featureIndex});
  });
  return result;
}

export function isInsideFramePolygonParts(
  x: number,
  y: number,
  parts: readonly (readonly [number, number][])[][]
): boolean {
  return parts.some(rings => rings.length && isInsideFramePolygon(x, y, rings));
}

export function computeLineLengthPerPolygon(
  polygons: CatalogFeatureCollection,
  lines: CatalogFeatureCollection,
  coordinateSystem: CatalogCoordinateSystem
) {
  const diagnostics: CatalogDiagnostic[] = [];
  const positions: CatalogPosition[] = [
    ...collectAllPositions(polygons),
    ...collectAllPositions(lines)
  ];
  const frame = makeLocalFrame(coordinateSystem, positions, {shift: 'minimum'});
  const regions = readFramePolygons(polygons, frame);
  const {segments, owners, droppedCount} = collectLineSegments(lines, frame);
  addDroppedDiagnostic(
    diagnostics,
    droppedCount,
    'Non-line features or degenerate lines are skipped.'
  );
  let edgeCount = 0;
  for (const region of regions)
    for (const rings of region.rings) for (const ring of rings) edgeCount += ring.length;
  // Segment-times-edge work exceeds the CPU reference capacity.
  if (segments.length * edgeCount > MAXIMUM_CPU_WORK)
    throw new KeplerAllocationError('capacity', {message: 'Work capacity exceeded'});
  const lengths = new Float64Array(regions.length);
  const lineCounts = new Uint32Array(regions.length);
  regions.forEach((region, regionIndex) => {
    const touched = new Set<number>();
    segments.forEach(([ax, ay, bx, by], segmentIndex) => {
      const dx = bx - ax;
      const dy = by - ay;
      const cuts = [0, 1];
      for (const rings of region.rings)
        for (const ring of rings)
          for (let index = 0; index < ring.length - 1; index++) {
            const [cx, cy] = ring[index];
            const ex = ring[index + 1][0] - cx;
            const ey = ring[index + 1][1] - cy;
            const denominator = dx * ey - dy * ex;
            if (denominator === 0) continue;
            const t = ((cx - ax) * ey - (cy - ay) * ex) / denominator;
            const u = ((cx - ax) * dy - (cy - ay) * dx) / denominator;
            if (t > 0 && t < 1 && u >= 0 && u <= 1) cuts.push(t);
          }
      cuts.sort((a, b) => a - b);
      const total = Math.hypot(dx, dy);
      for (let index = 1; index < cuts.length; index++) {
        const middle = (cuts[index - 1] + cuts[index]) / 2;
        if (!isInsideFramePolygonParts(ax + dx * middle, ay + dy * middle, region.rings)) continue;
        lengths[regionIndex] += (cuts[index] - cuts[index - 1]) * total;
        touched.add(owners[segmentIndex]);
      }
    });
    lineCounts[regionIndex] = touched.size;
  });
  return {regions, lengths, lineCounts, diagnostics};
}

export function addGridDiagnostics(
  diagnostics: CatalogDiagnostic[],
  nodataCount: number,
  message = 'Cells without enough contributing samples are nodata and are not published.'
): void {
  if (nodataCount) addCatalogDiagnostic(diagnostics, 'nodata-cells', 'info', message, nodataCount);
}

export function makeGridRunOutput(
  table: ReturnType<typeof makeGridTable>,
  diagnostics: CatalogDiagnostic[],
  grid: GridSpec,
  sampleCount: number,
  dispatchedContributors?: readonly string[]
): CatalogRunOutput {
  return {
    tables: [table],
    diagnostics,
    summary: {
      columns: grid.columns,
      rows: grid.rows,
      cellSize: grid.cellSize,
      sampleCount,
      outputCellCount: table.rowCount
    },
    ...(dispatchedContributors ? {dispatchedContributors} : {})
  };
}
