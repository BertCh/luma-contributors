// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRasterOperationArray} from '../raster-grid/raster-operation-band';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {validateOperationGraphNames} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH} from '../gpu-raster-fractional-coverage';
import {
  calibrateRasterOperationBand,
  readRasterOperationBand,
  snapshotRasterOperationBand,
  validateRasterOperationBand,
  type CPURasterOperationBand
} from '../raster-grid/raster-operation-band';
import type {RasterFractionalZonalStatisticsOperationProps} from './gpu-raster-fractional-zonal-statistics-operation';
import {validateRasterCellSizeMode} from '../../analysis-core/contracts/raster-grid';
import {validateRasterGridAxisAligned} from '../raster-band-utils';
import {
  getRasterRowAreaScale,
  isRasterEffectiveAxisAligned
} from '../raster-grid/raster-cell-area-scale';

/** Packed inputs of the fractional aggregate. Coverage status is borrowed, never changed. */
export type CPURasterFractionalZonalStatisticsInput = {
  input: CPURasterOperationBand;
  parameters?: Float32Array;
  cellAreas?: Float32Array;
  coverage: {
    zoneIds: Uint32Array;
    cellIds: Uint32Array;
    fractions: Float32Array;
    zoneValidity: Uint32Array;
    count: Uint32Array;
    requiredCount: Uint32Array;
    overflow: Uint32Array;
    candidateOverflow?: Uint32Array;
  };
};

export type CPURasterFractionalZonalStatisticsOptions = Pick<
  RasterFractionalZonalStatisticsOperationProps,
  'grid' | 'zoneCount' | 'quantity' | 'cellSizeMode'
>;
export type CPURasterFractionalZonalStatisticsResult = {
  totals: Float32Array;
  means: Float32Array;
  coveredArea: Float32Array;
  validArea: Float32Array;
  validity: Uint32Array;
};

/**
 * Aggregates all zones in one tuple pass, preserving the contributor's count clamp, ignored
 * out-of-range IDs and completeness rule (matching in-capacity counts, overflow and optional
 * candidateOverflow zero, zoneValidity nonzero). Fractions are used as supplied,
 * including negative and nonfinite fractions; they are not silently sanitized. Coverage cellIds
 * and fractions may have surplus rows beyond zoneIds.length, zoneValidity beyond zoneCount,
 * and scalar coverage statuses beyond their first row; these tails are ignored as on the GPU.
 *
 * What differs from the GPU contributor:
 * - Band calibration, sums and division use float64, with one rounding at each final output;
 *   compare ordinary finite outputs with relative tolerance 3e-6. Float32 intermediate overflow
 *   and underflow can therefore produce different values and validity near representability limits.
 * - A calibrated observation beyond float32 range remains usable when finite in float64.
 * - Subnormal inputs and intermediate quantities are retained; GPU implementations may flush them.
 * - Operation outputs require at least one zone, as required by GPUContributorOperation.
 *
 * @throws Error naming an argument when its array lengths or options are inconsistent.
 */
export function computeRasterFractionalZonalStatisticsOnCPU(
  input: CPURasterFractionalZonalStatisticsInput,
  options: CPURasterFractionalZonalStatisticsOptions
): CPURasterFractionalZonalStatisticsResult {
  const id = 'computeRasterFractionalZonalStatisticsOnCPU';
  for (const name of [
    'zoneIds',
    'cellIds',
    'zoneValidity',
    'count',
    'requiredCount',
    'overflow'
  ] as const) {
    validateRasterOperationArray(id, `coverage.${name}`, input.coverage[name], 'Uint32Array');
  }
  if (input.coverage.candidateOverflow !== undefined) {
    validateRasterOperationArray(
      id,
      'coverage.candidateOverflow',
      input.coverage.candidateOverflow,
      'Uint32Array'
    );
  }
  validateRasterOperationArray(id, 'coverage.fractions', input.coverage.fractions, 'Float32Array');
  if (input.parameters !== undefined)
    validateRasterOperationArray(id, 'parameters', input.parameters, 'Float32Array');
  if (input.cellAreas !== undefined)
    validateRasterOperationArray(id, 'cellAreas', input.cellAreas, 'Float32Array');
  validateRasterOperationBand(id, input.input, options.grid);
  validateStatisticsOptions(id, options, input.cellAreas !== undefined);
  if (input.cellAreas && input.cellAreas.length !== options.grid.width * options.grid.height)
    throw new Error(`${id} cellAreas must contain grid.width * grid.height rows`);
  const {coverage, parameters} = input;
  const effectiveAligned = isRasterEffectiveAxisAligned(
    options.grid,
    options.cellSizeMode ?? 'uniform',
    parameters
  );
  if (coverage.candidateOverflow && coverage.candidateOverflow.length < 1)
    throw new Error(`${id} coverage.candidateOverflow must have at least one row`);
  if (
    coverage.cellIds.length < coverage.zoneIds.length ||
    coverage.fractions.length < coverage.zoneIds.length
  ) {
    throw new Error(
      `${id} coverage.cellIds and coverage.fractions must have at least coverage.zoneIds length rows`
    );
  }
  if (coverage.zoneValidity.length < options.zoneCount)
    throw new Error(`${id} coverage.zoneValidity must have at least zoneCount rows`);
  for (const name of ['count', 'requiredCount', 'overflow'] as const) {
    if (coverage[name].length < 1)
      throw new Error(`${id} coverage.${name} must have at least one row`);
  }
  if (parameters && parameters.length < GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH)
    throw new Error(`${id} parameters must contain the coverage georeference layout`);
  const values = calibrateRasterOperationBand(
    input.input,
    options.grid.width * options.grid.height
  );
  const [columnX, rowX, , columnY, rowY] = options.grid.affine;
  // getGPURasterFractionalCoverageParameterValues: [originX, originY, a, b, d, e, determinant, padding].
  // Word 7 is ignored legacy padding. Version 2 appends [2, absolute Y offset] in words 8/9;
  // the same ten-word packet supports local clipping and absolute-latitude ground areas.
  const [, , , , , , determinant] = parameters ?? [];
  const cellArea = Math.abs(parameters ? determinant : columnX * rowY - rowX * columnY);
  const weighted = new Float64Array(options.zoneCount);
  const covered = new Float64Array(options.zoneCount);
  const observed = new Float64Array(options.zoneCount);
  const observedFractions = new Float64Array(options.zoneCount);
  const totals = new Float64Array(options.zoneCount);
  for (let tuple = 0; tuple < Math.min(coverage.count[0], coverage.zoneIds.length); tuple++) {
    const zone = coverage.zoneIds[tuple];
    const cell = coverage.cellIds[tuple];
    if (zone >= options.zoneCount || cell >= values.length) continue;
    const fraction = coverage.fractions[tuple];
    const area =
      input.cellAreas?.[cell] ??
      cellArea *
        getRasterRowAreaScale(
          options.grid,
          options.cellSizeMode ?? 'uniform',
          Math.floor(cell / options.grid.width),
          parameters
        );
    if (input.cellAreas && !(Number.isFinite(area) && area > 0)) {
      covered[zone] = NaN;
      observed[zone] = NaN;
      continue;
    }
    covered[zone] += fraction * area;
    if (Number.isFinite(values[cell])) {
      const value = values[cell];
      weighted[zone] += value * fraction;
      totals[zone] += value * fraction * (options.quantity === 'extensive' ? 1 : area);
      observed[zone] += fraction * area;
      observedFractions[zone] += fraction;
    }
  }
  const result: CPURasterFractionalZonalStatisticsResult = {
    totals: new Float32Array(options.zoneCount).fill(NaN),
    means: new Float32Array(options.zoneCount).fill(NaN),
    coveredArea: new Float32Array(options.zoneCount).fill(NaN),
    validArea: new Float32Array(options.zoneCount).fill(NaN),
    validity: new Uint32Array(options.zoneCount)
  };
  for (let zone = 0; zone < options.zoneCount; zone++) {
    const complete =
      coverage.overflow[0] === 0 &&
      (coverage.candidateOverflow?.[0] ?? 0) === 0 &&
      coverage.count[0] <= coverage.zoneIds.length &&
      coverage.requiredCount[0] === coverage.count[0] &&
      coverage.zoneValidity[zone] !== 0 &&
      effectiveAligned;
    const area = observed[zone];
    const mean = observedFractions[zone] > 0 ? weighted[zone] / observedFractions[zone] : NaN;
    if (
      !Number.isFinite(Math.fround(covered[zone])) ||
      !Number.isFinite(Math.fround(area)) ||
      (area > 0 &&
        (!Number.isFinite(Math.fround(totals[zone])) || !Number.isFinite(Math.fround(mean))))
    )
      continue;
    if (complete) {
      result.coveredArea[zone] = covered[zone];
      result.validArea[zone] = area;
      if (area > 0) {
        result.validity[zone] = 1;
        result.totals[zone] = totals[zone];
        result.means[zone] = mean;
      }
    }
  }
  return result;
}

function validateStatisticsOptions(
  id: string,
  options: CPURasterFractionalZonalStatisticsOptions,
  hasCellAreas: boolean
): void {
  validateRasterCellSizeMode(id, options.cellSizeMode);
  validateRasterGridAxisAligned(id, options.grid, options.cellSizeMode);
  validateRequirement(
    id,
    !hasCellAreas || (options.cellSizeMode ?? 'uniform') === 'uniform',
    'cellSizeMode must be uniform when cellAreas is given (cellAreas already declares the areas)'
  );
  validateCapacity(id, 'zoneCount', options.zoneCount);
  validateRequirement(
    id,
    ['extensive', 'density', 'intensive'].includes(options.quantity),
    'quantity must be extensive, density or intensive'
  );
}

/** CPU aggregate with the same packed graph values as the GPU operation. */
export class CPURasterFractionalZonalStatisticsOperation extends GPUCPUOperation {
  constructor(props: RasterFractionalZonalStatisticsOperationProps) {
    props = snapshotConstructionFields(props, [
      'id',
      'grid',
      'input',
      'zoneCount',
      'coverage',
      'quantity',
      'parameters',
      'cellSizeMode',
      'cellAreas',
      'output'
    ]);
    const output = Object.freeze({
      totals: props.output.totals,
      means: props.output.means,
      coveredArea: props.output.coveredArea,
      validArea: props.output.validArea,
      validity: props.output.validity
    });
    const coverage = snapshotConstructionFields(props.coverage, [
      'zoneIds',
      'cellIds',
      'fractions',
      'zoneValidity',
      'status'
    ]);
    props = Object.freeze({
      ...props,
      input: snapshotRasterOperationBand(props.input),
      coverage: Object.freeze({
        ...coverage,
        status: snapshotConstructionFields(coverage.status, [
          'count',
          'requiredCount',
          'overflow',
          'candidateOverflow'
        ])
      }),
      output
    });
    const id = props.id ?? 'fractional-zonal-statistics';
    validateRasterOperationBand(id, props.input, props.grid);
    validateStatisticsOptions(id, props, props.cellAreas !== undefined);
    const inputs = {
      values: props.input.storage.values,
      ...(props.input.validity !== undefined ? {inputValidity: props.input.validity} : {}),
      zoneIds: props.coverage.zoneIds,
      cellIds: props.coverage.cellIds,
      fractions: props.coverage.fractions,
      zoneValidity: props.coverage.zoneValidity,
      count: props.coverage.status.count,
      requiredCount: props.coverage.status.requiredCount,
      overflow: props.coverage.status.overflow,
      ...(props.coverage.status.candidateOverflow !== undefined
        ? {candidateOverflow: props.coverage.status.candidateOverflow}
        : {}),
      ...(props.parameters !== undefined ? {parameters: props.parameters} : {}),
      ...(props.cellAreas !== undefined ? {cellAreas: props.cellAreas} : {})
    };
    validateOperationGraphNames(id, inputs, props.output);
    super({
      id,
      inputs,
      outputs: Object.fromEntries(
        Object.entries(props.output).map(([port, value]) => [
          value,
          gpuGraph.vector(port === 'validity' ? 'uint32' : 'float32', {length: props.zoneCount})
        ])
      ),
      execute({inputs: rows}) {
        const result = computeRasterFractionalZonalStatisticsOnCPU(
          {
            input: readRasterOperationBand(
              id,
              {...rows, validity: rows['inputValidity']!},
              props.input
            ),
            parameters:
              props.parameters !== undefined
                ? getCPUOperationFloat32Rows(id, rows, 'parameters')
                : undefined,
            cellAreas:
              props.cellAreas !== undefined
                ? getCPUOperationFloat32Rows(id, rows, 'cellAreas')
                : undefined,
            coverage: {
              zoneIds: getCPUOperationUint32Rows(id, rows, 'zoneIds'),
              cellIds: getCPUOperationUint32Rows(id, rows, 'cellIds'),
              fractions: getCPUOperationFloat32Rows(id, rows, 'fractions'),
              zoneValidity: getCPUOperationUint32Rows(id, rows, 'zoneValidity'),
              count: getCPUOperationUint32Rows(id, rows, 'count'),
              requiredCount: getCPUOperationUint32Rows(id, rows, 'requiredCount'),
              overflow: getCPUOperationUint32Rows(id, rows, 'overflow'),
              candidateOverflow:
                props.coverage.status.candidateOverflow !== undefined
                  ? getCPUOperationUint32Rows(id, rows, 'candidateOverflow')
                  : undefined
            }
          },
          props
        );
        return {
          [props.output.totals]: result.totals,
          [props.output.means]: result.means,
          [props.output.coveredArea]: result.coveredArea,
          [props.output.validArea]: result.validArea,
          [props.output.validity]: result.validity
        };
      }
    });
  }
}
