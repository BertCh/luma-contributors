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
import {GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH} from '../gpu-raster-fractional-coverage';
import {
  calibrateRasterOperationBand,
  readRasterOperationBand,
  validateRasterOperationBand,
  type CPURasterOperationBand
} from '../raster-grid/raster-operation-band';
import type {RasterFractionalZonalStatisticsOperationProps} from './gpu-raster-fractional-zonal-statistics-operation';

/** Packed inputs of the fractional aggregate. Coverage status is borrowed, never changed. */
export type CPURasterFractionalZonalStatisticsInput = {
  input: CPURasterOperationBand;
  parameters?: Float32Array;
  coverage: {
    zoneIds: Uint32Array;
    cellIds: Uint32Array;
    fractions: Float32Array;
    zoneValidity: Uint32Array;
    count: Uint32Array;
    requiredCount: Uint32Array;
    overflow: Uint32Array;
  };
};

export type CPURasterFractionalZonalStatisticsOptions = Pick<
  RasterFractionalZonalStatisticsOperationProps,
  'grid' | 'zoneCount' | 'quantity'
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
 * out-of-range IDs and completeness rule (overflow == 0 and zoneValidity != 0). requiredCount
 * is a graph input but does not affect this contributor's result. Fractions are used as supplied,
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
  validateRasterOperationBand(id, input.input, options.grid);
  validateStatisticsOptions(id, options);
  const {coverage, parameters} = input;
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
  const [, , , , , , determinant] = parameters ?? [];
  const cellArea = Math.abs(parameters ? determinant : columnX * rowY - rowX * columnY);
  const weighted = new Float64Array(options.zoneCount);
  const covered = new Float64Array(options.zoneCount);
  const observed = new Float64Array(options.zoneCount);
  const totals = new Float64Array(options.zoneCount);
  for (let tuple = 0; tuple < Math.min(coverage.count[0], coverage.zoneIds.length); tuple++) {
    const zone = coverage.zoneIds[tuple];
    const cell = coverage.cellIds[tuple];
    if (zone >= options.zoneCount || cell >= values.length) continue;
    const fraction = coverage.fractions[tuple];
    covered[zone] += fraction;
    if (Number.isFinite(values[cell])) {
      const value = values[cell];
      weighted[zone] += value * fraction;
      totals[zone] += value * fraction * (options.quantity === 'extensive' ? 1 : cellArea);
      observed[zone] += fraction;
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
    const complete = coverage.overflow[0] === 0 && coverage.zoneValidity[zone] !== 0;
    const area = observed[zone] * cellArea;
    if (complete) {
      result.coveredArea[zone] = covered[zone] * cellArea;
      result.validArea[zone] = area;
      if (area > 0) {
        result.validity[zone] = 1;
        result.totals[zone] = totals[zone];
        result.means[zone] =
          observed[zone] > 0 ? weighted[zone] / Math.max(observed[zone], 1e-30) : NaN;
      }
    }
  }
  return result;
}

function validateStatisticsOptions(
  id: string,
  options: CPURasterFractionalZonalStatisticsOptions
): void {
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
    const id = props.id ?? 'fractional-zonal-statistics';
    validateRasterOperationBand(id, props.input, props.grid);
    validateStatisticsOptions(id, props);
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
      ...(props.parameters !== undefined ? {parameters: props.parameters} : {})
    };
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
            coverage: {
              zoneIds: getCPUOperationUint32Rows(id, rows, 'zoneIds'),
              cellIds: getCPUOperationUint32Rows(id, rows, 'cellIds'),
              fractions: getCPUOperationFloat32Rows(id, rows, 'fractions'),
              zoneValidity: getCPUOperationUint32Rows(id, rows, 'zoneValidity'),
              count: getCPUOperationUint32Rows(id, rows, 'count'),
              requiredCount: getCPUOperationUint32Rows(id, rows, 'requiredCount'),
              overflow: getCPUOperationUint32Rows(id, rows, 'overflow')
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
