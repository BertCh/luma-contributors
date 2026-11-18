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
import {GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH} from '../gpu-raster-fractional-coverage';
import {GPU_RASTER_FRACTIONAL_ZONAL_DISTRIBUTION_STATUS as STATUS} from '../gpu-raster-fractional-zonal-distribution';
import type {RasterFractionalZonalDistributionOperationProps} from './gpu-raster-fractional-zonal-distribution-operation';
import {validateRasterGridAxisAligned} from '../raster-band-utils';
import {
  getRasterRowAreaScale,
  isRasterEffectiveAxisAligned
} from '../raster-grid/raster-cell-area-scale';

export type CPURasterFractionalZonalDistributionOptions = Pick<
  RasterFractionalZonalDistributionOperationProps,
  'grid' | 'zoneCount' | 'quantileMethod' | 'cellSizeMode'
>;
export type CPURasterFractionalZonalDistributionInput = {
  input: CPURasterOperationBand;
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
  probabilities: Float32Array;
  observationWeights?: Float32Array;
  cellAreas?: Float32Array;
  parameters?: Float32Array;
};
export type CPURasterFractionalZonalDistributionResult = {
  means: Float32Array;
  populationVariances: Float32Array;
  sampleVariances: Float32Array;
  effectiveSampleSizes: Float32Array;
  weightSums: Float32Array;
  squaredWeightSums: Float32Array;
  coveredArea: Float32Array;
  validArea: Float32Array;
  quantiles: Float32Array;
  quantileValidity: Uint32Array;
  validity: Uint32Array;
  status: Uint32Array;
};

type CoverageRecord = {
  cell: number;
  value: number;
  fraction: number;
  weight: number;
  area: number;
  flags: number;
};

/**
 * Computes coverage/reliability weighted moments and exactextract distinct-value interpolated
 * quantiles (or inverse weighted CDF) in the contributor's zone-major output layout. Non-uniform
 * cellSizeMode uses getRasterRowAreaScale, with the row edges and the area determinant from
 * parameters when supplied and from grid.affine otherwise, just as the contributor does.
 *
 * What differs from the GPU contributor:
 * - Calibration, weighted moments and quantile interpolation use float64; final arrays round once.
 *   NON_FINITE_RESULT checks final float32 representability rather than float32 intermediates.
 *   Calibrated observations beyond float32 range can contribute instead of becoming GPU nodata.
 * - Ground-area row scales use the existing float64 helper before final area output rounding.
 * - GPU tuple, quantile and zone caps are dropped; no tuple is approximated or discarded.
 * - The twin requires only a finite nonzero grid.affine determinant. The contributor also
 *   refuses a determinant that rounds to zero in float32 or whose absolute value times the
 *   coverage capacity is not finite in float32; the twin accepts both and reports any area that
 *   overflows float32 as NON_FINITE_RESULT.
 *
 * @throws If any input array has an inconsistent length, or grid/options are invalid.
 */
export function computeRasterFractionalZonalDistributionOnCPU(
  input: CPURasterFractionalZonalDistributionInput,
  options: CPURasterFractionalZonalDistributionOptions
): CPURasterFractionalZonalDistributionResult {
  const id = 'computeRasterFractionalZonalDistributionOnCPU';
  validateRasterOperationBand(id, input.input, options.grid);
  validateDistributionOptions(id, options, input.cellAreas !== undefined);
  const {coverage, probabilities} = input;
  const capacity = coverage.zoneIds.length;
  validateCapacity(id, 'coverage.zoneIds length', capacity);
  validateCapacity(id, 'probabilities length', probabilities.length);
  if (coverage.cellIds.length !== capacity || coverage.fractions.length !== capacity)
    throw new Error(
      `${id} coverage.cellIds and coverage.fractions must match coverage.zoneIds length`
    );
  if (coverage.zoneValidity.length !== options.zoneCount)
    throw new Error(`${id} coverage.zoneValidity must have zoneCount rows`);
  for (const name of ['count', 'requiredCount', 'overflow'] as const)
    if (coverage[name].length < 1)
      throw new Error(`${id} coverage.${name} must have at least one row`);
  if (coverage.candidateOverflow && coverage.candidateOverflow.length < 1)
    throw new Error(`${id} coverage.candidateOverflow must have at least one row`);
  const cellCount = options.grid.width * options.grid.height;
  for (const name of ['observationWeights', 'cellAreas'] as const)
    if (input[name] && input[name].length !== cellCount)
      throw new Error(`${id} ${name} must have one row per cell`);
  // The packer's final word is padding; like the contributor only the preceding words are needed.
  if (
    input.parameters &&
    input.parameters.length < GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH - 1
  )
    throw new Error(`${id} parameters must contain the coverage georeference layout`);
  const elevations = calibrateRasterOperationBand(input.input, cellCount);
  const [columnX, rowX, , columnY, rowY] = options.grid.affine;
  // Layout of getGPURasterFractionalCoverageParameterValues: originX, originY, a, b, d, e, determinant, padding.
  const [, , , , , , parameterDeterminant] = input.parameters ?? [];
  const defaultArea = Math.abs(
    input.parameters ? parameterDeterminant : columnX * rowY - rowX * columnY
  );
  const zones: CoverageRecord[][] = Array.from({length: options.zoneCount}, () => []);
  let invalidZone = false;
  const completeCoverage =
    isRasterEffectiveAxisAligned(
      options.grid,
      options.cellSizeMode ?? 'uniform',
      input.parameters
    ) &&
    (coverage.candidateOverflow?.[0] ?? 0) === 0 &&
    coverage.count[0] <= capacity &&
    coverage.requiredCount[0] === coverage.count[0] &&
    coverage.overflow[0] === 0;
  for (let tuple = 0; tuple < Math.min(capacity, coverage.count[0]); tuple++) {
    const zone = coverage.zoneIds[tuple];
    if (zone >= options.zoneCount) {
      invalidZone = true;
      continue;
    }
    const cell = coverage.cellIds[tuple];
    const fraction = coverage.fractions[tuple];
    const value = cell < cellCount ? elevations[cell] : NaN;
    let flags =
      cell >= cellCount || !Number.isFinite(fraction) || fraction < 0 || fraction > 1
        ? 2
        : Number(Number.isFinite(value) && fraction > 0);
    const observationWeight = cell < cellCount ? (input.observationWeights?.[cell] ?? 1) : 1;
    const area =
      cell < cellCount
        ? (input.cellAreas?.[cell] ??
          defaultArea *
            getRasterRowAreaScale(
              options.grid,
              options.cellSizeMode ?? 'uniform',
              Math.floor(cell / options.grid.width),
              input.parameters
            ))
        : defaultArea;
    const weight = fraction * observationWeight;
    // The contributor validates dynamic support only when either optional support column exists.
    if (
      cell < cellCount &&
      (input.observationWeights || input.cellAreas) &&
      (!Number.isFinite(observationWeight) ||
        observationWeight < 0 ||
        !Number.isFinite(weight) ||
        !Number.isFinite(area) ||
        area <= 0)
    )
      flags = 2;
    zones[zone].push({cell, value: value === 0 ? 0 : value, fraction, weight, area, flags});
  }
  const result: CPURasterFractionalZonalDistributionResult = {
    means: new Float32Array(options.zoneCount).fill(NaN),
    populationVariances: new Float32Array(options.zoneCount).fill(NaN),
    sampleVariances: new Float32Array(options.zoneCount).fill(NaN),
    effectiveSampleSizes: new Float32Array(options.zoneCount).fill(NaN),
    weightSums: new Float32Array(options.zoneCount).fill(NaN),
    squaredWeightSums: new Float32Array(options.zoneCount).fill(NaN),
    coveredArea: new Float32Array(options.zoneCount).fill(NaN),
    validArea: new Float32Array(options.zoneCount).fill(NaN),
    quantiles: new Float32Array(options.zoneCount * probabilities.length).fill(NaN),
    quantileValidity: new Uint32Array(options.zoneCount * probabilities.length),
    validity: new Uint32Array(options.zoneCount),
    status: new Uint32Array(options.zoneCount)
  };
  zones.forEach((records, zone) => {
    records.sort(
      (first, second) =>
        (Number.isFinite(first.value) ? first.value : Infinity) -
          (Number.isFinite(second.value) ? second.value : Infinity) || first.cell - second.cell
    );
    let status =
      completeCoverage && coverage.zoneValidity[zone] !== 0 ? 0 : STATUS.INCOMPLETE_COVERAGE;
    if (invalidZone) status |= STATUS.INVALID_TUPLE;
    let covered = 0;
    let validArea = 0;
    let weight = 0;
    let squaredWeight = 0;
    const observations: CoverageRecord[] = [];
    for (let cursor = 0; cursor < records.length; cursor++) {
      const record = records[cursor];
      if (record.flags === 2 || (cursor > 0 && record.cell === records[cursor - 1].cell))
        status |= STATUS.INVALID_TUPLE;
      if (record.flags === 2) continue;
      covered += record.fraction * record.area;
      if (record.flags !== 1) continue;
      validArea += record.fraction * record.area;
      if (record.weight <= 0) continue;
      observations.push(record);
      weight += record.weight;
      squaredWeight += record.weight ** 2;
    }
    let mean = NaN;
    let populationVariance = NaN;
    let sampleVariance = NaN;
    let effectiveSupport = 0;
    let denominator = 0;
    if (weight > 0) {
      const anchor = observations[0].value;
      const centeredMean =
        observations.reduce((sum, record) => sum + record.weight * (record.value - anchor), 0) /
        weight;
      mean = anchor + centeredMean;
      const secondMoment = observations.reduce(
        (sum, record) => sum + record.weight * (record.value - anchor - centeredMean) ** 2,
        0
      );
      populationVariance = secondMoment / weight;
      effectiveSupport = weight ** 2 / squaredWeight;
      denominator = weight - squaredWeight / weight;
      if (observations.length >= 2 && denominator > 0) sampleVariance = secondMoment / denominator;
      if (
        ![
          mean,
          populationVariance,
          effectiveSupport,
          weight,
          squaredWeight,
          ...(observations.length >= 2 && denominator > 0 ? [sampleVariance] : [])
        ].every(value => Number.isFinite(Math.fround(value)))
      )
        status |= STATUS.NON_FINITE_RESULT;
    } else status |= STATUS.NO_VALID_SUPPORT;
    if (observations.length < 2 || denominator <= 0) status |= STATUS.INSUFFICIENT_SAMPLE_SUPPORT;
    if (![covered, validArea].every(value => Number.isFinite(Math.fround(value))))
      status |= STATUS.NON_FINITE_RESULT;
    const complete = (status & (STATUS.INCOMPLETE_COVERAGE | STATUS.INVALID_TUPLE)) === 0;
    const valid =
      (status &
        (STATUS.INCOMPLETE_COVERAGE |
          STATUS.INVALID_TUPLE |
          STATUS.NO_VALID_SUPPORT |
          STATUS.NON_FINITE_RESULT)) ===
      0;
    if (valid) {
      result.means[zone] = mean;
      result.populationVariances[zone] = populationVariance;
      result.sampleVariances[zone] = sampleVariance;
    }
    if (complete && (status & STATUS.NON_FINITE_RESULT) === 0)
      result.effectiveSampleSizes[zone] = effectiveSupport;
    if (complete) {
      result.weightSums[zone] = weight;
      result.squaredWeightSums[zone] = squaredWeight;
      result.coveredArea[zone] = covered;
      result.validArea[zone] = validArea;
    }
    result.validity[zone] = Number(valid);
    result.status[zone] = status;
    const groups: {value: number; weight: number}[] = [];
    for (const observation of observations) {
      const previous = groups[groups.length - 1];
      if (previous && previous.value === observation.value) previous.weight += observation.weight;
      else groups.push({value: observation.value, weight: observation.weight});
    }
    probabilities.forEach((probability, probabilityIndex) => {
      if (!valid || !Number.isFinite(probability) || probability < 0 || probability > 1) return;
      let cumulativeWeight = 0;
      let previousPosition = 0;
      let previousValue = 0;
      let quantile = NaN;
      const quantilePosition = probability * (groups.length - 1) * weight;
      for (let groupIndex = 0; groupIndex < groups.length; groupIndex++) {
        const group = groups[groupIndex];
        if (
          options.quantileMethod === 'inverse-cdf' &&
          cumulativeWeight + group.weight >= probability * weight
        ) {
          quantile = group.value;
          break;
        }
        const position = groupIndex * group.weight + (groups.length - 1) * cumulativeWeight;
        if (
          options.quantileMethod !== 'inverse-cdf' &&
          groupIndex > 0 &&
          quantilePosition <= position
        ) {
          quantile =
            previousValue +
            ((group.value - previousValue) * (quantilePosition - previousPosition)) /
              (position - previousPosition);
          break;
        }
        quantile = group.value;
        previousValue = group.value;
        previousPosition = position;
        cumulativeWeight += group.weight;
      }
      const outputIndex = zone * probabilities.length + probabilityIndex;
      result.quantiles[outputIndex] = quantile;
      result.quantileValidity[outputIndex] = Number(Number.isFinite(Math.fround(quantile)));
    });
  });
  return result;
}

function validateDistributionOptions(
  id: string,
  options: CPURasterFractionalZonalDistributionOptions,
  hasCellAreas: boolean
): void {
  validateCapacity(id, 'zoneCount', options.zoneCount, 0);
  validateRequirement(
    id,
    ['uniform', 'geographic', 'web-mercator'].includes(options.cellSizeMode ?? 'uniform'),
    'cellSizeMode must be uniform, geographic or web-mercator'
  );
  validateRasterGridAxisAligned(id, options.grid, options.cellSizeMode);
  validateRequirement(
    id,
    !hasCellAreas || (options.cellSizeMode ?? 'uniform') === 'uniform',
    'cellSizeMode must be uniform when cellAreas is given (cellAreas already declares the areas)'
  );
  validateRequirement(
    id,
    ['exactextract', 'inverse-cdf'].includes(options.quantileMethod ?? 'exactextract'),
    "quantileMethod must be 'exactextract' or 'inverse-cdf'"
  );
  const [columnX, rowX, , columnY, rowY] = options.grid.affine;
  validateRequirement(
    id,
    Number.isFinite(columnX * rowY - rowX * columnY) && columnX * rowY - rowX * columnY !== 0,
    'grid.affine must have a finite nonzero determinant'
  );
}

/** CPU twin without GPU tuple, probability or zone caps. */
export class CPURasterFractionalZonalDistributionOperation extends GPUCPUOperation {
  constructor(props: RasterFractionalZonalDistributionOperationProps) {
    const id = props.id ?? 'raster-fractional-zonal-distribution';
    validateRasterOperationBand(id, props.input, props.grid);
    validateDistributionOptions(id, props, props.cellAreas !== undefined);
    // The operation base requires nonempty graph outputs, even though the pure function accepts zero zones.
    validateCapacity(id, 'zoneCount', props.zoneCount);
    validateCapacity(id, 'probabilityCount', props.probabilityCount);
    validateCapacity(
      id,
      'zoneCount times probabilityCount',
      props.zoneCount * props.probabilityCount,
      0
    );
    super({
      id,
      inputs: {
        values: props.input.storage.values,
        probabilities: props.probabilities,
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
        ...(props.input.validity !== undefined ? {inputValidity: props.input.validity} : {}),
        ...(props.observationWeights !== undefined
          ? {observationWeights: props.observationWeights}
          : {}),
        ...(props.cellAreas !== undefined ? {cellAreas: props.cellAreas} : {}),
        ...(props.parameters !== undefined ? {parameters: props.parameters} : {})
      },
      outputs: Object.fromEntries(
        Object.entries(props.output).map(([port, value]) => [
          value,
          gpuGraph.vector(
            port === 'validity' || port === 'status' || port === 'quantileValidity'
              ? 'uint32'
              : 'float32',
            {
              length:
                props.zoneCount *
                (port === 'quantiles' || port === 'quantileValidity' ? props.probabilityCount : 1)
            }
          )
        ])
      ),
      execute({inputs}) {
        const input = readRasterOperationBand(
          id,
          {...inputs, validity: inputs['inputValidity']},
          props.input
        );
        const zoneIds = getCPUOperationUint32Rows(id, inputs, 'zoneIds');
        const probabilities = getCPUOperationFloat32Rows(id, inputs, 'probabilities');
        if (probabilities.length !== props.probabilityCount)
          throw new Error(`${id} probabilities must have probabilityCount rows`);
        const result = computeRasterFractionalZonalDistributionOnCPU(
          {
            input,
            probabilities,
            coverage: {
              zoneIds,
              cellIds: getCPUOperationUint32Rows(id, inputs, 'cellIds'),
              fractions: getCPUOperationFloat32Rows(id, inputs, 'fractions'),
              zoneValidity: getCPUOperationUint32Rows(id, inputs, 'zoneValidity'),
              count: getCPUOperationUint32Rows(id, inputs, 'count'),
              requiredCount: getCPUOperationUint32Rows(id, inputs, 'requiredCount'),
              overflow: getCPUOperationUint32Rows(id, inputs, 'overflow'),
              candidateOverflow:
                props.coverage.status.candidateOverflow !== undefined
                  ? getCPUOperationUint32Rows(id, inputs, 'candidateOverflow')
                  : undefined
            },
            observationWeights:
              props.observationWeights !== undefined
                ? getCPUOperationFloat32Rows(id, inputs, 'observationWeights')
                : undefined,
            cellAreas:
              props.cellAreas !== undefined
                ? getCPUOperationFloat32Rows(id, inputs, 'cellAreas')
                : undefined,
            parameters:
              props.parameters !== undefined
                ? getCPUOperationFloat32Rows(id, inputs, 'parameters')
                : undefined
          },
          props
        );
        return Object.fromEntries(
          Object.entries(props.output).map(([port, value]) => [
            value,
            result[port as keyof typeof result]
          ])
        );
      }
    });
  }
}
