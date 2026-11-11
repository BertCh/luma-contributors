// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {solveCPUEquilibratedSystem} from '../../analysis-core/kernels/cpu-equilibrated-solve';
import {KRIGING_PIVOT_RATIO_TOLERANCE} from './kriging-parameters';
import {
  GPU_LOCAL_KRIGING_PARAMETER_LENGTH,
  GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH,
  MATERN_3_2_RANGE_FACTOR,
  MATERN_5_2_RANGE_FACTOR
} from './prediction-covariance';

// The packers expose total lengths, but no field-offset constants. These named sections follow
// getGPULocalKrigingParameterValues and getGPUSpaceTimeKrigingParameterValues exactly.
const COVARIANCE_STRUCTURE_LENGTH = 4;
const LOCAL_STRUCTURE_START = GPU_LOCAL_KRIGING_PARAMETER_LENGTH - COVARIANCE_STRUCTURE_LENGTH;
const SPACE_TIME_HEADER_LENGTH = 4;
const SPACE_TIME_STRUCTURE_COUNT = 3;
const GNEITING_START =
  SPACE_TIME_HEADER_LENGTH + SPACE_TIME_STRUCTURE_COUNT * COVARIANCE_STRUCTURE_LENGTH;
const GNEITING_LENGTH = 8;

/** Packed per-site observations and ordered CSR conditioning sets. @internal */
export type CPUVecchiaRows = {
  positions: Float32Array;
  observations: Float32Array;
  neighborhoodOffsets: Uint32Array;
  neighborhoods: Uint32Array;
  parameters: Float32Array;
};

/** Same six output arrays as GPUVecchiaLikelihoodOutput. @internal */
export type CPUVecchiaLikelihoodRows = {
  conditionals: Float32Array;
  conditionalStatus: Uint32Array;
  siteTerms: Float32Array;
  siteStatus: Uint32Array;
  logLikelihood: Float32Array;
  status: Uint32Array;
};

/** Construction-time CPU props checks, with the GPU-only neighborhood capacity dropped. @internal */
export function validateCPUVecchiaProps(
  id: string,
  props: {rowCount: number; neighborCount: number; spatialContext?: GPUSpatialContext}
): void {
  validateCapacity(id, 'rowCount', props.rowCount);
  validateCapacity(id, 'neighborCount', props.neighborCount);
  resolveGPUSpatialContext(id, props.spatialContext, {
    ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
    measure: 'distance'
  });
}

/** Checks array shapes; spatial invalid CSR spans have an explicit GPU status instead. @internal */
export function validateCPUVecchiaRows(
  input: CPUVecchiaRows,
  times?: Float32Array,
  allowInvalidSpans = false
): number {
  const rowCount = input.positions.length / 2;
  if (!Number.isInteger(rowCount)) throw new Error('positions length must be a multiple of 2');
  if (input.observations.length !== rowCount * 4) {
    throw new Error('observations length must equal positions row count * 4');
  }
  if (times && times.length !== rowCount)
    throw new Error('times length must equal positions row count');
  if (input.neighborhoodOffsets.length !== rowCount + 1) {
    throw new Error('neighborhoodOffsets length must equal positions row count + 1');
  }
  if (!allowInvalidSpans) {
    for (let row = 0; row < rowCount; row++) {
      if (!hasCPUVecchiaSpan(input, row)) {
        throw new Error(
          'neighborhoodOffsets must be nondecreasing and within neighborhoods length'
        );
      }
    }
  }
  const parameterLength = times
    ? GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH
    : GPU_LOCAL_KRIGING_PARAMETER_LENGTH;
  if (input.parameters.length !== parameterLength) {
    throw new Error(`parameters length must equal ${parameterLength}`);
  }
  return rowCount;
}

/** Whether a CSR span is valid; the spatial kernel explicitly returns status 1 otherwise. @internal */
export function hasCPUVecchiaSpan(input: CPUVecchiaRows, site: number): boolean {
  const begin = input.neighborhoodOffsets[site];
  const end = input.neighborhoodOffsets[site + 1];
  return begin <= end && end <= input.neighborhoods.length;
}

/** The GPU's observation filter, independent of the position filter. @internal */
export function isCPUVecchiaObservationUsable(observations: Float32Array, row: number): boolean {
  const [value, error, , validity] = observations.subarray(row * 4, row * 4 + 4);
  return validity > 0 && Number.isFinite(value) && Number.isFinite(error) && error >= 0;
}

/** The GPU's finite point filter, optionally including relative time. @internal */
export function isCPUVecchiaPositionUsable(
  positions: Float32Array,
  row: number,
  times?: Float32Array
): boolean {
  return (
    Number.isFinite(positions[row * 2]) &&
    Number.isFinite(positions[row * 2 + 1]) &&
    (!times || Number.isFinite(times[row]))
  );
}

/** All usable conditioning rows, keeping duplicate slots and ignoring out-of-range IDs. @internal */
export function getCPUVecchiaNeighbors(
  input: CPUVecchiaRows,
  site: number,
  times?: Float32Array
): number[] {
  const rows: number[] = [];
  for (
    let slot = input.neighborhoodOffsets[site];
    slot < input.neighborhoodOffsets[site + 1];
    slot++
  ) {
    const sample = input.neighborhoods[slot];
    if (
      sample < input.positions.length / 2 &&
      isCPUVecchiaObservationUsable(input.observations, sample) &&
      isCPUVecchiaPositionUsable(input.positions, sample, times)
    )
      rows.push(sample);
  }
  return rows;
}

/**
 * Exact closed-form effective-range correlation in float64 (no truncated Matern series).
 * Match WGSL u32 conversion of finite negative structure codes: clamp to zero before truncation.
 * @internal
 */
export function getCPUVecchiaCorrelation(modelCode: number, scaled: number): number {
  switch (Math.trunc(Math.max(0, modelCode))) {
    case 0:
      return scaled >= 1 ? 0 : 1 - 1.5 * scaled + 0.5 * scaled ** 3;
    case 1:
      return Math.exp(-3 * scaled);
    case 2:
      return Math.exp(-3 * scaled * scaled);
    case 3: {
      const argument = MATERN_3_2_RANGE_FACTOR * scaled;
      if (argument === Infinity) return 0;
      return argument > 50
        ? Math.exp(Math.log1p(argument) - argument)
        : (1 + argument) * Math.exp(-argument);
    }
    default: {
      const argument = MATERN_5_2_RANGE_FACTOR * scaled;
      if (argument === Infinity) return 0;
      // In the tail, neither square a huge argument nor underflow exp(-argument) before its
      // polynomial factor has been applied. This also retains finite float64 subnormal tails.
      return argument > 50
        ? Math.exp(
            2 * Math.log(argument) -
              Math.log(3) +
              Math.log1p(3 / argument + 3 / argument / argument) -
              argument
          )
        : (1 + argument + (argument * argument) / 3) * Math.exp(-argument);
    }
  }
}

/** Decodes the local packer layout and matches loadCovarianceParameters' acceptance. @internal */
export function getCPUVecchiaCovariance(input: CPUVecchiaRows): {
  valid: boolean;
  scale: number;
  modelCode: number;
  nugget: number;
  sill: number;
  range: number;
  covariance(first: number, second: number): number;
} {
  const [scale] = input.parameters;
  const [modelCode, nugget, sill, range] = input.parameters.subarray(LOCAL_STRUCTURE_START);
  return {
    valid: scale > 0 && modelCode >= 0 && modelCode <= 4 && nugget >= 0 && sill >= 0 && range > 0,
    scale,
    modelCode,
    nugget,
    sill,
    range,
    covariance(first, second) {
      const differenceX = input.positions[first * 2] - input.positions[second * 2];
      const differenceY = input.positions[first * 2 + 1] - input.positions[second * 2 + 1];
      return (
        sill *
          getCPUVecchiaCorrelation(
            modelCode,
            Math.sqrt(differenceX * differenceX + differenceY * differenceY) / range
          ) +
        (differenceX === 0 && differenceY === 0 ? nugget : 0)
      );
    }
  };
}

/** Decodes the space-time packer directly; unused zero structures are never evaluated. @internal */
export function getCPUVecchiaSpaceTimeCovariance(
  input: CPUVecchiaRows,
  times: Float32Array
): {
  valid: boolean;
  scale: number;
  covariance(first: number, second: number): number;
} {
  const [scale, typeCode, anisotropy, coefficient] = input.parameters;
  const structures = Array.from({length: SPACE_TIME_STRUCTURE_COUNT}, (_, slot) => {
    const start = SPACE_TIME_HEADER_LENGTH + slot * COVARIANCE_STRUCTURE_LENGTH;
    return input.parameters.subarray(start, start + COVARIANCE_STRUCTURE_LENGTH);
  });
  const [sill, nugget, temporalScale, spatialScale, alpha, gamma, beta, tau] =
    input.parameters.subarray(GNEITING_START, GNEITING_START + GNEITING_LENGTH);
  const structureCovariance = (slot: number, lag: number, coincident: boolean): number => {
    const [modelCode, structureNugget, structureSill, range] = structures[slot];
    return (
      structureSill * getCPUVecchiaCorrelation(modelCode, lag / range) +
      (coincident ? structureNugget : 0)
    );
  };
  return {
    valid: scale > 0 && typeCode >= 0 && typeCode <= 4,
    scale,
    covariance(first, second) {
      const differenceX = input.positions[first * 2] - input.positions[second * 2];
      const differenceY = input.positions[first * 2 + 1] - input.positions[second * 2 + 1];
      const spatialLag = Math.sqrt(differenceX * differenceX + differenceY * differenceY);
      const timeLag = Math.abs(times[first] - times[second]);
      const samePlace = differenceX === 0 && differenceY === 0;
      const sameTime = timeLag === 0;
      switch (Math.trunc(typeCode)) {
        case 2:
        case 3: {
          const joint = structureCovariance(
            2,
            Math.sqrt(spatialLag * spatialLag + anisotropy * anisotropy * timeLag * timeLag),
            samePlace && sameTime
          );
          return Math.trunc(typeCode) === 2
            ? joint
            : structureCovariance(0, spatialLag, samePlace) +
                structureCovariance(1, timeLag, sameTime) +
                joint;
        }
        case 4: {
          const psi = temporalScale * (timeLag <= 0 ? 0 : timeLag ** (2 * alpha)) + 1;
          const spatialPower = spatialLag <= 0 ? 0 : spatialLag ** (2 * gamma);
          return (
            sill * psi ** -tau * Math.exp(-spatialScale * spatialPower * psi ** (-beta * gamma)) +
            (samePlace && sameTime ? nugget : 0)
          );
        }
        default: {
          const spatial = structureCovariance(0, spatialLag, samePlace);
          const temporal = structureCovariance(1, timeLag, sameTime);
          return (
            coefficient * spatial * temporal + (Math.trunc(typeCode) === 0 ? 0 : spatial + temporal)
          );
        }
      }
    }
  };
}

/** Builds the scaled conditioning system, adding measurement error only on its diagonal. @internal */
export function solveCPUVecchiaConditioning(
  input: CPUVecchiaRows,
  rows: readonly number[],
  covariance: (first: number, second: number) => number,
  scale: number,
  right: Float64Array
): ReturnType<typeof solveCPUEquilibratedSystem> {
  const matrix = new Float64Array(rows.length * rows.length);
  rows.forEach((sample, row) => {
    rows.forEach((other, column) => {
      matrix[row * rows.length + column] = covariance(sample, other) / scale;
    });
    matrix[row * rows.length + row] += input.observations[sample * 4 + 1] / scale;
  });
  return solveCPUEquilibratedSystem(
    matrix,
    Float64Array.from(right, value => value / scale),
    KRIGING_PIVOT_RATIO_TOLERANCE
  );
}

/** Float64 compensated totals, split into the contributor's high/low output layout. @internal */
export function sumCPUVecchiaCells(
  terms: Float64Array,
  cellCount: number,
  status: Uint32Array
): Float32Array {
  const totals = new Float32Array(cellCount * 2).fill(NaN);
  if (status[0] > 0 || status[2] === 0) return totals;
  for (let cell = 0; cell < cellCount; cell++) {
    let total = 0;
    let correction = 0;
    for (let index = cell; index < terms.length; index += cellCount) {
      const value = terms[index];
      if (!Number.isFinite(value)) continue;
      const next = total + value;
      correction +=
        Math.abs(total) >= Math.abs(value) ? total - next + value : value - next + total;
      total = next;
    }
    const high = Math.fround(total + correction);
    totals[cell * 2] = high;
    totals[cell * 2 + 1] = total - high + correction;
  }
  return totals;
}

/** Shared conditionals and likelihood term evaluation, without a conditioning-row cap. @internal */
export function evaluateCPUVecchiaLikelihood(
  input: CPUVecchiaRows,
  times?: Float32Array
): CPUVecchiaLikelihoodRows {
  const rowCount = validateCPUVecchiaRows(input, times, !times);
  const model = times
    ? getCPUVecchiaSpaceTimeCovariance(input, times)
    : getCPUVecchiaCovariance(input);
  const conditionals = new Float64Array(rowCount * 4);
  const conditionalStatus = new Uint32Array(rowCount).fill(1);
  const siteTerms = new Float64Array(rowCount).fill(NaN);
  const siteStatus = new Uint32Array(rowCount);
  const status = new Uint32Array(3);
  for (let site = 0; site < rowCount; site++) {
    const start = site * 4;
    conditionals.set([NaN, NaN, 0, 0], start);
    if (
      model.valid &&
      isCPUVecchiaPositionUsable(input.positions, site, times) &&
      hasCPUVecchiaSpan(input, site)
    ) {
      const rows = getCPUVecchiaNeighbors(input, site, times);
      if (!times) {
        const distanceToSite = (row: number): number => {
          const differenceX = input.positions[row * 2] - input.positions[site * 2];
          const differenceY = input.positions[row * 2 + 1] - input.positions[site * 2 + 1];
          return Math.sqrt(differenceX * differenceX + differenceY * differenceY);
        };
        // Stable sort: the explicit CSR path breaks equal distances by visit order, not row ID.
        rows.sort((first, second) => distanceToSite(first) - distanceToSite(second));
      }
      if (rows.length === 0) {
        const variance = model.covariance(site, site);
        if (Number.isFinite(variance) && (times || variance >= 0)) {
          conditionals.set([0, variance, 1, 0], start);
          conditionalStatus[site] = 0;
        }
      } else {
        conditionalStatus[site] = 2;
        const cross = Float64Array.from(rows, row => model.covariance(row, site));
        const solved = solveCPUVecchiaConditioning(
          input,
          rows,
          model.covariance,
          model.scale,
          cross
        );
        if (!solved.solution) {
          conditionals[start + 2] = solved.pivotRatio;
        } else {
          let mean = 0;
          const priorVariance = model.covariance(site, site);
          let variance = priorVariance;
          rows.forEach((row, index) => {
            mean += solved.solution![index] * input.observations[row * 4];
            variance -= solved.solution![index] * cross[index];
          });
          if (
            Number.isFinite(mean) &&
            Number.isFinite(variance) &&
            (times || variance >= -1e-4 * Math.max(priorVariance, 1e-20))
          ) {
            conditionals.set(
              [mean, times ? variance : Math.max(variance, 0), solved.pivotRatio, rows.length],
              start
            );
            conditionalStatus[site] = 0;
          }
        }
      }
    }
    if (!isCPUVecchiaObservationUsable(input.observations, site)) {
      siteStatus[site] = 1;
      status[1]++;
      continue;
    }
    const marginal = input.neighborhoodOffsets[site] === input.neighborhoodOffsets[site + 1];
    let mean = 0;
    let variance = model.scale;
    if (marginal ? !(variance > 0) : conditionalStatus[site] !== 0) {
      siteStatus[site] = 2;
      status[0]++;
      continue;
    }
    if (!marginal) {
      mean = conditionals[start];
      variance = conditionals[start + 1];
    }
    variance += input.observations[start + 1];
    const residual = input.observations[start] - mean;
    const term =
      -0.5 * (Math.log(2 * Math.PI) + Math.log(variance) + (residual * residual) / variance);
    if (!(variance > 0) || !Number.isFinite(variance) || !Number.isFinite(term)) {
      siteStatus[site] = 3;
      status[0]++;
      continue;
    }
    siteTerms[site] = term;
    status[2]++;
  }
  return {
    conditionals: Float32Array.from(conditionals),
    conditionalStatus,
    siteTerms: Float32Array.from(siteTerms),
    siteStatus,
    logLikelihood: sumCPUVecchiaCells(siteTerms, 1, status),
    status
  };
}
