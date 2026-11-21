// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateRasterGridAxisAligned} from '../raster-band-utils';
import type {GPURasterFractionalZonalDistributionOutput} from '../gpu-raster-fractional-zonal-distribution';

/** Capture all required destinations/views, including missing fields and inherited getters. @internal */
export function snapshotRasterFractionalDistributionOutput<
  Output extends Record<keyof GPURasterFractionalZonalDistributionOutput, unknown>
>(output: Output): Output {
  return Object.freeze({
    means: output.means,
    populationVariances: output.populationVariances,
    sampleVariances: output.sampleVariances,
    effectiveSampleSizes: output.effectiveSampleSizes,
    weightSums: output.weightSums,
    squaredWeightSums: output.squaredWeightSums,
    coveredArea: output.coveredArea,
    validArea: output.validArea,
    quantiles: output.quantiles,
    quantileValidity: output.quantileValidity,
    validity: output.validity,
    status: output.status
  }) as Output;
}

/** Capture the coverage evidence used by both declaration and dispatch. @internal */
export function snapshotRasterFractionalDistributionCoverage<
  Coverage extends {
    zoneIds: unknown;
    cellIds: unknown;
    fractions: unknown;
    zoneValidity: unknown;
    status: {
      count: unknown;
      requiredCount: unknown;
      overflow: unknown;
      candidateOverflow?: unknown;
    };
  }
>(coverage: Coverage): Coverage {
  const status = coverage.status;
  return Object.freeze({
    zoneIds: coverage.zoneIds,
    cellIds: coverage.cellIds,
    fractions: coverage.fractions,
    zoneValidity: coverage.zoneValidity,
    status: Object.freeze({
      count: status.count,
      requiredCount: status.requiredCount,
      overflow: status.overflow,
      candidateOverflow: status.candidateOverflow
    })
  }) as Coverage;
}

/** Shared construction-time options for CPU and GPU operation forms. @internal */
export function validateRasterFractionalDistributionOptions(
  id: string,
  options: {
    grid: GPURasterMetadata;
    zoneCount: number;
    cellSizeMode?: 'uniform' | 'geographic' | 'web-mercator';
    quantileMethod?: 'exactextract' | 'inverse-cdf';
  },
  hasCellAreas: boolean
): void {
  validateCapacity(id, 'zoneCount', options.zoneCount, 0);
  validateRequirement(
    id,
    ['uniform', 'geographic', 'web-mercator'].includes(options.cellSizeMode ?? 'uniform'),
    `cellSizeMode must be uniform, geographic or web-mercator; received ${String(options.cellSizeMode)}`
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
    `quantileMethod must be 'exactextract' or 'inverse-cdf'; received ${String(options.quantileMethod)}`
  );
  const [columnX, rowX, , columnY, rowY] = options.grid.affine;
  validateRequirement(
    id,
    Number.isFinite(columnX * rowY - rowX * columnY) && columnX * rowY - rowX * columnY !== 0,
    'grid.affine must have a finite nonzero determinant'
  );
}
