// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUSolverStatusPort} from '../../analysis-core/contracts/inference';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Supported IV sandwiches; CLUSTER uses one-way CR1 finite-sample correction. */
export type GPUInstrumentalVariablesCovarianceMethod =
  | 'HC0'
  | 'HC1'
  | 'HC2'
  | 'HC3'
  | 'HAC'
  | 'CLUSTER';

/** Stable solver codes, shared by the GPU covariance and the spatial 2SLS CPU twin. */
export const GPU_INSTRUMENTAL_VARIABLES_COVARIANCE_STATUS = {
  SUCCESS: 0,
  SINGULAR_PROJECTION: 1,
  INSUFFICIENT_ROWS: 2,
  NONFINITE_INPUT: 3,
  INVALID_KERNEL: 4,
  INVALID_LEVERAGE: 5,
  INVALID_CLUSTERS: 6,
  NOT_POSITIVE_SEMIDEFINITE: 7
} as const;

/** Small dense projection arrays, including the intercept, fit the shared 16-wide solver. */
export const INSTRUMENTAL_VARIABLES_COVARIANCE_MAXIMUM_COEFFICIENT_COUNT = 15;
/** The intercept is centered out of the at-most-16-wide instrument factorization. */
export const INSTRUMENTAL_VARIABLES_COVARIANCE_MAXIMUM_INSTRUMENT_COUNT = 16;
/** Float32 projection error near a unit leverage cannot resolve a usable HC2/HC3 denominator. */
export const INSTRUMENTAL_VARIABLES_COVARIANCE_LEVERAGE_TOLERANCE = 1e-6;
/** Certify V + tolerance * max(abs(V)) I positive definite, accepting zero/rank-deficient V. */
export const INSTRUMENTAL_VARIABLES_COVARIANCE_PSD_TOLERANCE = 1e-5;

/** Legacy scalar status is accepted; new callers should use the named solver port. @internal */
export function getInstrumentalVariablesCovarianceStatusView(
  status: GraphDataView<'uint32'> | Pick<GPUSolverStatusPort, 'code'>
): GraphDataView<'uint32'> {
  return 'code' in status ? status.code : status;
}

/** Shared construction checks; clusterCount is a capacity, not the number of occupied groups. @internal */
export function validateInstrumentalVariablesCovarianceOptions(
  id: string,
  rowCount: number,
  options: {method?: string; kernel?: unknown; clusters?: unknown; clusterCount?: number}
): void {
  const method = options.method ?? 'HC0';
  validateRequirement(
    id,
    ['HC0', 'HC1', 'HC2', 'HC3', 'HAC', 'CLUSTER'].includes(method),
    'method must be HC0, HC1, HC2, HC3, HAC or CLUSTER'
  );
  validateRequirement(
    id,
    (method === 'HAC') === Boolean(options.kernel),
    'kernel must be present exactly for method HAC'
  );
  validateRequirement(
    id,
    (method === 'CLUSTER') === Boolean(options.clusters),
    'clusters must be present exactly for method CLUSTER'
  );
  validateRequirement(
    id,
    (method === 'CLUSTER') === (options.clusterCount !== undefined),
    'clusterCount must be present exactly for method CLUSTER'
  );
  if (options.clusterCount !== undefined)
    validateCapacity(id, 'clusterCount', options.clusterCount, 1, rowCount);
}

// Fixed stages plus a conservative allowance for the published <=24-bit radix sort and searches.
const INSTRUMENTAL_VARIABLES_COVARIANCE_SORT_NODE_ALLOWANCE = 256;
// Fixed stages for HC/HAC, including row-status and covariance reductions.
const INSTRUMENTAL_VARIABLES_COVARIANCE_FIXED_NODE_ALLOWANCE = 16;

/** Includes extraction, gather, segment reduction and packing for each score/count. @internal */
export function getInstrumentalVariablesCovarianceCommandNodeEstimate(
  coefficientCount: number,
  clustered: boolean
): number {
  return clustered
    ? INSTRUMENTAL_VARIABLES_COVARIANCE_SORT_NODE_ALLOWANCE + 4 * (coefficientCount + 2)
    : INSTRUMENTAL_VARIABLES_COVARIANCE_FIXED_NODE_ALLOWANCE;
}

/** All HC/HAC/cluster scratch shapes, reusable before a composing fit declares its inputs. */
export function getInstrumentalVariablesCovarianceAllocationLengths(
  rows: number,
  columns: number,
  instruments: number,
  options: {clusterCount?: number; kernel?: unknown} = {}
): Readonly<Record<string, number>> {
  const coefficients = columns + 1;
  const tileCount = Math.ceil(rows / Math.max(64, Math.ceil(rows / 4096)));
  const meatRows = options.clusterCount ?? rows;
  const meatTileCount = Math.ceil(meatRows / Math.max(64, Math.ceil(meatRows / 4096)));
  const momentCount = (instruments * (instruments + 1)) / 2 + instruments * columns + columns;
  const meatCount = coefficients ** 2;
  return {
    'covariance workspace': columns + instruments + 1 + momentCount + meatCount + 3,
    'covariance projection': 2 * columns * instruments,
    'covariance score rows': rows * coefficients,
    'covariance row status/contributions/sort scratch': rows,
    'covariance sum partials': tileCount * (columns + instruments + 1),
    'covariance moment partials': tileCount * momentCount,
    'covariance leverage partials': tileCount,
    'covariance meat partials': meatTileCount * (meatCount + (options.kernel ? 1 : 0)),
    ...(options.clusterCount
      ? {
          'cluster scores': meatRows * coefficients,
          'cluster totals': meatRows,
          'cluster segment offsets': meatRows + 1,
          'cluster occupied partials': meatTileCount
        }
      : {})
  };
}
