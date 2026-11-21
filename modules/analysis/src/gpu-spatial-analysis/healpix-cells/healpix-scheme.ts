// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUHealpixIndexScheme} from './gpu-healpix-point-to-cell';

/** Resolve only an omitted scheme; HEALPix RING numbering is not implemented. */
export function resolveHealpixIndexScheme(
  id: string,
  scheme: GPUHealpixIndexScheme | undefined
): GPUHealpixIndexScheme {
  validateRequirement(
    id,
    scheme === undefined || scheme === 'nested' || scheme === 'uniq',
    `scheme must be 'nested' or 'uniq'; RING is unsupported; received ${String(scheme)}`
  );
  return scheme === undefined ? 'nested' : scheme;
}
