// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
/** Center used for the radius of gyration. */
export type GyrationCenter = 'coordinate-mean' | 'spherical-mean';
/** @internal One default and validation rule for every form. */
export function resolveGyrationCenter(
  id: string,
  value: GyrationCenter | undefined,
  spatialContext?: GPUSpatialContext
): GyrationCenter {
  validateRequirement(
    id,
    value === undefined || value === 'coordinate-mean' || value === 'spherical-mean',
    `gyrationCenter must be coordinate-mean or spherical-mean; received ${value}`
  );
  validateRequirement(
    id,
    value !== 'spherical-mean' || spatialContext?.metric === 'great-circle',
    'gyrationCenter spherical-mean requires a great-circle spatialContext; use coordinate-mean for planar positions'
  );
  return (
    value ?? (spatialContext?.metric === 'great-circle' ? 'spherical-mean' : 'coordinate-mean')
  );
}
