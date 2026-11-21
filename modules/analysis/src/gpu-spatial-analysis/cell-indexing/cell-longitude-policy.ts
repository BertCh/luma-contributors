// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getExactDegreeReductionWGSL} from './cell-index-wgsl';

/** Keep the family's native clipping/wrapping, or wrap all longitudes to [-180, 180) degrees. */
export type CellLongitudePolicy = 'family-default' | 'wrap';

export function resolveCellLongitudePolicy(
  id: string,
  policy: CellLongitudePolicy | undefined
): CellLongitudePolicy {
  validateRequirement(
    id,
    policy === undefined || policy === 'family-default' || policy === 'wrap',
    `longitudePolicy must be 'family-default' or 'wrap'; received ${String(policy)}`
  );
  return policy ?? 'family-default';
}

/** Remainder first preserves large finite degrees and avoids changing already canonical values. */
export function wrapCellLongitudeDegrees(longitude: number): number {
  const remainder = longitude % 360;
  return remainder >= 180 ? remainder - 360 : remainder < -180 ? remainder + 360 : remainder;
}

export const CELL_WRAP_LONGITUDE_WGSL = /* wgsl */ `
${getExactDegreeReductionWGSL('cellWrapReduceDegrees')}
fn cellWrapLongitudeDegrees(longitude: f32) -> f32 {
  let bits = bitcast<u32>(longitude);
  // Canonical inputs, including subnormals, must reach the tile indexer's bit tests unchanged.
  if ((bits & 0x7fffffffu) < 0x43340000u || bits == 0xc3340000u) {
    return longitude;
  }
  let reduced = cellWrapReduceDegrees(longitude);
  let remainder = reduced % 360.0;
  if (remainder >= 180.0) { return remainder - 360.0; }
  if (remainder < -180.0) { return remainder + 360.0; }
  return remainder;
}
`;
