// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type SphericalOverlayDisplayStatus = {overflow?: number; candidateOverflow?: number};
export const SPHERICAL_OVERLAY_DISPLAY_COUNT_WGSL =
  'displayCount[displayCountOffset]=select(0u,count[countOffset],overflow[overflowOffset]==0u && candidateOverflow[candidateOverflowOffset]==0u);';

export function getSphericalOverlayStatusLabel(status?: SphericalOverlayDisplayStatus): string {
  if (status?.overflow === undefined || status.candidateOverflow === undefined)
    return 'checking result';
  if (status.overflow !== 0) return 'incomplete: output capacity';
  if (status.candidateOverflow !== 0) return 'incomplete: input or precision loss';
  return '';
}

/** Keep operand context visible; result layers are usable only after both flags are known zero. */
export function selectSphericalOverlayLayers<Layer>(
  status: SphericalOverlayDisplayStatus | undefined,
  operands: readonly Layer[],
  result: readonly Layer[]
): Layer[] {
  return getSphericalOverlayStatusLabel(status) === '' ? [...operands, ...result] : [...operands];
}
