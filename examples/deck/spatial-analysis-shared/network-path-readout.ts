// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Preserve certified target costs; batch loss only qualifies targets without a path. */
export function formatNetworkPathReadout(
  status: {found: boolean; converged: number; candidateOverflow: number; overflow: number},
  foundText: string
): string {
  if (!status.converged) return 'search incomplete';
  if (status.found) return foundText;
  if (status.candidateOverflow) return 'incomplete batch (path walk budget exhausted)';
  if (status.overflow) return 'incomplete batch (path output capacity exceeded)';
  return 'no route found';
}
