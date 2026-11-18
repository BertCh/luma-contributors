// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** One completeness decision for CSR and dependent-result readbacks. */
export function getRegionalizationWeightsReadout(overflow: number, candidateOverflow: number) {
  const complete = overflow === 0 && candidateOverflow === 0;
  return {
    complete,
    capacity: complete ? 'ok' : 'Incomplete weights: raise neighbor or pair capacity',
    unavailable: 'Unavailable (incomplete weights)'
  };
}
