// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Independent fixed Epanechnikov scene: unit cells, h=1.5, point at (3.5,2.5).
 * K=2/(pi*h²)*(1-d²/h²) for d²<h², otherwise zero. Squared distances are integers.
 * Nonzero cells are rows 1..3, columns 2..4: center 8/(9pi), cardinals 40/(81pi),
 * corners 8/(81pi); total midpoint mass is 88/(27pi). All coordinates are represented exactly.
 * The 12x8 index excludes the true bin (3,2) from distant queries and a false far-corner bin
 * from these nonzero cells, even with the one-cell candidate expansion.
 */
export const TRANSLATED_DENSITY_FIXTURE = {
  width: 12,
  height: 8,
  extent: [0, 0, 12, 8] as const,
  point: [3.5, 2.5] as const,
  bandwidth: 1.5,
  values: [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.03143801345025095, 0.15719006725125467,
    0.03143801345025095, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.15719006725125467, 0.2829421210522584,
    0.15719006725125467, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.03143801345025095, 0.15719006725125467,
    0.03143801345025095, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  ],
  mass: 1.037454443858281
};
