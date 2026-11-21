// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Membership map pass, shared with device regression fixtures. ZONES and CLUSTERS are declarations. */
export const SCAN_MEMBERSHIP_BODY = /* wgsl */ `
  if (index >= ZONES || invalidCount[invalidCountOffset] != 0u) {
    membership[membershipOffset + index] = 0xffffffffu;
    return;
  }
  var member = 0u;
  let px = positions[positionsOffset + index * 2u];
  let py = positions[positionsOffset + index * 2u + 1u];
  for (var cluster = 0u; cluster < CLUSTERS; cluster++) {
    if (indices[indicesOffset + cluster * 4u + 1u] == 0u) {
      continue;
    }
    if (stats[statsOffset + cluster * 8u + 3u] > alpha[alphaOffset]) {
      continue;
    }
    let center = indices[indicesOffset + cluster * 4u];
    let dx = px - positions[positionsOffset + center * 2u];
    let dy = py - positions[positionsOffset + center * 2u + 1u];
    let radius = stats[statsOffset + cluster * 8u + 4u];
    if (dx * dx + dy * dy <= radius * radius * 1.00002 + 1.0) {
      member = cluster + 1u;
      break;
    }
  }
  membership[membershipOffset + index] = member;`;
