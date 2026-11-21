// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Select a neighborhood statistic while preserving unavailable rows for the map and extent. */
export function getNeighborhoodSelectionBody(columnCount: number, noMode = 0xffffffff): string {
  return /* wgsl */ `
  let choice = source[sourceOffset];
  var value = 0.0;
  if (choice < ${columnCount}u) {
    value = table[tableOffset + index * ${columnCount}u + choice];
  } else if (choice == ${columnCount}u) {
    let mode = modes[modesOffset + index];
    value = f32(mode);
    if (mode == ${noMode}u) {
      value = bitcast<f32>(0x7fc00000u | index);
    }
  } else {
    value = entropy[entropyOffset + index];
  }
  // Preserve unavailable numeric rows; the extent reduction ignores nonfinite inputs.
  selected[selectedOffset + index] = value;`;
}
