// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Count published usable rows only when the intersection checks are complete. */
export function getUsablePolygonCount(validity: Uint32Array, overflow: number): number | undefined {
  if (overflow !== 0) return undefined;
  return validity.reduce((count, value) => count + Number(value !== 0), 0);
}
