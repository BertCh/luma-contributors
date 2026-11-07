// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A key for a run's picture: color swatches with names. A run lists it in its `.parameters.ts` file
// (`export const legend`), which has no luma.gl imports, so prerendered pages carry it too.

export type RunLegendItem = {
  label: string;
  /** CSS color of the swatch. */
  color: string;
};
