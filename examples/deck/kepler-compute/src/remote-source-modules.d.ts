// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** `d3-dsv` ships no types in this checkout; only the used surface is declared. */
declare module 'd3-dsv' {
  export function dsvFormat(delimiter: string): {parseRows(text: string): string[][]};
}
