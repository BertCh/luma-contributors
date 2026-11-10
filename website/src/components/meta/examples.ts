// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Code on the how-it-works pages comes from `*.example.ts` files that the website typecheck compiles
// against the public packages, so an example cannot drift from the API. A file is a whole module;
// the page shows it without its licence header and without the regions marked as hidden:
//
//   // example:hide-start
//   declare const device: Device;   // setup the reader does not need to see
//   // example:hide-end
//
// A lens scenario file (modules/lenses/src/computation-lenses/scenarios/**/<name>.example.ts) can be
// imported the same way: its `// #region example` block is the code that produced the recorded run.

const HIDE_START = /^\s*\/\/ example:hide-start\s*$/;
const HIDE_END = /^\s*\/\/ example:hide-end\s*$/;
const HEADER = /^\/\/ (SPDX-|luma\.gl$)/;
const REGION_START = /^\s*\/\/ #region example\s*$/;
const REGION_END = /^\s*\/\/ #endregion example\s*$/;

/** The lines of a `// #region example` block without their shared indentation, or null. */
function extractRegion(lines: readonly string[]): string[] | null {
  const start = lines.findIndex(line => REGION_START.test(line));
  const end = lines.findIndex(line => REGION_END.test(line));
  if (start < 0 || end <= start) return null;
  const region = lines.slice(start + 1, end);
  const indents = region.filter(line => line.trim() !== '').map(line => line.search(/\S/));
  const indent = indents.length > 0 ? Math.min(...indents) : 0;
  return region.map(line => line.slice(indent));
}

/** The text a page shows for an example file: header and hidden regions removed, blank runs collapsed. */
export function extractExample(source: string): string {
  const kept: string[] = [];
  let hidden = false;
  const sourceLines = source.split('\n');
  for (const line of extractRegion(sourceLines) ?? sourceLines) {
    if (HIDE_START.test(line)) hidden = true;
    else if (HIDE_END.test(line)) hidden = false;
    else if (!hidden && !HEADER.test(line)) kept.push(line.replace(/\s+$/, ''));
  }
  const collapsed: string[] = [];
  for (const line of kept) {
    if (line === '' && (collapsed.length === 0 || collapsed.at(-1) === '')) continue;
    collapsed.push(line);
  }
  while (collapsed.at(-1) === '') collapsed.pop();
  return collapsed.join('\n');
}

/**
 * 1-based inclusive line range of the first line containing `from` through the next line containing
 * `to` (or the same line when `to` is omitted). Pages use it to point at code without counting lines.
 */
export function findExampleLines(code: string, from: string, to?: string): [number, number] {
  const lines = code.split('\n');
  const start = lines.findIndex(line => line.includes(from));
  if (start < 0) return [1, 1];
  if (to === undefined) return [start + 1, start + 1];
  const end = lines.findIndex((line, index) => index >= start && line.includes(to));
  return [start + 1, (end < 0 ? start : end) + 1];
}
