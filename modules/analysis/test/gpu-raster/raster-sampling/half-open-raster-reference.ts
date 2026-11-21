// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Independent edge search: no reciprocal, quotient or candidate correction from the shader. */
export function findHalfOpenRasterCell(
  position: number,
  origin: number,
  step: number,
  count: number
): number {
  if (!Number.isFinite(position)) return -1;
  // Search the ordered packed cell edges in the original coordinate frame. The first edge
  // strictly after position bounds its cell; the final outer edge therefore returns count.
  let begin = 0;
  let end = count + 1;
  while (begin < end) {
    const middle = begin + Math.floor((end - begin) / 2);
    const edge = Math.fround(origin + Math.fround(middle * step));
    const atOrAfterEdge = step > 0 ? position >= edge : position <= edge;
    if (atOrAfterEdge) begin = middle + 1;
    else end = middle;
  }
  return begin - 1;
}
