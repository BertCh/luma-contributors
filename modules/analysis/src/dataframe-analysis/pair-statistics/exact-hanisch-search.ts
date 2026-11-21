// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Rectangle Hanisch tail search. Requires the shared lattice and cell-ordered points. @internal */
export const EXACT_HANISCH_SEARCH_WGSL = /* wgsl */ `
fn getExactHanischCellNearest(point: vec2f, excludedRow: u32, cell: u32, limit: f32) -> vec2f {
  var nearestSquared = limit;
  var found = false;
  let begin = cellOffsets[cellOffsetsOffset + cell];
  let end = cellOffsets[cellOffsetsOffset + cell + 1u];
  for (var candidate = begin; candidate < end; candidate++) {
    let base = sortedPointsOffset + candidate * 3u;
    if (sortedPoints[base + 2u] == excludedRow) { continue; }
    let delta = vec2f(bitcast<f32>(sortedPoints[base]), bitcast<f32>(sortedPoints[base + 1u])) - point;
    let squared = dot(delta, delta);
    if (squared <= nearestSquared) { nearestSquared = squared; found = true; }
  }
  return vec2f(nearestSquared, select(0.0, 1.0, found));
}

// Only distances at most the boundary distance can contribute to the Hanisch denominator.
// The returned y is 1 iff such an event exists; x is its exact nearest distance.
// Each cell is visited once: at most columns*rows <= 2^24-1 cell visits and four side
// checks per ring (<4*max(columns,rows)). Thin lattices do not rescan their interior rows.
fn getExactHanischDistance(lattice: Lattice, point: vec2f, excludedRow: u32, border: f32) -> vec2f {
  let column = getCellColumn(lattice, point.x);
  let row = getCellRow(lattice, point.y);
  var nearestSquared = border * border;
  var found = false;
  for (var ring = 0u; ring < max(lattice.columns, lattice.rows); ring++) {
    let firstColumn = column - min(column, ring);
    let lastColumn = min(column + ring, lattice.columns - 1u);
    let firstRow = row - min(row, ring);
    let lastRow = min(row + ring, lattice.rows - 1u);
    for (var side = 0u; side < 4u; side++) {
      var first = firstColumn;
      var last = lastColumn;
      var fixed = firstRow;
      if (side == 0u && row < ring) { continue; }
      if (side == 1u) {
        if (ring == 0u || row + ring >= lattice.rows) { continue; }
        fixed = row + ring;
      }
      if (side >= 2u) {
        if (ring == 0u) { continue; }
        // Vertical sides exclude the two corner rows already covered horizontally.
        first = row - min(row, ring - 1u);
        last = min(row + ring - 1u, lattice.rows - 1u);
        fixed = firstColumn;
        if (side == 2u && column < ring) { continue; }
        if (side == 3u) {
          if (column + ring >= lattice.columns) { continue; }
          fixed = column + ring;
        }
      }
      for (var along = first; along <= last; along++) {
        let cell = select(fixed * lattice.columns + along, along * lattice.columns + fixed, side >= 2u);
        let nearest = getExactHanischCellNearest(point, excludedRow, cell, nearestSquared);
        nearestSquared = nearest.x;
        found = found || nearest.y > 0.0;
      }
    }
    // Every unvisited cell lies beyond one of these four sides. No need to search farther
    // than the nearest event already found, or the censoring boundary if none was found.
    // Index assignment floors fl(fl(x-minimum)/cellWidth), not the exact real quotient.
    // Bound subtraction, division (including reciprocal compilation), plane construction and
    // final distance rounding by 32*f32-epsilon times the coordinate/extent magnitudes.
    // Subtraction costs <=epsilon*(|x|+|minimum|); quotient accuracy (2.5 ulps plus a
    // reciprocal multiply) costs <8*epsilon*columns cell widths. Reconstructing the plane
    // and subtracting the focus adds <4*epsilon*(|minimum|+columns*cellWidth+|focus|).
    // For included x, |x| <= |minimum|+columns*cellWidth. The factor 32 covers this sum,
    // including rounding the bound itself. If the magnitude sum overflows, no exit is proved.
    // This deliberately becomes loose under cancellation at a large origin; a nominal plane
    // alone is not a lower bound on coordinates assigned to the next cell (B1).
    let errorX = 0.000003814697265625 * (max(abs(lattice.minimumX), abs(point.x)) + f32(lattice.columns) * lattice.cellWidth);
    let errorY = 0.000003814697265625 * (max(abs(lattice.minimumY), abs(point.y)) + f32(lattice.rows) * lattice.cellHeight);
    var unvisitedDistance = border + lattice.cellWidth + lattice.cellHeight;
    if (firstColumn > 0u) { unvisitedDistance = min(unvisitedDistance, max(0.0, point.x - (lattice.minimumX + f32(firstColumn) * lattice.cellWidth) - errorX)); }
    if (lastColumn + 1u < lattice.columns) { unvisitedDistance = min(unvisitedDistance, max(0.0, lattice.minimumX + f32(lastColumn + 1u) * lattice.cellWidth - point.x - errorX)); }
    if (firstRow > 0u) { unvisitedDistance = min(unvisitedDistance, max(0.0, point.y - (lattice.minimumY + f32(firstRow) * lattice.cellHeight) - errorY)); }
    if (lastRow + 1u < lattice.rows) { unvisitedDistance = min(unvisitedDistance, max(0.0, lattice.minimumY + f32(lastRow + 1u) * lattice.cellHeight - point.y - errorY)); }
    // A zero/overflowed uncertainty bound cannot establish an early exit.
    if (unvisitedDistance > 0.0 && unvisitedDistance * unvisitedDistance > nearestSquared * 1.000001) { break; }
  }
  return vec2f(sqrt(nearestSquared), select(0.0, 1.0, found));
}`;
