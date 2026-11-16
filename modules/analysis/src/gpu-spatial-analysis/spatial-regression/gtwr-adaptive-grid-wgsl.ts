// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_GTWR_MAXIMUM_NEIGHBOR_COUNT} from './gtwr-bandwidth-search';

/** Square root of the smallest normal f32: squared distances below this can flush to zero. @internal */
export const GPU_GTWR_MINIMUM_CANDIDATE_RADIUS = 1.0842021724855044e-19;
/** Smallest normal f32: a timeScale * deltaTime intermediate below it can flush to zero. @internal */
export const GPU_GTWR_MINIMUM_NORMAL_VALUE = 1.1754943508222875e-38;

/**
 * Exact kth usable space-time distance through expanding grid shells. The candidate list has
 * the reference route's bounded size. A conservative lower bound to every unvisited slab stops
 * the walk; degenerate/overflowing bounds disable early exit and shells eventually cover the
 * entire grid. No per-shell shader, dispatch or scratch is generated.
 *
 * @internal
 */
export function getGTWRGridAdaptiveBandwidthWGSL(): string {
  return /* wgsl */ `
// Distance to a not-yet-visited slab, padded by a whole cell for float bin/edge rounding.
fn getUnvisitedAxisGap(origin: f32, lower: f32, upper: f32, firstCell: i32, lastCell: i32) -> f32 {
  let width = (upper - lower) / f32(GRID_DIMENSION);
  if (!isFiniteBits(width) || !(width > 0.0)) { return 0.0; }
  var gap = 3.402823466e+38;
  if (firstCell > 0) {
    gap = min(gap, origin - (lower + f32(firstCell) * width));
  }
  if (lastCell < i32(GRID_DIMENSION) - 1) {
    gap = min(gap, lower + f32(lastCell + 1) * width - origin);
  }
  return max(0.0, gap - width - abs(origin) * 1.0e-6);
}

fn getAdaptiveBandwidth(focal: u32, requested: f32, timeScale: f32) -> f32 {
  if (!isFiniteBits(requested) || requested < 1.0 || requested != round(requested)
      || requested > ${GPU_GTWR_MAXIMUM_NEIGHBOR_COUNT}.0 || !isFiniteBits(timeScale) || timeScale < 0.0) {
    return getNaN();
  }
  let count = u32(requested);
  if (count > cellOffsets[cellOffsetsOffset + GRID_DIMENSION * GRID_DIMENSION * GRID_DIMENSION]) { return getNaN(); }
  let originBase = recordsOffset + focal * RECORD_WORDS;
  let originX = records[originBase];
  if (!isFiniteBits(originX)) { return getNaN(); }
  let originY = records[originBase + 1u];
  let originTime = records[originBase + 2u];
  let minimumX = spatialBounds[spatialBoundsOffset];
  let maximumX = spatialBounds[spatialBoundsOffset + 2u];
  let minimumY = spatialBounds[spatialBoundsOffset + 1u];
  let maximumY = spatialBounds[spatialBoundsOffset + 3u];
  let minimumTime = temporalBounds[temporalBoundsOffset];
  let maximumTime = temporalBounds[temporalBoundsOffset + 2u];
  let centerColumn = i32(getGTWRBinIndex(originX, minimumX, maximumX));
  let centerRow = i32(getGTWRBinIndex(originY, minimumY, maximumY));
  var centerTime = 0;
  var lastTime = 0;
  if (timeScale > 0.0 && maximumTime > minimumTime) {
    centerTime = i32(getGTWRBinIndex(originTime, minimumTime, maximumTime));
    lastTime = i32(GRID_DIMENSION) - 1;
  }
  let lastColumnCell = select(0, i32(GRID_DIMENSION) - 1, maximumX > minimumX);
  let lastRowCell = select(0, i32(GRID_DIMENSION) - 1, maximumY > minimumY);
  let lastRing = max(max(lastColumnCell, lastRowCell), lastTime);
  var distances: array<f32, ${GPU_GTWR_MAXIMUM_NEIGHBOR_COUNT}>;
  for (var neighbor = 0u; neighbor < count; neighbor++) { distances[neighbor] = 3.402823466e+38; }
  for (var ring = 0; ring <= lastRing; ring++) {
    let firstColumn = max(0, centerColumn - ring);
    let lastColumn = min(lastColumnCell, centerColumn + ring);
    let firstRow = max(0, centerRow - ring);
    let lastRow = min(lastRowCell, centerRow + ring);
    let firstTimeCell = max(0, centerTime - ring);
    let lastTimeCell = min(lastTime, centerTime + ring);
    for (var timeCell = firstTimeCell; timeCell <= lastTimeCell; timeCell++) {
      let timeEdge = timeScale > 0.0 && abs(timeCell - centerTime) == ring;
      for (var gridRow = firstRow; gridRow <= lastRow; gridRow++) {
        let fullColumns = timeEdge || abs(gridRow - centerRow) == ring;
        for (var side = 0; side < 2; side++) {
          var columnLow = firstColumn;
          var columnHigh = lastColumn;
          if (fullColumns) {
            if (side == 1) { continue; }
          } else {
            let column = select(centerColumn + ring, centerColumn - ring, side == 0);
            if (column < 0 || column > lastColumnCell) { continue; }
            columnLow = column;
            columnHigh = column;
          }
          let cellBase = (u32(timeCell) * GRID_DIMENSION + u32(gridRow)) * GRID_DIMENSION;
          let firstSlot = cellOffsets[cellOffsetsOffset + cellBase + u32(columnLow)];
          let endSlot = cellOffsets[cellOffsetsOffset + cellBase + u32(columnHigh) + 1u];
          for (var slot = firstSlot; slot < endSlot; slot++) {
            let row = sortedRows[sortedRowsOffset + slot];
            let base = recordsOffset + row * RECORD_WORDS;
            let deltaX = records[base] - originX;
            let deltaY = records[base + 1u] - originY;
            let deltaTime = records[base + 2u] - originTime;
            let distance = sqrt(deltaX * deltaX + deltaY * deltaY + timeScale * deltaTime * deltaTime);
            if (distance < distances[count - 1u]) {
              var position = count - 1u;
              loop {
                if (position == 0u || distances[position - 1u] <= distance) { break; }
                distances[position] = distances[position - 1u];
                position--;
              }
              distances[position] = distance;
            }
          }
        }
      }
    }
    var gap = 3.402823466e+38;
    if (firstColumn > 0 || lastColumn < lastColumnCell) {
      gap = min(gap, getUnvisitedAxisGap(originX, minimumX, maximumX, firstColumn, lastColumn));
    }
    if (firstRow > 0 || lastRow < lastRowCell) {
      gap = min(gap, getUnvisitedAxisGap(originY, minimumY, maximumY, firstRow, lastRow));
    }
    if (firstTimeCell > 0 || lastTimeCell < lastTime) {
      let rawTimeGap = getUnvisitedAxisGap(originTime, minimumTime, maximumTime, firstTimeCell, lastTimeCell);
      var timeGap = 0.0;
      if (rawTimeGap > max(${GPU_GTWR_MINIMUM_CANDIDATE_RADIUS}, ${GPU_GTWR_MINIMUM_NORMAL_VALUE} / timeScale)) {
        timeGap = rawTimeGap * sqrt(timeScale);
      }
      gap = min(gap, timeGap);
    }
    // Never stop on a distance that could be affected by subnormal squared-distance flushing.
    gap = max(0.0, gap - ${2 * GPU_GTWR_MINIMUM_CANDIDATE_RADIUS});
    if (distances[count - 1u] < gap) { break; }
  }
  return distances[count - 1u] * 1.0000001;
}`;
}
