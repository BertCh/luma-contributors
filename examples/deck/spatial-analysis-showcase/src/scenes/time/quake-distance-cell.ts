// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Quake distance readers use the producer's signed, original-coordinate-frame cell edges. */
export const QUAKE_DISTANCE_CELL_WGSL = /* wgsl */ `
const NONE: u32 = 0xffffffffu;
fn getQuakeDistanceCell(position: f32, origin: f32, step: f32, inverseStep: f32) -> f32 {
  var cell = floor((position - origin) * inverseStep);
  let firstEdge = origin + cell * step;
  let nextEdge = origin + (cell + 1.0) * step;
  if (step > 0.0) {
    if (position < firstEdge) { cell -= 1.0; }
    else if (position >= nextEdge) { cell += 1.0; }
  } else {
    if (position > firstEdge) { cell -= 1.0; }
    else if (position <= nextEdge) { cell += 1.0; }
  }
  return cell;
}`;

export const QUAKE_EVENT_DISTANCE_WGSL = /* wgsl */ `
  let cellX = getQuakeDistanceCell(positions[positionsOffset + index * 2u], parameters[parametersOffset + 8u], parameters[parametersOffset + 11u], parameters[parametersOffset + 10u]);
  let cellY = getQuakeDistanceCell(positions[positionsOffset + index * 2u + 1u], parameters[parametersOffset + 9u], parameters[parametersOffset + 11u], parameters[parametersOffset + 10u]);
  var kilometers = -1.0;
  if (cellX >= 0.0 && cellY >= 0.0 && cellX < parameters[parametersOffset + 12u] && cellY < parameters[parametersOffset + 13u]) {
    let meters = distances[distancesOffset + u32(cellY) * u32(parameters[parametersOffset + 12u]) + u32(cellX)];
    if (meters < 3.0e38) {
      kilometers = meters * 0.001;
    }
  }
  distanceKm[distanceKmOffset + index] = kilometers;`;

export const QUAKE_LINKS_WGSL = /* wgsl */ `
  linkFade[linkFadeOffset + index] = 0.0;
  linkSegments[linkSegmentsOffset + index * 4u] = 0.0;
  linkSegments[linkSegmentsOffset + index * 4u + 1u] = 0.0;
  linkSegments[linkSegmentsOffset + index * 4u + 2u] = 0.0;
  linkSegments[linkSegmentsOffset + index * 4u + 3u] = 0.0;
  if (parameters[parametersOffset + 6u] < 0.5 || distanceKm[distanceKmOffset + index] < 0.0) {
    return;
  }
  if (depth[depthOffset + index] < parameters[parametersOffset + 7u]) {
    return;
  }
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let cellX = u32(getQuakeDistanceCell(x, parameters[parametersOffset + 8u], parameters[parametersOffset + 11u], parameters[parametersOffset + 10u]));
  let cellY = u32(getQuakeDistanceCell(y, parameters[parametersOffset + 9u], parameters[parametersOffset + 11u], parameters[parametersOffset + 10u]));
  let width = u32(parameters[parametersOffset + 12u]);
  let nearest = nearestCells[nearestCellsOffset + cellY * width + cellX];
  if (nearest == NONE) {
    return;
  }
  let size = parameters[parametersOffset + 11u];
  linkSegments[linkSegmentsOffset + index * 4u] = x;
  linkSegments[linkSegmentsOffset + index * 4u + 1u] = y;
  linkSegments[linkSegmentsOffset + index * 4u + 2u] = parameters[parametersOffset + 8u] + (f32(nearest % width) + 0.5) * size;
  linkSegments[linkSegmentsOffset + index * 4u + 3u] = parameters[parametersOffset + 9u] + (f32(nearest / width) + 0.5) * size;
  linkFade[linkFadeOffset + index] = 1.0;`;
