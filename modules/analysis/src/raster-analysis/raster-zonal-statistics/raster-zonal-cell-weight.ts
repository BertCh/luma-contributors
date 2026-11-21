// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_EARTH_AUTHALIC_RADIUS_METERS} from '../../analysis-core/contracts/earth';
import {getSphericalCellAreaWGSL} from '../../analysis-core/kernels/spherical-cell-area';

/**
 * WGSL `fn getCellWeight(cell: u32) -> f32`: the spherical area of a geographic cell from the
 * `settings` binding (the `parameters` view), through `getCellArea` of the shared spherical cell-area pattern (which reads
 * `[originLongitude, originLatitude, cellWidth, cellHeight]` from `extent`). Non-finite or
 * non-positive areas weigh 0. Needs `isFiniteValue`.
 */
export function getCellWeightWGSL(
  width: number,
  height: number,
  parameterLayoutVersion: 1 | 2
): string {
  return `const ZONAL_GRID_WIDTH: u32 = ${width}u;
const ZONAL_GRID_HEIGHT: f32 = ${height}.0;
const ZONAL_PARAMETER_LAYOUT_VERSION: u32 = ${parameterLayoutVersion}u;
var<private> extent: array<f32, 4>;
const extentOffset: u32 = 0u;
${getSphericalCellAreaWGSL(GPU_EARTH_AUTHALIC_RADIUS_METERS)}
fn getCellWeight(cell: u32) -> f32 {
  if (ZONAL_PARAMETER_LAYOUT_VERSION == 2u) {
    let row = f32(cell / ZONAL_GRID_WIDTH);
    let rowStep = settings[settingsOffset + 4u];
    let originPoleDistance = settings[settingsOffset + 5u];
    let signedPoleStep = settings[settingsOffset + 6u] * rowStep;
    let rowDisplacement = row * signedPoleStep;
    let originPoleEdgeDistance = originPoleDistance - rowDisplacement;
    var middlePoleDistance = originPoleEdgeDistance - 0.5 * signedPoleStep;
    if (originPoleEdgeDistance > 90.0) {
      // Switch pole BEFORE adding the half-row. abs is also a magnitude gate: the compiler
      // cannot reassociate originPoleDistance into the 180-degree reference subtraction.
      // For valid latitude strips this reference distance is nonnegative.
      let oppositeReferenceDistance = abs(180.0 + rowDisplacement);
      let oppositeEdgeDistance = oppositeReferenceDistance - originPoleDistance;
      middlePoleDistance = oppositeEdgeDistance + 0.5 * signedPoleStep;
    }
    let area = abs(2.0 * SPHERE_RADIUS * SPHERE_RADIUS * settings[settingsOffset] * DEGREES_TO_RADIANS *
      sin(middlePoleDistance * DEGREES_TO_RADIANS) * sin(0.5 * rowStep * DEGREES_TO_RADIANS));
    return select(0.0, area, isFiniteValue(area) && area > 0.0);
  }
  let northEdge = settings[settingsOffset + 2u];
  // Signed row height: negative when rows advance south, which makes getCellArea negative.
  let rowHeight = (settings[settingsOffset + 3u] - northEdge) / ZONAL_GRID_HEIGHT;
  extent = array<f32, 4>(0.0, northEdge, settings[settingsOffset], rowHeight);
  let area = abs(getCellArea(cell / ZONAL_GRID_WIDTH));
  return select(0.0, area, isFiniteValue(area) && area > 0.0);
}`;
}
