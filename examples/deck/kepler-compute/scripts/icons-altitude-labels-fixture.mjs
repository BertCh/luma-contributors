// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';

export const ICON_ALTITUDE_ROWS = [[-73.995, 40.735, 0], [-73.965, 40.735, 6000], [-73.995, 40.765, -1000], [-73.965, 40.765, null], [-73.98, 40.75, NaN]];
export const CUSTOM_ICON_DEFINITIONS = [{id: 'custom-square', mesh: {positions: [[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]], cells: [[0,1,2],[0,2,3]]}}];

export function makeIconAltitudeTable() {
  return new arrow.Table({
    east: arrow.vectorFromArray(ICON_ALTITUDE_ROWS.map(row => Math.fround(row[0])), new arrow.Float32()),
    north: arrow.vectorFromArray(ICON_ALTITUDE_ROWS.map(row => Math.fround(row[1])), new arrow.Float32()),
    height: arrow.vectorFromArray(ICON_ALTITUDE_ROWS.map(row => row[2] === null ? null : Math.fround(row[2])), new arrow.Float32()),
    symbol: arrow.vectorFromArray(ICON_ALTITUDE_ROWS.map(() => CUSTOM_ICON_DEFINITIONS[0].id)),
    name: arrow.vectorFromArray(ICON_ALTITUDE_ROWS.map((_, index) => `row ${index}`))
  });
}
