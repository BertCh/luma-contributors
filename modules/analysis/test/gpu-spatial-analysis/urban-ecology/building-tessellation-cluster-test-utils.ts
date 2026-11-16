// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Fixed fixture width; the last grid building is replaced with a distant outlier. */
export const CLUSTERED_TESSELLATION_COLUMNS = 100;

export function createClusteredTessellationInput(buildingCount: number) {
  const segments = new Float32Array(buildingCount * 16);
  const owners = new Uint32Array(buildingCount * 4);
  for (let building = 0; building < buildingCount; building++) {
    const horizontal =
      building === buildingCount - 1 ? 1e6 : (building % CLUSTERED_TESSELLATION_COLUMNS) + 0.25;
    const vertical =
      building === buildingCount - 1
        ? 1e6
        : Math.floor(building / CLUSTERED_TESSELLATION_COLUMNS) + 0.25;
    const corners = [
      [horizontal, vertical],
      [horizontal + 0.125, vertical],
      [horizontal + 0.125, vertical + 0.125],
      [horizontal, vertical + 0.125]
    ];
    for (let edge = 0; edge < 4; edge++) {
      segments.set([...corners[edge], ...corners[(edge + 1) % 4]], building * 16 + edge * 4);
      owners[building * 4 + edge] = building;
    }
  }
  return {segments, owners};
}

/**
 * Independent closed form: adjacent buildings' sample coordinates end at k+.375 and start at
 * k+1.25, so their Voronoi boundary is k+.8125. Left/bottom cells meet the nearby study boundary
 * at zero. Only the last column, last row and outlier have cells extending far outside the
 * radius certificate. Every other building has four bounded sample cells.
 */
export function getClusteredTessellationOracle(buildingCount: number) {
  const rows = buildingCount / CLUSTERED_TESSELLATION_COLUMNS;
  return {
    status: {
      count: (CLUSTERED_TESSELLATION_COLUMNS - 1) * (rows - 1),
      requiredCount: buildingCount,
      overflow: 0,
      candidateOverflow: 1,
      invalidCount: 0
    },
    isCertified: (building: number) =>
      building % CLUSTERED_TESSELLATION_COLUMNS < CLUSTERED_TESSELLATION_COLUMNS - 1 &&
      Math.floor(building / CLUSTERED_TESSELLATION_COLUMNS) < rows - 1,
    bounds: (building: number) => {
      const column = building % CLUSTERED_TESSELLATION_COLUMNS;
      const row = Math.floor(building / CLUSTERED_TESSELLATION_COLUMNS);
      return {
        lower: [Math.max(0, column - 0.1875), Math.max(0, row - 0.1875)],
        upper: [column + 0.8125, row + 0.8125]
      };
    }
  };
}
