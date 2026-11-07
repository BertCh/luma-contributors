// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A small synthetic city for the urban-form recipes: a grid of rectangular buildings separated by
// streets, with wall segments, heights and viewpoints in the street intersections.

export type World = {width: number; height: number};
export type Random = () => number;

export type Buildings = {
  count: number;
  columns: number;
  rows: number;
  /** (x0, y0, x1, y1) of every building. */
  rectangles: Float32Array;
  /** Wall segments, four per building, counter-clockwise. */
  segments: Float32Array;
  /** Owning building of every wall segment. */
  segmentBuildingIds: Uint32Array;
  heights: Float32Array;
};

export type CityLayout = {
  margin: number;
  cellWidth: number;
  cellHeight: number;
  columns: number;
  rows: number;
  /** Per-building size jitter in [0, 1), fixed so a slider only rescales. */
  jitter: Float32Array;
  heights: Float32Array;
};

export function createCityLayout(
  random: Random,
  world: World,
  columns: number,
  rows: number,
  margin = 0.04
): CityLayout {
  return {
    margin,
    cellWidth: (world.width - 2 * margin) / columns,
    cellHeight: (1 - 2 * margin) / rows,
    columns,
    rows,
    jitter: Float32Array.from({length: columns * rows * 2}, () => random()),
    heights: Float32Array.from({length: columns * rows}, () => 6 + 30 * random())
  };
}

/** Buildings that fill `fill` of their cell (the rest is street), with a little size jitter. */
export function getBuildings(layout: CityLayout, fill: number): Buildings {
  const count = layout.columns * layout.rows;
  const rectangles = new Float32Array(count * 4);
  const segments = new Float32Array(count * 16);
  for (let row = 0; row < layout.rows; row++) {
    for (let column = 0; column < layout.columns; column++) {
      const building = row * layout.columns + column;
      const widthShare = fill * (0.85 + 0.15 * layout.jitter[building * 2]);
      const heightShare = fill * (0.85 + 0.15 * layout.jitter[building * 2 + 1]);
      const centerX = layout.margin + (column + 0.5) * layout.cellWidth;
      const centerY = layout.margin + (row + 0.5) * layout.cellHeight;
      const x0 = centerX - (layout.cellWidth * widthShare) / 2;
      const x1 = centerX + (layout.cellWidth * widthShare) / 2;
      const y0 = centerY - (layout.cellHeight * heightShare) / 2;
      const y1 = centerY + (layout.cellHeight * heightShare) / 2;
      rectangles.set([x0, y0, x1, y1], building * 4);
      segments.set([x0, y0, x1, y0, x1, y0, x1, y1, x1, y1, x0, y1, x0, y1, x0, y0], building * 16);
    }
  }
  return {
    count,
    columns: layout.columns,
    rows: layout.rows,
    rectangles,
    segments,
    segmentBuildingIds: Uint32Array.from({length: count * 4}, (_, segment) =>
      Math.floor(segment / 4)
    ),
    heights: layout.heights
  };
}

/** Positions in the street intersections of the layout (between four buildings). */
export function getIntersections(layout: CityLayout): Float32Array {
  const points: number[] = [];
  for (let row = 1; row < layout.rows; row++) {
    for (let column = 1; column < layout.columns; column++) {
      points.push(
        layout.margin + column * layout.cellWidth,
        layout.margin + row * layout.cellHeight
      );
    }
  }
  return Float32Array.from(points);
}
