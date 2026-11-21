// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

type Extent = readonly [number, number, number, number];
type Vector = readonly [number, number];

/** Binary64 bilinear sampler of the cell-centre convention, `undefined` where invalid. */
function sampleReference(
  field: Float32Array,
  width: number,
  height: number,
  extent: Extent,
  x: number,
  y: number
): Vector | undefined {
  const localX = (x - extent[0]) / extent[2];
  const localY = (y - extent[1]) / extent[3];
  if (!(localX >= 0 && localY >= 0 && localX < width && localY < height)) {
    return undefined;
  }
  const gridX = Math.min(Math.max(localX - 0.5, 0), width - 1);
  const gridY = Math.min(Math.max(localY - 0.5, 0), height - 1);
  const column0 = Math.min(Math.floor(gridX), width - 1);
  const row0 = Math.min(Math.floor(gridY), height - 1);
  const column1 = Math.min(column0 + 1, width - 1);
  const row1 = Math.min(row0 + 1, height - 1);
  const fractionX = gridX - column0;
  const fractionY = gridY - row0;
  const read = (column: number, row: number): Vector => [
    field[2 * (row * width + column)],
    field[2 * (row * width + column) + 1]
  ];
  const corners = [
    read(column0, row0),
    read(column1, row0),
    read(column0, row1),
    read(column1, row1)
  ];
  if (corners.some(corner => Number.isNaN(corner[0]) || Number.isNaN(corner[1]))) {
    return undefined;
  }
  const mix = (axis: 0 | 1): number => {
    const bottom = corners[0][axis] * (1 - fractionX) + corners[1][axis] * fractionX;
    const top = corners[2][axis] * (1 - fractionX) + corners[3][axis] * fractionX;
    return bottom * (1 - fractionY) + top * fractionY;
  };
  return [mix(0), mix(1)];
}

/** Binary64 reference of the flow map: midpoint steps, linear time interpolation of two slices. */
export function integrateReference(options: {
  start: Float32Array;
  end: Float32Array;
  width: number;
  height: number;
  extent: Extent;
  seeds: readonly Vector[];
  startTime: number;
  timeStep: number;
  sliceTimes: Vector;
  stepCount: number;
}): (Vector | undefined)[] {
  const {start, end, width, height, extent, timeStep, sliceTimes, stepCount} = options;
  const sample = (x: number, y: number, time: number): Vector | undefined => {
    const weight =
      sliceTimes[0] === sliceTimes[1]
        ? 0
        : Math.min(Math.max((time - sliceTimes[0]) / (sliceTimes[1] - sliceTimes[0]), 0), 1);
    const first = sampleReference(start, width, height, extent, x, y);
    const second = sampleReference(end, width, height, extent, x, y);
    if (!first || !second) {
      return undefined;
    }
    return [
      first[0] * (1 - weight) + second[0] * weight,
      first[1] * (1 - weight) + second[1] * weight
    ];
  };
  return options.seeds.map((seed): Vector | undefined => {
    let [x, y] = seed;
    for (let step = 0; step < stepCount; step++) {
      const time = options.startTime + step * timeStep;
      const first = sample(x, y, time);
      if (!first) {
        return undefined;
      }
      const second = sample(
        x + 0.5 * timeStep * first[0],
        y + 0.5 * timeStep * first[1],
        time + 0.5 * timeStep
      );
      if (!second) {
        return undefined;
      }
      x += timeStep * second[0];
      y += timeStep * second[1];
      if (!sample(x, y, time + timeStep)) {
        return undefined;
      }
    }
    return [x, y];
  });
}
