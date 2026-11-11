// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Chicago tracts drawn on the left of the canvas, leaving the right side for a chart. Shared by the
// runs of this group that classify, stratify or regionalize tracts.

import type {RunBuffer, RunBuilder, RunLayer} from '../../../define-run';
import {createTracts, type Random, type TractsInput, type World} from '../../../inputs';

/** Share of the canvas width the tract map occupies. */
export const MAP_SHARE = 0.6;
/** Plot rectangle of the chart on the right (fractions of the view, y up). */
export const CHART_RECTANGLE = [0.66, 0.16, 0.96, 0.84] as const;

/** All 156 sample tracts, fitted into the left part of the canvas. */
export function createTractMap(random: Random, world: World): TractsInput {
  return createTracts(random, {width: world.width * MAP_SHARE, height: world.height});
}

export type TractMapBuffers = {
  vertices: RunBuffer<'float32x2'>;
  indices: RunBuffer<'uint32'>;
  owners: RunBuffer<'uint32'>;
  centers: RunBuffer<'float32x2'>;
  outline: RunBuffer<'float32x4'>;
};

/** Uploads the tract mesh, centers and outline segments. */
export function uploadTractMap(builder: RunBuilder, tracts: TractsInput): TractMapBuffers {
  return {
    vertices: builder.upload('tract-vertices', 'float32x2', tracts.vertices),
    indices: builder.upload('tract-indices', 'uint32', tracts.triangleIndices),
    owners: builder.upload('tract-owners', 'uint32', tracts.triangleOwners),
    centers: builder.upload('tract-centers', 'float32x2', tracts.centers),
    outline: builder.upload('tract-outline', 'float32x4', tracts.outline)
  };
}

/** Fills the tracts with one value per tract. */
export function drawTractValues(
  map: TractMapBuffers,
  tracts: TractsInput,
  values: RunBuffer<'float32' | 'uint32'>,
  look: Partial<Extract<RunLayer, {layer: 'polygons'}>> = {}
): RunLayer {
  return {
    layer: 'polygons',
    positions: map.vertices,
    indices: map.indices,
    owners: map.owners,
    values,
    triangleCount: tracts.triangleCount,
    ...look
  };
}

/** Thin tract boundaries. */
export function drawTractOutline(map: TractMapBuffers, tracts: TractsInput): RunLayer {
  return {layer: 'lines', context: true, segments: map.outline, count: tracts.outlineCount};
}

/** Class of each value by equal-count quantiles, `0 .. classCount - 1`. */
export function getQuantileClasses(values: ArrayLike<number>, classCount: number): Uint32Array {
  const order = Array.from({length: values.length}, (_, row) => row).sort(
    (first, second) => values[first] - values[second]
  );
  const classes = new Uint32Array(values.length);
  order.forEach((row, rank) => {
    classes[row] = Math.min(classCount - 1, Math.floor((rank * classCount) / values.length));
  });
  return classes;
}
