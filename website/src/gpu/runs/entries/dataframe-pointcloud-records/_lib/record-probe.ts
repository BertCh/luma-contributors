// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Pointer support for the computation-record runs. Their pictures are CPU geometry that `afterRun`
// uploads, and `resolve` cannot see that result, so each run's `afterRun` also leaves a plain CPU
// description of what it drew in `inputs.latest`. This file has the hit tests and the dimming
// overlay shared by those runs.

import type {RunLayer} from '../../../define-run';
import type {RunGlyph} from '../../../interaction';
import {ShapeBatch, type Rectangle} from '../../../record-demo';

export type Box = Rectangle;

/** The box nearest to (x, y): the one containing it, or the closest by distance to its edge. */
export function findBoxAt<Item extends Box>(boxes: readonly Item[], x: number, y: number) {
  let nearest: Item | null = null;
  let nearestDistance = Infinity;
  for (const box of boxes) {
    const distance = Math.hypot(
      Math.max(box.x0 - x, 0, x - box.x1),
      Math.max(box.y0 - y, 0, y - box.y1)
    );
    if (distance < nearestDistance) {
      nearest = box;
      nearestDistance = distance;
    }
  }
  return nearest;
}

export function getBoxGlyph(box: Box, dashed = false): RunGlyph {
  return {shape: 'box', minimum: [box.x0, box.y0], maximum: [box.x1, box.y1], dashed};
}

export function getCenter(box: Box): [number, number] {
  return [(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2];
}

/** The column of a grid of `columns` equal columns over [x0, x1] under world x, clamped. */
export function getColumnAt(x0: number, x1: number, columns: number, x: number): number {
  return Math.min(columns - 1, Math.max(0, Math.floor(((x - x0) / (x1 - x0)) * columns)));
}

/** A rectangle over `area`, uploaded once, that `getDimLayer` draws to dim part of a picture. */
export function createDimShape(
  upload: Parameters<ShapeBatch['uploadShapes']>[0],
  id: string,
  area: Box
) {
  const batch = new ShapeBatch();
  batch.addRectangle(area, 0);
  return batch.uploadShapes(upload, id);
}

/** The dimming overlay: the background at 72% over everything right of `curtain`. */
export function getDimLayer(
  shape: ReturnType<typeof createDimShape>,
  background: readonly number[],
  curtain: number
): RunLayer {
  return {
    layer: 'polygons',
    positions: shape.positions,
    indices: shape.indices,
    owners: shape.owners,
    values: shape.categories,
    triangleCount: shape.triangleCount,
    categories: [[background[0], background[1], background[2]]],
    alpha: 0.72,
    clip: {side: 'right', x: curtain}
  };
}
