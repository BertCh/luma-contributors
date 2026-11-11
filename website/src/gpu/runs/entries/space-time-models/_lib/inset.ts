// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The backing panel of a chart inset or timeline strip over a raster map: two triangles in the theme
// background at 0.88, so a chart reads over any map and in both themes.

import type {RunBuffer, RunBuilder, RunLayer} from '../../../define-run';
import type {DemoTheme} from '../../../../demos/kit';

export type InsetRectangle = {x0: number; y0: number; x1: number; y1: number};
export type InsetBacking = {
  positions: RunBuffer;
  indices: RunBuffer;
  owners: RunBuffer;
  values: RunBuffer;
};

export function createInsetBacking(
  builder: RunBuilder,
  id: string,
  rectangle: InsetRectangle
): InsetBacking {
  return {
    positions: builder.upload(`${id}-positions`, 'float32x2', getRectangleCorners(rectangle)),
    indices: builder.upload(`${id}-indices`, 'uint32', Uint32Array.of(0, 1, 2, 0, 2, 3)),
    owners: builder.upload(`${id}-owners`, 'uint32', Uint32Array.of(0, 0)),
    values: builder.upload(`${id}-values`, 'float32', Float32Array.of(1))
  };
}

export function getRectangleCorners(rectangle: InsetRectangle): Float32Array {
  const {x0, y0, x1, y1} = rectangle;
  return Float32Array.of(x0, y0, x1, y0, x1, y1, x0, y1);
}

export function getInsetBackingLayer(backing: InsetBacking, theme: DemoTheme): RunLayer {
  return {
    layer: 'polygons',
    context: true,
    positions: backing.positions,
    indices: backing.indices,
    owners: backing.owners,
    values: backing.values,
    triangleCount: 2,
    color: [theme.background[0], theme.background[1], theme.background[2], 0.88]
  };
}
