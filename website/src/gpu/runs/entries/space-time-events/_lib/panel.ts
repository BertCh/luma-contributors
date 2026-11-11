// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// An opaque backing panel for charts and strips drawn over a map: one quad as a polygons layer in
// the theme background at alpha 0.88, so bars and lines stay readable in both themes.

import type {DemoTheme} from '../../../../demos/kit';
import type {RunBuffer, RunBuilder, RunLayer} from '../../../define-run';

export type Panel = {
  positions: RunBuffer;
  indices: RunBuffer;
  owners: RunBuffer;
  values: RunBuffer;
};

/** Uploads the quad of `rectangle` ([x0, y0, x1, y1] as fractions of the world). */
export function uploadPanel(
  builder: RunBuilder,
  rectangle: readonly [number, number, number, number],
  world: {width: number; height: number},
  id = 'panel'
): Panel {
  const [x0, y0, x1, y1] = rectangle;
  const left = (x0 - 0.02) * world.width;
  const right = (x1 + 0.02) * world.width;
  const bottom = Math.max(0, y0 - 0.02) * world.height;
  const top = (y1 + 0.03) * world.height;
  return {
    positions: builder.upload(
      `${id}-positions`,
      'float32x2',
      Float32Array.of(left, bottom, right, bottom, right, top, left, top)
    ),
    indices: builder.upload(`${id}-indices`, 'uint32', Uint32Array.of(0, 1, 2, 0, 2, 3)),
    owners: builder.upload(`${id}-owners`, 'uint32', Uint32Array.of(0, 0)),
    values: builder.upload(`${id}-values`, 'float32', Float32Array.of(0))
  };
}

/** The panel layer; draw it before the chart. */
export function drawPanel(panel: Panel, theme: DemoTheme): RunLayer {
  return {
    layer: 'polygons',
    positions: panel.positions,
    indices: panel.indices,
    owners: panel.owners,
    values: panel.values,
    triangleCount: 2,
    color: [theme.background[0], theme.background[1], theme.background[2], 0.88]
  };
}
