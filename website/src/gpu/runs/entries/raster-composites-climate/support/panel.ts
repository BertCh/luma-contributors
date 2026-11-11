// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A backing panel for chart insets: a quad in the theme background (alpha 0.88) with a thin frame
// in the muted theme color, drawn under the chart layers so lines and bars stay legible over a
// raster in both themes. The geometry is made once in `createInputs` (it needs the world size),
// uploaded in `build`, and turned into layers in `draw` with the theme colors.

import type {DemoTheme} from '../../../../demos/kit';
import type {PolygonsRunLayer, RunBuffer, RunBuilder, LinesRunLayer} from '../../../define-run';

export type PanelRectangle = readonly [number, number, number, number];

export type PanelData = {
  positions: Float32Array;
  indices: Uint32Array;
  owners: Uint32Array;
  values: Float32Array;
  frame: Float32Array;
  rectangleCount: number;
};

export type PanelBuffers = {
  positions: RunBuffer<'float32x2'>;
  indices: RunBuffer<'uint32'>;
  owners: RunBuffer<'uint32'>;
  values: RunBuffer<'float32'>;
  frame: RunBuffer<'float32x4'>;
  rectangleCount: number;
};

/** Quad and frame geometry for panels given as fractions of the view (x0, y0, x1, y1; y up). */
export function createPanelData(
  world: {width: number; height: number},
  rectangles: readonly PanelRectangle[]
): PanelData {
  const positions = new Float32Array(rectangles.length * 8);
  const indices = new Uint32Array(rectangles.length * 6);
  const frame = new Float32Array(rectangles.length * 16);
  rectangles.forEach(([left, bottom, right, top], index) => {
    const x0 = left * world.width;
    const y0 = bottom * world.height;
    const x1 = right * world.width;
    const y1 = top * world.height;
    positions.set([x0, y0, x1, y0, x1, y1, x0, y1], index * 8);
    indices.set(
      [0, 1, 2, 0, 2, 3].map(corner => corner + index * 4),
      index * 6
    );
    frame.set([x0, y0, x1, y0, x1, y0, x1, y1, x1, y1, x0, y1, x0, y1, x0, y0], index * 16);
  });
  return {
    positions,
    indices,
    owners: new Uint32Array(rectangles.length * 2),
    values: Float32Array.of(1),
    frame,
    rectangleCount: rectangles.length
  };
}

/** Uploads the panel geometry; call from `build` and return the buffers among the outputs. */
export function uploadPanel(builder: RunBuilder, data: PanelData): PanelBuffers {
  return {
    positions: builder.upload('panel-positions', 'float32x2', data.positions),
    indices: builder.upload('panel-indices', 'uint32', data.indices),
    owners: builder.upload('panel-owners', 'uint32', data.owners),
    values: builder.upload('panel-values', 'float32', data.values),
    frame: builder.upload('panel-frame', 'float32x4', data.frame),
    rectangleCount: data.rectangleCount
  };
}

/** The panel's quad and frame in theme colors; put them before the chart layers. */
export function getPanelLayers(
  panel: PanelBuffers,
  theme: DemoTheme
): [PolygonsRunLayer, LinesRunLayer] {
  return [
    {
      layer: 'polygons',
      positions: panel.positions,
      indices: panel.indices,
      owners: panel.owners,
      values: panel.values,
      triangleCount: panel.rectangleCount * 2,
      color: [theme.background[0], theme.background[1], theme.background[2], 0.88]
    },
    {
      layer: 'lines',
      segments: panel.frame,
      count: panel.rectangleCount * 4,
      color: [theme.muted[0], theme.muted[1], theme.muted[2], 0.9]
    }
  ];
}

/** An RGBA chart color from the theme, for bars, lines and dots. */
export function getThemeColor(
  color: readonly number[],
  alpha = 0.95
): [number, number, number, number] {
  return [color[0], color[1], color[2], alpha];
}

/** The panel rectangle around a chart rectangle: a small margin on every side. */
export function getPanelRectangle(chart: PanelRectangle): PanelRectangle {
  return [
    Math.max(0, chart[0] - 0.01),
    Math.max(0, chart[1] - 0.025),
    Math.min(1, chart[2] + 0.01),
    Math.min(1, chart[3] + 0.025)
  ];
}
