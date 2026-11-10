// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPolygonRasterization: Chicago tract polygons are burned into a grid of
// zone ids; the boundary cells are flagged. The grid width is a compile-time prop (the run
// rebuilds); the boundary scale rewrites the polygon vertices. Zone ids are graph-coloured so
// neighbouring tracts differ, boundary cells are outlined, and the pointer probes a cell's zone.

import {
  getGPUPolygonRasterizationParameterValues,
  GPUPolygonRasterization
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun, type RunLayer} from '../../define-run';
import {createTracts} from '../../inputs';
import {parameters} from './polygon-rasterization.parameters';
import {colorLabels, createLabelColors, getCellAt, toUint32Array} from './support/gestures';
import {getLabelCategories, HIGHLIGHT_CATEGORIES} from './support/labels';
import {makeWorldGrid} from './support/bands';
import {createTractFeatures} from './support/tract-zones';

export default defineOperationRun({
  parameters,
  caption: 'Polygons in; a raster of zone ids out. Cells without a polygon stay empty.',
  createInputs: ({random, world}) => ({
    world,
    features: createTractFeatures(createTracts(random, world, 7)),
    focus: {zone: -1}
  }),
  build(builder, {world, features}, values) {
    const {grid, columns, rows, cellCount} = makeWorldGrid(world, values.gridWidth);
    const positions = builder.upload('positions', 'float32x2', features.positions);
    const featureOffsets = builder.upload('feature-offsets', 'uint32', features.featureOffsets);
    const polygonOffsets = builder.upload('polygon-offsets', 'uint32', features.polygonOffsets);
    const ringOffsets = builder.upload('ring-offsets', 'uint32', features.ringOffsets);
    const extent = builder.upload(
      'extent',
      'float32',
      getGPUPolygonRasterizationParameterValues(grid)
    );
    const zones = builder.allocate('zones', 'uint32', cellCount);
    const boundary = builder.allocate('boundary', 'uint32', cellCount);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    const outline = builder.upload('outline', 'float32x4', features.outline);
    builder.graph.add(
      new GPUPolygonRasterization({
        grid,
        parameters: extent.view,
        positions: positions.view,
        featureOffsets: featureOffsets.view,
        polygonOffsets: polygonOffsets.view,
        ringOffsets: ringOffsets.view,
        crossingCapacity: 4 * cellCount,
        output: {zones: zones.view, boundary: boundary.view, status: {overflow: overflow.view}}
      })
    );
    const hover = builder.allocate('hover', 'uint32', cellCount);
    return {
      outputs: {zones, boundary, outline, hover, columns, rows},
      update: current => {
        const moved = features.getPositions(current.boundaryScale);
        positions.write(moved);
        outline.write(features.getOutline(moved));
      }
    };
  },
  rebuildOn: ['gridWidth'],
  async afterRun({inputs, outputs, upload}) {
    const zones = toUint32Array(await outputs.zones.buffer.readAsync());
    const labels = Uint32Array.from(zones, zone =>
      zone < inputs.features.featureCount ? zone + 1 : 0
    );
    // A rebuild clears the highlight; rewrite it from the last probed zone.
    if (inputs.focus.zone >= 0) {
      outputs.hover.write(Uint32Array.from(zones, zone => (zone === inputs.focus.zone ? 1 : 0)));
    }
    return {
      display: upload(
        'display',
        'uint32',
        colorLabels(labels, createLabelColors(labels, outputs.columns, outputs.rows))
      )
    };
  },
  interaction: {
    kind: 'probe',
    hint: 'probe a cell',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, outputs, values, world, peek}) => {
      const zones = peek(outputs.zones);
      const boundary = peek(outputs.boundary);
      if (!zones) return {readout: 'polygon rasterization'};
      const zone = zones[getCellAt(outputs, world, pointer).index];
      inputs.focus.zone = zone < inputs.features.featureCount ? zone : -1;
      if (zone >= inputs.features.featureCount) {
        outputs.hover.write(new Uint32Array(zones.length));
        return {readout: 'no polygon in this cell'};
      }
      let cells = 0;
      let edge = 0;
      const mask = new Uint32Array(zones.length);
      for (let index = 0; index < zones.length; index++) {
        if (zones[index] !== zone) continue;
        mask[index] = 1;
        cells++;
        if (boundary?.[index]) edge++;
      }
      outputs.hover.write(mask);
      const ring = inputs.features
        .getPositions(values.boundaryScale)
        .slice(inputs.features.ringOffsets[zone] * 2, inputs.features.ringOffsets[zone + 1] * 2);
      return {
        glyphs: [{shape: 'polyline', closed: true, points: ring}],
        readout: `zone ${zone} | ${cells} cells | ${edge} boundary`
      };
    }
  },
  draw: ({outputs, extra, theme}): RunLayer[] => [
    ...(extra
      ? [
          {
            layer: 'raster' as const,
            values: extra.display,
            columns: outputs.columns,
            rows: outputs.rows,
            ramp: 'categorical' as const,
            categories: getLabelCategories(theme)
          }
        ]
      : []),
    {
      layer: 'raster',
      values: outputs.boundary,
      columns: outputs.columns,
      rows: outputs.rows,
      ramp: 'categorical',
      categories: [
        [0, 0, 0, 0],
        [theme.ink[0], theme.ink[1], theme.ink[2], 0.55]
      ]
    },
    {
      layer: 'raster',
      values: outputs.hover,
      columns: outputs.columns,
      rows: outputs.rows,
      ramp: 'categorical',
      categories: HIGHLIGHT_CATEGORIES
    },
    {layer: 'lines', context: true, segments: outputs.outline, color: [...theme.ink, 0.9]}
  ]
});
