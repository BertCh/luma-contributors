// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterPatchMetrics on a patch label raster: one area and one perimeter per patch (the bars:
// areas bottom left, perimeters bottom right, one bar per patch label). Labels are drawn behind.
// The pointer probes a patch: it is highlighted and its own area and perimeter bars light up.
// The patch count rewrites the labels; `countRasterBorder` is a construction prop.

import {GPURasterPatchMetrics} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createElevationRaster} from '../../inputs';
import {parameters} from './patch-metrics.parameters';
import {drawRaster, makeGridOfRaster} from './support/bands';
import {colorLabels, createLabelColors, getCellAt} from './support/gestures';
import {getLabelCategories, HIGHLIGHT_CATEGORIES, maskLabel, readFloat32} from './support/labels';
import {createPatchSeeds, getPatchLabels} from './support/patches';

const PATCH_CAPACITY = 41;
const AREA_RECTANGLE = [0.03, 0.03, 0.48, 0.3] as const;
const PERIMETER_RECTANGLE = [0.52, 0.03, 0.97, 0.3] as const;

export default defineOperationRun({
  parameters,
  rebuildOn: ['countRasterBorder'],
  caption: 'Patch labels in (graph-coloured); area and perimeter of every patch out.',
  createInputs: ({random, world}) => {
    const raster = createElevationRaster(world, 48);
    return {
      ...raster,
      seeds: createPatchSeeds(random, raster.columns, raster.rows, 40, raster.values),
      labels: new Uint32Array(raster.values.length) as Uint32Array,
      // Bar-chart ranges, taken from the latest readback so the highlighted bar matches its row.
      maximumArea: 0.3,
      maximumPerimeter: 3,
      focus: {label: 0}
    };
  },
  build(builder, inputs, values) {
    const cellCount = inputs.values.length;
    const labels = builder.upload('labels', 'uint32', new Uint32Array(cellCount));
    const colored = builder.allocate('colored', 'uint32', cellCount);
    const hover = builder.allocate('hover', 'uint32', cellCount);
    const hoverArea = builder.allocate('hover-area', 'float32', PATCH_CAPACITY);
    const hoverPerimeter = builder.allocate('hover-perimeter', 'float32', PATCH_CAPACITY);
    const areas = builder.allocate('areas', 'float32', PATCH_CAPACITY);
    const perimeters = builder.allocate('perimeters', 'float32', PATCH_CAPACITY);
    const pixelCounts = builder.allocate('pixel-counts', 'uint32', PATCH_CAPACITY);
    builder.graph.add(
      new GPURasterPatchMetrics({
        grid: makeGridOfRaster(inputs),
        labels: labels.view,
        countRasterBorder: values.countRasterBorder === 1,
        output: {areas: areas.view, perimeters: perimeters.view, pixelCounts: pixelCounts.view}
      })
    );
    return {
      outputs: {labels, colored, hover, hoverArea, hoverPerimeter, areas, perimeters, pixelCounts},
      update: current => {
        const patchLabels = getPatchLabels(inputs.seeds, current.patchCount);
        inputs.labels = patchLabels;
        labels.write(patchLabels);
        colored.write(
          colorLabels(patchLabels, createLabelColors(patchLabels, inputs.columns, inputs.rows))
        );
        // A rebuild clears the highlight; rewrite it from the last probed patch.
        hover.write(maskLabel(patchLabels, inputs.focus.label));
      }
    };
  },
  // The highlighted bars are rewritten from the metrics after every run.
  async afterRun({inputs, outputs}) {
    const label = inputs.focus.label;
    const areas = await readFloat32(outputs.areas);
    const perimeters = await readFloat32(outputs.perimeters);
    inputs.maximumArea = Math.max(1e-6, ...areas);
    inputs.maximumPerimeter = Math.max(1e-6, ...perimeters);
    const area = new Float32Array(PATCH_CAPACITY);
    const perimeter = new Float32Array(PATCH_CAPACITY);
    if (label > 0 && label < PATCH_CAPACITY) {
      area[label] = areas[label];
      perimeter[label] = perimeters[label];
    }
    outputs.hoverArea.write(area);
    outputs.hoverPerimeter.write(perimeter);
  },
  interaction: {
    kind: 'probe',
    hint: 'probe a patch',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, outputs, world, peek}) => {
      const label = inputs.labels[getCellAt(inputs, world, pointer).index];
      inputs.focus.label = label;
      outputs.hover.write(maskLabel(inputs.labels, label));
      const areas = peek(outputs.areas);
      const perimeters = peek(outputs.perimeters);
      const pixelCounts = peek(outputs.pixelCounts);
      if (areas && perimeters) {
        inputs.maximumArea = Math.max(1e-6, ...areas);
        inputs.maximumPerimeter = Math.max(1e-6, ...perimeters);
      }
      const area = new Float32Array(PATCH_CAPACITY);
      const perimeter = new Float32Array(PATCH_CAPACITY);
      if (areas && perimeters && label < PATCH_CAPACITY) {
        area[label] = areas[label];
        perimeter[label] = perimeters[label];
      }
      outputs.hoverArea.write(area);
      outputs.hoverPerimeter.write(perimeter);
      const cells = pixelCounts && label < PATCH_CAPACITY ? pixelCounts[label] : 0;
      const edges = perimeters && label < PATCH_CAPACITY ? perimeters[label] / inputs.cellSize : 0;
      return {
        readout: `patch ${label} | area ${cells} cells | perimeter ${edges.toFixed(0)} edges`
      };
    }
  },
  draw: ({inputs, outputs, theme}) => [
    drawRaster(inputs, outputs.colored, {
      ramp: 'categorical',
      categories: getLabelCategories(theme),
      alpha: 0.85
    }),
    drawRaster(inputs, outputs.hover, {ramp: 'categorical', categories: HIGHLIGHT_CATEGORIES}),
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.areas,
      ramp: 'sequential',
      range: [0, inputs.maximumArea],
      rectangle: AREA_RECTANGLE
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.hoverArea,
      range: [0, inputs.maximumArea],
      color: [1, 1, 1, 1],
      rectangle: AREA_RECTANGLE
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.perimeters,
      range: [0, inputs.maximumPerimeter],
      ramp: 'sequential',
      rectangle: PERIMETER_RECTANGLE
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.hoverPerimeter,
      range: [0, inputs.maximumPerimeter],
      color: [1, 1, 1, 1],
      rectangle: PERIMETER_RECTANGLE
    }
  ]
});
