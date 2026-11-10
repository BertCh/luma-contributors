// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterWatershedSegmentation: basins flood upward from marker cells (the lowest local minima of
// the surface) and meet along ridges. Each basin gets the label of its marker. The number of
// markers rewrites the marker raster; method and connectivity are construction props.

import {GPURasterWatershedSegmentation} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createElevationRaster} from '../../inputs';
import {parameters} from './watershed-segmentation.parameters';
import {
  colorLabels,
  createLabelColors,
  getCellAt,
  getCellCenter,
  toUint32Array
} from './support/gestures';
import {getLabelCategories} from './support/labels';
import {drawRaster, makeGridOfRaster, uploadBand} from './support/bands';
import {findLocalMinima} from './support/minima';

export default defineOperationRun({
  parameters,
  rebuildOn: ['method', 'connectivity'],
  caption: 'A surface and basin markers in; the basin of every cell out.',
  createInputs: ({world}) => {
    const raster = createElevationRaster(world, 54);
    // The pointer adds one more marker, a pointer-owned input; it starts at the rest position.
    return {
      ...raster,
      minima: findLocalMinima(raster),
      extra: [Math.floor(raster.columns * 0.5), Math.floor(raster.rows * 0.5)] as [number, number]
    };
  },
  build(builder, inputs, values) {
    const cellCount = inputs.values.length;
    const surface = uploadBand(builder, inputs, 'surface', inputs.values);
    const markers = builder.allocate('markers', 'uint32', cellCount);
    const labels = builder.allocate('labels', 'uint32', cellCount);
    const converged = builder.allocate('converged', 'uint32', 1);
    builder.graph.add(
      new GPURasterWatershedSegmentation({
        grid: makeGridOfRaster(inputs),
        input: surface.band,
        markers: markers.view,
        connectivity: values.connectivity === 1 ? 8 : 4,
        method: values.method === 1 ? 'minimum-spanning-forest' : 'fifo',
        output: {labels: labels.view, status: {converged: converged.view}}
      })
    );
    return {
      outputs: {surface: surface.buffer, labels},
      update: current => {
        const data = new Uint32Array(cellCount);
        inputs.minima.slice(0, current.markerCount).forEach((cell, index) => {
          data[cell] = index + 1;
        });
        data[inputs.extra[1] * inputs.columns + inputs.extra[0]] = current.markerCount + 1;
        markers.write(data);
      }
    };
  },
  // Basin labels are graph-coloured after the run, so neighbouring basins never share a colour.
  async afterRun({inputs, outputs, upload}) {
    const labels = toUint32Array(await outputs.labels.buffer.readAsync());
    return {
      display: upload(
        'display',
        'uint32',
        colorLabels(labels, createLabelColors(labels, inputs.columns, inputs.rows))
      )
    };
  },
  // The pointer places one extra marker; its basin floods out and splits its neighbours.
  interaction: {
    kind: 'place',
    hint: 'place a marker',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, outputs, values, world, peek}) => {
      const {column, row} = getCellAt(inputs, world, pointer);
      inputs.extra = [column, row];
      const labels = peek(outputs.labels);
      let basin = 0;
      if (labels) for (const label of labels) if (label === values.markerCount + 1) basin++;
      return {
        rerun: true,
        glyphs: [{shape: 'crosshair', at: getCellCenter(inputs, world, column, row), size: 10}],
        readout: `marker at [${column}, ${row}] | basin ${basin} cells`
      };
    }
  },
  draw: ({inputs, outputs, extra, theme}) => [
    drawRaster(inputs, outputs.surface, {context: true, range: [0, 1]}),
    ...(extra
      ? [
          drawRaster(inputs, extra.display, {
            ramp: 'categorical',
            categories: getLabelCategories(theme),
            alpha: 0.75,
            hillshade: 0.4
          })
        ]
      : [])
  ]
});
