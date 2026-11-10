// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterSieve on a patch label raster: patches smaller than `minimumPixels` are removed (set
// to background) or merged into their largest neighbor. The size threshold is a per-frame
// parameter; mode and connectivity are construction props (the run rebuilds).

import {
  getGPURasterSieveParameterValues,
  GPURasterSieve
} from '@luma-contributors/analysis/raster-analysis';
import {bindAxes} from '../../interaction';
import {defineOperationRun} from '../../define-run';
import {createElevationRaster} from '../../inputs';
import {parameters} from './sieve.parameters';
import {drawRaster, makeGridOfRaster} from './support/bands';
import {colorLabels, createLabelColors, getCellAt, toUint32Array} from './support/gestures';
import {
  countLabelCells,
  getLabelCategories,
  HIGHLIGHT_CATEGORIES,
  maskLabel
} from './support/labels';
import {createPatchSeeds, getPatchLabels} from './support/patches';

const PATCH_COUNT = 90;

const PAD = bindAxes(parameters, {
  kind: 'level',
  hint: 'raise the size threshold',
  x: {parameter: 'minimumPixels', scale: 'logarithmic'}
});

export default defineOperationRun({
  parameters,
  rebuildOn: ['mode', 'connectivity'],
  caption:
    'Patch labels in (graph-coloured so neighbours differ); small patches removed or merged into neighbors out.',
  createInputs: ({random, world}) => {
    const raster = createElevationRaster(world, 54);
    const seeds = createPatchSeeds(random, raster.columns, raster.rows, PATCH_COUNT, raster.values);
    const labels = getPatchLabels(seeds, PATCH_COUNT);
    return {
      ...raster,
      labels,
      sizes: countLabelCells(labels),
      colors: createLabelColors(labels, raster.columns, raster.rows),
      focus: {label: 0}
    };
  },
  build(builder, inputs, values) {
    const cellCount = inputs.values.length;
    const labels = builder.upload('labels', 'uint32', inputs.labels);
    const parameterBuffer = builder.allocate('parameters', 'uint32', 4);
    const sieved = builder.allocate('sieved', 'uint32', cellCount);
    builder.graph.add(
      new GPURasterSieve({
        grid: makeGridOfRaster(inputs),
        labels: labels.view,
        patchCapacity: PATCH_COUNT + 1,
        mode: values.mode === 1 ? 'merge' : 'remove',
        connectivity: values.connectivity === 1 ? 8 : 4,
        parameters: parameterBuffer.view,
        output: {labels: sieved.view}
      })
    );
    const colored = builder.upload('colored', 'uint32', colorLabels(inputs.labels, inputs.colors));
    const hover = builder.allocate('hover', 'uint32', cellCount);
    return {
      outputs: {labels, sieved, colored, hover},
      update: current => {
        parameterBuffer.write(
          getGPURasterSieveParameterValues({minimumPixels: current.minimumPixels})
        );
        // A rebuild clears the highlight; rewrite it from the last probed patch.
        hover.write(maskLabel(inputs.labels, inputs.focus.label));
      }
    };
  },
  // The surviving labels keep their original colour, looked up after the run, so a patch that
  // merges into a neighbour visibly takes that neighbour's colour.
  async afterRun({inputs, outputs, upload}) {
    const survivors = toUint32Array(await outputs.sieved.buffer.readAsync());
    return {display: upload('display', 'uint32', colorLabels(survivors, inputs.colors))};
  },
  // The pointer is the size threshold (x): every patch smaller than minimumPixels is sieved; the
  // patch under the pointer is highlighted with its size.
  interaction: {
    ...PAD,
    resolve: (pointer, state) => {
      const base = PAD.resolve(pointer, state);
      const current = {...state.values, ...base.values};
      const {inputs, outputs, world} = state;
      const {index} = getCellAt(inputs, world, pointer);
      const label = inputs.labels[index];
      inputs.focus.label = label;
      outputs.hover.write(maskLabel(inputs.labels, label));
      const size = inputs.sizes[label];
      let removed = 0;
      for (let patch = 1; patch < inputs.sizes.length; patch++) {
        if (inputs.sizes[patch] > 0 && inputs.sizes[patch] < current.minimumPixels) removed++;
      }
      return {
        ...base,
        readout: `minimumPixels ${current.minimumPixels} | ${removed} patches sieved | this one ${size} px`
      };
    }
  },
  draw: ({inputs, outputs, extra, theme}) => [
    drawRaster(inputs, outputs.colored, {
      context: true,
      ramp: 'categorical',
      categories: getLabelCategories(theme)
    }),
    ...(extra
      ? [
          drawRaster(inputs, extra.display, {
            ramp: 'categorical',
            categories: getLabelCategories(theme)
          })
        ]
      : []),
    drawRaster(inputs, outputs.hover, {ramp: 'categorical', categories: HIGHLIGHT_CATEGORIES})
  ]
});
