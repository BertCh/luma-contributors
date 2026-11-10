// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterReconstruction: the marker is the mask lowered (dilation) or raised (erosion) by an
// offset, and the operation grows it back under the mask. Dilation with an offset cuts every
// peak lower than the offset flat (h-maxima); erosion fills shallow pits. The result is
// published only once the growth has converged, so too few `maximumIterations` leave it blank.
// Method and iterations are compile-time.

import {GPURasterReconstruction} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createElevationRaster} from '../../inputs';
import {bindAxes, formatNumber} from '../../interaction';
import {parameters} from './morphological-reconstruction.parameters';
import {drawRaster, makeGridOfRaster} from './support/bands';
import {addArithmetic} from './support/indices';

const PAD = bindAxes(parameters, {kind: 'level', hint: 'lower the marker', x: 'height'});

export default defineOperationRun({
  parameters,
  rebuildOn: ['method', 'maximumIterations'],
  caption: 'A mask and a marker in; the marker reconstructed under the mask out.',
  createInputs: ({world}) => createElevationRaster(world, 36),
  build(builder, inputs, values) {
    const cellCount = inputs.values.length;
    const mask = builder.upload('mask', 'float32', inputs.values);
    const marker = builder.allocate('marker', 'float32', cellCount);
    const result = builder.allocate('values', 'float32', cellCount);
    const validity = builder.allocate('validity', 'uint32', cellCount);
    const converged = builder.allocate('converged', 'uint32', 1);
    const erosion = values.method === 1;
    builder.graph.add(
      new GPURasterReconstruction({
        grid: makeGridOfRaster(inputs),
        marker: marker.view,
        mask: mask.view,
        method: erosion ? 'erosion' : 'dilation',
        connectivity: 8,
        maximumIterations: values.maximumIterations,
        output: {values: result.view, validity: validity.view, status: {converged: converged.view}}
      })
    );
    const removed = addArithmetic(builder, 'removed', 'absoluteDifference', mask, result);
    return {
      outputs: {mask, marker, result, removed},
      update: current =>
        marker.write(
          inputs.values.map(value => (erosion ? value + current.height : value - current.height))
        )
    };
  },
  // The pointer is the marker offset (x): the further right, the lower the marker sits under the
  // mask and the taller the domes that are cut flat.
  interaction: {
    ...PAD,
    resolve: (pointer, state) => {
      const base = PAD.resolve(pointer, state);
      const current = {...state.values, ...base.values};
      const {inputs, outputs, peek} = state;
      const removed = peek(outputs.removed);
      let cut = 0;
      if (removed) for (const value of removed) if (value > 0.005) cut++;
      return {
        ...base,
        glyphs: [{shape: 'crosshair', at: [pointer.x, pointer.y]}],
        readout: `height ${current.height.toFixed(2)} | ${formatNumber((100 * cut) / inputs.values.length, 0)}% of cells cut`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.result, {range: [0, 1], hillshade: 0.6, elevation: 1.5}),
    drawRaster(inputs, outputs.removed, {ramp: 'sequential', range: [0, 0.5], fade: true})
  ]
});
