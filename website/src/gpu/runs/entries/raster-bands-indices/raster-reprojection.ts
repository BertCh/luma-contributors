// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterReprojection resamples a raster (with a coordinate grid painted on it) into a
// destination grid. Here the destination grid is the source rotated and scaled about its center,
// so the output is the same scene turned on the screen; cells that fall outside the source are
// nodata. The destination grid is a construction prop, so the run rebuilds on every change. The
// pointer aims the rotation: the compass angle around the center snaps to five rotations, applied
// once the pointer rests, and five builds are cached.

import {
  GPURasterReprojection,
  makeGPURasterBand
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createElevationRaster} from '../../inputs';
import {findParameter} from '../../interaction';
import {parameters} from './raster-reprojection.parameters';
import {drawRaster, makeGridOfRaster} from './support/bands';
import {makeRotatedGrid, paintGridLines} from './support/reproject';

const ROTATION = findParameter(parameters, 'rotation');
const ROTATIONS = [-50, -25, 0, 25, 50] as const;

export default defineOperationRun({
  buildCache: 5,
  parameters,
  rebuildOn: ['rotation', 'scale', 'method'],
  caption: 'A raster in; resampled into a rotated, rescaled grid out.',
  createInputs: ({world}) => {
    const raster = createElevationRaster(world, 48);
    return {...raster, painted: paintGridLines(raster)};
  },
  build(builder, inputs, values) {
    const cellCount = inputs.values.length;
    const sourceGrid = makeGridOfRaster(inputs);
    const destinationGrid = makeRotatedGrid(inputs, values.rotation, values.scale);
    const source = builder.upload('source', 'float32', inputs.painted);
    const result = builder.allocate('values', 'float32', cellCount);
    const validity = builder.allocate('validity', 'uint32', cellCount);
    builder.graph.add(
      builder.own(
        new GPURasterReprojection({
          sourceGrid,
          destinationGrid,
          input: makeGPURasterBand(source.view, sourceGrid, {id: 'source'}),
          method: values.method === 1 ? 'bilinear' : 'nearest',
          output: {values: result.view, validity: validity.view}
        }),
        {reusable: true}
      )
    );
    return {outputs: {source, result}};
  },
  interaction: {
    kind: 'aim',
    hint: 'aim the rotation',
    rest: ({toUnit, world}) => {
      const radians = (ROTATION.default * Math.PI) / 180;
      return toUnit([
        world.width / 2 + Math.sin(radians) * world.height * 0.3,
        world.height / 2 + Math.cos(radians) * world.height * 0.3
      ]);
    },
    resolve: (pointer, {values, world}) => {
      const deltaX = pointer.x - world.width / 2;
      const deltaY = pointer.y - world.height / 2;
      // Compass angle: 0 up, clockwise, folded to the half turn the slider covers.
      let degrees = (Math.atan2(deltaX, deltaY) * 180) / Math.PI;
      if (degrees > 90) degrees = 180 - degrees;
      if (degrees < -90) degrees = -180 - degrees;
      const rotation = ROTATIONS.reduce((best, candidate) =>
        Math.abs(candidate - degrees) < Math.abs(best - degrees) ? candidate : best
      );
      const radians = (rotation * Math.PI) / 180;
      const length = Math.max(0.15, Math.hypot(deltaX, deltaY));
      return {
        settle: {rotation},
        glyphs: [
          {
            shape: 'arrow',
            from: [world.width / 2, world.height / 2],
            to: [
              world.width / 2 + Math.sin(radians) * length,
              world.height / 2 + Math.cos(radians) * length
            ]
          },
          {shape: 'dot', at: [world.width / 2, world.height / 2]}
        ],
        readout: `rotation ${rotation}° | cell size ×${values.scale.toFixed(2)}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.source, {context: true, range: [0, 1]}),
    drawRaster(inputs, outputs.result, {range: [0, 1]})
  ]
});
