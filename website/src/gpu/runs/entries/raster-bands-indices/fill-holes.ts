// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterFillHoles on a binary mask (a landmass with lakes, peppered with pits): background
// regions not connected to the border are filled. The pointer stamps a closed ring of land into the
// mask, and the ring's interior fills. Few iterations stop at the limit: the result publishes what
// it has so far and reports `converged = 0`.

import {GPURasterFillHoles} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {formatNumber} from '../../interaction';
import {parameters} from './fill-holes.parameters';
import {drawRaster, makeGridOfRaster} from './support/bands';
import {getCellAt, getCellCenter} from './support/gestures';
import {createShapeRaster, stampDisc} from './support/shapes';
import {readUints} from '../categories-cells-tests/_lib/readback';

const RING_RADIUS = 6;
const LAND = [0.72, 0.76, 0.84, 1] as const;
const FILLED = [1, 0.6, 0.2, 1] as const;
const NOTHING = [0, 0, 0, 0] as const;

export default defineOperationRun({
  parameters,
  rebuildOn: ['maximumIterations'],
  caption:
    'A binary mask in; enclosed holes filled out (orange), including inside the stamped ring.',
  createInputs: ({random, world}) => {
    const raster = createShapeRaster(random, world, 72);
    return {
      ...raster,
      draws: Float32Array.from(raster.values, () => random()),
      // The ring stamp is a pointer-owned input of the mask; it starts at the rest position.
      stamp: [Math.floor(raster.columns * 0.5), Math.floor(raster.rows * 0.5)] as [number, number]
    };
  },
  build(builder, inputs, values) {
    const cellCount = inputs.values.length;
    const mask = builder.allocate('mask', 'uint32', cellCount);
    const result = builder.allocate('values', 'uint32', cellCount);
    const converged = builder.allocate('converged', 'uint32', 1);
    builder.graph.add(
      new GPURasterFillHoles({
        grid: makeGridOfRaster(inputs),
        input: mask.view,
        connectivity: 8,
        maximumIterations: values['maximumIterations'],
        output: {values: result.view, status: {converged: converged.view}}
      })
    );
    return {
      outputs: {mask, result, converged},
      update: current => {
        const data = Uint32Array.from(inputs.values, (value, index) =>
          value > current['threshold'] && inputs.draws[index] >= current['pits'] ? 1 : 0
        );
        stampDisc(
          data,
          inputs.columns,
          inputs.rows,
          inputs.stamp[0],
          inputs.stamp[1],
          RING_RADIUS,
          1.6
        );
        mask.write(data);
      }
    };
  },
  async afterRun({outputs}) {
    return {converged: (await readUints(outputs.converged))[0] !== 0};
  },
  interaction: {
    kind: 'drag',
    hint: 'stamp a ring of land',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, outputs, world, peek}) => {
      const {column, row} = getCellAt(inputs, world, pointer);
      inputs.stamp = [column, row];
      const mask = peek(outputs.mask);
      const result = peek(outputs.result);
      const converged = peek(outputs.converged)?.[0];
      let filled = 0;
      if (converged && mask && result) {
        for (let index = 0; index < mask.length; index++)
          if (result[index] === 1 && mask[index] === 0) filled++;
      }
      return {
        rerun: true,
        glyphs: [
          {
            shape: 'ring',
            at: getCellCenter(inputs, world, column, row),
            radius: (RING_RADIUS * world.height) / inputs.rows,
            dashed: true
          }
        ],
        readout: `ring at [${column}, ${row}] | ${converged === undefined ? 'status pending' : converged === 0 ? 'unfinished propagation; filled count withheld' : `filled ${formatNumber(filled, 0)} cells`}`
      };
    }
  },
  draw: ({inputs, outputs, extra}) => [
    ...(extra?.converged
      ? [drawRaster(inputs, outputs.result, {ramp: 'categorical', categories: [NOTHING, FILLED]})]
      : []),
    drawRaster(inputs, outputs.mask, {ramp: 'categorical', categories: [NOTHING, LAND]})
  ]
});
