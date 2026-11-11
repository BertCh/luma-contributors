// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addContiguityOrderCorrelogramRecipe on Chicago tracts: global Moran's I for the neighbours at
// exactly 1, 2, ... `maximumOrder` contiguity steps. The bars are I per order (the dashed line is zero).
// Hover a tract to color every tract by its contiguity order from it: those are the rings the
// recipe builds and measures. Autocorrelation fades as the ring moves away.

import {
  addContiguityOrderCorrelogramRecipe,
  GPU_CONTIGUITY_ORDER_CORRELOGRAM_COLUMNS
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import type {RunGlyph} from '../../interaction';
import {findTractUnder, getTractOutline} from '../regression-weights/shared/probes';
import {readFloats} from './_lib/readback';
import {
  CHART_RECTANGLE,
  createTractMap,
  drawTractOutline,
  drawTractValues,
  uploadTractMap
} from './_lib/tract-map';
import {parameters} from './contiguity-order-correlogram-recipe.parameters';

const COLUMNS = GPU_CONTIGUITY_ORDER_CORRELOGRAM_COLUMNS;
const MAXIMUM_ORDER = 6;

/** Contiguity order of every tract from `start` (0 for the start, NaN beyond `limit`). */
function getOrders(
  offsets: Uint32Array,
  neighbors: Uint32Array,
  start: number,
  limit: number
): Float32Array {
  const orders = new Float32Array(offsets.length - 1).fill(Number.NaN);
  orders[start] = 0;
  let frontier = [start];
  for (let order = 1; order <= limit; order++) {
    const next: number[] = [];
    for (const tract of frontier) {
      for (let slot = offsets[tract]; slot < offsets[tract + 1]; slot++) {
        const neighbor = neighbors[slot];
        if (Number.isNaN(orders[neighbor])) {
          orders[neighbor] = order;
          next.push(neighbor);
        }
      }
    }
    frontier = next;
  }
  return orders;
}

export default defineOperationRun({
  parameters,
  caption: "Moran's I of Chicago tract variables for neighbours at exactly 1, 2, 3 ... steps.",
  createInputs: ({random, world}) => ({...createTractMap(random, world), focus: {tract: -1}}),
  build(builder, tracts, values) {
    const weights = uploadSpatialWeights(builder, tracts.contiguity);
    const attribute = builder.allocate('attribute', 'float32', tracts.count);
    const correlogram = builder.allocate(
      'correlogram',
      'float32',
      values.maximumOrder * COLUMNS.length
    );
    const incomplete = builder.allocate('incomplete-orders', 'uint32', values.maximumOrder);
    addContiguityOrderCorrelogramRecipe(builder.graph, {
      weights,
      values: attribute.view,
      maximumOrder: values.maximumOrder,
      slotCapacity: tracts.contiguity.neighbors.length * values.maximumOrder * 3,
      outputs: {correlogram: correlogram.view, incompleteOrders: incomplete.view}
    });
    return {
      outputs: {
        correlogram,
        rings: builder.allocate('rings', 'float32', tracts.count),
        map: uploadTractMap(builder, tracts)
      },
      update: current => {
        attribute.write(
          [
            tracts.povertyShare,
            tracts.unemploymentShare,
            Float32Array.from(tracts.age65, (count, tract) => count / tracts.population[tract])
          ][current.variable]
        );
      }
    };
  },
  rebuildOn: ['maximumOrder'],
  buildCache: 3,
  async afterRun({outputs, values, upload}) {
    const table = await readFloats(outputs.correlogram);
    const statistics = Float32Array.from({length: MAXIMUM_ORDER}, (_, order) =>
      order < values.maximumOrder ? table[order * COLUMNS.length + COLUMNS.statistic] : 0
    );
    return {statistics: upload('moran', 'float32', statistics), table};
  },
  interaction: {
    kind: 'probe',
    hint: 'hover a tract',
    rest: [0.3, 0.5],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const found = findTractUnder(inputs, pointer.x, pointer.y);
      if (found >= 0) inputs.focus.tract = found;
      const tract = inputs.focus.tract;
      const table = peek(outputs.correlogram);
      const zero =
        (CHART_RECTANGLE[1] + (CHART_RECTANGLE[3] - CHART_RECTANGLE[1]) / 2) * world.height;
      const glyphs: RunGlyph[] = [
        {
          shape: 'line',
          from: [CHART_RECTANGLE[0] * world.width, zero],
          to: [CHART_RECTANGLE[2] * world.width, zero],
          dashed: true
        }
      ];
      let rings: Float32Array | null = null;
      if (tract >= 0) {
        rings = getOrders(
          inputs.contiguity.offsets,
          inputs.contiguity.neighbors,
          tract,
          values.maximumOrder
        );
        outputs.rings.write(rings);
        glyphs.push(getTractOutline(inputs, tract));
      }
      const sizes = Array.from({length: values.maximumOrder}, (_, index) =>
        rings ? rings.filter(order => order === index + 1).length : 0
      );
      const moran = table
        ? Array.from({length: values.maximumOrder}, (_, order) => {
            const value = table[order * COLUMNS.length + COLUMNS.statistic];
            return Number.isFinite(value) ? value.toFixed(2) : 'incomplete';
          }).join(' ')
        : '…';
      const readout = `rings ${sizes.join(', ')} · I ${moran}`;
      return {glyphs, readout, spotlight: null};
    }
  },
  draw: ({inputs, outputs, extra}) => [
    drawTractValues(outputs.map, inputs, outputs.rings, {
      ramp: 'sequential',
      range: [0, MAXIMUM_ORDER],
      alpha: 0.9
    }),
    drawTractOutline(outputs.map, inputs),
    {
      layer: 'chart',
      kind: 'bars',
      values: extra.statistics,
      ramp: 'diverging',
      range: [-1, 1],
      rectangle: CHART_RECTANGLE
    }
  ]
});
