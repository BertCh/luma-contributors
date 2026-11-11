// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUColocationQuotient over the k nearest neighbours of Honshu earthquakes. The categories are the
// thirds of the 2020-2024 timeline (early, middle, late). A dot of the `from` category is colored by
// its local quotient toward `to`: red means its neighbours are of `to` more often than chance (above 1),
// blue less often. Where aftershock sequences cluster in time, the quotient of a category toward itself
// is above 1. Hover an event to see its neighbours.

import {GPUColocationQuotient} from '@luma-contributors/analysis/gpu-spatial-analysis';
import coastline from '../../../data/japan-coast.json';
import {HONSHU_WINDOW, projectRows} from '../../../data/point-observations';
import {defineOperationRun} from '../../define-run';
import {createKnnWeights, uploadSpatialWeights} from '../../inputs';
import type {RunGlyph} from '../../interaction';
import {createPointIndex} from '../../pick';
import {getHonshuEventsWithTime} from './_lib/honshu';
import {getQuantileClasses} from './_lib/tract-map';
import {drawPanel, uploadPanel} from './_lib/grid-panel';
import {readFloats} from './_lib/readback';
import {parameters} from './colocation-quotient.parameters';

const CATEGORY_COUNT = 3;
const CHART = [0.03, 0.62, 0.3, 0.95] as const;

export default defineOperationRun({
  parameters,
  caption:
    'Earthquakes off Honshu, 2020-2024, in three eras of equal size; dots show local colocation quotients.',
  createInputs: ({world}) => {
    const events = getHonshuEventsWithTime();
    const toRows = (pairs: ArrayLike<number>) =>
      Array.from({length: pairs.length / 2}, (_, row) => ({
        longitude: pairs[row * 2],
        latitude: pairs[row * 2 + 1]
      }));
    const positions = projectRows(toRows(events.positions), [...HONSHU_WINDOW], world);
    return {
      ...events,
      positions,
      categories: getQuantileClasses(events.day, CATEGORY_COUNT),
      index: createPointIndex(positions),
      world,
      focus: {event: -1},
      coastline: projectRows(toRows(coastline.flat()), [...HONSHU_WINDOW], world)
    };
  },
  build(builder, inputs, values) {
    const csr = createKnnWeights(inputs.positions, values.neighborCount);
    const global = builder.allocate('global-quotients', 'float32', CATEGORY_COUNT ** 2);
    const local = builder.allocate('local-quotients', 'float32', inputs.count * CATEGORY_COUNT);
    builder.graph.add(
      new GPUColocationQuotient({
        weights: uploadSpatialWeights(builder, csr),
        categories: builder.upload('categories', 'uint32', inputs.categories).view,
        categoryCount: CATEGORY_COUNT,
        output: {globalQuotients: global.view, localQuotients: local.view}
      })
    );
    return {
      outputs: {
        global,
        local,
        csr,
        panel: uploadPanel(builder, CHART, inputs.world),
        positions: builder.upload('positions', 'float32x2', inputs.positions),
        coast: builder.upload('coast', 'float32x4', inputs.coastline),
        categories: builder.upload('category-colors', 'uint32', inputs.categories)
      }
    };
  },
  rebuildOn: ['neighborCount'],
  buildCache: 4,
  async afterRun({inputs, outputs, values, upload}) {
    const local = await readFloats(outputs.local);
    const selected = Float32Array.from({length: inputs.count}, (_, event) =>
      inputs.categories[event] === values.from
        ? local[event * CATEGORY_COUNT + values.to]
        : Number.NaN
    );
    return {selected: upload('selected', 'float32', selected)};
  },
  interaction: {
    kind: 'probe',
    hint: 'hover an event',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const found = inputs.index.nearest(pointer.x, pointer.y);
      if (found >= 0) inputs.focus.event = found;
      const event = inputs.focus.event;
      // The dashed rule is a quotient of 1: independence, neither attraction nor avoidance.
      const independence = (CHART[1] + (CHART[3] - CHART[1]) / 2) * world.height;
      const glyphs: RunGlyph[] = [
        {
          shape: 'line',
          from: [CHART[0] * world.width, independence],
          to: [CHART[2] * world.width, independence],
          dashed: true
        }
      ];
      if (event < 0) return {glyphs, spotlight: null};
      const {csr} = outputs;
      let sameCount = 0;
      const neighborCount = csr.offsets[event + 1] - csr.offsets[event];
      for (let slot = csr.offsets[event]; slot < csr.offsets[event + 1]; slot++) {
        const neighbor = csr.neighbors[slot];
        if (inputs.categories[neighbor] === values.to) sameCount++;
        glyphs.push({
          shape: 'line',
          from: [inputs.positions[event * 2], inputs.positions[event * 2 + 1]],
          to: [inputs.positions[neighbor * 2], inputs.positions[neighbor * 2 + 1]]
        });
      }
      const global = peek(outputs.global);
      const category = inputs.categories[event];
      const quotient = peek(outputs.local)?.[event * CATEGORY_COUNT + values.to];
      return {
        glyphs,
        spotlight: null,
        readout: `category ${category} · ${sameCount} of ${neighborCount} neighbours in ${values.to} · local ${
          quotient !== undefined && Number.isFinite(quotient) ? quotient.toFixed(2) : '–'
        } · CLQ ${global ? global[values.from * CATEGORY_COUNT + values.to].toFixed(2) : '…'}`
      };
    }
  },
  draw: ({outputs, extra, theme}) => [
    {layer: 'lines', context: true, segments: outputs.coast},
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2.4},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: extra.selected,
      ramp: 'diverging',
      range: [0, 2],
      radius: 4.2
    },
    drawPanel(outputs.panel, theme.background),
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.global,
      ramp: 'sequential',
      range: [0, 2],
      rectangle: CHART
    }
  ]
});
