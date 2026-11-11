// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialEntropy of poverty classes over rook-contiguous Chicago tracts. The bars are the summary
// words in nats: Shannon entropy of the classes, origin and destination entropy of a link, the
// co-occurrence entropy of the class pairs at links, and their difference, the spatial mutual
// information. Sweeping the pointer right swaps the classes of more and more tracts: the map becomes
// random and the mutual information falls toward zero while the Shannon entropy does not move.

import {
  GPUSpatialEntropy,
  GPU_SPATIAL_ENTROPY_LAYOUT
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import {bindAxes} from '../../interaction';
import {
  CHART_RECTANGLE,
  createTractMap,
  drawTractOutline,
  drawTractValues,
  getQuantileClasses,
  uploadTractMap
} from './_lib/tract-map';
import {readFloats} from './_lib/readback';
import {parameters} from './spatial-entropy.parameters';

const LAYOUT = GPU_SPATIAL_ENTROPY_LAYOUT;
const BAR_WORDS = [
  LAYOUT.shannon,
  LAYOUT.originEntropy,
  LAYOUT.destinationEntropy,
  LAYOUT.cooccurrence,
  LAYOUT.mutualInformation
];

export default defineOperationRun({
  parameters,
  caption:
    'Poverty classes of Chicago tracts. Bars (nats): Shannon, origin, destination, co-occurrence, mutual information.',
  createInputs: ({random, world}) => {
    const tracts = createTractMap(random, world);
    // A fixed order of tracts: the first `scrambled * count` of them swap classes with a partner.
    const order = Array.from({length: tracts.count}, (_, tract) => tract);
    for (let index = order.length - 1; index > 0; index--) {
      const other = Math.floor(random() * (index + 1));
      [order[index], order[other]] = [order[other], order[index]];
    }
    return {...tracts, order};
  },
  build(builder, tracts, values) {
    const categories = builder.allocate('categories', 'uint32', tracts.count);
    const summary = builder.allocate('summary', 'float32', LAYOUT.length);
    builder.graph.add(
      new GPUSpatialEntropy({
        weights: uploadSpatialWeights(builder, tracts.contiguity),
        categories: categories.view,
        categoryCount: values.categoryCount,
        output: {summary: summary.view}
      })
    );
    return {
      outputs: {categories, summary, map: uploadTractMap(builder, tracts)},
      update: current => {
        const classes = getQuantileClasses(tracts.povertyShare, current.categoryCount);
        const swapped = Math.round(current.scrambled * tracts.count);
        // Rotate the classes of the swapped tracts among themselves.
        const subset = tracts.order.slice(0, swapped);
        const original = subset.map(tract => classes[tract]);
        subset.forEach((tract, index) => {
          classes[tract] = original[(index + 1) % original.length];
        });
        categories.write(classes);
      }
    };
  },
  rebuildOn: ['categoryCount'],
  buildCache: 4,
  async afterRun({outputs, upload}) {
    const summary = await readFloats(outputs.summary);
    const bars = Float32Array.from(BAR_WORDS, word =>
      Number.isFinite(summary[word]) ? summary[word] : 0
    );
    return {bars: upload('bars', 'float32', bars), summary};
  },
  interaction: bindAxes<unknown, {summary: RunBuffer<'float32'>}>(parameters, {
    kind: 'time',
    hint: 'scramble the map',
    x: 'scrambled',
    decorate: (current, _pointer, state) => {
      const summary = state.peek(state.outputs.summary);
      return {
        readout: `scrambled ${Math.round(current.scrambled * 100)}% · mutual information ${
          summary ? summary[LAYOUT.mutualInformation].toFixed(3) : '…'
        } nats`
      };
    }
  }),
  draw: ({inputs, outputs, extra}) => [
    drawTractValues(outputs.map, inputs, outputs.categories),
    drawTractOutline(outputs.map, inputs),
    {
      layer: 'chart',
      kind: 'bars',
      values: extra.bars,
      ramp: 'sequential',
      range: [0, Math.log(5)],
      rectangle: CHART_RECTANGLE
    }
  ]
});
