// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULocalCategoryDependence: each Chicago tract is cut into one of `categoryCount` quantile classes.
// The fill is how surprising the tract's neighbourhood is: -log10 of the binomial upper tail of its
// like-class neighbours (bright means more like neighbours than the class share predicts). The dots are
// the classes. Hover a tract to see which neighbours share its class.

import {
  GPULocalCategoryDependence,
  GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts, uploadSpatialWeights} from '../../inputs';
import type {RunGlyph} from '../../interaction';
import {
  findTractUnder,
  getTractNeighbors,
  getTractOutline
} from '../regression-weights/shared/probes';
import {getQuantileClasses} from './_lib/tract-map';
import {readFloats} from './_lib/readback';
import {parameters} from './local-category-dependence.parameters';

const COLUMNS = GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS;

export default defineOperationRun({
  parameters,
  caption:
    'Chicago tract classes (quantiles of poverty, unemployment or the 65+ share) and how unusual each neighbourhood is.',
  createInputs: ({random, world}) => ({
    ...createTracts(random, world),
    focus: {tract: -1}
  }),
  build(builder, tracts, values) {
    const weights = uploadSpatialWeights(builder, tracts.contiguity);
    const categories = builder.allocate('categories', 'uint32', tracts.count);
    const statistics = builder.allocate('statistics', 'float32', tracts.count * COLUMNS.length);
    builder.graph.add(
      new GPULocalCategoryDependence({
        weights,
        categories: categories.view,
        categoryCount: values.categoryCount,
        output: {statistics: statistics.view}
      })
    );
    return {
      outputs: {
        categories,
        statistics,
        vertices: builder.upload('vertices', 'float32x2', tracts.vertices),
        indices: builder.upload('indices', 'uint32', tracts.triangleIndices),
        owners: builder.upload('owners', 'uint32', tracts.triangleOwners),
        centers: builder.upload('centers', 'float32x2', tracts.centers),
        outline: builder.upload('outline', 'float32x4', tracts.outline)
      },
      update: current => {
        const column = [
          tracts.povertyShare,
          tracts.unemploymentShare,
          Float32Array.from(tracts.age65, (count, tract) => count / tracts.population[tract])
        ][current.variable];
        categories.write(getQuantileClasses(column, current.categoryCount));
      }
    };
  },
  rebuildOn: ['categoryCount'],
  buildCache: 3,
  async afterRun({outputs, upload}) {
    const statistics = await readFloats(outputs.statistics);
    const score = Float32Array.from({length: statistics.length / COLUMNS.length}, (_, tract) => {
      const tail = statistics[tract * COLUMNS.length + COLUMNS.upperTailProbability];
      return Number.isFinite(tail) ? -Math.log10(Math.max(tail, 1e-4)) : 0;
    });
    return {score: upload('score', 'float32', score)};
  },
  interaction: {
    kind: 'probe',
    hint: 'hover a tract',
    rest: [0.3, 0.5],
    resolve(pointer, {inputs, outputs, peek}) {
      const found = findTractUnder(inputs, pointer.x, pointer.y);
      if (found >= 0) inputs.focus.tract = found;
      const tract = inputs.focus.tract;
      if (tract < 0) return {};
      const statistics = peek(outputs.statistics);
      const categories = peek(outputs.categories);
      const glyphs: RunGlyph[] = [getTractOutline(inputs, tract)];
      if (categories) {
        for (const neighbor of getTractNeighbors(inputs, tract)) {
          glyphs.push(
            getTractOutline(inputs, neighbor, 0, categories[neighbor] !== categories[tract])
          );
        }
      }
      if (!statistics) return {glyphs, spotlight: null};
      const row = tract * COLUMNS.length;
      const like = statistics[row + COLUMNS.likeCount];
      const neighborCount = statistics[row + COLUMNS.neighborCount];
      const expected = statistics[row + COLUMNS.expectedProportion];
      const tail = statistics[row + COLUMNS.upperTailProbability];
      return {
        glyphs,
        spotlight: null,
        readout: `like ${like}/${neighborCount} · expected share ${expected.toFixed(2)} · p ${
          Number.isFinite(tail) ? tail.toPrecision(2) : '–'
        }`
      };
    }
  },
  draw: ({inputs, outputs, extra}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: extra.score,
      triangleCount: inputs.triangleCount,
      ramp: 'sequential',
      range: [0, 3]
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.outlineCount},
    {layer: 'dots', positions: outputs.centers, values: outputs.categories, radius: 4.5}
  ]
});
