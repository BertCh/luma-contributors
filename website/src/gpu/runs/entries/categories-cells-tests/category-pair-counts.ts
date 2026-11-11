// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCategoryPairCounts: the join-count table of a categorical map. Every rook link between two
// Chicago tracts is counted once in each direction, by the classes of its ends. The bars are the
// categoryCount x categoryCount ordered pair counts (row = origin class, column = destination
// class). Hover a tract to light up the row of its class and outline its neighbours.

import {GPUCategoryPairCounts} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import type {RunGlyph} from '../../interaction';
import {findTractUnder} from '../regression-weights/shared/probes';
import {
  CHART_RECTANGLE,
  createTractMap,
  drawTractOutline,
  drawTractValues,
  getQuantileClasses,
  uploadTractMap
} from './_lib/tract-map';
import {createGridPanel} from './_lib/grid-panel';
import {parameters} from './category-pair-counts.parameters';

/** The largest ordered count of any pair, for one fixed chart scale. */
const CHART_MAXIMUM = 360;

export default defineOperationRun({
  parameters,
  caption:
    'Ordered class-pair link counts of Chicago tracts: table rows are origin class, columns destination class. Amber links are the pairs counted.',
  createInputs: ({random, world}) => ({
    ...createTractMap(random, world),
    world,
    linkCount: 0,
    focus: {origin: 0, destination: 0}
  }),
  build(builder, tracts, values) {
    const categories = builder.allocate('categories', 'uint32', tracts.count);
    const pairCounts = builder.allocate('pair-counts', 'uint32', values.categoryCount ** 2);
    const rowCounts = builder.allocate('row-counts', 'uint32', values.categoryCount);
    builder.graph.add(
      new GPUCategoryPairCounts({
        weights: uploadSpatialWeights(builder, tracts.contiguity),
        categories: categories.view,
        categoryCount: values.categoryCount,
        output: {pairCounts: pairCounts.view, categoryRowCounts: rowCounts.view}
      })
    );
    const grid = createGridPanel(values.categoryCount, CHART_RECTANGLE, tracts.world);
    return {
      outputs: {
        categories,
        pairCounts,
        rowCounts,
        links: builder.allocate('links', 'float32x4', tracts.contiguity.neighbors.length),
        grid,
        gridPositions: builder.upload('grid-positions', 'float32x2', grid.positions),
        gridIndices: builder.upload('grid-indices', 'uint32', grid.indices),
        gridOwners: builder.upload('grid-owners', 'uint32', grid.owners),
        map: uploadTractMap(builder, tracts)
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
  interaction: {
    kind: 'probe',
    hint: 'hover the table or a tract',
    rest: [0.84, 0.35],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const count = values.categoryCount;
      const categories = peek(outputs.categories);
      const pairCounts = peek(outputs.pairCounts);
      const cell = outputs.grid.findCell(pointer.x, pointer.y);
      const found = findTractUnder(inputs, pointer.x, pointer.y);
      if (cell) inputs.focus = {origin: cell.row, destination: cell.column};
      else if (found >= 0 && categories)
        inputs.focus = {origin: categories[found], destination: -1};
      const {origin, destination} = inputs.focus;
      inputs.linkCount = 0;
      if (origin < 0 || origin >= count || !categories || !pairCounts) return {spotlight: null};
      // Table cells in the selection, and the map links they count.
      const glyphs: RunGlyph[] = [];
      for (let column = 0; column < count; column++) {
        if (destination >= 0 && column !== destination) continue;
        const {minimum, maximum} = outputs.grid.getCell(origin, column);
        glyphs.push({shape: 'box', minimum, maximum});
      }
      const {offsets, neighbors} = inputs.contiguity;
      const links = new Float32Array(neighbors.length * 4);
      for (let tract = 0; tract < inputs.count; tract++) {
        if (categories[tract] !== origin) continue;
        for (let slot = offsets[tract]; slot < offsets[tract + 1]; slot++) {
          const other = neighbors[slot];
          if (destination >= 0 && categories[other] !== destination) continue;
          links.set(
            [
              inputs.centers[tract * 2],
              inputs.centers[tract * 2 + 1],
              inputs.centers[other * 2],
              inputs.centers[other * 2 + 1]
            ],
            inputs.linkCount * 4
          );
          inputs.linkCount++;
        }
      }
      outputs.links.write(links);
      let total = 0;
      for (let column = 0; column < count; column++) total += pairCounts[origin * count + column];
      const selected = destination >= 0 ? pairCounts[origin * count + destination] : total;
      return {
        glyphs,
        spotlight: null,
        readout:
          destination >= 0
            ? `class ${origin} to ${destination}: ${selected} links (${Math.round((100 * selected) / Math.max(total, 1))}% of class ${origin})`
            : `class ${origin} links ${total} · to own class ${pairCounts[origin * count + origin]}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    drawTractValues(outputs.map, inputs, outputs.categories),
    drawTractOutline(outputs.map, inputs),
    {
      layer: 'lines',
      segments: outputs.links,
      count: Math.max(inputs.linkCount, 0),
      color: [1, 0.72, 0.2, 1],
      radius: 2
    },
    {
      layer: 'polygons',
      positions: outputs.gridPositions,
      indices: outputs.gridIndices,
      owners: outputs.gridOwners,
      values: outputs.pairCounts,
      triangleCount: outputs.grid.size * outputs.grid.size * 2,
      ramp: 'sequential',
      logarithmic: true,
      range: [2, Math.log2(CHART_MAXIMUM)]
    }
  ]
});
