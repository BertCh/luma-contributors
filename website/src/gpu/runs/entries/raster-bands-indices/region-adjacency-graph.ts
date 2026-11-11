// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterRegionGraph on a segmentation: region adjacency (shared boundary faces) and the mean
// difference across every adjacent pair, with regions closer than `mergeThreshold` merged. Dots mark
// the regions and take the color of the region they were merged into (the mapping output);
// regions with close means (`mergeThreshold`) end up sharing a color.

import {GPURasterRegionGraph} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createElevationRaster} from '../../inputs';
import {parameters} from './region-adjacency-graph.parameters';
import {colorLabels, createLabelColors, getCellAt, toUint32Array} from './support/gestures';
import {getLabelCategories} from './support/labels';
import {drawRaster, makeGridOfRaster} from './support/bands';
import {createPatchSeeds, getPatchLabels} from './support/patches';

const REGION_COUNT = 24;

/** Segments from a region's center to the centers of the regions it touches. */
function getLinks(centers: Float32Array, adjacency: ArrayLike<number>, region: number) {
  const links = new Float32Array(REGION_COUNT * 4);
  let neighbors = 0;
  for (let other = 0; other < REGION_COUNT; other++) {
    if (other === region) continue;
    if (
      adjacency[region * REGION_COUNT + other] > 0 ||
      adjacency[other * REGION_COUNT + region] > 0
    ) {
      links.set(
        [centers[region * 2], centers[region * 2 + 1], centers[other * 2], centers[other * 2 + 1]],
        neighbors * 4
      );
      neighbors++;
    }
  }
  return {links, neighbors};
}

export default defineOperationRun({
  parameters,
  rebuildOn: ['mergeThreshold', 'mergeMode', 'maximumIterations'],
  caption: 'A segmentation and a value raster in; region merges out.',
  createInputs: ({random, world}) => {
    const raster = createElevationRaster(world, 48);
    const seeds = createPatchSeeds(
      random,
      raster.columns,
      raster.rows,
      REGION_COUNT,
      raster.values
    );
    const labels = getPatchLabels(seeds, REGION_COUNT).map(label => label - 1);
    const values = raster.values;
    const centers = Float32Array.from({length: REGION_COUNT * 2}, (_, index) =>
      index % 2 === 0
        ? ((seeds.seeds[(index >> 1) * 3] + 0.5) / raster.columns) * world.width
        : world.height * (1 - (seeds.seeds[(index >> 1) * 3 + 1] + 0.5) / raster.rows)
    );
    return {...raster, labels, values, centers, focus: {region: -1}};
  },
  build(builder, inputs, values) {
    const labels = builder.upload('labels', 'uint32', inputs.labels);
    const cellValues = builder.upload('values', 'float32', inputs.values);
    const adjacency = builder.allocate('adjacency', 'uint32', REGION_COUNT ** 2);
    const weights = builder.allocate('weights', 'float32', REGION_COUNT ** 2);
    const mapping = builder.allocate('mapping', 'uint32', REGION_COUNT);
    const status = builder.allocate('status', 'uint32', 1);
    builder.graph.add(
      new GPURasterRegionGraph({
        grid: makeGridOfRaster(inputs),
        labels: labels.view,
        values: cellValues.view,
        regionCount: REGION_COUNT,
        mergeMode: values.mergeMode === 1 ? 'adaptive' : 'fixed',
        mergeThreshold: values.mergeThreshold,
        maximumIterations: values.maximumIterations,
        output: {
          adjacency: adjacency.view,
          weights: weights.view,
          mapping: mapping.view,
          status: {iterationLimitReached: status.view}
        }
      })
    );
    const centers = builder.upload('centers', 'float32x2', inputs.centers);
    const links = builder.allocate('links', 'float32x4', REGION_COUNT);
    return {outputs: {labels, weights, mapping, adjacency, centers, links}};
  },
  // Regions are filled by the group they merged into (graph-coloured), and the adjacency the
  // operation found is drawn between region centers.
  async afterRun({inputs, outputs, upload}) {
    const mapping = toUint32Array(await outputs.mapping.buffer.readAsync());
    const adjacency = toUint32Array(await outputs.adjacency.buffer.readAsync());
    const merged = Uint32Array.from(inputs.labels, label => mapping[label] + 1);
    const colors = createLabelColors(merged, inputs.columns, inputs.rows);
    const segments: number[] = [];
    for (let first = 0; first < REGION_COUNT; first++) {
      for (let second = first + 1; second < REGION_COUNT; second++) {
        if (
          adjacency[first * REGION_COUNT + second] > 0 ||
          adjacency[second * REGION_COUNT + first] > 0
        ) {
          segments.push(
            inputs.centers[first * 2],
            inputs.centers[first * 2 + 1],
            inputs.centers[second * 2],
            inputs.centers[second * 2 + 1]
          );
        }
      }
    }
    // A rebuild clears the links; rewrite them from the last probed region.
    if (inputs.focus.region >= 0) {
      outputs.links.write(getLinks(inputs.centers, adjacency, inputs.focus.region).links);
    }
    return {
      display: upload('display', 'uint32', colorLabels(merged, colors)),
      edges: upload(
        'edges',
        'float32x4',
        Float32Array.from(segments.length ? segments : [0, 0, 0, 0])
      ),
      edgeCount: Math.max(1, segments.length / 4)
    };
  },
  // The pointer probes a region: its adjacent regions are linked from its center.
  interaction: {
    kind: 'probe',
    hint: 'probe a region',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, outputs, world, peek}) => {
      const region = inputs.labels[getCellAt(inputs, world, pointer).index];
      const adjacency = peek(outputs.adjacency);
      const mapping = peek(outputs.mapping);
      inputs.focus.region = region;
      const {links, neighbors} = adjacency
        ? getLinks(inputs.centers, adjacency, region)
        : {links: new Float32Array(REGION_COUNT * 4), neighbors: 0};
      outputs.links.write(links);
      let group = 0;
      if (mapping) for (const target of mapping) if (target === mapping[region]) group++;
      return {
        focus: neighbors,
        readout: `region ${region} | ${neighbors} neighbours | merged with ${Math.max(0, group - 1)}`
      };
    }
  },
  draw: ({inputs, outputs, extra, interaction, theme}) => [
    ...(extra
      ? [
          drawRaster(inputs, extra.display, {
            ramp: 'categorical',
            categories: getLabelCategories(theme),
            alpha: 0.55
          }),
          {layer: 'lines' as const, segments: extra.edges, count: extra.edgeCount, alpha: 0.5}
        ]
      : []),
    {
      layer: 'lines',
      segments: outputs.links,
      count: typeof interaction.focus === 'number' ? interaction.focus : 0,
      radius: 2.5
    },
    {layer: 'dots', positions: outputs.centers, values: outputs.mapping, radius: 7}
  ]
});
