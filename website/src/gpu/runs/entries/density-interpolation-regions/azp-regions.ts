// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUAZPRegions improves a starting partition by moving border tracts between regions while the
// within-region variance falls. With one iteration the picture is close to the starting partition (a
// Voronoi split of random seed tracts); more iterations straighten the borders onto the data.

import {
  getGPUAZPRegionsParameterValues,
  GPU_AZP_STATUS_LAYOUT,
  GPUAZPRegions
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {makeRandom} from '../../../demos/kit';
import {uploadSpatialWeights} from '../../inputs';
import {parameters} from './azp-regions.parameters';
import {uploadTractMesh} from './support/geometry';
import {createPartitionBorders, createRegionScene, drawLabels} from './support/regions';

// The iteration limits the pointer can settle on; the default is the last.
const ITERATIONS = [1, 3, 5, 9, 31] as const;

export default defineOperationRun({
  parameters,
  caption: 'Starting partition (white borders) improved by moving border tracts.',
  createInputs: ({world}) => createRegionScene(makeRandom(7), world, 8, 2),
  build(builder, scene, current) {
    const {tracts, attributes, columnCount} = scene;
    // Starting partition: every tract joins its nearest seed tract, so regions start contiguous.
    const random = makeRandom(current.seed);
    const seeds = Array.from({length: current.regionCapacity}, () =>
      Math.floor(random() * tracts.count)
    );
    const initialLabels = Uint32Array.from({length: tracts.count}, (_, tract) => {
      let best = 0;
      let bestDistance = Infinity;
      seeds.forEach((seed, index) => {
        const distance = Math.hypot(
          tracts.centers[tract * 2] - tracts.centers[seed * 2],
          tracts.centers[tract * 2 + 1] - tracts.centers[seed * 2 + 1]
        );
        if (distance < bestDistance) {
          bestDistance = distance;
          best = index;
        }
      });
      return best;
    });
    const labels = builder.allocate('labels', 'uint32', tracts.count);
    builder.graph.add(
      new GPUAZPRegions({
        weights: uploadSpatialWeights(builder, tracts.contiguity),
        values: builder.upload('attributes', 'float32', attributes).view,
        columnCount,
        standardize: true,
        initialLabels: builder.upload('initial-labels', 'uint32', initialLabels).view,
        regionCapacity: current.regionCapacity,
        maximumIterations: current.maximumIterations,
        parameters: builder.upload(
          'parameters',
          'uint32',
          getGPUAZPRegionsParameterValues({seed: current.seed})
        ).view,
        output: {
          labels: labels.view,
          objective: builder.allocate('objective', 'float32', 1).view,
          status: builder.allocate('status', 'uint32', GPU_AZP_STATUS_LAYOUT.length).view
        }
      })
    );
    // Borders of the starting partition, to see which tracts the iterations moved.
    const startBorders = createPartitionBorders(scene, initialLabels);
    return {
      outputs: {
        labels,
        startBorders: builder.upload('start-borders', 'float32x4', startBorders),
        startBorderCount: startBorders.length / 4,
        ...uploadTractMesh(builder, tracts)
      }
    };
  },
  buildCache: 5,
  rebuildOn: ['regionCapacity', 'maximumIterations', 'seed'],
  // x gives the search more iterations: the regions are rebuilt once the pointer rests on a
  // limit, and the borders of the starting partition stay drawn for comparison.
  interaction: {
    kind: 'time',
    hint: 'allow more iterations',
    rest: [0.9, 0.5],
    resolve(pointer, {world}) {
      const index = Math.min(ITERATIONS.length - 1, Math.floor(pointer.unitX * ITERATIONS.length));
      const glyphs: RunGlyph[] = ITERATIONS.map((_, stop) => ({
        shape: 'dot',
        at: [((stop + 0.5) / ITERATIONS.length) * world.width, 0.03]
      }));
      glyphs.push({
        shape: 'crosshair',
        at: [((index + 0.5) / ITERATIONS.length) * world.width, 0.03]
      });
      return {
        settle: {maximumIterations: ITERATIONS[index]},
        glyphs,
        readout: `maximumIterations ${ITERATIONS[index]}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    ...drawLabels(inputs, outputs, outputs.labels),
    {
      layer: 'lines',
      segments: outputs.startBorders,
      count: outputs.startBorderCount,
      radius: 1.6,
      color: [1, 1, 1, 0.9]
    }
  ]
});
