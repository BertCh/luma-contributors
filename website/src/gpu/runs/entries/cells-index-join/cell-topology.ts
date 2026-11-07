// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCellTopology expands three seed cells into their k-ring disk (colored by grid distance) or
// their ring at distance k. The neighbor keys are decoded with GPUCellGeometry only to draw them.

import {GPUCellTopology} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addCellFill, addPointKeys, getCellFamily} from './support/cells';
import {parameters} from './cell-topology.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Grid neighbors of three seed cells: disk colored by distance, or the ring at k.',
  createInputs: ({world}) => ({
    seeds: Float32Array.of(
      world.width * 0.24,
      0.5,
      world.width * 0.56,
      0.38,
      world.width * 0.8,
      0.66
    )
  }),
  build(builder, inputs, values) {
    const family = getCellFamily(values.family);
    const isDisk = values.operation === 0;
    const k = values.k;
    const stride = family === 'h3' ? 3 * k * (k + 1) + 1 : (2 * k + 1) ** 2;
    const ringStride = (family === 'h3' ? 6 : 8) * k;
    const seedCount = inputs.seeds.length / 2;
    const rowStride = isDisk ? stride : ringStride;
    const seeds = builder.upload('seeds', 'float32x2', inputs.seeds);
    const seedCells = addPointKeys(builder, family, family === 'h3' ? 5 : 11, seeds);
    const neighbors = builder.allocate('neighbors', 'uint32x2', seedCount * rowStride);
    const distances = builder.allocate('distances', 'uint32', seedCount * rowStride);
    builder.graph.add(
      new GPUCellTopology({
        family,
        operation: {type: isDisk ? 'disk' : 'ring', k},
        cells: seedCells.view,
        output: isDisk
          ? {cells: neighbors.view, distances: distances.view}
          : {cells: neighbors.view}
      })
    );
    return {outputs: {seeds, distances, fill: addCellFill(builder, family, neighbors)}};
  },
  draw: ({outputs, values}) => [
    {
      layer: 'polygons',
      positions: outputs.fill.boundaries,
      indices: outputs.fill.indices,
      owners: outputs.fill.owners,
      values: outputs.distances,
      triangleCount: outputs.fill.triangleCount,
      color: values.operation === 0 ? undefined : [0.95, 0.6, 0.25, 0.9]
    },
    {layer: 'dots', positions: outputs.seeds, radius: 4, color: [1, 1, 1, 1]}
  ]
});
