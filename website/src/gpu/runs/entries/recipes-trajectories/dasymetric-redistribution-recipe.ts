// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addDasymetricRedistributionRecipe: source zone totals redistributed to target zones, weighted by
// an ancillary raster (bright = built-up land). Targets are filled by their redistributed total;
// the ancillary raster shows through the fill.

import {addDasymetricRedistributionRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createQuadZones, getQuadFill, uploadQuadZones} from './lib/quads';
import {createWorldGrid, getCellCenter, rasterizeZones} from './lib/rasterize';
import {parameters} from './dasymetric-redistribution-recipe.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Zone totals pushed down to target zones where the ancillary raster is bright.',
  createInputs: ({random, world}) => {
    const grid = createWorldGrid(world);
    const source = createQuadZones(random, world, 4, 3);
    const target = createQuadZones(random, world, 9, 5, 0.16);
    return {
      world,
      grid,
      source,
      target,
      targetZones: rasterizeZones(target.positions, target.count, 4, grid),
      centers: [
        [(0.25 + 0.3 * random()) * world.width, 0.3 + 0.4 * random()],
        [(0.55 + 0.3 * random()) * world.width, 0.3 + 0.4 * random()]
      ]
    };
  },
  build(builder, {grid, source, target, targetZones, centers}, values) {
    const cellCount = grid.width * grid.height;
    const sourceMesh = uploadQuadZones(builder, source, 'source');
    const targetMesh = uploadQuadZones(builder, target, 'target');
    const sourceTotals = builder.allocate('source-totals', 'float32', source.count);
    const ancillary = builder.allocate('ancillary', 'float32', cellCount);
    const targetTotals = builder.allocate('target-totals', 'float32', target.count);
    addDasymetricRedistributionRecipe(builder.graph, {
      grid: grid.grid,
      source: {
        kind: 'polygons',
        zones: {
          positions: sourceMesh.positions.view,
          featureOffsets: sourceMesh.featureOffsets.view,
          polygonOffsets: sourceMesh.featureOffsets.view,
          ringOffsets: sourceMesh.ringOffsets.view
        },
        tupleCapacity: 8192
      },
      sourceTotals: sourceTotals.view,
      targetZones: builder.upload('target-zones', 'uint32', targetZones).view,
      targetZoneCount: target.count,
      ancillary: {
        id: 'ancillary',
        format: 'float32',
        storage: {kind: 'buffer', values: ancillary.view}
      },
      denominator: values.denominator === 1 ? 'overlap' : 'zone',
      pairCapacity: 4096,
      outputs: {targetTotals: targetTotals.view}
    });
    return {
      outputs: {targetTotals, ancillary, targetMesh},
      update: current => {
        sourceTotals.write(
          Float32Array.from(source.values, value => 1000 * (0.3 + current.totals * value))
        );
        ancillary.write(
          Float32Array.from({length: cellCount}, (_, index) => {
            const [x, y] = getCellCenter(grid, index);
            const core = centers.reduce(
              (sum, [centerX, centerY]) =>
                sum + Math.exp(-(((x - centerX) / 0.18) ** 2 + ((y - centerY) / 0.2) ** 2)),
              0
            );
            return 0.05 + current.core * core + 0.02 * Math.sin(x * 40 + y * 30) ** 2;
          })
        );
      }
    };
  },
  rebuildOn: ['denominator'],
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.ancillary,
      columns: inputs.grid.width,
      rows: inputs.grid.height
    },
    getQuadFill(outputs.targetMesh, inputs.target, outputs.targetTotals, {
      ramp: 'sequential',
      alpha: 0.7,
      range: [0, 600]
    }),
    {
      layer: 'lines',
      context: true,
      segments: outputs.targetMesh.outline,
      count: inputs.target.outlineCount
    }
  ]
});
