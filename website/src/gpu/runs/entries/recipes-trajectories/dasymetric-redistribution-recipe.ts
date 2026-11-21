// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addDasymetricRedistributionRecipe: source zone totals redistributed to target zones, weighted by
// an ancillary raster (bright = built-up land). Targets are filled by their redistributed total;
// the ancillary raster shows through the fill. The heavy outlines are the source zones, each with the
// total that is redistributed.

import {addDasymetricRedistributionRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getGPURasterFractionalCoverageParameterValues} from '@luma-contributors/analysis/raster-analysis';
import {getBuiltFraction} from './lib/land-cover';
import {defineOperationRun} from '../../define-run';
import {createLegend, formatTick, mergeText, type RunLabel} from '../../text';
import {createBuiltUpZones} from './lib/built-up-zones';
import {createQuadZones, getQuadFill, uploadQuadZones} from './lib/quads';
import {createWorldGrid, getCellCenter, rasterizeZones} from './lib/rasterize';
import {parameters} from './dasymetric-redistribution-recipe.parameters';

/** The most built-up point of the map: the default center of the ancillary surface. */
function findBuiltUpPeak(world: {width: number; height: number}): [number, number] {
  let best: [number, number] = [world.width / 2, 0.5];
  let bestValue = -1;
  for (let row = 0; row < 32; row++) {
    for (let column = 0; column < 32; column++) {
      const x = (column + 0.5) / 32;
      const y = (row + 0.5) / 32;
      const value = getBuiltFraction(x, y);
      if (value > bestValue) {
        bestValue = value;
        best = [Math.round(x * world.width * 50) / 50, Math.round(y * 50) / 50];
      }
    }
  }
  return best;
}

/** Radius (world units) of the bump the pointer adds to the ancillary surface. */
const BUMP_RADIUS = 0.16;

export default defineOperationRun({
  parameters,
  caption: 'Zone totals pushed down to target zones where the ancillary raster is bright.',
  createInputs: ({random, world}) => {
    const grid = createWorldGrid(world);
    const source = createBuiltUpZones(random, world, 4, 3);
    const target = createQuadZones(random, world, 9, 5, 0.16);
    return {
      world,
      grid,
      source,
      target,
      peak: findBuiltUpPeak(world),
      targetZones: rasterizeZones(target.positions, target.count, 4, grid)
    };
  },
  build(builder, inputs, values) {
    const {grid, source, target, targetZones, world} = inputs;
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
        tupleCapacity: 8192,
        parameters: builder.upload(
          'coverage-parameters',
          'float32',
          getGPURasterFractionalCoverageParameterValues(grid.grid)
        ).view
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
      output: {targetTotals: targetTotals.view}
    });
    return {
      outputs: {targetTotals, ancillary, targetMesh, sourceMesh},
      update: current => {
        sourceTotals.write(
          Float32Array.from(source.values, value => 1000 * (0.3 + current.totals * value))
        );
        ancillary.write(
          Float32Array.from({length: cellCount}, (_, index) => {
            const [x, y] = getCellCenter(grid, index);
            // The built-up land, plus a bump the pointer places on the ancillary surface.
            const bump = Math.exp(
              -(((x - inputs.peak[0]) ** 2 + (y - inputs.peak[1]) ** 2) / BUMP_RADIUS ** 2)
            );
            return (
              0.05 +
              current.core * Math.max(getBuiltFraction(x / world.width, y / world.height), bump)
            );
          })
        );
      }
    };
  },
  rebuildOn: ['denominator'],
  // The pointer places the center of the built-up peak on the ancillary surface; the totals flow
  // toward the cells under it.
  interaction: {
    kind: 'drag',
    hint: 'move the built-up peak',
    rest: ({toUnit, inputs}) => toUnit(inputs.peak),
    resolve(pointer, {inputs, outputs, peek}) {
      inputs.peak = [Math.round(pointer.x * 50) / 50, Math.round(pointer.y * 50) / 50];
      const totals = peek(outputs.targetTotals);
      const largest = totals ? Math.max(0, ...Array.from(totals).filter(Number.isFinite)) : 0;
      return {
        rerun: true,
        glyphs: [
          {shape: 'ring', at: inputs.peak, radius: BUMP_RADIUS, dashed: true},
          {shape: 'crosshair', at: inputs.peak, size: 5}
        ],
        readout: `peak (${inputs.peak[0].toFixed(2)}, ${inputs.peak[1].toFixed(2)}) · largest target ${Math.round(largest)}`
      };
    }
  },
  draw: ({inputs, outputs, values, theme}) => {
    // The total each source zone hands down, from the same sourceContrast the operation reads.
    const sourceLabels: RunLabel[] = Array.from({length: inputs.source.count}, (_, zone) => ({
      text: formatTick(Math.round(1000 * (0.3 + values.totals * inputs.source.values[zone]))),
      at: [inputs.source.centers[zone * 2], inputs.source.centers[zone * 2 + 1]] as const,
      role: 'value' as const
    }));
    return [
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
      },
      {
        layer: 'lines',
        segments: outputs.sourceMesh.outline,
        count: inputs.source.outlineCount,
        radius: 3,
        color: [...theme.ink, 0.9]
      },
      {
        layer: 'text',
        ...mergeText(
          {labels: sourceLabels},
          createLegend({
            corner: 'bottom-right',
            entries: [
              {label: 'source zone and its total', swatch: [...theme.ink]},
              {label: 'target fill: redistributed total', swatch: [...theme.ramp[4]]},
              {label: 'texture: built-up land (ancillary)', swatch: [...theme.muted]}
            ]
          })
        )
      }
    ];
  }
});
