// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainHydrologicIndices on the filled DEM and area accumulation of GPUTerrainFlow: slope,
// specific catchment area, wetness index and stream power. All four views are written; the
// slider picks which one is drawn, and minimumSlope clamps flat cells before the division.

import {
  getGPUTerrainHydrologicIndicesParameterValues,
  GPUTerrainHydrologicIndices
} from '@luma-contributors/analysis/gpu-terrain';
import {makeGPURasterBand} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './terrain-hydrologic-indices.parameters';
import {addTerrainFlow, createTerrainInputs, getTerrainContextLayer} from './shared/terrain';

/** Fixed value ranges, in the order of the `output` options. */
const INDEX_RANGES = [
  [-4, 1],
  [0, 0.1],
  [0, 1],
  [0.02, 0.25]
] as const;

export default defineOperationRun({
  parameters,
  caption: 'Wetness peaks in flat, high-accumulation valleys; stream power on steep channels.',
  createInputs: context => ({
    ...createTerrainInputs(context),
    /** The highlighted copy of the drawn index and what it was made from. */
    highlight: {
      source: null as Float32Array | null,
      threshold: Number.NaN,
      output: -1,
      sorted: new Float32Array(0)
    }
  }),
  build(builder, terrain) {
    const flow = addTerrainFlow(builder, terrain, {accumulationUnits: 'area'});
    flow.writeStreamThreshold(25);
    const parameterBuffer = builder.allocate('index-parameters', 'float32', 8);
    const slope = builder.allocate('slope', 'float32', terrain.cellCount);
    const specificCatchmentArea = builder.allocate('catchment-area', 'float32', terrain.cellCount);
    const wetnessIndex = builder.allocate('wetness', 'float32', terrain.cellCount);
    const streamPowerIndex = builder.allocate('stream-power', 'float32', terrain.cellCount);
    builder.graph.add(
      new GPUTerrainHydrologicIndices({
        grid: terrain.grid,
        elevation: makeGPURasterBand(flow.filledElevation.view, terrain.grid, {id: 'filled'}),
        accumulation: flow.accumulation.view,
        parameters: parameterBuffer.view,
        output: {
          slope: slope.view,
          specificCatchmentArea: specificCatchmentArea.view,
          wetnessIndex: wetnessIndex.view,
          streamPowerIndex: streamPowerIndex.view
        }
      })
    );
    // The drawn index where it is at least the threshold, NaN elsewhere.
    const highlighted = builder.allocate('highlighted', 'float32', terrain.cellCount);
    return {
      outputs: {
        filledElevation: flow.filledElevation,
        indices: [wetnessIndex, streamPowerIndex, slope, specificCatchmentArea],
        highlighted
      },
      update: values =>
        parameterBuffer.write(
          getGPUTerrainHydrologicIndicesParameterValues(
            {minimumSlope: values.minimumSlope},
            terrain.grid
          )
        )
    };
  },
  // The pointer reads the drawn index: its value under the pointer is the threshold, and every cell
  // that is at least as high is lit in full color.
  interaction: {
    kind: 'level',
    hint: 'raise the level',
    rest: [0.5, 0.88],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const output = Math.round(values.output);
      const index = peek(outputs.indices[output]) as Float32Array | null;
      if (!index) return {};
      const name = parameters[1].options?.[output] ?? 'index';
      // The height of the pointer is a rank: the share of cells that are lit, from none (bottom) to
      // all (top). The threshold is the value at that rank.
      const share = Math.min(1, Math.max(0, (1 - pointer.unitY - 0.05) / 0.9));
      const highlight = inputs.highlight;
      if (highlight.source !== index || highlight.output !== output) {
        inputs.highlight = {
          source: index,
          threshold: Number.NaN,
          output,
          sorted: Float32Array.from(index.filter(Number.isFinite)).sort()
        };
      }
      const sorted = inputs.highlight.sorted;
      const rank = Math.min(sorted.length - 1, Math.floor((1 - share) * sorted.length));
      const threshold = share <= 0 ? Number.POSITIVE_INFINITY : sorted[rank];
      if (!Object.is(inputs.highlight.threshold, threshold)) {
        outputs.highlighted.write(
          Float32Array.from(index, value => (value >= threshold ? value : Number.NaN))
        );
        inputs.highlight.threshold = threshold;
      }
      return {
        glyphs: [{shape: 'line', from: [0, pointer.y], to: [world.width, pointer.y], dashed: true}],
        readout: `${name} >= ${Number.isFinite(threshold) ? threshold.toFixed(2) : 'max'} · ${Math.round(share * 100)}% of cells`
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
    getTerrainContextLayer(inputs, outputs.filledElevation),
    {
      layer: 'raster',
      values: outputs.indices[Math.round(values.output)],
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      range: INDEX_RANGES[Math.round(values.output)],
      alpha: 0.5,
      smoothing: true
    },
    {
      layer: 'raster',
      values: outputs.highlighted,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      range: INDEX_RANGES[Math.round(values.output)],
      alpha: 1
    }
  ]
});
