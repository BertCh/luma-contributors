// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainHeightAboveDrainage (HAND) on the filled DEM, D8 directions and stream mask of
// GPUTerrainFlow: each cell's height above the stream cell its flow path reaches first. Raising the
// stage of the streams floods every cell whose HAND is not above it.

import {GPUTerrainHeightAboveDrainage} from '@luma-contributors/analysis/gpu-terrain';
import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import {makeGPURasterBand} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {clamp} from '../../interaction';
import {parameters} from './terrain-height-above-drainage.parameters';
import {addTerrainFlow, createTerrainInputs, getTerrainContextLayer} from './shared/terrain';

const MAXIMUM_STAGE = 0.06;
const DEFAULT_STAGE = 0.012;
const WATER_COLORS = [
  [0, 0, 0, 0],
  [0.25, 0.55, 1, 0.8]
] as const;

/** Stage of the flood for a pointer y (fraction of the height, 0 at the top): the inner 90% spans it. */
function getStage(unitY: number): number {
  return clamp((1 - unitY - 0.05) / 0.9, 0, 1) * MAXIMUM_STAGE;
}

export default defineOperationRun({
  parameters,
  caption: 'Height above the nearest stream: low cells near streams light up.',
  createInputs: context => ({...createTerrainInputs(context), stage: DEFAULT_STAGE}),
  build(builder, terrain) {
    const flow = addTerrainFlow(builder, terrain);
    const heightAboveDrainage = builder.allocate('hand', 'float32', terrain.cellCount);
    builder.graph.add(
      new GPUTerrainHeightAboveDrainage({
        grid: terrain.grid,
        elevation: makeGPURasterBand(flow.filledElevation.view, terrain.grid, {id: 'filled'}),
        flowDirections: flow.flowDirections.view,
        streams: flow.streams.view,
        output: {heightAboveDrainage: heightAboveDrainage.view}
      })
    );
    // Cells whose height above drainage is within the stage: the flooded area.
    const stage = builder.allocate('stage', 'float32', 1);
    const water = builder.allocate('water', 'uint32', terrain.cellCount);
    builder.graph.add({
      getCommandNodes: graph => [
        createWGSLKernelNode(graph, {
          id: 'flood-stage',
          operation: 'flood-stage',
          bindings: [
            {name: 'hand', view: heightAboveDrainage.view, type: 'f32', access: 'read'},
            {name: 'stage', view: stage.view, type: 'f32', access: 'read'},
            {name: 'water', view: water.view, type: 'u32', access: 'read_write'}
          ],
          invocationCount: terrain.cellCount,
          body: 'water[waterOffset + index] = select(0u, 1u, hand[handOffset + index] <= stage[stageOffset]);'
        })
      ]
    });
    return {
      outputs: {filledElevation: flow.filledElevation, heightAboveDrainage, water},
      update: values => {
        flow.writeStreamThreshold(values.streamThreshold);
        stage.write(Float32Array.of(terrain.stage));
      }
    };
  },
  // The pointer height is the flood stage: cells that stand no higher above their stream than the
  // stage are filled with water.
  interaction: {
    kind: 'level',
    hint: 'raise the flood',
    rest: [0.5, 1 - (0.05 + 0.9 * (DEFAULT_STAGE / MAXIMUM_STAGE))],
    resolve(pointer, {inputs, outputs, world, peek}) {
      const stage = Math.round(getStage(pointer.unitY) * 1000) / 1000;
      const changed = inputs.stage !== stage;
      inputs.stage = stage;
      const hand = peek(outputs.heightAboveDrainage);
      let flooded = 0;
      if (hand) {
        for (const value of hand) if (value <= stage) flooded++;
      }
      return {
        rerun: changed,
        glyphs: [{shape: 'line', from: [0, pointer.y], to: [world.width, pointer.y], dashed: true}],
        readout: `stage ${stage.toFixed(3)}${hand ? ` · flooded ${Math.round((100 * flooded) / hand.length)}%` : ''}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    getTerrainContextLayer(inputs, outputs.filledElevation),
    {
      layer: 'raster',
      values: outputs.heightAboveDrainage,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      range: [0, 0.06],
      alpha: 0.85,
      smoothing: true
    },
    {
      layer: 'raster',
      values: outputs.water,
      columns: inputs.columns,
      rows: inputs.rows,
      categories: WATER_COLORS
    }
  ]
});
