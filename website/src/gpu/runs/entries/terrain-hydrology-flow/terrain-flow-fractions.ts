// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainFlowFractions on the filled DEM and D8 directions of GPUTerrainFlow writes the eight
// per-neighbor fractions of D-infinity or MFD routing. The picture shows what they do: unit rain
// on every cell is passed down those fractions by GPUTerrainBranchingFlow (transport), so
// concentrated routings give sharp channels and a high exponent or D-infinity keeps them narrow.

import {
  GPUTerrainBranchingFlow,
  GPUTerrainFlowFractions
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {parameters} from './terrain-flow-fractions.parameters';
import {addTerrainFlow, createTerrainInputs, getTerrainContextLayer} from './shared/terrain';

const ROUTINGS = ['d-infinity', 'mfd-freeman', 'mfd-quinn'] as const;

export default defineOperationRun({
  parameters,
  caption:
    'Unit rain routed through the fractions: flow splits across neighbors on divergent slopes.',
  createInputs: context => createTerrainInputs(context),
  build(builder, terrain, values) {
    const flow = addTerrainFlow(builder, terrain);
    const fractions = builder.allocate('fractions', 'float32', terrain.cellCount * 8);
    const edgeLengths = builder.allocate('edge-lengths', 'float32', terrain.cellCount * 8);
    builder.graph.add(
      new GPUTerrainFlowFractions({
        grid: terrain.grid,
        elevation: flow.filledElevation.view,
        flowDirections: flow.flowDirections.view,
        parameters: flow.parameters.view,
        routing: ROUTINGS[Math.round(values.routing)],
        output: {fractions: fractions.view, edgeLengths: edgeLengths.view}
      })
    );
    const rain = builder.upload('rain', 'float32', new Float32Array(terrain.cellCount).fill(1));
    const transported = builder.allocate('transported', 'float32', terrain.cellCount);
    builder.graph.add(
      new GPUTerrainBranchingFlow({
        grid: terrain.grid,
        fractions: fractions.view,
        values: rain.view,
        mode: 'transport',
        output: {
          values: transported.view,
          status: builder.allocate('status', 'uint32', terrain.cellCount).view,
          unresolvedCount: builder.allocate('unresolved-count', 'uint32', 1).view
        }
      })
    );
    return {
      outputs: {filledElevation: flow.filledElevation, transported},
      update: current => flow.writeSettings({flowExponent: current.flowExponent})
    };
  },
  rebuildOn: ['routing'],
  draw: ({inputs, outputs}) => [
    getTerrainContextLayer(inputs, outputs.filledElevation),
    {
      layer: 'raster',
      values: outputs.transported,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      range: [0, 60],
      fade: true,
      smoothing: true
    }
  ]
});
