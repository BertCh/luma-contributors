// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainBranchingFlow over the eight-neighbor fractions of GPUTerrainFlowFractions (MFD on the
// filled DEM): transport of unit rain, expected / minimum / maximum flow distance to an outlet,
// or the chance that a cell drains to the outlets on the upper half of the border.

import {
  GPUTerrainBranchingFlow,
  GPUTerrainFlowFractions
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {parameters} from './terrain-branching-flow.parameters';
import {addTerrainFlow, createTerrainInputs, getTerrainContextLayer} from './shared/terrain';

const MODES = [
  'transport',
  'expected-distance',
  'minimum-distance',
  'maximum-distance',
  'fractional-catchment'
] as const;

export default defineOperationRun({
  parameters,
  caption: 'Mass fans out and rejoins, or distance and outlet probability, through the fractions.',
  createInputs: context => {
    const terrain = createTerrainInputs(context);
    // Outlet probability: 1 on the upper half of the border, 0 elsewhere.
    const outletProbability = new Float32Array(terrain.cellCount);
    for (let cell = 0; cell < terrain.cellCount; cell++) {
      outletProbability[cell] = Math.floor(cell / terrain.columns) < terrain.rows / 2 ? 1 : 0;
    }
    return {...terrain, outletProbability};
  },
  build(builder, terrain, values) {
    const mode = MODES[Math.round(values.mode)];
    const flow = addTerrainFlow(builder, terrain);
    const fractions = builder.allocate('fractions', 'float32', terrain.cellCount * 8);
    const edgeLengths = builder.allocate('edge-lengths', 'float32', terrain.cellCount * 8);
    builder.graph.add(
      new GPUTerrainFlowFractions({
        grid: terrain.grid,
        elevation: flow.filledElevation.view,
        flowDirections: flow.flowDirections.view,
        parameters: flow.parameters.view,
        routing: 'mfd-freeman',
        output: {fractions: fractions.view, edgeLengths: edgeLengths.view}
      })
    );
    const source = builder.upload(
      'source',
      'float32',
      mode === 'fractional-catchment'
        ? terrain.outletProbability
        : new Float32Array(terrain.cellCount).fill(1)
    );
    const result = builder.allocate('result', 'float32', terrain.cellCount);
    builder.graph.add(
      new GPUTerrainBranchingFlow({
        grid: terrain.grid,
        fractions: fractions.view,
        values: source.view,
        mode,
        edgeLengths: mode === 'transport' ? undefined : edgeLengths.view,
        output: {
          values: result.view,
          status: builder.allocate('status', 'uint32', terrain.cellCount).view,
          unresolvedCount: builder.allocate('unresolved-count', 'uint32', 1).view
        }
      })
    );
    return {
      outputs: {filledElevation: flow.filledElevation, result},
      update: current => flow.writeSettings({flowExponent: current.flowExponent})
    };
  },
  rebuildOn: ['mode'],
  draw: ({inputs, outputs, values}) => [
    getTerrainContextLayer(inputs, outputs.filledElevation),
    {
      layer: 'raster',
      values: outputs.result,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      range: values.mode === 0 ? [0, 60] : values.mode === 4 ? [0, 1] : 'auto',
      fade: values.mode === 0,
      alpha: 0.9,
      smoothing: true
    }
  ]
});
