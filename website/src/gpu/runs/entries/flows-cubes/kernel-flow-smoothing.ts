// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUKernelFlowSmoothing, in the manner of Guo and Zhu, on the 226 Chicago tracts. Hover an origin:
// tract dots are colored by the smoothed flow from it to every tract, arrows are its raw flows. The
// ring is the origin's neighborhood; its strongest flow is re-estimated from the flows between the
// tracts inside the ring at the origin and inside the ring at the destination.

import {
  GPUKernelFlowSmoothing,
  getGPUKernelFlowSmoothingParameterValues
} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {createTractFlows, findBusiestOrigin, findNearestOrigin} from './_lib/tract-flows';
import {addMatrixRow} from './_lib/zone-kernels';
import {parameters} from './kernel-flow-smoothing.parameters';

type Inputs = ReturnType<typeof createInputs>;
type Outputs = {
  row: RunBuffer;
  smoothed: RunBuffer;
  radii: RunBuffer;
  positions: RunBuffer;
  lines: RunBuffer;
};

function createInputs({world}: {world: {width: number; height: number}}) {
  const flows = createTractFlows(world);
  return {flows, origin: findBusiestOrigin(flows), restOrigin: findBusiestOrigin(flows)};
}

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Flows out of one Chicago tract after kernel smoothing of the 1,200 largest tract-to-tract flows (LODES, 2021): each is re-estimated from neighboring origins and destinations, and pairs not listed count as zero.',
  createInputs,
  build(builder, inputs) {
    const {flows} = inputs;
    const positions = builder.upload('positions', 'float32x2', flows.positions);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const smoothed = builder.allocate('smoothed', 'float32', flows.zoneCount ** 2);
    const radii = builder.allocate('radii', 'float32', flows.zoneCount);
    builder.graph.add(
      new GPUKernelFlowSmoothing({
        positions: positions.view,
        locationWeights: builder.upload(
          'location-weights',
          'float32',
          new Float32Array(flows.zoneCount).fill(1)
        ).view,
        flows: {
          kind: 'pairs',
          originZoneIds: builder.upload('origin-zones', 'uint32', flows.origins).view,
          destinationZoneIds: builder.upload('destination-zones', 'uint32', flows.destinations)
            .view,
          weights: builder.upload('weights', 'float32', flows.jobs).view,
          validPairCount: builder.upload('pair-count', 'uint32', Uint32Array.of(flows.jobs.length))
            .view
        },
        parameters: parameterBuffer.view,
        output: {smoothed: smoothed.view, radii: radii.view}
      })
    );
    const selection = builder.allocate('selection', 'uint32', 1);
    return {
      outputs: {
        row: addMatrixRow(builder, {
          id: 'origin',
          matrix: smoothed,
          zoneCount: flows.zoneCount,
          selection
        }),
        smoothed,
        radii,
        positions,
        lines: builder.upload('flow-lines', 'float32x4', flows.segments)
      },
      update: current => {
        parameterBuffer.write(
          getGPUKernelFlowSmoothingParameterValues({
            neighborhoodSize: current.neighborhoodSize,
            bandwidthRatio: current.bandwidthRatio
          })
        );
        selection.write(Uint32Array.of(inputs.origin));
      }
    };
  },
  // Probe: the nearest tract with flows is the origin.
  interaction: {
    kind: 'probe',
    hint: 'hover an origin tract',
    rest: ({inputs, toUnit}) =>
      toUnit([
        inputs.flows.positions[inputs.restOrigin * 2],
        inputs.flows.positions[inputs.restOrigin * 2 + 1]
      ]),
    resolve(pointer, {inputs, outputs, peek}) {
      const {flows} = inputs;
      const origin = findNearestOrigin(flows, pointer.x, pointer.y);
      const changed = origin !== inputs.origin;
      inputs.origin = origin;
      const from: [number, number] = [flows.positions[origin * 2], flows.positions[origin * 2 + 1]];
      const glyphs: RunGlyph[] = [{shape: 'dot', at: from}];
      const strongest = flows.flowsByOrigin[origin][0];
      for (const flow of flows.flowsByOrigin[origin].slice(0, 6)) {
        const destination = flows.destinations[flow];
        glyphs.push({
          shape: 'arrow',
          from,
          to: [flows.positions[destination * 2], flows.positions[destination * 2 + 1]]
        });
      }
      const radii = peek(outputs.radii);
      const matrix = peek(outputs.smoothed);
      let readout = `origin ${origin}`;
      if (radii && strongest !== undefined) {
        const destination = flows.destinations[strongest];
        const at: [number, number] = [
          flows.positions[destination * 2],
          flows.positions[destination * 2 + 1]
        ];
        glyphs.push(
          {shape: 'ring', at: from, radius: radii[origin], dashed: true},
          {shape: 'ring', at: at, radius: radii[destination], dashed: true}
        );
        let tracts = 0;
        for (let zone = 0; zone < flows.zoneCount; zone++) {
          if (
            Math.hypot(
              flows.positions[zone * 2] - from[0],
              flows.positions[zone * 2 + 1] - from[1]
            ) <= radii[origin]
          ) {
            tracts++;
          }
        }
        const smoothed = matrix ? matrix[origin * flows.zoneCount + destination] : Number.NaN;
        readout += ` · top flow raw ${Math.round(flows.jobs[strongest])} → smoothed ${
          Number.isFinite(smoothed) ? smoothed.toFixed(1) : '…'
        } · pools ${tracts} tracts at the origin`;
      }
      return {rerun: changed, spotlight: null, glyphs, readout};
    }
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'lines',
      context: true,
      segments: outputs.lines,
      radius: 0.8,
      color: [0.5, 0.55, 0.65, 0.06]
    },
    {
      layer: 'dots',
      context: true,
      positions: outputs.positions,
      radius: 3.5
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.row,
      count: inputs.flows.zoneCount,
      logarithmic: true,
      range: [0, 6],
      radius: 6
    }
  ]
});
