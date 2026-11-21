// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURadiationFlowModel on the 226 Chicago tracts. Tract dots are colored by the expected flow from
// the chosen origin, a prediction from populations and distances alone. The dashed ring runs through
// the likeliest destination: the population inside it is what the model counts as intervening
// opportunities. Arrows are the strongest flows actually observed from the origin.

import {GPURadiationFlowModel} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {
  createTractFlows,
  findBusiestOrigin,
  findNearestOrigin,
  writeTractTotals
} from './_lib/tract-flows';
import {addMatrixRow, addRowShares} from './_lib/zone-kernels';
import {parameters} from './radiation-flow-model.parameters';

type Inputs = ReturnType<typeof createInputs>;
const PANEL = [0.7, 0.08, 0.96, 0.36] as const;
const SHARES = [0.03, 0.04, 0.3, 0.14] as const;

type Outputs = {
  row: RunBuffer;
  positions: RunBuffer;
  populations: RunBuffer;
  lines: RunBuffer;
  shares: RunBuffer;
};

const CAPACITY = 64;
const format = (value: number) => Math.round(value).toLocaleString('en-US');

function createInputs({world}: {world: {width: number; height: number}}) {
  const flows = createTractFlows(world);
  return {flows, origin: findBusiestOrigin(flows), restOrigin: findBusiestOrigin(flows)};
}

export default defineOperationRun<
  Inputs,
  Outputs,
  {predicted: RunBuffer; observed: RunBuffer; limit: number}
>({
  parameters,
  caption:
    'Expected flow out of one Chicago tract by the radiation model. A tract’s population here is the workers living plus the jobs located in it, summed over the 1,200 largest LODES 2021 flows that pass the minimum, not census totals. Bars: the share of each tract’s outflow the model places. Without normalization the model places all but about the origin’s own share of the population (97 to 99.9% here); finite-size normalization places exactly 100%, so the bars barely change.',
  createInputs,
  build(builder, inputs, values) {
    const {flows} = inputs;
    const populations = builder.allocate('populations', 'float32', flows.zoneCount);
    const outflows = builder.allocate('outflows', 'float32', flows.zoneCount);
    const positions = builder.upload('positions', 'float32x2', flows.positions);
    const expected = builder.allocate('expected-flows', 'float32', flows.zoneCount ** 2);
    builder.graph.add(
      new GPURadiationFlowModel({
        positions: positions.view,
        populations: populations.view,
        outflows: outflows.view,
        normalization: values.normalization === 1 ? 'finite-size' : 'none',
        output: {flows: expected.view}
      })
    );
    const selection = builder.allocate('selection', 'uint32', 1);
    return {
      outputs: {
        row: addMatrixRow(builder, {
          id: 'origin',
          matrix: expected,
          zoneCount: flows.zoneCount,
          selection
        }),
        positions,
        populations,
        shares: addRowShares(builder, {
          id: 'radiation',
          matrix: expected,
          outflows,
          zoneCount: flows.zoneCount
        }),
        lines: builder.upload('flow-lines', 'float32x4', flows.segments)
      },
      update: current => {
        writeTractTotals(flows, current.minimumJobs);
        populations.write(flows.populations);
        outflows.write(flows.outflows);
        selection.write(Uint32Array.of(inputs.origin));
      }
    };
  },
  rebuildOn: ['normalization'],
  async afterRun({inputs, outputs, values, upload}) {
    const {flows} = inputs;
    const bytes = await outputs.row.buffer.readAsync();
    const expected = new Float32Array(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
    );
    // One point per observed flow of the origin: observed workers against the model's expected flow.
    const list = flows.flowsByOrigin[inputs.origin].filter(
      flow => flows.jobs[flow] >= values.minimumJobs
    );
    const observed = new Float32Array(CAPACITY);
    const predicted = new Float32Array(CAPACITY);
    let limit = 1;
    for (let row = 0; row < CAPACITY; row++) {
      const flow = list[Math.min(row, list.length - 1)];
      observed[row] = flow === undefined ? 0 : flows.jobs[flow];
      predicted[row] = flow === undefined ? 0 : expected[flows.destinations[flow]];
      limit = Math.max(limit, observed[row], predicted[row]);
    }
    limit *= 1.15;
    return {
      observed: upload('observed', 'float32', observed),
      predicted: upload('predicted', 'float32', predicted),
      limit
    };
  },
  // Probe: the nearest tract with flows is the origin; its expected flows color every tract.
  interaction: {
    kind: 'probe',
    hint: 'hover an origin tract',
    rest: ({inputs, toUnit}) =>
      toUnit([
        inputs.flows.positions[inputs.restOrigin * 2],
        inputs.flows.positions[inputs.restOrigin * 2 + 1]
      ]),
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const {flows} = inputs;
      const origin = findNearestOrigin(flows, pointer.x, pointer.y);
      const changed = origin !== inputs.origin;
      inputs.origin = origin;
      const expected = peek(outputs.row);
      const from: [number, number] = [flows.positions[origin * 2], flows.positions[origin * 2 + 1]];
      const glyphs: RunGlyph[] = [{shape: 'dot', at: from}];
      const strongest = flows.flowsByOrigin[origin].filter(
        flow => flows.jobs[flow] >= values.minimumJobs
      );
      for (const flow of strongest.slice(0, 6)) {
        const destination = flows.destinations[flow];
        glyphs.push({
          shape: 'arrow',
          from,
          to: [flows.positions[destination * 2], flows.positions[destination * 2 + 1]]
        });
      }
      let readout = `origin ${origin} · ${format(flows.outflows[origin])} leave`;
      const panel = (rectangle: readonly number[]) => ({
        minimum: [rectangle[0] * world.width, rectangle[1]] as [number, number],
        maximum: [rectangle[2] * world.width, rectangle[3]] as [number, number]
      });
      glyphs.push(
        {shape: 'box', ...panel(PANEL)},
        {
          shape: 'line',
          from: [PANEL[0] * world.width, PANEL[1]],
          to: [PANEL[2] * world.width, PANEL[3]],
          dashed: true
        }
      );
      if (expected) {
        const top = strongest[0];
        const destination = top === undefined ? origin : flows.destinations[top];
        const radius = Math.hypot(
          flows.positions[destination * 2] - from[0],
          flows.positions[destination * 2 + 1] - from[1]
        );
        // Intervening opportunities: the population strictly nearer to the origin than the destination.
        let between = 0;
        for (let zone = 0; zone < flows.zoneCount; zone++) {
          if (zone === origin || zone === destination) continue;
          const distance = Math.hypot(
            flows.positions[zone * 2] - from[0],
            flows.positions[zone * 2 + 1] - from[1]
          );
          if (distance < radius) between += flows.populations[zone];
        }
        glyphs.push({shape: 'ring', at: from, radius, dashed: true});
        let rowSum = 0;
        for (let zone = 0; zone < expected.length; zone++) {
          if (zone !== origin) rowSum += expected[zone];
        }
        const share = ((rowSum / Math.max(flows.outflows[origin], 1)) * 100).toFixed(1);
        readout += ` · model places ${share}% of them`;
        if (top !== undefined) {
          readout += ` · to tract ${destination}: expected ${format(expected[destination])}, observed ${format(flows.jobs[top])} · ${format(between)} people nearer`;
        }
      }
      return {rerun: changed, spotlight: null, glyphs, readout};
    }
  },
  draw: ({inputs, outputs, extra}) => [
    {
      layer: 'lines',
      context: true,
      segments: outputs.lines,
      radius: 0.6,
      color: [0.5, 0.55, 0.65, 0.04]
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
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.shares,
      range: [0.7, 1],
      rectangle: SHARES,
      color: [0.95, 0.7, 0.25, 0.9]
    },
    {
      layer: 'chart',
      kind: 'scatter',
      values: extra.predicted,
      xValues: extra.observed,
      xRange: [0, extra.limit],
      range: [0, extra.limit],
      rectangle: PANEL,
      color: [0.95, 0.7, 0.25, 0.95],
      radius: 3
    }
  ]
});
