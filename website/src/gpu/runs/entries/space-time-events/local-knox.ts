// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULocalKnox on the 2018 Kīlauea earthquakes. A radius-mode GPUNeighborSearch lists the events
// close in space; for every event the operation counts the neighbors that are also close in time and
// tests that count against permuted times. Events with pseudo p <= 0.05 are colored by their local
// Knox count; the rest stay grey. The pointer is the pad of the two thresholds.

import {
  getGPUNeighborSearchParameterValues,
  getGPUSpaceTimeParameterValues,
  GPULocalKnox,
  GPUNeighborSearch,
  GPU_LOCAL_KNOX_SUMMARY,
  GPU_LOCAL_KNOX_SUMMARY_LENGTH,
  GPU_SPACE_TIME_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {bindAxes} from '../../interaction';
import {
  allocateNeighborWeights,
  createKilaueaEvents,
  findWithin,
  shuffleTimes
} from './_lib/events';
import {addKernelPass} from './_lib/kernel';
import {parameters} from './local-knox.parameters';

const MAXIMUM_PERMUTATIONS = 199;
const SIGNIFICANCE = 0.05;
/** Links drawn from the hovered event; the readout counts all. */
const MAXIMUM_LINKS = 12;

type Inputs = ReturnType<typeof createKilaueaEvents> & {width: number};
type Outputs = {positions: RunBuffer; significant: RunBuffer; summary: RunBuffer};

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Kīlauea summit earthquakes, May–August 2018: colored by local Knox count where it beats 199 time permutations (p ≤ 0.05, uncorrected).',
  createInputs: ({world}) => ({
    ...createKilaueaEvents([0, 0, world.width, world.height]),
    width: world.width
  }),
  build(builder, events) {
    const count = events.count;
    const positions = builder.upload('positions', 'float32x2', events.positions);
    const searchParameters = builder.allocate('search-parameters', 'float32', 12);
    const pairs = allocateNeighborWeights(builder, 'pairs', count, count * count);
    builder.graph.add(
      new GPUNeighborSearch({
        mode: 'radius',
        positions: positions.view,
        gridSize: [24, 16],
        parameters: searchParameters.view,
        output: pairs.port
      })
    );
    const parameterBuffer = builder.allocate(
      'parameters',
      'uint32',
      GPU_SPACE_TIME_PARAMETER_LENGTH
    );
    const times = builder.allocate('times', 'float32', count);
    const summary = builder.allocate('summary', 'float32', count * GPU_LOCAL_KNOX_SUMMARY_LENGTH);
    builder.graph.add(
      new GPULocalKnox({
        pairs: pairs.input,
        times: times.view,
        parameters: parameterBuffer.view,
        maximumPermutations: MAXIMUM_PERMUTATIONS,
        output: {summary: summary.view}
      })
    );
    // Local Knox count of each event where its pseudo p-value passes, NaN (not drawn) otherwise.
    const significant = builder.allocate('significant', 'float32', count);
    addKernelPass(builder, {
      id: 'local-knox-significant',
      invocationCount: count,
      bindings: [
        {name: 'summary', buffer: summary, access: 'read'},
        {name: 'significant', buffer: significant, access: 'read_write'}
      ],
      body: `let row = summaryOffset + index * ${GPU_LOCAL_KNOX_SUMMARY_LENGTH}u;
  let observed = summary[row + ${GPU_LOCAL_KNOX_SUMMARY.observed}u];
  let pValue = summary[row + ${GPU_LOCAL_KNOX_SUMMARY.pseudoPGreater}u];
  var notANumberBits = 0x7fc00000u;
  significant[significantOffset + index] = select(bitcast<f32>(notANumberBits), observed, observed > 0.0 && pValue <= ${SIGNIFICANCE});`
    });
    return {
      outputs: {positions, significant, summary},
      update: current => {
        times.write(shuffleTimes(events.days, current.observedShare));
        searchParameters.write(
          getGPUNeighborSearchParameterValues({
            bounds: [0, 0, events.width, 1],
            radius: current.radius / events.kilometersPerUnit
          })
        );
        parameterBuffer.write(
          getGPUSpaceTimeParameterValues({
            seed: 7,
            permutationCount: MAXIMUM_PERMUTATIONS,
            timeThreshold: current.timeThreshold
          })
        );
      }
    };
  },
  // Pad: time threshold across, radius up. The ring at the pointer is the radius to scale, and the
  // nearest event is linked to its neighbors that are also within the time threshold.
  interaction: bindAxes<Inputs, Outputs>(parameters, {
    kind: 'pad',
    hint: 'time across, radius up',
    x: 'timeThreshold',
    y: 'radius',
    decorate: (values, pointer, {inputs, outputs, peek}) => {
      const radius = values.radius / inputs.kilometersPerUnit;
      const row = inputs.index.nearest(pointer.x, pointer.y);
      const times = shuffleTimes(inputs.days, values.observedShare);
      const glyphs = [];
      let closeCount = 0;
      let neighborCount = 0;
      if (row >= 0) {
        const at = [inputs.positions[row * 2], inputs.positions[row * 2 + 1]] as const;
        for (const other of findWithin(inputs.positions, row, radius)) {
          const isClose = Math.abs(times[other] - times[row]) <= values.timeThreshold;
          neighborCount++;
          if (!isClose) continue;
          closeCount++;
          if (closeCount > MAXIMUM_LINKS) continue;
          glyphs.push({
            shape: 'line' as const,
            from: at,
            to: [inputs.positions[other * 2], inputs.positions[other * 2 + 1]] as const
          });
        }
        glyphs.push({shape: 'ring' as const, at, radius, dashed: true});
      }
      const significant = peek(outputs.significant);
      const significantCount = significant
        ? Array.from(significant).filter(Number.isFinite).length
        : undefined;
      return {
        glyphs,
        readout: `${significantCount ?? '…'} events p ≤ ${SIGNIFICANCE} · ${values.radius} km, ${values.timeThreshold} d · here ${closeCount}/${neighborCount}`
      };
    }
  }),
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2.5, alpha: 0.4},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.significant,
      range: [1, 7],
      logarithmic: true,
      radius: 4.5
    }
  ]
});
