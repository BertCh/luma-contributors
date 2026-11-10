// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUInequality over workplace job counts grouped by tract: one inequality index per zone, written by
// the GPU and drawn as the tract fill. `output` picks which index is shown.

import {
  GPUInequality,
  GPU_INEQUALITY_PARAMETER_LENGTH,
  getGPUInequalityParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {getWorkplaceRows} from '../../../data/workplace-rows';
import {createTractProbe, uploadTractGeometry} from './_lib/helpers';
import {parameters} from './inequality.parameters';

// Fixed color ranges: gini, atkinson and hoover lie in [0, 1]; theilT is bounded by ln of the block count.
const OUTPUT_RANGES = [
  [0, 1],
  [0, 2],
  [0, 1],
  [0, 1]
] as const;
const OUTPUT_NAMES = ['Gini', 'Theil T', 'Atkinson', 'Hoover'] as const;
const LORENZ_POINT_LIMIT = 40;

function createInputs({
  world,
  random
}: {
  world: {width: number; height: number};
  random: () => number;
}) {
  const rows = getWorkplaceRows(random, world);
  return {...rows, world, probe: createTractProbe(rows.tracts, rows.keys), focus: {tract: -1}};
}
type Inputs = ReturnType<typeof createInputs>;
type Outputs = ReturnType<typeof uploadTractGeometry> &
  Record<'gini' | 'theilT' | 'atkinson' | 'hoover', RunBuffer>;

/** The Lorenz curve of a tract's block jobs as packed (x, y) points inside the inset box. */
function getLorenzPoints(jobs: Float32Array, rows: readonly number[], inset: readonly number[]) {
  const sorted = rows.map(row => jobs[row]).sort((first, second) => first - second);
  const total = sorted.reduce((sum, value) => sum + value, 0) || 1;
  const step = Math.max(1, Math.ceil(sorted.length / LORENZ_POINT_LIMIT));
  const points: number[] = [inset[0], inset[1]];
  let cumulative = 0;
  sorted.forEach((value, index) => {
    cumulative += value;
    if ((index + 1) % step === 0 || index === sorted.length - 1) {
      points.push(
        inset[0] + ((index + 1) / sorted.length) * (inset[2] - inset[0]),
        inset[1] + (cumulative / total) * (inset[3] - inset[1])
      );
    }
  });
  return points;
}

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption: 'Concentration of jobs across observed workplace blocks in each Chicago tract.',
  createInputs,
  build(builder, inputs) {
    const {tracts} = inputs;
    const jobs = builder.upload('jobs', 'float32', inputs.jobs);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_INEQUALITY_PARAMETER_LENGTH
    );
    const gini = builder.allocate('gini', 'float32', tracts.count);
    const theilT = builder.allocate('theil-t', 'float32', tracts.count);
    const atkinson = builder.allocate('atkinson', 'float32', tracts.count);
    const hoover = builder.allocate('hoover', 'float32', tracts.count);
    builder.graph.add(
      new GPUInequality({
        values: jobs.view,
        zoneIds: builder.upload('zone-ids', 'uint32', inputs.keys).view,
        zoneCount: tracts.count,
        parameters: parameterBuffer.view,
        output: {gini: gini.view, theilT: theilT.view, atkinson: atkinson.view, hoover: hoover.view}
      })
    );
    return {
      outputs: {...uploadTractGeometry(builder, tracts), gini, theilT, atkinson, hoover},
      update: current => {
        parameterBuffer.write(getGPUInequalityParameterValues({epsilon: current.epsilon}));
      }
    };
  },
  rebuildOn: [],
  // The pointer is a tract: its outline is traced and the Lorenz curve of its blocks is drawn in an
  // inset, with the indices the GPU wrote for it in the readout.
  interaction: {
    kind: 'probe',
    hint: 'hover a tract',
    rest: ({inputs}) => {
      const center = inputs.tracts.centers.subarray(
        inputs.probe.richest * 2,
        inputs.probe.richest * 2 + 2
      );
      return [center[0] / inputs.world.width, 1 - center[1] / inputs.world.height];
    },
    resolve(pointer, {inputs, outputs, values, peek}) {
      const found = inputs.probe.findTract(pointer.x, pointer.y);
      if (found >= 0) inputs.focus.tract = found;
      const tract = inputs.focus.tract;
      if (tract < 0) return {readout: 'hover a tract'};
      const inset = [0.03 * inputs.world.width, 0.04, 0.03 * inputs.world.width + 0.24, 0.28];
      const glyphs: RunGlyph[] = [
        {shape: 'polyline', points: inputs.probe.getRing(tract), closed: true},
        {shape: 'box', minimum: [inset[0], inset[1]], maximum: [inset[2], inset[3]]},
        {shape: 'line', from: [inset[0], inset[1]], to: [inset[2], inset[3]], dashed: true},
        {
          shape: 'polyline',
          points: getLorenzPoints(inputs.jobs, inputs.probe.members[tract], inset)
        }
      ];
      const text = (buffer: RunBuffer) => {
        const value = peek(buffer)?.[tract];
        return value === undefined ? '…' : value.toFixed(2);
      };
      const shown = [outputs.gini, outputs.theilT, outputs.atkinson, outputs.hoover];
      return {
        glyphs,
        readout: `tract ${inputs.tracts.geoids[tract].slice(-6)} · ${OUTPUT_NAMES[values.output]} ${text(
          shown[values.output]
        )} · ${inputs.probe.members[tract].length} blocks`
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: [outputs.gini, outputs.theilT, outputs.atkinson, outputs.hoover][values.output],
      triangleCount: inputs.tracts.triangleCount,
      range: OUTPUT_RANGES[values.output]
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.tracts.outlineCount}
  ]
});
