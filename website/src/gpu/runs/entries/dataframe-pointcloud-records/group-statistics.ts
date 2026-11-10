// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGroupStatistics over workplace blocks keyed by tract (dense mode). The tract fill is the
// per-group statistic the GPU wrote; the dots are the rows themselves.

import {GPUGroupStatistics} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {getWorkplaceRows} from '../../../data/workplace-rows';
import {
  createBoxMask,
  createTractProbe,
  getBoxMaskBounds,
  uploadTractGeometry,
  writeRowPositions
} from './_lib/helpers';
import {parameters} from './group-statistics.parameters';

// Approximate 90th percentiles across tracts for mean, median, standard deviation and maximum.
const STATISTICS = ['mean', 'median', 'std dev', 'max'] as const;
const DISPLAY_LIMITS = [485, 100, 1685, 7750];

type Inputs = ReturnType<typeof createInputs>;
type Outputs = ReturnType<typeof uploadTractGeometry> &
  Record<'means' | 'medians' | 'standardDeviations' | 'maximums' | 'rows' | 'members', RunBuffer>;

function getMaskBounds(inputs: Pick<Inputs, 'world'>, share: number) {
  return getBoxMaskBounds(inputs.world.width, inputs.world.height, share);
}

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

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption: 'Jobs per workplace block, grouped by Chicago tract.',
  createInputs,
  build(builder, inputs) {
    const {tracts} = inputs;
    const column = builder.upload('jobs', 'float32', inputs.jobs);
    const mask = builder.allocate('mask', 'uint32', inputs.jobs.length);
    const means = builder.allocate('means', 'float32', tracts.count);
    const medians = builder.allocate('medians', 'float32', tracts.count);
    const standardDeviations = builder.allocate('standard-deviations', 'float32', tracts.count);
    const maximums = builder.allocate('maximums', 'float32', tracts.count);
    builder.graph.add(
      new GPUGroupStatistics({
        keys: builder.upload('keys', 'uint32', inputs.keys).view,
        keyCount: tracts.count,
        mask: mask.view,
        columns: [
          {
            values: column.view,
            statistics: ['mean', 'median', 'standardDeviation', 'maximum'],
            output: {
              means: means.view,
              medians: medians.view,
              standardDeviations: standardDeviations.view,
              maximums: maximums.view
            }
          }
        ],
        output: {
          keys: builder.allocate('group-keys', 'uint32', tracts.count).view,
          counts: builder.allocate('group-counts', 'uint32', tracts.count).view,
          status: {
            count: builder.allocate('group-count', 'uint32', 1).view,
            overflow: builder.allocate('group-overflow', 'uint32', 1).view
          }
        }
      })
    );
    return {
      outputs: {
        ...uploadTractGeometry(builder, tracts),
        means,
        medians,
        standardDeviations,
        maximums,
        rows: builder.upload('rows', 'float32x2', inputs.positions),
        members: builder.allocate('members', 'float32x2', inputs.jobs.length)
      },
      update: current => {
        mask.write(createBoxMask(inputs.positions, getMaskBounds(inputs, current.mask)));
      }
    };
  },
  rebuildOn: [],
  // The pointer is a tract: its outline is traced, its blocks drawn bright, and the readout gives
  // the statistic the GPU wrote for that group.
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
      const bounds = getMaskBounds(inputs, values.mask);
      const box = {
        shape: 'box' as const,
        minimum: [bounds.minimumX, bounds.minimumY] as const,
        maximum: [bounds.maximumX, bounds.maximumY] as const,
        dashed: true
      };
      if (tract < 0) return {glyphs: [box], readout: 'hover a tract'};
      const kept = createBoxMask(inputs.positions, bounds);
      const rows = inputs.probe.members[tract].filter(row => kept[row]);
      const count = writeRowPositions(outputs.members, inputs.positions, rows);
      const statistic = peek(
        [outputs.means, outputs.medians, outputs.standardDeviations, outputs.maximums][
          values.statistic
        ]
      )?.[tract];
      return {
        focus: count,
        glyphs: [{shape: 'polyline', points: inputs.probe.getRing(tract), closed: true}, box],
        readout: `tract ${inputs.tracts.geoids[tract].slice(-6)} · ${count} blocks · ${STATISTICS[values.statistic]} ${
          statistic === undefined ? '…' : Math.round(statistic)
        } jobs`
      };
    }
  },
  draw: ({inputs, outputs, values, interaction}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: [outputs.means, outputs.medians, outputs.standardDeviations, outputs.maximums][
        values.statistic
      ],
      triangleCount: inputs.tracts.triangleCount,
      range: [0, DISPLAY_LIMITS[values.statistic]]
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.tracts.outlineCount},
    {layer: 'dots', context: true, positions: outputs.rows, color: [1, 1, 1, 0.55], radius: 2},
    {
      layer: 'dots',
      positions: outputs.members,
      count: typeof interaction.focus === 'number' ? Math.max(1, interaction.focus) : 1,
      color: [1, 0.75, 0.2, 1],
      radius: 3.6
    }
  ]
});
