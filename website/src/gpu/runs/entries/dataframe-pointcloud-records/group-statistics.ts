// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGroupStatistics over household rows keyed by tract (dense mode). The tract fill is the
// per-group statistic the GPU wrote; the dots are the rows themselves.

import {GPUGroupStatistics} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts} from '../../inputs';
import {assignNearestSites, drawNormal, hashToUnit, uploadTractGeometry} from './_lib/helpers';
import {parameters} from './group-statistics.parameters';

const ROW_COUNT = 4000;

export default defineOperationRun({
  parameters,
  caption: 'One statistic per tract, computed from the rows keyed to it.',
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, world, 12, 0.1);
    const positions = Float32Array.from({length: ROW_COUNT * 2}, (_, index) =>
      index % 2 === 0 ? random() * world.width : random()
    );
    return {
      tracts,
      positions,
      keys: assignNearestSites(positions, tracts.centers),
      normals: Float32Array.from({length: ROW_COUNT}, () => drawNormal(random))
    };
  },
  build(builder, inputs) {
    const {tracts} = inputs;
    const column = builder.allocate('values', 'float32', ROW_COUNT);
    const mask = builder.allocate('mask', 'uint32', ROW_COUNT);
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
        rows: builder.upload('rows', 'float32x2', inputs.positions)
      },
      update: current => {
        column.write(
          Float32Array.from(inputs.keys, (key, row) =>
            Math.max(0, 10 + 40 * tracts.values[key] + inputs.normals[row] * 8 * current.noise)
          )
        );
        mask.write(
          Uint32Array.from({length: ROW_COUNT}, (_, row) =>
            hashToUnit(row) * 100 < current.mask ? 1 : 0
          )
        );
      }
    };
  },
  rebuildOn: [],
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: [outputs.means, outputs.medians, outputs.standardDeviations, outputs.maximums][
        values.statistic
      ],
      triangleCount: inputs.tracts.triangleCount,
      range: values.statistic === 2 ? [0, 14] : [10, values.statistic === 3 ? 80 : 50]
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.tracts.outlineCount},
    {layer: 'dots', context: true, positions: outputs.rows, color: [1, 1, 1, 0.25], radius: 1.2}
  ]
});
