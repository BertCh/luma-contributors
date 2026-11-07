// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUKeyJoin of parcel points (left, keyed by tract) to a tract table (right). Matched parcels take
// the tract's gathered value; parcels whose key is missing from the right table stay muted gray.

import {GPUKeyJoin} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts} from '../../inputs';
import {assignNearestSites, hashToUnit} from './_lib/helpers';
import {parameters} from './key-join.parameters';

const PARCEL_COUNT = 3000;

export default defineOperationRun({
  parameters,
  caption: 'Parcels colored by the value joined from their tract.',
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, world, 12, 0.1);
    const positions = Float32Array.from({length: PARCEL_COUNT * 2}, (_, index) =>
      index % 2 === 0 ? random() * world.width : random()
    );
    return {tracts, positions, leftKeys: assignNearestSites(positions, tracts.centers)};
  },
  build(builder, inputs) {
    const {tracts} = inputs;
    const rightMask = builder.allocate('right-mask', 'uint32', tracts.count);
    const leftMask = builder.allocate('left-mask', 'uint32', PARCEL_COUNT);
    const joined = builder.allocate('joined-value', 'float32', PARCEL_COUNT);
    builder.graph.add(
      new GPUKeyJoin({
        leftKeys: builder.upload('left-keys', 'uint32', inputs.leftKeys).view,
        rightKeys: builder.upload(
          'right-keys',
          'uint32',
          Uint32Array.from({length: tracts.count}, (_, row) => row)
        ).view,
        leftMask: leftMask.view,
        rightMask: rightMask.view,
        kind: 'left',
        gather: [
          {
            column: builder.upload('right-value', 'float32', Float32Array.from(tracts.values)).view,
            output: joined.view
          }
        ],
        output: {
          matched: builder.allocate('matched', 'uint32', PARCEL_COUNT).view,
          rightRows: builder.allocate('right-rows', 'uint32', PARCEL_COUNT).view
        }
      })
    );
    return {
      outputs: {
        joined,
        parcels: builder.upload('parcels', 'float32x2', inputs.positions),
        outline: builder.upload('outline', 'float32x4', tracts.outline)
      },
      update: current => {
        rightMask.write(
          Uint32Array.from({length: tracts.count}, (_, row) =>
            hashToUnit(row) * 100 < current.rightMask ? 1 : 0
          )
        );
        leftMask.write(
          Uint32Array.from({length: PARCEL_COUNT}, (_, row) =>
            hashToUnit(row + 5000) * 100 < current.leftMask ? 1 : 0
          )
        );
      }
    };
  },
  rebuildOn: [],
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.tracts.outlineCount},
    {layer: 'dots', positions: outputs.parcels, values: outputs.joined, range: [0, 1], radius: 3.2}
  ]
});
