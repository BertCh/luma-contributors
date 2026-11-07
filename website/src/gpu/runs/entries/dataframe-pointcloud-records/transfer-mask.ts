// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTransferMask carrying a parcel selection (bright dots) up to the tracts that contain them,
// with no readback. In `any` mode a tract is lit when any of its parcels is selected; in `count`
// mode the fill is the number of selected parcels.

import {GPUTransferMask} from '@luma-contributors/analysis/computation-record';
import {defineOperationRun} from '../../define-run';
import {createTracts} from '../../inputs';
import {addIntegerToFloat} from './_lib/convert';
import {assignNearestSites, uploadTractGeometry} from './_lib/helpers';
import {parameters} from './transfer-mask.parameters';

const PARCEL_COUNT = 3000;

export default defineOperationRun({
  parameters,
  caption: 'Tracts touched by the selected parcels.',
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, world, 12, 0.1);
    const positions = Float32Array.from({length: PARCEL_COUNT * 2}, (_, index) =>
      index % 2 === 0 ? random() * world.width : random()
    );
    return {tracts, positions, tractOfParcel: assignNearestSites(positions, tracts.centers)};
  },
  build(builder, inputs, values) {
    const {tracts} = inputs;
    const parcelFlags = builder.allocate('parcel-flags', 'uint32', PARCEL_COUNT);
    const tractFlags = builder.allocate('tract-flags', 'uint32', tracts.count);
    builder.graph.add(
      new GPUTransferMask({
        id: 'carry-selection',
        source: parcelFlags.view,
        relation: {
          layout: 'index',
          targets: builder.upload('tract-of-parcel', 'uint32', inputs.tractOfParcel).view
        },
        target: tractFlags.view,
        mode: values.mode === 0 ? 'any' : 'count'
      })
    );
    return {
      outputs: {
        ...uploadTractGeometry(builder, tracts),
        parcels: builder.upload('parcels', 'float32x2', inputs.positions),
        parcelFlags,
        tractFill: addIntegerToFloat(builder, 'tract-fill', tractFlags)
      },
      update: current => {
        parcelFlags.write(
          Uint32Array.from({length: PARCEL_COUNT}, (_, row) =>
            Math.hypot(
              inputs.positions[row * 2] - current.selectionCenter,
              inputs.positions[row * 2 + 1] - 0.5
            ) < current.selectionRadius
              ? 1
              : 0
          )
        );
      }
    };
  },
  rebuildOn: ['mode'],
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.tractFill,
      triangleCount: inputs.tracts.triangleCount,
      range: [0, values.mode === 0 ? 1 : 25]
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.tracts.outlineCount},
    {
      layer: 'dots',
      positions: outputs.parcels,
      values: outputs.parcelFlags,
      radius: 1.6,
      alpha: 0.9
    }
  ]
});
