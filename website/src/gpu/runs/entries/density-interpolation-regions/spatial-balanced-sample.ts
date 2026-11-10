// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialBalancedSample picks units from a finite frame, spread evenly over space within each
// stratum (west and east halves). The output frame ids are gathered back to positions for drawing.

import {GPUSpatialBalancedSample} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createUniformPoints} from '../../inputs';
import {parameters} from './spatial-balanced-sample.parameters';
import {addPositionGather} from './support/gather';

export default defineOperationRun({
  parameters,
  caption:
    'Selected units (bright) are evenly spread inside each stratum; each stratum is sampled at its own rate.',
  createInputs: ({random, world}) => ({
    world,
    frame: createUniformPoints(random, world, 900, 0),
    // The stratum boundary is owned by the pointer; it rests at the middle.
    split: {x: world.width / 2}
  }),
  build(builder, {world, frame, split}, values) {
    const counts = [values.westCount, values.eastCount];
    const total = counts[0] + counts[1];
    const strata = builder.allocate('strata', 'uint32', frame.count);
    const positions = builder.upload('positions', 'float32x2', frame.positions);
    const selected = builder.allocate('selected', 'uint32', total);
    builder.graph.add(
      new GPUSpatialBalancedSample({
        positions: positions.view,
        frameIds: builder.upload(
          'frame-ids',
          'uint32',
          Uint32Array.from({length: frame.count}, (_, row) => row)
        ).view,
        strata: strata.view,
        allocations: [
          {stratumId: 0, sampleCount: counts[0]},
          {stratumId: 1, sampleCount: counts[1]}
        ],
        bounds: [0, 0, world.width, world.height],
        seed: values.seed,
        output: {
          frameIds: selected.view,
          inclusionProbabilities: builder.allocate('inclusion', 'float32', total).view,
          designWeights: builder.allocate('design-weights', 'float32', total).view,
          status: builder.allocate('status', 'uint32', 4).view
        }
      })
    );
    return {
      outputs: {positions, chosen: addPositionGather(builder, selected, positions, 'chosen')},
      update: () =>
        strata.write(
          Uint32Array.from({length: frame.count}, (_, row) =>
            frame.positions[row * 2] < split.x ? 0 : 1
          )
        )
    };
  },
  rebuildOn: ['westCount', 'eastCount', 'seed'],
  // The pointer moves the boundary between the two strata; each side keeps its own allocation.
  interaction: {
    kind: 'place',
    hint: 'move the stratum split',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, values}) {
      inputs.split.x = Math.min(
        0.75 * inputs.world.width,
        Math.max(0.25 * inputs.world.width, pointer.x)
      );
      return {
        rerun: true,
        glyphs: [{shape: 'curtain', x: inputs.split.x}],
        readout: `west ${values.westCount} · east ${values.eastCount} · split ${inputs.split.x.toFixed(2)}`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'dots', positions: outputs.positions, radius: 2.4, color: [0.75, 0.8, 0.92, 0.6]},
    {layer: 'dots', positions: outputs.chosen, radius: 5, color: [1, 0.85, 0.3, 1]}
  ]
});
