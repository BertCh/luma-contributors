// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialBalancedSample picks units from a finite frame, spread evenly over space within each
// stratum (west and east halves). The output frame ids are gathered back to positions for drawing.
// The frame is colored by stratum (the boundary follows the pointer), the selected units are drawn large in the ink color.

import {GPUSpatialBalancedSample} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createUniformPoints} from '../../inputs';
import {mergeText} from '../../text';
import {parameters} from './spatial-balanced-sample.parameters';
import {addPositionGather} from './support/gather';

export default defineOperationRun({
  parameters,
  caption:
    'Selected units (dark or white) are evenly spread inside each stratum; each stratum is sampled at its own rate.',
  createInputs: ({random, world}) => {
    const frame = createUniformPoints(random, world, 900, 0);
    return {
      world,
      frame,
      // Sorted x of the frame: the split is kept where each side holds at least its allocation.
      sortedX: Float32Array.from(
        {length: frame.count},
        (_, row) => frame.positions[row * 2]
      ).sort(),
      // The stratum boundary is owned by the pointer; it rests at the middle.
      split: {x: world.width / 2}
    };
  },
  build(builder, {world, frame, split}, values) {
    const counts = [values.westCount, values.eastCount];
    const total = counts[0] + counts[1];
    const strata = builder.allocate('strata', 'uint32', frame.count);
    const positions = builder.upload('positions', 'float32x2', frame.positions);
    const selected = builder.allocate('selected', 'uint32', total);
    const status = builder.allocate('status', 'uint32', 4);
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
          status: status.view
        }
      })
    );
    return {
      outputs: {
        status,
        strata,
        positions,
        chosen: addPositionGather(builder, selected, positions, 'chosen')
      },
      update: () =>
        strata.write(
          Uint32Array.from({length: frame.count}, (_, row) =>
            frame.positions[row * 2] < split.x ? 0 : 1
          )
        )
    };
  },
  // What the operation selected on each side of the boundary: its output rows are positions, and a
  // row it could not fill is NaN.
  async afterRun({outputs, inputs}) {
    const bytes = await outputs.chosen.buffer.readAsync();
    const chosen = new Float32Array(bytes.buffer, bytes.byteOffset, outputs.chosen.length * 2);
    const sampled = [0, 0];
    for (let row = 0; row < outputs.chosen.length; row++) {
      const x = chosen[row * 2];
      if (Number.isFinite(x)) sampled[x < inputs.split.x ? 0 : 1]++;
    }
    // Per stratum [selected count, status]; status 1 is an allocation larger than the stratum.
    const statusBytes = await outputs.status.buffer.readAsync();
    const status = new Uint32Array(statusBytes.buffer, statusBytes.byteOffset, 4);
    return {sampled, rejected: [status[1] !== 0, status[3] !== 0]};
  },
  rebuildOn: ['westCount', 'eastCount', 'seed'],
  // The pointer moves the boundary between the two strata; each side keeps its own allocation.
  interaction: {
    kind: 'place',
    hint: 'move the stratum split',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, values}) {
      // The pointer spans the whole range of valid splits: west keeps at least westCount units and
      // east at least eastCount, so the operation never rejects the design.
      const count = inputs.sortedX.length;
      const lowest = inputs.sortedX[Math.min(count - 1, values.westCount)];
      const highest = inputs.sortedX[Math.max(0, count - values.eastCount - 1)];
      const share = Math.min(1, Math.max(0, pointer.x / inputs.world.width));
      inputs.split.x = lowest + share * Math.max(0, highest - lowest);
      const westFrame = countWestUnits(inputs.frame.positions, inputs.split.x);
      return {
        rerun: true,
        glyphs: [{shape: 'curtain', x: inputs.split.x}],
        readout: `west ${westFrame} units, ${values.westCount} asked · east ${inputs.frame.count - westFrame} units, ${values.eastCount} asked · split ${inputs.split.x.toFixed(2)}`
      };
    }
  },
  draw: ({inputs, outputs, extra, theme, values, world}) => {
    const westFrame = countWestUnits(inputs.frame.positions, inputs.split.x);
    return [
      {
        layer: 'dots',
        positions: outputs.positions,
        values: outputs.strata,
        radius: 3.2,
        alpha: 0.85
      },
      {layer: 'dots', positions: outputs.chosen, radius: 5.5, color: [...theme.ink, 1]},
      {
        layer: 'text',
        ...mergeText({
          labels: [
            {
              text: extra.rejected[0]
                ? `west: rejected, ${values.westCount} asked of ${westFrame}`
                : `west: ${extra.sampled[0]} sampled of ${westFrame}`,
              at: [inputs.split.x / 2, world.height],
              anchor: 'top',
              offset: [0, -4],
              role: 'title',
              swatch: 0
            },
            {
              text: extra.rejected[1]
                ? `east: rejected, ${values.eastCount} asked of ${inputs.frame.count - westFrame}`
                : `east: ${extra.sampled[1]} sampled of ${inputs.frame.count - westFrame}`,
              at: [(inputs.split.x + world.width) / 2, world.height],
              anchor: 'top',
              offset: [0, -4],
              role: 'title',
              swatch: 1
            }
          ]
        })
      }
    ];
  }
});

/** Frame units west of the stratum boundary (the CPU copy of what `update` assigns to stratum 0). */
function countWestUnits(positions: Float32Array, boundary: number): number {
  let count = 0;
  for (let row = 0; row < positions.length / 2; row++) if (positions[row * 2] < boundary) count++;
  return count;
}
