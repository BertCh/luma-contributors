// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrajectoryThinning in mode 'minimum-distance' on the four longest San Francisco Bay vessel tracks. The
// first and last fix of a track are kept, and each kept fix is followed by the first later fix at
// least `tolerance` away. Kept fixes are dots; the thinned path joins them.

import {
  getGPUTrajectoryThinningParameterValues,
  GPUTrajectoryThinning
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {clamp, findParameter, type RunGlyph, snapToParameter} from '../../interaction';
import {createMovementFleet, readUint32, uploadFleet} from './_lib/movement';
import {formatDistance} from './_lib/units';
import {parameters} from './trajectory-thinning.parameters';

const TRACK_COUNT = 4;

type Outputs = {
  segments: RunBuffer;
  positions: RunBuffer;
  rowIds: RunBuffer;
  count: RunBuffer;
};

export default defineOperationRun({
  parameters,
  caption:
    'The four longest Bay vessel tracks thinned with mode minimum-distance. Small dots are the raw fixes, large dots the fixes kept. Distances and speeds are approximate: the frame stretches x and y differently.',
  createInputs: ({world}) => {
    const fleet = createMovementFleet(world, TRACK_COUNT);
    // The pointer measures from the start of the nearest track; rest sits one default tolerance
    // to the right of the first track's start.
    const starts = Array.from(fleet.trackOffsets.slice(0, -1), row => [
      fleet.positions[row * 2],
      fleet.positions[row * 2 + 1]
    ]);
    return {...fleet, starts};
  },
  build(builder, fleet) {
    const buffers = uploadFleet(builder, fleet);
    const parameterBuffer = builder.allocate('thinning-parameters', 'float32', 4);
    const rowIds = builder.allocate('kept-row-ids', 'uint32', fleet.rowCount);
    const keptOffsets = builder.allocate('kept-offsets', 'uint32', fleet.trackCount + 1);
    const count = builder.allocate('kept-count', 'uint32', 1);
    const overflow = builder.allocate('kept-overflow', 'uint32', 1);
    const candidateOverflow = builder.allocate('kept-candidate-overflow', 'uint32', 1);
    builder.graph.add(
      new GPUTrajectoryThinning({
        mode: 'minimum-distance',
        positions: buffers.positions.view,
        timestamps: buffers.timestamps.view,
        trackOffsets: buffers.trackOffsets.view,
        parameters: parameterBuffer.view,
        output: {
          rowIds: rowIds.view,
          trackOffsets: keptOffsets.view,
          status: {
            count: count.view,
            overflow: overflow.view,
            candidateOverflow: candidateOverflow.view
          }
        }
      })
    );
    const outputs: Outputs & {keptOffsets: RunBuffer} = {
      segments: builder.upload('segments', 'float32x4', fleet.segments),
      positions: buffers.positions,
      rowIds,
      count,
      keptOffsets
    };
    return {
      outputs,
      update: current =>
        parameterBuffer.write(
          getGPUTrajectoryThinningParameterValues({tolerance: current.tolerance})
        )
    };
  },
  // The thinned path joins consecutive kept fixes of each track.
  async afterRun({outputs, inputs, upload}) {
    const [count, rowIds, offsets] = await Promise.all([
      readUint32(outputs.count),
      readUint32(outputs.rowIds),
      readUint32(outputs.keptOffsets)
    ]);
    const edges: number[] = [];
    for (let track = 0; track + 1 < offsets.length; track++) {
      for (let kept = offsets[track]; kept + 1 < Math.min(offsets[track + 1], count[0]); kept++) {
        edges.push(rowIds[kept], rowIds[kept + 1]);
      }
    }
    const kept = new Float32Array(Math.max(1, count[0]) * 2);
    for (let row = 0; row < count[0]; row++) {
      kept.set([inputs.positions[rowIds[row] * 2], inputs.positions[rowIds[row] * 2 + 1]], row * 2);
    }
    return {
      keptCount: count[0],
      kept: upload('kept-positions', 'float32x2', kept),
      edgeCount: edges.length / 2,
      edges: upload('thinned-edges', 'uint32x2', Uint32Array.from(edges.length ? edges : [0, 0]))
    };
  },
  // Reach: the distance from the start of the nearest track is the tolerance; the ring shows how
  // far the next kept fix has to be from the last one.
  interaction: {
    kind: 'reach',
    hint: 'reach from a track start',
    rest: ({toUnit, inputs}) =>
      toUnit([inputs.starts[0][0] + parameters[0].default, inputs.starts[0][1]]),
    resolve(pointer, {inputs, peek, outputs}) {
      let nearest = 0;
      let nearestDistance = Infinity;
      inputs.starts.forEach(([x, y], track) => {
        const distance = Math.hypot(x - pointer.x, y - pointer.y);
        if (distance < nearestDistance) {
          nearest = track;
          nearestDistance = distance;
        }
      });
      const parameter = findParameter(parameters, 'tolerance');
      const tolerance = snapToParameter(
        parameter,
        clamp(nearestDistance, parameter.min, parameter.max)
      );
      const at = inputs.starts[nearest] as [number, number];
      const count = peek(outputs.count);
      const keptRows = peek(outputs.rowIds);
      const keptOffsets = peek(outputs.keptOffsets);
      // The chain: a ring of the tolerance around every kept fix of this track.
      const chain: RunGlyph[] = [];
      if (keptRows && keptOffsets) {
        for (let kept = keptOffsets[nearest]; kept < keptOffsets[nearest + 1]; kept++) {
          chain.push({
            shape: 'ring',
            at: [inputs.positions[keptRows[kept] * 2], inputs.positions[keptRows[kept] * 2 + 1]],
            radius: tolerance,
            dashed: true
          });
        }
      }
      return {
        values: {tolerance},
        glyphs: [...chain, {shape: 'line', from: at, to: [pointer.x, pointer.y], dashed: true}],
        readout: `tolerance ${formatDistance(tolerance, inputs.metersPerUnit)} · ${count ? count[0] : 0} of ${inputs.rowCount} fixes kept`
      };
    }
  },
  draw: ({inputs, outputs, extra}) => [
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.segmentCount},
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2.5},
    ...(extra
      ? [
          {
            layer: 'segments' as const,
            nodes: outputs.positions,
            edges: extra.edges,
            radius: 2.5,
            color: [0.25, 0.8, 0.95, 1] as const
          },
          {
            layer: 'dots' as const,
            positions: extra.kept,
            count: extra.keptCount,
            radius: 6,
            color: [1, 0.7, 0.2, 1] as const
          }
        ]
      : [])
  ]
});
