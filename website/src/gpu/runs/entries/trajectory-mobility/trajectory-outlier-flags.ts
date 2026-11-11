// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrajectoryOutlierFlags on ten San Francisco Bay vessel tracks with ten synthetic GPS jumps
// added (labeled in the caption). A fix is flagged as displaced when both steps around it exceed
// maximumSpeed (bit 1), or when the step speed rises by more than maximumAcceleration into the fix
// and falls by more than that after the step out of it (bit 2); the good fix after a jump is left
// alone.

import {
  getGPUTrajectoryOutlierFlagsParameterValues,
  GPUTrajectoryOutlierFlags
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {clamp, findParameter} from '../../interaction';
import {
  createMovementFleet,
  createSegmentsOfPositions,
  getStepSpeeds,
  getTrackOfRow,
  readUint32
} from './_lib/movement';
import {formatSpeed} from './_lib/units';
import {parameters} from './trajectory-outlier-flags.parameters';

const TRACK_COUNT = 10;
const SPIKE_COUNT = 10;
const MAXIMUM_STEP_DURATION = 8;

type Outputs = {
  segments: RunBuffer;
  positions: RunBuffer;
  flags: RunBuffer;
};

export default defineOperationRun({
  parameters,
  caption:
    'Ten Bay vessel tracks with 10 synthetic GPS jumps (a fix moved 0.5 of the frame). Colored fixes are flagged as displaced: both steps around them are faster than maximumSpeed, or (maximumAcceleration above 0) the speed rises into the fix and falls out of it by more than that limit. Distances and speeds are approximate: the frame stretches x and y differently.',
  createInputs: ({world}) => {
    const fleet = createMovementFleet(world, TRACK_COUNT);
    const positions = Float32Array.from(fleet.positions);
    // Jumps go on interior fixes whose steps are short in time, so a jump is clearly faster than
    // the fastest honest vessel.
    const candidates: number[] = [];
    for (let track = 0; track < fleet.trackCount; track++) {
      for (
        let row = fleet.trackOffsets[track] + 2;
        row < fleet.trackOffsets[track + 1] - 2;
        row++
      ) {
        const before = fleet.timestamps[row] - fleet.timestamps[row - 1];
        const after = fleet.timestamps[row + 1] - fleet.timestamps[row];
        if (Math.max(before, after) <= MAXIMUM_STEP_DURATION) candidates.push(row);
      }
    }
    const spikeRows = Array.from(
      {length: SPIKE_COUNT},
      (_, spike) => candidates[Math.floor(((spike + 0.5) * candidates.length) / SPIKE_COUNT)]
    );
    const displace = (row: number, angle: number, magnitude: number) => {
      [world.width, 1].forEach((limit, axis) => {
        const start = fleet.positions[row * 2 + axis];
        let delta = (axis === 0 ? Math.cos(angle) : Math.sin(angle)) * magnitude;
        if (start + delta < 0 || start + delta > limit) delta = -delta;
        positions[row * 2 + axis] = clamp(start + delta, 0, limit);
      });
    };
    const getLevel = (row: number) => {
      const speeds = getStepSpeeds({...fleet, positions});
      return Math.min(speeds[row], speeds[row + 1]);
    };
    // The first jump grows until its slower step is just above the default limit: the pointer's
    // rest position. The others are large.
    const defaultLimit = findParameter(parameters, 'maximumSpeed').default;
    spikeRows.forEach((row, spike) => {
      const angle = spike * 2.399963; // golden angle: spread directions
      if (spike > 0) return displace(row, angle, 0.5);
      for (let magnitude = 0.1; magnitude <= 0.6; magnitude += 0.01) {
        displace(row, angle, magnitude);
        if (getLevel(row) >= defaultLimit + 0.002) break;
      }
    });
    const edited = {...fleet, positions};
    const speeds = getStepSpeeds(edited);
    const restFix = spikeRows[0];
    return {
      ...edited,
      segments: createSegmentsOfPositions(positions, fleet.trackOffsets),
      speeds,
      restFix
    };
  },
  build(builder, fleet) {
    const positions = builder.upload('positions', 'float32x2', fleet.positions);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const flags = builder.allocate('flags', 'uint32', fleet.rowCount);
    builder.graph.add(
      new GPUTrajectoryOutlierFlags({
        positions: positions.view,
        timestamps: builder.upload('timestamps', 'float32', fleet.timestamps).view,
        trackOffsets: builder.upload('offsets', 'uint32', fleet.trackOffsets).view,
        parameters: parameterBuffer.view,
        output: {flags: flags.view}
      })
    );
    const outputs: Outputs = {
      segments: builder.upload('segments', 'float32x4', fleet.segments),
      positions,
      flags
    };
    return {
      outputs,
      update: current =>
        parameterBuffer.write(
          getGPUTrajectoryOutlierFlagsParameterValues({
            maximumSpeed: current.maximumSpeed,
            maximumAcceleration: current.maximumAcceleration
          })
        )
    };
  },
  // Flagged fixes are gathered from the flag column, with the bit that flagged them as the color.
  async afterRun({outputs, inputs, upload}) {
    const flags = await readUint32(outputs.flags);
    const positions: number[] = [];
    const kinds: number[] = [];
    for (let row = 0; row < flags.length; row++) {
      if (!flags[row]) continue;
      positions.push(inputs.positions[row * 2], inputs.positions[row * 2 + 1]);
      kinds.push(flags[row]);
    }
    return {
      flaggedCount: kinds.length,
      flagged: upload('flagged', 'float32x2', Float32Array.from(kinds.length ? positions : [0, 0])),
      kinds: upload('kinds', 'uint32', Uint32Array.from(kinds.length ? kinds : [0]))
    };
  },
  // Level: the slower of the two steps around the fix under the pointer becomes maximumSpeed, so
  // that fix is the first to be flagged.
  interaction: {
    kind: 'level',
    hint: 'pick a fix',
    rest: ({toUnit, inputs}) =>
      toUnit([inputs.positions[inputs.restFix * 2], inputs.positions[inputs.restFix * 2 + 1]]),
    resolve(pointer, {inputs, values, peek, outputs}) {
      let nearest = 0;
      let nearestDistance = Infinity;
      for (let row = 0; row < inputs.rowCount; row++) {
        const distance = Math.hypot(
          inputs.positions[row * 2] - pointer.x,
          inputs.positions[row * 2 + 1] - pointer.y
        );
        if (distance < nearestDistance) {
          nearest = row;
          nearestDistance = distance;
        }
      }
      const track = getTrackOfRow(inputs.trackOffsets, nearest);
      const first = inputs.trackOffsets[track];
      const last = inputs.trackOffsets[track + 1] - 1;
      const stepIn = nearest > first ? inputs.speeds[nearest] : Infinity;
      const stepOut = nearest < last ? inputs.speeds[nearest + 1] : Infinity;
      const parameter = findParameter(parameters, 'maximumSpeed');
      const level = Math.min(stepIn, stepOut);
      const maximumSpeed = clamp(
        Number((Math.floor(level / parameter.step) * parameter.step).toPrecision(12)),
        parameter.min,
        parameter.max
      );
      const at: [number, number] = [
        inputs.positions[nearest * 2],
        inputs.positions[nearest * 2 + 1]
      ];
      const flags = peek(outputs.flags);
      let flaggedCount = 0;
      for (let row = 0; flags && row < flags.length; row++) flaggedCount += flags[row] ? 1 : 0;
      return {
        values: {maximumSpeed},
        glyphs: [
          // Reachable discs: the fix is flagged exactly when it lies outside both.
          ...(nearest > first
            ? [
                {
                  shape: 'ring' as const,
                  dashed: true,
                  at: [inputs.positions[nearest * 2 - 2], inputs.positions[nearest * 2 - 1]] as [
                    number,
                    number
                  ],
                  radius:
                    maximumSpeed * (inputs.timestamps[nearest] - inputs.timestamps[nearest - 1])
                }
              ]
            : []),
          ...(nearest < last
            ? [
                {
                  shape: 'ring' as const,
                  dashed: true,
                  at: [inputs.positions[nearest * 2 + 2], inputs.positions[nearest * 2 + 3]] as [
                    number,
                    number
                  ],
                  radius:
                    maximumSpeed * (inputs.timestamps[nearest + 1] - inputs.timestamps[nearest])
                }
              ]
            : []),
          ...(nearest > first
            ? [
                {
                  shape: 'line' as const,
                  from: [inputs.positions[nearest * 2 - 2], inputs.positions[nearest * 2 - 1]] as [
                    number,
                    number
                  ],
                  to: at
                }
              ]
            : []),
          ...(nearest < last
            ? [
                {
                  shape: 'line' as const,
                  from: at,
                  to: [inputs.positions[nearest * 2 + 2], inputs.positions[nearest * 2 + 3]] as [
                    number,
                    number
                  ]
                }
              ]
            : []),
          {shape: 'crosshair' as const, at}
        ],
        readout: `in ${Number.isFinite(stepIn) ? formatSpeed(stepIn, inputs.metersPerUnit) : '-'} · out ${Number.isFinite(stepOut) ? formatSpeed(stepOut, inputs.metersPerUnit) : '-'} · limit ${formatSpeed(maximumSpeed, inputs.metersPerUnit)} · ${flaggedCount} flagged`
      };
    }
  },
  draw: ({inputs, outputs, extra}) => [
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.segmentCount},
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2.5},
    ...(extra
      ? [
          {
            layer: 'dots' as const,
            positions: extra.flagged,
            values: extra.kinds,
            count: extra.flaggedCount,
            radius: 14,
            alpha: 0.85
          }
        ]
      : [])
  ]
});
