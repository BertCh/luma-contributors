// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrajectorySplit on San Francisco Bay vessel tracks. Each new segment gets a color. Mode 'gap'
// splits after any step longer than maximumGap, 'stop' splits at the staypoints of
// GPUTrajectoryStaypoints, and 'value-change' splits where a per-row value (here the quarter of the
// day, a stand-in for any uint32 label) changes, repeating the boundary row.

import {
  getGPUTrajectorySplitParameterValues,
  GPUTrajectorySplit,
  type GPUTrajectorySplitMode
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {findParameter} from '../../interaction';
import {findNearestSegment} from '../../pick';
import {
  addStaypoints,
  createMovementFleet,
  readUint32,
  uploadFleet,
  writeStaypointSettings
} from './_lib/movement';
import {formatDuration} from './_lib/units';
import {parameters} from './trajectory-split.parameters';

const MODES: readonly GPUTrajectorySplitMode[] = ['gap', 'stop', 'value-change'];
const TRACK_COUNT = 8;
const BLOCK_COUNT = 4;
const STAYPOINT_SETTINGS = {distanceThreshold: 0.004, timeThreshold: 10, maximumGap: 0};

type Outputs = {
  segments: RunBuffer;
  positions: RunBuffer;
  rowIds: RunBuffer;
  segmentOffsets: RunBuffer;
  segmentCount: RunBuffer;
  copiedPositions: RunBuffer;
};

export default defineOperationRun({
  parameters,
  caption:
    'Eight Bay vessel tracks. Colors are the new segments, amber lines the steps cut in mode gap; mode value-change uses the quarter of the day as its per-row value.',
  createInputs: ({world}) => {
    const fleet = createMovementFleet(world, TRACK_COUNT);
    // Duration of each drawn step, aligned with fleet.segments.
    const stepDurations: number[] = [];
    for (let track = 0; track < fleet.trackCount; track++) {
      for (let row = fleet.trackOffsets[track]; row < fleet.trackOffsets[track + 1] - 1; row++) {
        stepDurations.push(fleet.timestamps[row + 1] - fleet.timestamps[row]);
      }
    }
    const defaultGap = findParameter(parameters, 'maximumGap').default;
    let restStep = 0;
    stepDurations.forEach((duration, step) => {
      if (
        Math.abs(duration - defaultGap - 0.5) < Math.abs(stepDurations[restStep] - defaultGap - 0.5)
      ) {
        restStep = step;
      }
    });
    const blocks = Uint32Array.from(fleet.timestamps, time =>
      Math.min(BLOCK_COUNT - 1, Math.floor((time / fleet.duration) * BLOCK_COUNT))
    );
    return {...fleet, stepDurations, restStep, blocks};
  },
  build(builder, fleet, values) {
    const mode = MODES[values.mode];
    const buffers = uploadFleet(builder, fleet);
    const segmentCapacity = fleet.rowCount;
    const rowCapacity = fleet.rowCount * 2;
    const segmentOffsets = builder.allocate('segment-offsets', 'uint32', segmentCapacity + 1);
    const segmentTracks = builder.allocate('segment-tracks', 'uint32', segmentCapacity);
    const rowIds = builder.allocate('row-ids', 'uint32', rowCapacity);
    const segmentCount = builder.allocate('segment-count', 'uint32', 1);
    const overflow = builder.allocate('segment-overflow', 'uint32', 1);
    const copiedPositions = builder.allocate('copied-positions', 'float32x2', rowCapacity);
    const parameterBuffer = builder.allocate('split-parameters', 'float32', 4);
    const staypoints = mode === 'stop' ? addStaypoints(builder, fleet, buffers) : null;
    builder.graph.add(
      new GPUTrajectorySplit({
        mode,
        positions: buffers.positions.view,
        timestamps: buffers.timestamps.view,
        trackOffsets: buffers.trackOffsets.view,
        parameters: mode === 'gap' ? parameterBuffer.view : undefined,
        staypointIds: staypoints?.rowStaypointIds.view,
        values:
          mode === 'value-change'
            ? builder.upload('blocks', 'uint32', fleet.blocks).view
            : undefined,
        output: {
          segmentOffsets: segmentOffsets.view,
          segmentTracks: segmentTracks.view,
          rowIds: rowIds.view,
          status: {count: segmentCount.view, overflow: overflow.view},
          positions: copiedPositions.view
        }
      })
    );
    const outputs: Outputs = {
      segments: builder.upload('segments', 'float32x4', fleet.segments),
      positions: buffers.positions,
      rowIds,
      segmentOffsets,
      segmentCount,
      copiedPositions
    };
    return {
      outputs,
      update: current => {
        parameterBuffer.write(
          getGPUTrajectorySplitParameterValues({maximumGap: current.maximumGap})
        );
        if (staypoints) writeStaypointSettings(staypoints, STAYPOINT_SETTINGS);
      }
    };
  },
  // The segment table is read back to join the consecutive rows of each segment, labeled by segment.
  async afterRun({inputs, outputs, values, upload}) {
    const [count, offsets] = await Promise.all([
      readUint32(outputs.segmentCount),
      readUint32(outputs.segmentOffsets)
    ]);
    const segmentCount = Math.min(count[0], offsets.length - 1);
    const rowTotal = segmentCount > 0 ? offsets[segmentCount] : 0;
    const edges: number[] = [];
    const labels = new Uint32Array(outputs.copiedPositions.length);
    for (let segment = 0; segment < segmentCount; segment++) {
      for (let row = offsets[segment]; row < offsets[segment + 1]; row++) {
        labels[row] = segment;
        if (row + 1 < offsets[segment + 1]) edges.push(row, row + 1);
      }
    }
    // In mode 'gap' the cut steps are the ones longer than maximumGap; they are drawn as lines.
    const cuts: number[] = [];
    if (Math.round(values.mode) === 0) {
      inputs.stepDurations.forEach((duration, step) => {
        if (duration > values.maximumGap) {
          cuts.push(...inputs.segments.slice(step * 4, step * 4 + 4));
        }
      });
    }
    return {
      cutCount: cuts.length / 4,
      cuts: upload('cuts', 'float32x4', Float32Array.from(cuts.length ? cuts : [0, 0, 0, 0])),
      segmentCount,
      rowTotal,
      labels: upload('labels', 'uint32', labels),
      edges: upload('segment-edges', 'uint32x2', Uint32Array.from(edges.length ? edges : [0, 0]))
    };
  },
  rebuildOn: ['mode'],
  buildCache: 3,
  // Level: the duration of the step under the pointer becomes maximumGap, so that step is the
  // first one split (a split needs a step strictly longer than the gap).
  interaction: {
    kind: 'level',
    hint: 'pick a step',
    rest: ({toUnit, inputs}) =>
      toUnit([
        (inputs.segments[inputs.restStep * 4] + inputs.segments[inputs.restStep * 4 + 2]) / 2,
        (inputs.segments[inputs.restStep * 4 + 1] + inputs.segments[inputs.restStep * 4 + 3]) / 2
      ]),
    resolve(pointer, {inputs, values, peek, outputs}) {
      const hit = findNearestSegment(inputs.segments, pointer.x, pointer.y, inputs.segmentCount);
      if (!hit) return {};
      const parameter = findParameter(parameters, 'maximumGap');
      const duration = inputs.stepDurations[hit.index];
      const maximumGap = Math.min(parameter.max, Math.max(parameter.min, Math.floor(duration)));
      const count = peek(outputs.segmentCount);
      return {
        values: {maximumGap},
        glyphs: [
          {
            shape: 'line',
            from: [inputs.segments[hit.index * 4], inputs.segments[hit.index * 4 + 1]],
            to: [inputs.segments[hit.index * 4 + 2], inputs.segments[hit.index * 4 + 3]]
          },
          {shape: 'crosshair', at: hit.point}
        ],
        readout:
          values.mode === 0
            ? `maximumGap ${formatDuration(maximumGap)} · step ${formatDuration(duration)} · ${count ? count[0] : 0} segments`
            : `mode ${MODES[values.mode]}${values.mode === 2 ? ' (label: quarter of the day)' : ''} · ${count ? count[0] : 0} segments`
      };
    }
  },
  draw: ({inputs, outputs, extra}) => [
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.segmentCount},
    ...(extra
      ? [
          {
            layer: 'segments' as const,
            nodes: outputs.copiedPositions,
            edges: extra.edges,
            values: extra.labels,
            radius: 3
          },
          {
            layer: 'lines' as const,
            segments: extra.cuts,
            count: extra.cutCount,
            radius: 1.5,
            color: [1, 0.7, 0.2, 0.9] as const
          }
        ]
      : [])
  ]
});
