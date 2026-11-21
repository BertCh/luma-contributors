// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUModalSplit after GPUTrajectoryStaypoints and GPUTrajectorySegments on eight San Francisco Bay
// vessel tracks. Each tripleg is given a mode from its average speed (modeSpeedUpperBoundsFloat32: slow, medium,
// fast), and the bars add the chosen metric up per track and mode. The map is squeezed upward to
// leave room for the bars; the operation reads the true positions.

import {
  getGPUTrajectorySegmentsParameterValues,
  GPUModalSplit
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {findParameter, getFractionOf, getParameterAt} from '../../interaction';
import {
  addSegments,
  addStaypoints,
  createMovementFleet,
  createSegmentsOfPositions,
  readFloat32,
  readUint32,
  TRIPLEG_CAPACITY,
  uploadFleet,
  writeStaypointSettings
} from './_lib/movement';
import {formatSpeed} from './_lib/units';
import {parameters} from './trajectory-modal-split.parameters';

const TRACK_COUNT = 8;
const MODE_COUNT = 3;
const METRICS = ['count', 'distance', 'duration'] as const;
const STAYPOINT_SETTINGS = {distanceThreshold: 0.006, timeThreshold: 12, maximumGap: 0};
const MAP_OFFSET = 0.4;
const NONE = 0xffffffff;

type Outputs = {
  segments: RunBuffer;
  positions: RunBuffer;
  triplegCount: RunBuffer;
  triplegOverflow: RunBuffer;
  triplegCandidateOverflow: RunBuffer;
  triplegStartRows: RunBuffer;
  triplegEndRows: RunBuffer;
  modalSplit: RunBuffer;
  predictedModes: RunBuffer;
};

function squeeze(values: Float32Array): Float32Array {
  return Float32Array.from(values, (value, index) =>
    index % 2 === 1 ? MAP_OFFSET + value * (1 - MAP_OFFSET) : value
  );
}

export default defineOperationRun({
  parameters,
  caption:
    'Eight Bay vessel tracks. Triplegs are colored by the mode their average speed falls in; bars (slow, medium, fast) add up the metric per track. Distances and speeds are approximate: the frame stretches x and y differently.',
  createInputs: ({world}) => {
    const fleet = createMovementFleet(world, TRACK_COUNT);
    const displayPositions = squeeze(fleet.positions);
    return {
      ...fleet,
      displayPositions,
      displaySegments: createSegmentsOfPositions(displayPositions, fleet.trackOffsets)
    };
  },
  build(builder, fleet, values) {
    const buffers = uploadFleet(builder, fleet);
    const staypoints = addStaypoints(builder, fleet, buffers);
    const segments = addSegments(builder, fleet, buffers, staypoints);
    const modalSplit = builder.allocate('modal-split', 'float32', fleet.trackCount * MODE_COUNT);
    const predictedModes = builder.allocate('predicted-modes', 'uint32', TRIPLEG_CAPACITY);
    builder.graph.add(
      new GPUModalSplit({
        positions: buffers.positions.view,
        timestamps: buffers.timestamps.view,
        trackOffsets: buffers.trackOffsets.view,
        triplegs: {
          tracks: segments.triplegTracks.view,
          startRows: segments.triplegStartRows.view,
          endRows: segments.triplegEndRows.view,
          status: {
            count: segments.triplegCount.view,
            overflow: segments.triplegOverflow.view,
            candidateOverflow: segments.triplegCandidateOverflow.view
          }
        },
        // The breaks must be strictly ascending, so the fast break is kept above the slow one.
        modeSpeedUpperBoundsFloat32: [
          values.slowBreak,
          Math.max(values.fastBreak, values.slowBreak + 0.001),
          Infinity
        ],
        modeCount: MODE_COUNT,
        metric: METRICS[values.metric],
        output: {modalSplit: modalSplit.view, predictedModes: predictedModes.view}
      })
    );
    const outputs: Outputs = {
      segments: builder.upload('display-segments', 'float32x4', fleet.displaySegments),
      positions: builder.upload('display-positions', 'float32x2', fleet.displayPositions),
      triplegCount: segments.triplegCount,
      triplegOverflow: segments.triplegOverflow,
      triplegCandidateOverflow: segments.triplegCandidateOverflow,
      triplegStartRows: segments.triplegStartRows,
      triplegEndRows: segments.triplegEndRows,
      modalSplit,
      predictedModes
    };
    return {
      outputs,
      update: () => {
        writeStaypointSettings(staypoints, STAYPOINT_SETTINGS);
        segments.parameters.write(
          getGPUTrajectorySegmentsParameterValues({maximumGap: 0, activityMinimumDuration: 40})
        );
      }
    };
  },
  rebuildOn: ['slowBreak', 'fastBreak', 'metric'],
  buildCache: 6,
  // Triplegs are joined row to row and labeled by mode; the table is split into one column per mode.
  async afterRun({inputs, outputs, upload}) {
    const [count, starts, ends, modes, split, overflow, candidateOverflow] = await Promise.all([
      readUint32(outputs.triplegCount),
      readUint32(outputs.triplegStartRows),
      readUint32(outputs.triplegEndRows),
      readUint32(outputs.predictedModes),
      readFloat32(outputs.modalSplit),
      readUint32(outputs.triplegOverflow),
      readUint32(outputs.triplegCandidateOverflow)
    ]);
    const incomplete = Boolean(overflow[0] || candidateOverflow[0]);
    const labels = new Uint32Array(inputs.rowCount).fill(NONE);
    const edges: number[] = [];
    for (let tripleg = 0; !incomplete && tripleg < Math.min(count[0], starts.length); tripleg++) {
      for (let row = starts[tripleg]; row < ends[tripleg]; row++) {
        labels[row] = modes[tripleg];
        if (row + 1 < ends[tripleg]) edges.push(row, row + 1);
      }
    }
    const columns = Array.from({length: MODE_COUNT}, (_, mode) =>
      upload(
        `mode-${mode}`,
        'float32',
        Float32Array.from(
          {length: inputs.trackCount},
          (_, track) => split[track * MODE_COUNT + mode]
        )
      )
    );
    return {
      incomplete,
      labels: upload('mode-labels', 'uint32', labels),
      edges: upload('tripleg-edges', 'uint32x2', Uint32Array.from(edges.length ? edges : [0, 0])),
      columns,
      maximum: Math.max(1e-6, ...Array.from(split).filter(Number.isFinite))
    };
  },
  // A pad over the two speed breaks. They are compile-time, so they apply once the pointer rests.
  interaction: {
    kind: 'pad',
    hint: 'slow break across, fast break up',
    rest: () => [
      getFractionOf(findParameter(parameters, 'slowBreak'), 0.012),
      1 - getFractionOf(findParameter(parameters, 'fastBreak'), 0.03)
    ],
    resolve(pointer, {inputs, values, outputs, peek}) {
      const slowBreak = getParameterAt(findParameter(parameters, 'slowBreak'), pointer.unitX);
      const fastBreak = getParameterAt(findParameter(parameters, 'fastBreak'), 1 - pointer.unitY);
      return {
        settle: {slowBreak, fastBreak},
        readout:
          peek(outputs.triplegOverflow)?.[0] || peek(outputs.triplegCandidateOverflow)?.[0]
            ? 'Incomplete tripleg table; modal split unavailable'
            : `modeSpeedUpperBoundsFloat32 [${formatSpeed(slowBreak, inputs.metersPerUnit)}, ${formatSpeed(fastBreak, inputs.metersPerUnit)}] · ${METRICS[values.metric]}`
      };
    }
  },
  draw: ({inputs, outputs, extra, theme}) => [
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.segmentCount},
    ...(extra && !extra.incomplete
      ? [
          {
            layer: 'segments' as const,
            nodes: outputs.positions,
            edges: extra.edges,
            values: extra.labels,
            radius: 3.5
          },
          ...extra.columns.map((column, mode) => ({
            layer: 'chart' as const,
            kind: 'bars' as const,
            values: column,
            range: [0, extra.maximum] as const,
            rectangle: [0.04 + mode * 0.32, 0.05, 0.3 + mode * 0.32, 0.3] as const,
            color: [...theme.categories[mode], 1] as const
          }))
        ]
      : [])
  ]
});
