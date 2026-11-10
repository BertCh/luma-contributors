// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// createGPUComputationCaptures bundling what a snapshot arena read back. A real relaxation runs
// with a tap per iteration; the arena's frames (state and residual) are read back, `snapshots`
// of them go into the sidecar, and a `settledAt` stamp (the round each tract first changed by less
// than epsilon) is derived from the readbacks. The picture is the sidecar itself: its snapshots as
// small multiples, its stamp as a map, its array lengths as bars, and whether it validates.

import {
  createGPUComputationCaptures,
  createGPUComputationRecord,
  describeGPUCommandGraph,
  getGPUComputationAdapter,
  GPUComputationRecorder,
  GPUSnapshotArena,
  validateGPUComputationRecord,
  validateGPUComputationSidecars,
  type GPUComputationCapture
} from '@luma-contributors/analysis/computation-record';
import {defineOperationRun} from '../../define-run';
import {
  buildRelaxation,
  createRecordTracts,
  getGridCell,
  groupCapturesBySlot,
  ShapeBatch
} from '../../record-demo';
import {createDimShape, getBoxGlyph, getColumnAt, getDimLayer, type Box} from './_lib/record-probe';
import {parameters} from './computation-captures.parameters';

/** What the last run drew, for the pointer: its grid, the rounds of its frames, how many settled. */
type Latest = {
  gridArea: Box;
  columns: number;
  rounds: number[];
  settledByRound: number[];
  tractCount: number;
};
// The grid of captured frames spans these fractions of the picture's width.
const GRID_FRACTIONS = [0.03, 0.66] as const;

export default defineOperationRun({
  parameters,
  caption: 'The captures sidecar: snapshots, a settledAt stamp and how many of each it holds.',
  createInputs: ({random, world}) => ({
    tracts: createRecordTracts(random),
    world,
    latest: null as Latest | null
  }),
  build(builder, {tracts}, values) {
    const recorder = new GPUComputationRecorder();
    const arena = builder.own(
      new GPUSnapshotArena(builder.device, {
        id: 'captures',
        slots: [
          {name: 'state', byteLength: tracts.count * 4, elementType: 'f32'},
          {name: 'residual', byteLength: tracts.count * 4, elementType: 'f32'}
        ],
        capacity: values.iterations
      })
    );
    buildRelaxation(builder, tracts, {
      iterations: values.iterations,
      recorder,
      tap: iteration => ({
        arena,
        views: {state: iteration.state, residual: iteration.residual},
        iteration: iteration.iteration
      })
    });
    return {outputs: {arena, recorder}};
  },
  async afterRun({compiled, device, outputs, inputs, values, upload}) {
    const {tracts, world} = inputs;
    const described = describeGPUCommandGraph(compiled, {recorder: outputs.recorder});
    const record = createGPUComputationRecord({
      id: 'relaxation',
      adapter: getGPUComputationAdapter(device, 'portable'),
      domains: ['gpu-core'],
      graphs: [described]
    });
    const frames = await outputs.arena.read({graph: described.graph.id, runId: record.runId});

    // The settle round per tract: the first iteration whose residual is below epsilon.
    const residuals = groupCapturesBySlot(frames).get('residual') ?? [];
    const settledAt = new Uint32Array(tracts.count).fill(values.iterations + 1);
    residuals.forEach((frame, index) => {
      for (let tract = 0; tract < tracts.count; tract++) {
        if (settledAt[tract] > values.iterations && frame.values[tract] < values.epsilon) {
          settledAt[tract] = index + 1;
        }
      }
    });
    const stamp: GPUComputationCapture = {
      id: 'settled-at',
      name: 'settledAt',
      graph: described.graph.id,
      iteration: values.iterations,
      elementType: 'u32',
      values: settledAt,
      provenance: 'gpu-readback',
      meaning: 'settledAt round'
    };

    // Bundle `snapshots` evenly spaced frames (state and residual of each) into the sidecar.
    const frameCount = residuals.length;
    const chosen = new Set(
      Array.from({length: values.snapshots}, (_, index) =>
        Math.min(frameCount - 1, Math.round(((index + 1) * frameCount) / values.snapshots) - 1)
      )
    );
    const captures = createGPUComputationCaptures({
      runId: record.runId,
      snapshots: frames.filter((_, index) => chosen.has(Math.floor(index / 2))),
      stamps: values.stamps === 1 ? [stamp] : []
    });
    const isValid =
      validateGPUComputationRecord(record).valid &&
      validateGPUComputationSidecars(record, {captures}).valid;

    // Draw the sidecar: snapshots as columns, the stamp as a map, the array lengths as bars.
    const gridArea = {
      x0: GRID_FRACTIONS[0] * world.width,
      x1: GRID_FRACTIONS[1] * world.width,
      y0: 0.3,
      y1: 0.97
    };
    const columns = values.snapshots;
    const outlines = new ShapeBatch();
    const state = new ShapeBatch();
    const residual = new ShapeBatch();
    const bySlot = groupCapturesBySlot(captures.snapshots);
    let maximumResidual = 1e-6;
    for (const frame of bySlot.get('residual') ?? []) {
      for (const value of Array.from(frame.values))
        maximumResidual = Math.max(maximumResidual, value);
    }
    for (let column = 0; column < columns; column++) {
      for (const [row, batch] of [state, residual].entries()) {
        const cell = getGridCell(gridArea, columns, 2, column, row);
        outlines.addOutline(cell);
        const frame = bySlot.get(row === 0 ? 'state' : 'residual')?.[column];
        if (frame) batch.addTractMap(tracts, cell, frame.values);
      }
    }
    const stampArea = {x0: 0.69 * world.width, x1: 0.97 * world.width, y0: 0.3, y1: 0.97};
    outlines.addOutline(stampArea);
    const stampMap = new ShapeBatch();
    if (captures.stamps[0]) stampMap.addTractMap(tracts, stampArea, captures.stamps[0].values);
    const marker = new ShapeBatch();
    marker.addRectangle(
      {x0: 0.94 * world.width, y0: 0.965, x1: 0.97 * world.width, y1: 0.995},
      isValid ? 1 : 2
    );
    const counts = Float32Array.of(
      captures.snapshots.length,
      captures.stamps.length,
      captures.replays.length
    );
    const settledByRound = Array.from(
      {length: values.iterations + 1},
      (_, round) => settledAt.filter(settled => settled <= round).length
    );
    inputs.latest = {
      gridArea,
      columns,
      rounds: [...chosen].sort((first, second) => first - second).map(index => index + 1),
      settledByRound,
      tractCount: tracts.count
    };
    return {
      dim: createDimShape(upload, 'dim', gridArea),
      outlines: outlines.uploadSegments(upload, 'outlines'),
      outlineCount: outlines.segmentCount,
      state: state.uploadShapes(upload, 'state'),
      residual: residual.uploadShapes(upload, 'residual'),
      stampMap: stampMap.uploadShapes(upload, 'stamp'),
      marker: marker.uploadShapes(upload, 'valid'),
      counts: upload('counts', 'float32', counts),
      maximumResidual
    };
  },
  // The pointer is a playhead over the captured frames: frames to its right are dimmed, and the
  // settledAt map shows only the tracts that had settled by the playhead's round.
  interaction: {
    kind: 'time',
    hint: 'scrub the frames',
    rest: [GRID_FRACTIONS[1], 0.5],
    resolve(pointer, {inputs}) {
      const latest = inputs.latest;
      if (!latest) return {readout: 'scrub the frames'};
      const {gridArea, columns} = latest;
      const column = getColumnAt(gridArea.x0, gridArea.x1, columns, pointer.x);
      const width = (gridArea.x1 - gridArea.x0) / columns;
      const round = latest.rounds[column] ?? latest.rounds[latest.rounds.length - 1] ?? 1;
      const settled = latest.settledByRound[Math.min(round, latest.settledByRound.length - 1)];
      return {
        curtain: gridArea.x0 + (column + 1) * width,
        focus: round,
        glyphs: [
          getBoxGlyph({
            x0: gridArea.x0 + column * width,
            x1: gridArea.x0 + (column + 1) * width,
            y0: gridArea.y0,
            y1: gridArea.y1
          })
        ],
        readout: `snapshot ${column + 1} of ${columns} · round ${round} · ${settled} of ${latest.tractCount} tracts settled`
      };
    }
  },
  draw: ({extra, theme, interaction}) => [
    {layer: 'lines', context: true, segments: extra.outlines, count: extra.outlineCount, radius: 1},
    {
      layer: 'polygons',
      positions: extra.state.positions,
      indices: extra.state.indices,
      owners: extra.state.owners,
      values: extra.state.values,
      triangleCount: extra.state.triangleCount,
      range: [0, 1]
    },
    {
      layer: 'polygons',
      positions: extra.residual.positions,
      indices: extra.residual.indices,
      owners: extra.residual.owners,
      values: extra.residual.values,
      triangleCount: extra.residual.triangleCount,
      range: [0, extra.maximumResidual]
    },
    {
      layer: 'polygons',
      positions: extra.stampMap.positions,
      indices: extra.stampMap.indices,
      owners: extra.stampMap.owners,
      // The settle round is a label: one color per round; rounds after the playhead are muted.
      values: extra.stampMap.categories,
      triangleCount: extra.stampMap.triangleCount,
      categories: [theme.muted, ...theme.categories].map((color, round) =>
        round === 0 || typeof interaction.focus !== 'number' || round <= interaction.focus
          ? color
          : theme.muted
      )
    },
    ...(interaction.curtain === undefined
      ? []
      : [getDimLayer(extra.dim, theme.background, interaction.curtain)]),
    {
      layer: 'chart',
      kind: 'bars',
      values: extra.counts,
      range: [0, 12],
      rectangle: [0.05, 0.04, 0.5, 0.24],
      color: [...theme.categories[6], 1]
    },
    {
      layer: 'polygons',
      positions: extra.marker.positions,
      indices: extra.marker.indices,
      owners: extra.marker.owners,
      values: extra.marker.categories,
      triangleCount: extra.marker.triangleCount,
      categories: [theme.muted, theme.categories[4], theme.categories[2]]
    }
  ]
});
