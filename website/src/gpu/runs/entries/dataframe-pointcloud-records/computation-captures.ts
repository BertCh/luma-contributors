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
import {parameters} from './computation-captures.parameters';

const MAXIMUM_SNAPSHOTS = 6;

export default defineOperationRun({
  parameters,
  caption: 'The captures sidecar: snapshots, a settledAt stamp and how many of each it holds.',
  createInputs: ({random, world}) => ({tracts: createRecordTracts(random), world}),
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
    const gridArea = {x0: 0.03 * world.width, x1: 0.66 * world.width, y0: 0.3, y1: 0.97};
    const outlines = new ShapeBatch();
    const state = new ShapeBatch();
    const residual = new ShapeBatch();
    const bySlot = groupCapturesBySlot(captures.snapshots);
    let maximumResidual = 1e-6;
    for (const frame of bySlot.get('residual') ?? []) {
      for (const value of Array.from(frame.values))
        maximumResidual = Math.max(maximumResidual, value);
    }
    for (let column = 0; column < MAXIMUM_SNAPSHOTS; column++) {
      for (const [row, batch] of [state, residual].entries()) {
        const cell = getGridCell(gridArea, MAXIMUM_SNAPSHOTS, 2, column, row);
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
    return {
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
  draw: ({extra, theme}) => [
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
      // The settle round is a label: one color per round, muted for tracts that never settled.
      values: extra.stampMap.categories,
      triangleCount: extra.stampMap.triangleCount,
      categories: [theme.muted, ...theme.categories]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: extra.counts,
      range: [0, MAXIMUM_SNAPSHOTS * 2],
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
