// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSnapshotArena collecting frames of a real relaxation. The arena has `capacity` frames of
// `slots` named ranges; taps copy the state, the residual and the deviation of the relaxation into
// it as the graph runs, and one mapped read after the run returns every frame. The picture is the
// arena itself: one column per frame, one row per slot, with the frames still unused left empty,
// and a bar of how many bytes of the readback buffer were filled.

import {GPUSnapshotArena} from '@luma-contributors/analysis/computation-record';
import {defineOperationRun} from '../../define-run';
import {
  buildRelaxation,
  createRecordTracts,
  getGridCell,
  groupCapturesBySlot,
  RELAXATION_SLOT_NAMES,
  ShapeBatch
} from '../../record-demo';
import {createDimShape, getBoxGlyph, getColumnAt, getDimLayer, type Box} from './_lib/record-probe';
import {parameters} from './snapshot-arena.parameters';

/** What the last run drew, for the pointer: the grid, the frames it holds and the byte meter. */
type Latest = {
  area: Box;
  meter: Box;
  capacity: number;
  slotCount: number;
  frames: {iteration: number | undefined}[];
  bytesPerFrame: number;
};

export default defineOperationRun({
  parameters,
  caption: 'One column per frame the arena collected, one row per slot.',
  createInputs: ({random, world}) => ({
    tracts: createRecordTracts(random),
    world,
    latest: null as Latest | null
  }),
  build(builder, {tracts}, values) {
    const slotNames = RELAXATION_SLOT_NAMES.slice(0, values.slots);
    const arena = builder.own(
      new GPUSnapshotArena(builder.device, {
        id: 'relaxation',
        slots: slotNames.map(name => ({name, byteLength: tracts.count * 4, elementType: 'f32'})),
        capacity: values.capacity
      })
    );
    // Decimate: tap every n-th iteration so the frames never exceed the capacity.
    const step = Math.ceil(values.iterations / values.capacity);
    buildRelaxation(builder, tracts, {
      iterations: values.iterations,
      deviation: values.slots >= 3,
      tap: iteration =>
        iteration.iteration % step === 0
          ? {
              arena,
              views: Object.fromEntries(
                slotNames.map(name => [name, iteration[name] ?? iteration.state])
              ),
              label: `iteration ${iteration.iteration}`,
              iteration: iteration.iteration
            }
          : undefined
    });
    return {outputs: {arena}};
  },
  async afterRun({outputs, inputs, values, upload}) {
    const {arena} = outputs;
    const {tracts, world} = inputs;
    const captures = await arena.read();
    const bySlot = groupCapturesBySlot(captures);
    const area = {x0: 0.03 * world.width, x1: 0.97 * world.width, y0: 0.14, y1: 0.97};
    const outlines = new ShapeBatch();
    const state = new ShapeBatch();
    const residual = new ShapeBatch();
    const deviation = new ShapeBatch();
    const batches = [state, residual, deviation];
    const slotNames = RELAXATION_SLOT_NAMES.slice(0, values.slots);
    let maximumResidual = 1e-6;
    for (const frame of bySlot.get('residual') ?? []) {
      for (const value of Array.from(frame.values))
        maximumResidual = Math.max(maximumResidual, value);
    }
    slotNames.forEach((name, row) => {
      for (let column = 0; column < values.capacity; column++) {
        const cell = getGridCell(area, values.capacity, slotNames.length, column, row);
        outlines.addOutline(cell);
        const frame = bySlot.get(name)?.[column];
        if (frame) batches[row].addTractMap(tracts, cell, frame.values);
      }
    });
    // Bytes filled: one segment per slot, as wide as the frames collected times the slot's bytes.
    const bytes = new ShapeBatch();
    const frameCount = bySlot.get('state')?.length ?? 0;
    const meter = {x0: 0.03 * world.width, x1: 0.97 * world.width, y0: 0.04, y1: 0.09};
    const slotWidth = (meter.x1 - meter.x0) / (values.capacity * slotNames.length);
    for (let frame = 0; frame < frameCount; frame++) {
      slotNames.forEach((name, slot) => {
        const x0 = meter.x0 + (frame * slotNames.length + slot) * slotWidth;
        bytes.addRectangle({x0, x1: x0 + slotWidth * 0.9, y0: meter.y0, y1: meter.y1}, slot);
      });
    }
    bytes.addOutline(meter);
    inputs.latest = {
      area,
      meter,
      capacity: values.capacity,
      slotCount: slotNames.length,
      frames: (bySlot.get('state') ?? []).map(frame => ({iteration: frame.iteration})),
      bytesPerFrame: tracts.count * 4 * slotNames.length
    };
    return {
      dim: createDimShape(upload, 'dim', area),
      outlines: outlines.uploadSegments(upload, 'cell-outlines'),
      outlineCount: outlines.segmentCount,
      state: state.uploadShapes(upload, 'state'),
      residual: residual.uploadShapes(upload, 'residual'),
      deviation: deviation.uploadShapes(upload, 'deviation'),
      bytes: bytes.uploadShapes(upload, 'bytes'),
      meter: bytes.uploadSegments(upload, 'meter'),
      meterCount: bytes.segmentCount,
      maximumResidual
    };
  },
  // The pointer is a playhead over the frames the arena collected: later frames are dimmed, and
  // the bytes the playhead's frame occupies are outlined in the meter.
  interaction: {
    kind: 'time',
    hint: 'scrub the frames',
    rest: [0.97, 0.5],
    resolve(pointer, {inputs}) {
      const latest = inputs.latest;
      if (!latest || latest.frames.length === 0) return {readout: 'scrub the frames'};
      const {area, meter, capacity, slotCount, frames} = latest;
      const column = Math.min(
        frames.length - 1,
        getColumnAt(area.x0, area.x1, capacity, pointer.x)
      );
      const width = (area.x1 - area.x0) / capacity;
      const slotWidth = (meter.x1 - meter.x0) / (capacity * slotCount);
      const iteration = frames[column].iteration;
      return {
        curtain: area.x0 + (column + 1) * width,
        glyphs: [
          getBoxGlyph({
            x0: area.x0 + column * width,
            x1: area.x0 + (column + 1) * width,
            y0: area.y0,
            y1: area.y1
          }),
          getBoxGlyph({
            x0: meter.x0 + column * slotCount * slotWidth,
            x1: meter.x0 + (column + 1) * slotCount * slotWidth,
            y0: meter.y0,
            y1: meter.y1
          })
        ],
        readout: `frame ${column + 1} of ${frames.length}${iteration === undefined ? '' : ` · iteration ${iteration}`} · ${slotCount} slots · ${((frames.length * latest.bytesPerFrame) / 1000).toFixed(1)} kB filled`
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
      range: [0, extra.maximumResidual],
      ramp: 'sequential'
    },
    {
      layer: 'polygons',
      positions: extra.deviation.positions,
      indices: extra.deviation.indices,
      owners: extra.deviation.owners,
      values: extra.deviation.values,
      triangleCount: extra.deviation.triangleCount,
      range: [-0.4, 0.4],
      ramp: 'diverging'
    },
    {
      layer: 'polygons',
      positions: extra.bytes.positions,
      indices: extra.bytes.indices,
      owners: extra.bytes.owners,
      values: extra.bytes.categories,
      triangleCount: extra.bytes.triangleCount,
      categories: [theme.categories[0], theme.categories[1], theme.categories[2]]
    },
    ...(interaction.curtain === undefined
      ? []
      : [getDimLayer(extra.dim, theme.background, interaction.curtain)]),
    {layer: 'lines', segments: extra.meter, count: extra.meterCount, radius: 1}
  ]
});
