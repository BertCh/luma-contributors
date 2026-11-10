// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainBreach on the raw DEM: from each pit (strict local minimum) it cuts the cheapest
// bounded channel to lower ground. The picture is the depth it cut and the channel of every pit
// (lines); pit dots are green when breached and red when the pit stays closed because its outlet is
// beyond maximumSteps or needs more than maximumCutDepth.

import {GPUTerrainBreach} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {chooseSticky, createPointIndex} from '../../pick';
import type {RunGlyph} from '../../interaction';
import {parameters} from './terrain-breach.parameters';
import {
  createCraterTerrainInputs,
  getCellPolyline,
  getTerrainCellCenter,
  getTerrainContextLayer
} from './shared/terrain';

const PIT_LIMIT = 40;
const NO_CELL = 0xffffffff;
/** Pit status words of GPUTerrainBreach: 0 not run, 1 breached, 2 left closed. */
const STATUS_COLORS = [
  [0.6, 0.6, 0.65],
  [0.35, 0.95, 0.55],
  [1, 0.35, 0.3]
] as const;

export default defineOperationRun({
  parameters,
  caption: 'Cut depth along the channel from each pit (dots) to its outlet.',
  createInputs: context => {
    const terrain = createCraterTerrainInputs(context);
    // Spread the pits over the whole map, then place their dots in drawing coordinates.
    const stride = Math.max(1, Math.floor(terrain.pitCells.length / PIT_LIMIT));
    const pitCells = terrain.pitCells
      .filter((_, index) => index % stride === 0)
      .slice(0, PIT_LIMIT);
    const pitPositions = new Float32Array(pitCells.length * 2);
    pitCells.forEach((cell, index) => {
      pitPositions[index * 2] = ((cell % terrain.columns) + 0.5) * terrain.cellSize;
      pitPositions[index * 2 + 1] =
        context.world.height - (Math.floor(cell / terrain.columns) + 0.5) * terrain.cellSize;
    });
    return {
      ...terrain,
      world: context.world,
      pitCells,
      pitPositions,
      pitIndex: createPointIndex(pitPositions),
      focus: {pit: -1}
    };
  },
  build(builder, terrain, values) {
    const pitCount = terrain.pitCells.length;
    const maximumSteps = Math.round(values.maximumSteps);
    const elevation = builder.upload('dem', 'float32', terrain.elevation);
    const pitCells = builder.upload('pit-cells', 'uint32', terrain.pitCells);
    const breached = builder.allocate('breached', 'float32', terrain.cellCount);
    const cutDepth = builder.allocate('cut-depth', 'float32', terrain.cellCount);
    const pathCells = builder.allocate('path-cells', 'uint32', pitCount * (maximumSteps + 1));
    const pathLengths = builder.allocate('path-lengths', 'uint32', pitCount);
    const pitStatus = builder.allocate('pit-status', 'uint32', pitCount);
    builder.graph.add(
      new GPUTerrainBreach({
        grid: terrain.grid,
        elevation: elevation.view,
        pitCells: pitCells.view,
        maximumSteps,
        maximumCutDepth: values.maximumCutDepth,
        elevationDecrement: 1e-5,
        output: {
          elevation: breached.view,
          cutDepth: cutDepth.view,
          modificationMask: builder.allocate('mask', 'uint32', terrain.cellCount).view,
          pathCells: pathCells.view,
          pathLengths: pathLengths.view,
          pitStatus: pitStatus.view,
          unresolvedCount: builder.allocate('unresolved-count', 'uint32', 1).view
        }
      })
    );
    const pits = builder.upload('pits', 'float32x2', terrain.pitPositions);
    return {outputs: {elevation, cutDepth, pits, pathCells, pathLengths, pitStatus}};
  },
  async afterRun({inputs, outputs, values, upload}) {
    // The channel of every breached pit as line segments between consecutive cells.
    const [cellBytes, lengthBytes] = await Promise.all([
      outputs.pathCells.buffer.readAsync(),
      outputs.pathLengths.buffer.readAsync()
    ]);
    const cells = new Uint32Array(
      cellBytes.buffer,
      cellBytes.byteOffset,
      cellBytes.byteLength >> 2
    );
    const lengths = new Uint32Array(
      lengthBytes.buffer,
      lengthBytes.byteOffset,
      lengthBytes.byteLength >> 2
    );
    const stride = Math.round(values.maximumSteps) + 1;
    const segments: number[] = [];
    for (let pit = 0; pit < inputs.pitCells.length; pit++) {
      const points = getCellPolyline(
        inputs,
        inputs.world,
        Array.from(cells.subarray(pit * stride, pit * stride + lengths[pit]))
      );
      for (let point = 0; point + 1 < lengths[pit]; point++) {
        segments.push(
          points[point * 2],
          points[point * 2 + 1],
          points[point * 2 + 2],
          points[point * 2 + 3]
        );
      }
    }
    return {
      channels: upload(
        'channels',
        'float32x4',
        segments.length > 0 ? Float32Array.from(segments) : new Float32Array(4).fill(Number.NaN)
      ),
      channelCount: segments.length / 4
    };
  },
  // The pointer picks the nearest pit: it is ringed, and the channel cut from it is traced with the
  // depth it needed.
  interaction: {
    kind: 'probe',
    hint: 'hover a pit',
    rest: ({inputs, toUnit}) =>
      toUnit([inputs.pitPositions[0] ?? 0.5, inputs.pitPositions[1] ?? 0.5]),
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const pit = chooseSticky(inputs.pitIndex, pointer.x, pointer.y, inputs.focus.pit);
      inputs.focus.pit = pit;
      if (pit < 0) return {};
      const at = [inputs.pitPositions[pit * 2], inputs.pitPositions[pit * 2 + 1]] as const;
      const glyphs: RunGlyph[] = [{shape: 'ring', at, radius: inputs.cellSize * 2.5}];
      const status = peek(outputs.pitStatus) as Uint32Array | null;
      const lengths = peek(outputs.pathLengths) as Uint32Array | null;
      const paths = peek(outputs.pathCells) as Uint32Array | null;
      const cutDepth = peek(outputs.cutDepth);
      if (!status || !lengths || !paths || !cutDepth) return {glyphs, readout: `pit ${pit + 1}`};
      if (status[pit] !== 1) {
        return {
          glyphs,
          readout: `pit ${pit + 1} · stays closed · maximumCutDepth ${values.maximumCutDepth}`
        };
      }
      const stride = Math.round(values.maximumSteps) + 1;
      const cells = Array.from(paths.subarray(pit * stride, pit * stride + lengths[pit])).filter(
        cell => cell !== NO_CELL
      );
      let deepest = 0;
      for (const cell of cells) deepest = Math.max(deepest, cutDepth[cell]);
      glyphs.push(
        {shape: 'polyline', points: getCellPolyline(inputs, world, cells)},
        {shape: 'dot', at: getTerrainCellCenter(inputs, world, cells[cells.length - 1])}
      );
      return {
        glyphs,
        readout: `pit ${pit + 1} · breached · cut ${deepest.toFixed(3)} over ${cells.length - 1} cells`
      };
    }
  },
  draw: ({inputs, outputs, values, extra}) => [
    getTerrainContextLayer(inputs, outputs.elevation),
    {
      layer: 'raster',
      values: outputs.cutDepth,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      range: [0, values.maximumCutDepth],
      fade: true
    },
    {
      layer: 'lines',
      segments: extra.channels,
      count: extra.channelCount,
      radius: 2.4,
      color: [1, 0.95, 0.45, 1]
    },
    {
      layer: 'dots',
      positions: outputs.pits,
      values: outputs.pitStatus,
      categories: STATUS_COLORS,
      radius: 4
    }
  ]
});
