// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainTiledFlow accumulates D8 flow over a DEM that arrives as horizontal tiles: each tile
// only sees its own rows, and flow crosses the seams (lines) through the resident global graph.
// However many tiles there are, the accumulation is the same as for the whole DEM.

import {GPUTerrainTiledFlow} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {parameters} from './terrain-tiled-flow.parameters';
import {
  addTerrainFlow,
  createTerrainInputs,
  getCellPolyline,
  getTerrainCell,
  getTerrainCellCenter,
  traceDownstream,
  getTerrainContextLayer
} from './shared/terrain';

const COLUMNS = 128;
const ROWS = 72;
const MAXIMUM_TILES = 4;
const NO_REACH = 0xffffffff;

function isSameArray(first: Uint32Array | null, second: Uint32Array): boolean {
  return (
    first !== null &&
    first.length === second.length &&
    first.every((value, index) => value === second[index])
  );
}

function joinParts<Type extends Uint32Array | Float32Array>(
  Constructor: new (length: number) => Type,
  parts: Type[],
  length: number
): Type {
  const joined = new Constructor(length);
  let offset = 0;
  for (const part of parts) {
    joined.set(part as never, offset);
    offset += part.length;
  }
  return joined;
}

export default defineOperationRun({
  parameters,
  caption: 'Streams stay continuous across tile seams.',
  createInputs: context => ({
    ...createTerrainInputs(context, ROWS, COLUMNS),
    focus: {reach: -1, key: '', source: null as Uint32Array | null}
  }),
  build(builder, terrain, values) {
    const tileCount = Math.round(values.tiles);
    const flow = addTerrainFlow(builder, terrain);
    const runoff = builder.upload('runoff', 'float32', new Float32Array(terrain.cellCount).fill(1));
    // Tile i is a strip of whole rows, so its inputs are one contiguous range of each row-major
    // buffer. Each tile owns separate output buffers (outputs may not share a buffer).
    const tileRows = terrain.rows / tileCount;
    const tileLength = tileRows * terrain.columns;
    const getInputView = <Format extends 'float32' | 'uint32'>(
      source: RunBuffer<Format>,
      tileIndex: number
    ) =>
      builder.graph.createDataView(source.view.buffer, {
        format: source.format,
        length: tileLength,
        byteOffset: tileIndex * tileLength * 4
      });
    const tiles = Array.from({length: tileCount}, (_, tileIndex) => ({
      column: 0,
      row: tileIndex * tileRows,
      width: terrain.columns,
      height: tileRows,
      owned: [0, 0, terrain.columns, tileRows] as const,
      flowDirections: getInputView(flow.flowDirections, tileIndex),
      runoff: getInputView(runoff, tileIndex),
      streams: getInputView(flow.streams, tileIndex),
      accumulation: builder.allocate(`tile-${tileIndex}-accumulation`, 'float32', tileLength),
      reachIds: builder.allocate(`tile-${tileIndex}-reach-ids`, 'uint32', tileLength),
      status: builder.allocate(`tile-${tileIndex}-status`, 'uint32', tileLength)
    }));
    builder.graph.add(
      new GPUTerrainTiledFlow({
        grid: terrain.grid,
        routing: 'd8',
        tiles: tiles.map(tile => ({
          ...tile,
          output: {
            accumulation: tile.accumulation.view,
            status: tile.status.view,
            reachIds: tile.reachIds.view
          }
        }))
      })
    );
    // The tile outputs are copied side by side into whole-DEM buffers, only so they can be drawn.
    const accumulation = builder.allocate('assembled-accumulation', 'float32', terrain.cellCount);
    const reachIds = builder.allocate('assembled-reach-ids', 'uint32', terrain.cellCount);
    const assembleTiles = () => {
      const encoder = builder.device.createCommandEncoder();
      tiles.forEach((tile, tileIndex) => {
        for (const [source, destination] of [
          [tile.accumulation, accumulation],
          [tile.reachIds, reachIds]
        ]) {
          encoder.copyBufferToBuffer({
            sourceBuffer: source.buffer,
            destinationBuffer: destination.buffer,
            destinationOffset: tileIndex * tileLength * 4,
            size: tileLength * 4
          });
        }
      });
      builder.device.submit(encoder.finish());
    };
    const seams = builder.allocate('seams', 'float32x4', MAXIMUM_TILES);
    // A faint tint on every other tile, so each strip is visible; and the hovered reach (1 on its cells).
    const tileTint = builder.allocate('tile-tint', 'uint32', terrain.cellCount);
    const focusMask = builder.allocate('focus-mask', 'uint32', terrain.cellCount);
    return {
      outputs: {
        filledElevation: flow.filledElevation,
        flowDirections: flow.flowDirections,
        accumulation,
        reachIds,
        seams,
        tileTint,
        focusMask,
        tiles,
        assembleTiles
      },
      update: current => {
        flow.writeStreamThreshold(current.streamThreshold);
        tileTint.write(
          Uint32Array.from({length: terrain.cellCount}, (_, cell) =>
            Math.floor(Math.floor(cell / terrain.columns) / tileRows) % 2 === 1 ? 1 : 0
          )
        );
        const segments = new Float32Array(MAXIMUM_TILES * 4);
        for (let seam = 1; seam < tileCount; seam++) {
          const y = 1 - seam / tileCount;
          segments.set([0, y, terrain.columns * terrain.cellSize, y], (seam - 1) * 4);
        }
        seams.write(segments);
      }
    };
  },
  rebuildOn: ['tiles'],
  // The pointer picks a reach: every cell of it is lit across the tile seams, and the readout names
  // the tiles it spans.
  interaction: {
    kind: 'probe',
    hint: 'hover a reach',
    rest: [0.5, 0.5],
    // At rest nothing is hovered, so the poster is the plain picture.
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      // The assembled buffers are only filled when drawing, so read the tiles back and join them.
      const reachParts = outputs.tiles.map(tile => peek(tile.reachIds) as Uint32Array | null);
      const accumulationParts = outputs.tiles.map(tile => peek(tile.accumulation));
      if (reachParts.some(part => !part) || accumulationParts.some(part => !part)) return {};
      const reachIds = joinParts(Uint32Array, reachParts as Uint32Array[], inputs.cellCount);
      const accumulation = joinParts(
        Float32Array,
        accumulationParts as Float32Array[],
        inputs.cellCount
      );
      const cell = getTerrainCell(inputs, world, pointer.x, pointer.y);
      let nearest = -1;
      let nearestDistance = 6;
      for (let offsetRow = -5; offsetRow <= 5; offsetRow++) {
        for (let offsetColumn = -5; offsetColumn <= 5; offsetColumn++) {
          const column = cell.column + offsetColumn;
          const row = cell.row + offsetRow;
          if (column < 0 || row < 0 || column >= inputs.columns || row >= inputs.rows) continue;
          const candidate = row * inputs.columns + column;
          const distance = Math.hypot(offsetColumn, offsetRow);
          if (reachIds[candidate] !== NO_REACH && distance < nearestDistance) {
            nearest = candidate;
            nearestDistance = distance;
          }
        }
      }
      const isRest = Math.hypot(pointer.unitX - 0.5, pointer.unitY - 0.5) < 0.003;
      if (isRest) nearest = -1;
      const reach = nearest < 0 ? -1 : reachIds[nearest];
      const tileCount = Math.round(values.tiles);
      const tileRows = inputs.rows / tileCount;
      const tile = Math.min(tileCount - 1, Math.floor(cell.row / tileRows));
      const key = `${reach}:${tile}:${tileCount}`;
      if (inputs.focus.key !== key || !isSameArray(inputs.focus.source, reachIds)) {
        // 2 lights the hovered tile, 1 the cells of the hovered reach on every tile.
        outputs.focusMask.write(
          Uint32Array.from(reachIds, (value, index) => {
            if (reach >= 0 && value === reach) return 1;
            return !isRest && Math.floor(Math.floor(index / inputs.columns) / tileRows) === tile
              ? 2
              : 0;
          })
        );
        inputs.focus = {reach, key, source: reachIds};
      }
      const tileTop = 1 - (tile * tileRows) / inputs.rows;
      const tileBottom = 1 - ((tile + 1) * tileRows) / inputs.rows;
      if (isRest) return {focus: -1, spotlight: null, readout: `${tileCount} tiles`};
      const glyphs: RunGlyph[] = [
        {
          shape: 'box',
          minimum: [0, tileBottom * world.height],
          maximum: [world.width, tileTop * world.height],
          dashed: true
        }
      ];
      if (reach < 0)
        return {focus: -1, spotlight: null, glyphs, readout: `tile ${tile + 1} of ${tileCount}`};
      const directions = peek(outputs.flowDirections) as Uint32Array | null;
      if (directions) {
        const path = traceDownstream(inputs, directions, nearest);
        glyphs.push(
          {shape: 'polyline', points: getCellPolyline(inputs, world, path)},
          {shape: 'dot', at: getTerrainCellCenter(inputs, world, nearest)}
        );
      }
      let firstTile = Infinity;
      let lastTile = -Infinity;
      reachIds.forEach((value, index) => {
        if (value !== reach) return;
        const reachTile = Math.floor(Math.floor(index / inputs.columns) / tileRows) + 1;
        firstTile = Math.min(firstTile, reachTile);
        lastTile = Math.max(lastTile, reachTile);
      });
      return {
        focus: reach,
        spotlight: null,
        glyphs,
        readout: `reach ${reach} · ${firstTile === lastTile ? `tile ${firstTile}` : `tiles ${firstTile}-${lastTile}`} · accumulation ${Math.round(accumulation[nearest])}`
      };
    }
  },
  draw: ({inputs, outputs, values, interaction}) => {
    outputs.assembleTiles();
    const hasFocus = interaction.focus !== undefined && interaction.focus !== -1;
    return [
      getTerrainContextLayer(inputs, outputs.filledElevation),
      {
        layer: 'raster',
        values: outputs.tileTint,
        columns: inputs.columns,
        rows: inputs.rows,
        categories: [
          [0, 0, 0, 0],
          [1, 1, 1, 0.1]
        ]
      },
      {
        layer: 'raster',
        values: outputs.accumulation,
        columns: inputs.columns,
        rows: inputs.rows,
        ramp: 'sequential',
        range: [0, 150],
        fade: true,
        alpha: 0.7,
        smoothing: true
      },
      {
        layer: 'raster',
        values: outputs.reachIds,
        columns: inputs.columns,
        rows: inputs.rows,
        alpha: hasFocus ? 0.35 : 0.9
      },
      {
        layer: 'raster',
        values: outputs.focusMask,
        columns: inputs.columns,
        rows: inputs.rows,
        categories: [
          [0, 0, 0, 0],
          [1, 1, 1, 1],
          [1, 1, 1, 0.14]
        ]
      },
      {
        layer: 'lines',
        context: true,
        segments: outputs.seams,
        count: Math.round(values.tiles) - 1,
        radius: 1.5,
        color: [1, 1, 1, 0.8]
      }
    ];
  }
});
