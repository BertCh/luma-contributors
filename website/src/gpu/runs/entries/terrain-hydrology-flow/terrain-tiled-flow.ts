// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainTiledFlow accumulates D8 flow over a DEM that arrives as horizontal tiles: each tile
// only sees its own rows, and flow crosses the seams (lines) through the resident global graph.
// However many tiles there are, the accumulation is the same as for the whole DEM.

import {GPUTerrainTiledFlow} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {parameters} from './terrain-tiled-flow.parameters';
import {addTerrainFlow, createTerrainInputs, getTerrainContextLayer} from './shared/terrain';

const COLUMNS = 128;
const ROWS = 72;
const MAXIMUM_TILES = 4;

export default defineOperationRun({
  parameters,
  caption: 'Streams stay continuous across tile seams.',
  createInputs: context => createTerrainInputs(context, ROWS, COLUMNS),
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
    return {
      outputs: {
        filledElevation: flow.filledElevation,
        accumulation,
        reachIds,
        seams,
        assembleTiles
      },
      update: current => {
        flow.writeStreamThreshold(current.streamThreshold);
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
  draw: ({inputs, outputs, values}) => {
    outputs.assembleTiles();
    return [
      getTerrainContextLayer(inputs, outputs.filledElevation),
      {
        layer: 'raster',
        values: outputs.reachIds,
        columns: inputs.columns,
        rows: inputs.rows,
        alpha: 0.6
      },
      {
        layer: 'raster',
        values: outputs.accumulation,
        columns: inputs.columns,
        rows: inputs.rows,
        ramp: 'sequential',
        range: [0, 150],
        fade: true,
        smoothing: true
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
