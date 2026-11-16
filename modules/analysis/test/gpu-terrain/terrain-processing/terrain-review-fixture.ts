// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import type {GPUTerrainFlowTile} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow';
import {GPUTerrainTiledFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow';
import {GPUTerrainCatchmentSummary} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

/** Imported caller storage, including genuine empty buffers and arbitrary four-byte prefixes. */
export function createTerrainReviewFixture(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createOutputBuffer>[] = [];
  let identifier = 0;
  const input = <Format extends 'float32' | 'uint32'>(
    format: Format,
    values: ArrayLike<number>,
    prefix = 0
  ) => {
    const data =
      format === 'float32'
        ? new Float32Array(values.length + prefix)
        : new Uint32Array(values.length + prefix);
    data.fill(12345);
    data.set(values, prefix);
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    const backing = importGraphBuffer(graph, `review-${identifier++}`, buffer, format, data.length);
    return {
      buffer,
      view: graph.createDataView(backing.buffer, {
        format,
        length: values.length,
        byteOffset: prefix * 4
      })
    };
  };
  const output = <Format extends 'float32' | 'uint32'>(
    format: Format,
    length: number,
    prefix = 0
  ) => {
    const buffer = createOutputBuffer(device, length + prefix);
    const initial =
      format === 'float32' ? new Float32Array(length + prefix) : new Uint32Array(length + prefix);
    initial.fill(12345, 0, prefix);
    buffer.write(initial);
    buffers.push(buffer);
    const backing = importGraphBuffer(
      graph,
      `review-${identifier++}`,
      buffer,
      format,
      length + prefix
    );
    const view = graph.createDataView(backing.buffer, {format, length, byteOffset: prefix * 4});
    return {
      buffer,
      view,
      read: async () =>
        (
          await (format === 'float32'
            ? readFloat32(buffer, length + prefix)
            : readUint32(buffer, length + prefix))
        ).slice(prefix)
    };
  };
  return {
    graph,
    input,
    output,
    buffers,
    run() {
      const compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      return compiled;
    },
    submit(compiled: ReturnType<typeof graph.compile>) {
      submitGraph(device, compiled, undefined);
    },
    destroy() {
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

/** Poisoned halos are deliberately unrelated to neighbouring tiles' owned values. */
export function createTerrainReviewTiles(
  fixture: ReturnType<typeof createTerrainReviewFixture>,
  width: number,
  height: number,
  widths: number[],
  halos: number[],
  directions: ArrayLike<number>,
  runoff: ArrayLike<number>,
  streams: ArrayLike<number>
) {
  let column = 0;
  const rectangles = widths.map(ownedWidth => {
    const rectangle = [column, 0, ownedWidth, height] as const;
    column += ownedWidth;
    return rectangle;
  });
  return createTerrainReviewRectangles(
    fixture,
    width,
    rectangles,
    halos,
    directions,
    runoff,
    streams
  );
}

/** Rectangular partitions may change their column boundaries between row bands. */
export function createTerrainReviewRectangles(
  fixture: ReturnType<typeof createTerrainReviewFixture>,
  width: number,
  rectangles: readonly (readonly [number, number, number, number])[],
  halos: number[],
  directions: ArrayLike<number>,
  runoff: ArrayLike<number>,
  streams: ArrayLike<number>
) {
  return rectangles.map(([globalColumn, globalRow, ownedWidth, ownedHeight], tileIndex) => {
    const halo = halos[tileIndex];
    const tileWidth = ownedWidth + halo * 2;
    const tileHeight = ownedHeight + halo * 2;
    const count = tileWidth * tileHeight;
    const tileDirections = new Uint32Array(count).fill(0xffffffff);
    const tileRunoff = new Float32Array(count).fill(1e6);
    const tileStreams = new Uint32Array(count).fill(0xffffffff);
    for (let row = 0; row < ownedHeight; row++)
      for (let column = 0; column < ownedWidth; column++) {
        const local = (row + halo) * tileWidth + column + halo;
        const cell = (row + globalRow) * width + globalColumn + column;
        tileDirections[local] = directions[cell];
        tileRunoff[local] = runoff[cell];
        tileStreams[local] = streams[cell];
      }
    const accumulation = fixture.output('float32', count, tileIndex + 1);
    const status = fixture.output('uint32', count, tileIndex + 2);
    const reachIds = fixture.output('uint32', count, tileIndex + 3);
    const runoffInput = fixture.input('float32', tileRunoff, tileIndex + 2);
    const tile: GPUTerrainFlowTile = {
      column: globalColumn - halo,
      row: globalRow - halo,
      width: tileWidth,
      height: tileHeight,
      owned: [halo, halo, ownedWidth, ownedHeight],
      flowDirections: fixture.input('uint32', tileDirections, tileIndex + 1).view,
      runoff: runoffInput.view,
      streams: fixture.input('uint32', tileStreams, tileIndex + 3).view,
      output: {accumulation: accumulation.view, status: status.view, reachIds: reachIds.view}
    };
    const result = {
      tile,
      accumulation,
      status,
      reachIds,
      globalColumn,
      globalRow,
      writeRunoff(values: ArrayLike<number>) {
        for (let row = 0; row < ownedHeight; row++)
          for (let column = 0; column < ownedWidth; column++)
            tileRunoff[(row + halo) * tileWidth + column + halo] =
              values[(row + globalRow) * width + globalColumn + column];
        runoffInput.buffer.write(tileRunoff, runoffInput.view.byteOffset);
      }
    };
    return result;
  });
}

export function createTerrainReviewEmptyView(
  fixture: ReturnType<typeof createTerrainReviewFixture>,
  buffer: ReturnType<typeof createOutputBuffer>,
  name: string
): GraphDataView<'uint32'> {
  return importGraphBuffer(fixture.graph, name, buffer, 'uint32', 0);
}

/** Imports shared storage once. Only the writable layout is legal within one dispatch. */
export function createTerrainReviewPackedDispatch(
  fixture: ReturnType<typeof createTerrainReviewFixture>,
  kind: 'tiled' | 'catchment',
  byteOffset: number,
  storageUses: 'mixed' | 'writable',
  execution: 'auto' | 'resident' | 'tiled' = 'tiled'
) {
  const packed = fixture.output('uint32', byteOffset / 4 + 4);
  const view = <Format extends 'float32' | 'uint32'>(
    format: Format,
    length: number,
    offset: number
  ) => fixture.graph.createDataView(packed.view.buffer, {format, length, byteOffset: offset});
  const grid = {
    width: 1,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const invalidCount = fixture.output('uint32', 1);
  if (kind === 'tiled') {
    const status = fixture.output('uint32', 1);
    const props = {
      grid,
      execution,
      tiles: [
        {
          column: 0,
          row: 0,
          width: 1,
          height: 1,
          owned: [0, 0, 1, 1] as const,
          flowDirections: fixture.input('uint32', [0]).view,
          runoff: fixture.input('float32', [3]).view,
          streams:
            storageUses === 'mixed' ? view('uint32', 1, 0) : fixture.input('uint32', [2]).view,
          output: {
            accumulation: view('float32', 1, storageUses === 'mixed' ? byteOffset : 0),
            status: storageUses === 'mixed' ? status.view : view('uint32', 1, byteOffset),
            reachIds: fixture.output('uint32', 1).view
          }
        }
      ],
      output: {status: {invalidCount: invalidCount.view}}
    };
    return {
      create: () => new GPUTerrainTiledFlow(props),
      async read() {
        const values = await readFloat32(packed.buffer, byteOffset / 4 + 4);
        const words = await packed.read();
        return {
          values: [values[storageUses === 'mixed' ? byteOffset / 4 : 0]],
          status:
            storageUses === 'mixed' ? Array.from(await status.read()) : [words[byteOffset / 4]],
          invalidCount: Array.from(await invalidCount.read())
        };
      }
    };
  }
  const props = {
    grid,
    elevation: fixture.input('float32', [7]).view,
    basinIds: fixture.input('uint32', [0]).view,
    streams: fixture.input('uint32', [2]).view,
    flowDirections: fixture.input('uint32', [0]).view,
    basinCount: 1,
    histogramBins: 1,
    output: {
      statistics: fixture.output('float32', 6).view,
      hypsometry: fixture.output('float32', 1).view,
      profiles:
        storageUses === 'mixed'
          ? fixture.output('float32', 3).view
          : view('float32', 3, byteOffset),
      profileStatus: view('uint32', 1, 0),
      status: {
        invalidCount: storageUses === 'mixed' ? view('uint32', 1, byteOffset) : invalidCount.view
      }
    }
  };
  return {
    create: () => new GPUTerrainCatchmentSummary(props),
    async read() {
      const values = await readFloat32(packed.buffer, byteOffset / 4 + 4);
      const words = await packed.read();
      return {
        values: Array.from(values.slice(byteOffset / 4, byteOffset / 4 + 3)),
        status: [words[0]],
        invalidCount: Array.from(await invalidCount.read())
      };
    }
  };
}
