// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {makeGPURasterBand} from '@luma-contributors/analysis/raster-analysis';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUTerrainFlow,
  getGPUTerrainFlowParameterValues
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow';
import {GPUTerrainFlowDistance} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow-distance';
import {GPUTerrainHeightAboveDrainage} from '../../../src/gpu-terrain/hydrology/gpu-terrain-height-above-drainage';
import {GPUTerrainStreamNetwork} from '../../../src/gpu-terrain/hydrology/gpu-terrain-stream-network';
import {GPUTerrainStreamOrder} from '../../../src/gpu-terrain/hydrology/gpu-terrain-stream-order';
import {GPUTerrainWatersheds} from '../../../src/gpu-terrain/hydrology/gpu-terrain-watersheds';
import type {GPUTerrainFlowDirectionEncoding} from '../../../src/gpu-terrain/hydrology/terrain-flow-tracing';
import {GPUTerrainCatchmentSummary} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary';
import {GPUTerrainTiledFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Math correctness review 2026-11-10, terrain-hydrology-05 (direction encodings), -04 (one
// meaning of "stream cell") and -03 (stream network vertex positions).
//
// One 4 x 4 direction grid, written by hand in the three numberings from each tool's own table.
// Compass directions on a north-up raster (row 0 is the northernmost row), T a terminal cell:
//
//   E   SE  S   SW
//   NE  E   T   W
//   N   NW  SW  S
//   E   T   W   W
//
// ESRI (ArcGIS, pysheds, GPUTerrainFlow): E 1, SE 2, S 4, SW 8, W 16, NW 32, N 64, NE 128.
// WhiteboxTools default pointer:          NE 1, E 2, SE 4, S 8, SW 16, W 32, NW 64, N 128.
// TauDEM D8:                              E 1, NE 2, N 3, NW 4, W 5, SW 6, S 7, SE 8.
const WIDTH = 4;
const HEIGHT = 4;
const CELL_COUNT = WIDTH * HEIGHT;
const DIRECTIONS: Record<GPUTerrainFlowDirectionEncoding, number[]> = {
  esri: [1, 2, 4, 8, 128, 1, 0, 16, 64, 32, 8, 4, 1, 0, 16, 16],
  whitebox: [2, 4, 8, 16, 1, 2, 0, 32, 128, 64, 16, 8, 2, 0, 32, 32],
  taudem: [1, 8, 7, 6, 2, 1, 0, 5, 3, 4, 6, 7, 1, 0, 5, 5]
};
// Hand traced. The terminal cells are 6 (row 1, column 2) and 13 (row 3, column 1).
const EXPECTED_BASINS = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 13, 13, 13, 13, 13, 13];
// Strahler order with every cell a stream cell: cell 4 has two order-1 donors (cells 8 and 9), so
// it and the cells below it (1, then 6) are order 2; cell 13 has three order-1 donors.
const EXPECTED_ORDER = [1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1];
// Upstream cell counts including the cell itself.
const EXPECTED_ACCUMULATION = [1, 5, 1, 1, 3, 1, 10, 1, 1, 1, 1, 1, 1, 6, 3, 2];
// Fourteen cells have a receiver: nine cardinal moves and five diagonal ones, on unit cells.
const EXPECTED_STREAM_LENGTH = 9 + 5 * Math.SQRT2;
// A reach starts at each of the ten headwaters and at the two confluences that are not terminal.
const EXPECTED_REACH_COUNT = 12;

// A north-up raster: rows advance south (`affine[4] < 0`), so the compass names above are true.
const UNIT_GRID: GPURasterMetadata = {
  width: WIDTH,
  height: HEIGHT,
  affine: [1, 0, 0, 0, -1, HEIGHT],
  pixelInterpretation: 'area'
};
// The same cells with rows advancing north (`affine[4] > 0`): row + 1 is now geographic north.
const ROWS_NORTH_GRID: GPURasterMetadata = {...UNIT_GRID, affine: [1, 0, 0, 0, 1, 0]};

async function setup() {
  const candidateDevice = await getWebGPUTestDevice();
  requireTestDevice(candidateDevice);
  // A constant of the narrowed type: the helper functions below are hoisted closures, which do
  // not see the narrowing of `candidateDevice`.
  const device = candidateDevice;
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createOutputBuffer>[] = [];
  let identifier = 0;
  function input<Format extends 'uint32' | 'float32'>(
    values: number[],
    format: Format
  ): GraphDataView<Format> {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
    );
    buffers.push(buffer);
    return importGraphBuffer(graph, `input-${identifier++}`, buffer, format, values.length);
  }
  function output<Format extends 'uint32' | 'float32' | 'float32x2'>(
    length: number,
    format: Format
  ) {
    const components = format === 'float32x2' ? 2 : 1;
    const buffer = createOutputBuffer(device, length * components);
    buffers.push(buffer);
    return {
      view: importGraphBuffer(graph, `output-${identifier++}`, buffer, format, length),
      async read(count = length) {
        return Array.from(
          await (format === 'uint32'
            ? readUint32(buffer, count * components)
            : readFloat32(buffer, count * components))
        );
      }
    };
  }
  return {
    graph,
    input,
    output,
    destroyBuffers() {
      buffers.forEach(buffer => buffer.destroy());
    },
    run() {
      const compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      return () => {
        compiled.destroy();
        buffers.forEach(buffer => buffer.destroy());
      };
    }
  };
}

/** Outputs of a stream network with room for `reachCapacity` reaches of `vertexCapacity` vertices in all. */
function createStreamNetworkOutput(
  output: Awaited<ReturnType<typeof setup>>['output'],
  reachCapacity: number,
  vertexCapacity: number
) {
  return {
    sourceCells: output(reachCapacity, 'uint32'),
    targetCells: output(reachCapacity, 'uint32'),
    vertexOffsets: output(reachCapacity + 1, 'uint32'),
    positions: output(vertexCapacity, 'float32x2'),
    lengths: output(reachCapacity, 'float32'),
    reachCount: output(1, 'uint32'),
    vertexCount: output(1, 'uint32'),
    overflow: output(1, 'uint32'),
    unresolvedCount: output(1, 'uint32')
  };
}

function getStreamNetworkOutputViews(outputs: ReturnType<typeof createStreamNetworkOutput>) {
  return {
    sourceCells: outputs.sourceCells.view,
    targetCells: outputs.targetCells.view,
    vertexOffsets: outputs.vertexOffsets.view,
    positions: outputs.positions.view,
    lengths: outputs.lengths.view,
    reachCount: outputs.reachCount.view,
    vertexCount: outputs.vertexCount.view,
    status: {overflow: outputs.overflow.view},
    unresolvedCount: outputs.unresolvedCount.view
  };
}

describe('direction encodings (terrain-hydrology-05)', () => {
  for (const directionEncoding of ['esri', 'whitebox', 'taudem'] as const) {
    it(`a ${directionEncoding} direction grid gives the hand-traced drainage`, async () => {
      const {graph, input, output, run} = await setup();
      const flowDirections = input(DIRECTIONS[directionEncoding], 'uint32');
      const allStreams = input(Array(CELL_COUNT).fill(1), 'uint32');
      // Only the two terminal cells are streams, so each cell's drainage cell is its basin.
      const terminalStreams = input(
        EXPECTED_BASINS.map((basin, cell) => (basin === cell ? 1 : 0)),
        'uint32'
      );
      const elevation = input(Array(CELL_COUNT).fill(1), 'float32');
      const labels = output(CELL_COUNT, 'uint32'),
        drainageCells = output(CELL_COUNT, 'uint32'),
        streamOrder = output(CELL_COUNT, 'uint32'),
        terminalCells = output(CELL_COUNT, 'uint32'),
        distanceValidity = output(CELL_COUNT, 'uint32'),
        statistics = output(6, 'float32'),
        accumulation = output(CELL_COUNT, 'float32'),
        tiledStatus = output(CELL_COUNT, 'uint32');
      const network = createStreamNetworkOutput(output, CELL_COUNT, 2 * CELL_COUNT);
      graph.add(
        new GPUTerrainWatersheds({
          grid: UNIT_GRID,
          flowDirections,
          directionEncoding,
          output: {labels: labels.view}
        })
      );
      graph.add(
        new GPUTerrainHeightAboveDrainage({
          grid: UNIT_GRID,
          elevation: makeGPURasterBand(elevation, UNIT_GRID, {id: 'elevation'}),
          flowDirections,
          directionEncoding,
          streams: terminalStreams,
          output: {drainageCells: drainageCells.view}
        })
      );
      graph.add(
        new GPUTerrainStreamOrder({
          grid: UNIT_GRID,
          flowDirections,
          directionEncoding,
          streams: allStreams,
          output: {streamOrder: streamOrder.view}
        })
      );
      graph.add(
        new GPUTerrainFlowDistance({
          grid: UNIT_GRID,
          flowDirections,
          directionEncoding,
          output: {
            distance: output(CELL_COUNT, 'float32').view,
            validity: distanceValidity.view,
            terminalCells: terminalCells.view
          }
        })
      );
      graph.add(
        new GPUTerrainStreamNetwork({
          grid: UNIT_GRID,
          flowDirections,
          directionEncoding,
          streams: allStreams,
          output: getStreamNetworkOutputViews(network)
        })
      );
      graph.add(
        new GPUTerrainCatchmentSummary({
          grid: UNIT_GRID,
          elevation,
          basinIds: input(Array(CELL_COUNT).fill(0), 'uint32'),
          basinCount: 1,
          streams: allStreams,
          flowDirections,
          directionEncoding,
          histogramBins: 1,
          output: {
            statistics: statistics.view,
            hypsometry: output(1, 'float32').view,
            profiles: output(CELL_COUNT * 3, 'float32').view,
            profileStatus: output(CELL_COUNT, 'uint32').view
          }
        })
      );
      graph.add(
        new GPUTerrainTiledFlow({
          grid: UNIT_GRID,
          directionEncoding,
          tiles: [
            {
              column: 0,
              row: 0,
              width: WIDTH,
              height: HEIGHT,
              owned: [0, 0, WIDTH, HEIGHT],
              flowDirections,
              runoff: input(Array(CELL_COUNT).fill(1), 'float32'),
              streams: allStreams,
              output: {
                accumulation: accumulation.view,
                status: tiledStatus.view,
                reachIds: output(CELL_COUNT, 'uint32').view
              }
            }
          ]
        })
      );
      const destroy = run();
      try {
        expect(await labels.read(), 'watersheds').toEqual(EXPECTED_BASINS);
        expect(await drainageCells.read(), 'drainage cells').toEqual(EXPECTED_BASINS);
        expect(await streamOrder.read(), 'stream order').toEqual(EXPECTED_ORDER);
        expect(await distanceValidity.read()).toEqual(Array(CELL_COUNT).fill(1));
        expect(await terminalCells.read(), 'terminal cells').toEqual(EXPECTED_BASINS);
        expect(await tiledStatus.read()).toEqual(Array(CELL_COUNT).fill(1));
        expect(await accumulation.read(), 'accumulation').toEqual(EXPECTED_ACCUMULATION);
        expect((await statistics.read())[4], 'stream length').toBeCloseTo(
          EXPECTED_STREAM_LENGTH,
          4
        );
        expect(await network.unresolvedCount.read()).toEqual([0]);
        const [reachCount] = await network.reachCount.read();
        expect(reachCount, 'reach count').toBe(EXPECTED_REACH_COUNT);
        const lengths = await network.lengths.read(reachCount);
        expect(
          lengths.reduce((total, length) => total + length, 0),
          'reach lengths'
        ).toBeCloseTo(EXPECTED_STREAM_LENGTH, 4);
      } finally {
        destroy();
      }
    }, 120000);
  }

  it('a Whitebox grid read with the default numbering is silently rotated, which is why the prop exists', async () => {
    const {graph, input, output, run} = await setup();
    const labels = output(CELL_COUNT, 'uint32');
    graph.add(
      new GPUTerrainWatersheds({
        grid: UNIT_GRID,
        flowDirections: input(DIRECTIONS.whitebox, 'uint32'),
        output: {labels: labels.view}
      })
    );
    const destroy = run();
    try {
      expect(await labels.read()).not.toEqual(EXPECTED_BASINS);
    } finally {
      destroy();
    }
  });

  // Review follow-up fr-fields-08. The contract is raster-relative: a code names a row and column
  // offset ("south" is row + 1), whatever the row sign of the grid affine. `GPUTerrainFlow` writes
  // codes the same way, so producer and consumers agree on any grid.
  it('direction codes are raster offsets: a rows-north grid decodes the same words to the same cells', async () => {
    const {graph, input, output, run} = await setup();
    const results = (['esri', 'whitebox', 'taudem'] as const).flatMap(directionEncoding =>
      [UNIT_GRID, ROWS_NORTH_GRID].map((grid, gridIndex) => {
        const labels = output(CELL_COUNT, 'uint32');
        graph.add(
          new GPUTerrainWatersheds({
            id: `${directionEncoding}-${gridIndex}`,
            grid,
            flowDirections: input(DIRECTIONS[directionEncoding], 'uint32'),
            directionEncoding,
            output: {labels: labels.view}
          })
        );
        return labels;
      })
    );
    const destroy = run();
    try {
      for (const labels of results) {
        expect(await labels.read()).toEqual(EXPECTED_BASINS);
      }
    } finally {
      destroy();
    }
  });

  it('GPUTerrainFlow writes raster-relative codes on a rows-north grid and its consumers read them back', async () => {
    const {graph, input, output, run} = await setup();
    // 2 x 2 DEM in raster order. Hand computed steepest descents: cell 0 drops 4 over a diagonal
    // (2.83 per unit) to cell 3, against 1 east and 2 down a row; cell 1 drops 3 down a row to
    // cell 3; cell 2 drops 2 east to cell 3; cell 3 is the outlet. In raster terms: row + 1 and
    // column + 1 (code 2), row + 1 (code 4), column + 1 (code 1), terminal (0).
    const elevation = [5, 4, 3, 1];
    const expectedDirections = [2, 4, 1, 0];
    const results = [UNIT_GRID, ROWS_NORTH_GRID].map((baseGrid, gridIndex) => {
      const grid: GPURasterMetadata = {...baseGrid, width: 2, height: 2};
      const flowDirections = output(4, 'uint32'),
        labels = output(4, 'uint32');
      graph.add(
        new GPUTerrainFlow({
          id: `flow-${gridIndex}`,
          grid,
          elevation: makeGPURasterBand(input(elevation, 'float32'), grid, {
            id: `elevation-${gridIndex}`
          }),
          parameters: input(Array.from(getGPUTerrainFlowParameterValues({}, grid)), 'float32'),
          output: {flowDirections: flowDirections.view}
        })
      );
      graph.add(
        new GPUTerrainWatersheds({
          id: `basins-${gridIndex}`,
          grid,
          flowDirections: flowDirections.view,
          output: {labels: labels.view}
        })
      );
      return {flowDirections, labels};
    });
    const destroy = run();
    try {
      for (const result of results) {
        // The same words on both grids: on the rows-north grid code 4 ("south" in the ESRI table)
        // points geographically north, and every consumer still follows it to row + 1.
        expect(await result.flowDirections.read()).toEqual(expectedDirections);
        expect(await result.labels.read()).toEqual([3, 3, 3, 3]);
      }
    } finally {
      destroy();
    }
  }, 120000);

  it('zero and the sentinels mean the same in every numbering, and a word outside it has no receiver', async () => {
    const {graph, input, output, run} = await setup();
    const rowGrid: GPURasterMetadata = {...UNIT_GRID, width: 5, height: 1};
    // East, east, a word outside the numbering, invalid, unresolved.
    const words = {
      esri: [1, 1, 3, 0xffffffff, 0xfffffffe],
      whitebox: [2, 2, 3, 0xffffffff, 0xfffffffe],
      taudem: [1, 1, 9, 0xffffffff, 0xfffffffe]
    };
    const results = (['esri', 'whitebox', 'taudem'] as const).map(directionEncoding => {
      const labels = output(5, 'uint32');
      graph.add(
        new GPUTerrainWatersheds({
          id: directionEncoding,
          grid: rowGrid,
          flowDirections: input(words[directionEncoding], 'uint32'),
          directionEncoding,
          output: {labels: labels.view}
        })
      );
      return labels;
    });
    const destroy = run();
    try {
      for (const labels of results) {
        // Cells 0 and 1 drain to cell 2, which is terminal; 0xffffffff is invalid (no label) and
        // 0xfffffffe is a valid cell with an unknown destination.
        expect(await labels.read()).toEqual([2, 2, 2, 0xffffffff, 0xfffffffe]);
      }
    } finally {
      destroy();
    }
  });

  it('rejects an unknown numbering at construction', async () => {
    const {input, output, destroyBuffers} = await setup();
    expect(
      () =>
        new GPUTerrainWatersheds({
          grid: UNIT_GRID,
          flowDirections: input(DIRECTIONS.esri, 'uint32'),
          directionEncoding: 'grass' as GPUTerrainFlowDirectionEncoding,
          output: {labels: output(CELL_COUNT, 'uint32').view}
        })
    ).toThrow(/terrain-watersheds directionEncoding must be 'esri', 'whitebox' or 'taudem'/);
    destroyBuffers();
  });
});

describe('one meaning of "stream cell" (terrain-hydrology-04)', () => {
  it('a mask of Strahler orders is a stream mask for the catchment summary and tiled flow', async () => {
    const {graph, input, output, run} = await setup();
    const flowDirections = input(DIRECTIONS.esri, 'uint32');
    const results = [Array(CELL_COUNT).fill(1), EXPECTED_ORDER].map((streamWords, index) => {
      const streams = input(streamWords, 'uint32');
      const statistics = output(6, 'float32'),
        profileStatus = output(CELL_COUNT, 'uint32'),
        reachIds = output(CELL_COUNT, 'uint32'),
        tiledStatus = output(CELL_COUNT, 'uint32');
      graph.add(
        new GPUTerrainCatchmentSummary({
          id: `summary-${index}`,
          grid: UNIT_GRID,
          elevation: input(Array(CELL_COUNT).fill(1), 'float32'),
          basinIds: input(Array(CELL_COUNT).fill(0), 'uint32'),
          basinCount: 1,
          streams,
          flowDirections,
          histogramBins: 1,
          output: {
            statistics: statistics.view,
            hypsometry: output(1, 'float32').view,
            profiles: output(CELL_COUNT * 3, 'float32').view,
            profileStatus: profileStatus.view
          }
        })
      );
      graph.add(
        new GPUTerrainTiledFlow({
          id: `tiled-${index}`,
          grid: UNIT_GRID,
          tiles: [
            {
              column: 0,
              row: 0,
              width: WIDTH,
              height: HEIGHT,
              owned: [0, 0, WIDTH, HEIGHT],
              flowDirections,
              runoff: input(Array(CELL_COUNT).fill(1), 'float32'),
              streams,
              output: {
                accumulation: output(CELL_COUNT, 'float32').view,
                status: tiledStatus.view,
                reachIds: reachIds.view
              }
            }
          ]
        })
      );
      return {statistics, profileStatus, reachIds, tiledStatus};
    });
    const destroy = run();
    try {
      const [binary, orders] = results;
      // Every stream edge counts, whatever nonzero word marks its cells.
      expect((await orders.statistics.read())[4]).toBeCloseTo(EXPECTED_STREAM_LENGTH, 4);
      expect(await orders.profileStatus.read()).toEqual(Array(CELL_COUNT).fill(1));
      expect(await orders.tiledStatus.read()).toEqual(Array(CELL_COUNT).fill(1));
      // The order mask gives exactly what the 0/1 mask gives.
      expect(await orders.statistics.read()).toEqual(await binary.statistics.read());
      expect(await orders.reachIds.read()).toEqual(await binary.reachIds.read());
      // Reach identity: a headwater starts its own reach, a cell with one stream donor inherits it.
      const reachIds = await orders.reachIds.read();
      expect(reachIds[11]).toBe(11);
      expect(reachIds[15]).toBe(11);
      expect(reachIds[14]).toBe(11);
      expect(reachIds[13]).toBe(13);
    } finally {
      destroy();
    }
  });

  it('the unresolved stream words are not stream cells and stay unresolved', async () => {
    const {graph, input, output, run} = await setup();
    const rowGrid: GPURasterMetadata = {...UNIT_GRID, width: 4, height: 1};
    const statistics = output(6, 'float32'),
      profileStatus = output(4, 'uint32'),
      tiledStatus = output(4, 'uint32');
    const streams = input([1, 0xffffffff, 0xfffffffe, 1], 'uint32');
    const flowDirections = input([1, 1, 1, 0], 'uint32');
    graph.add(
      new GPUTerrainCatchmentSummary({
        grid: rowGrid,
        elevation: input([4, 3, 2, 1], 'float32'),
        basinIds: input([0, 0, 0, 0], 'uint32'),
        basinCount: 1,
        streams,
        flowDirections,
        histogramBins: 1,
        output: {
          statistics: statistics.view,
          hypsometry: output(1, 'float32').view,
          profiles: output(12, 'float32').view,
          profileStatus: profileStatus.view
        }
      })
    );
    graph.add(
      new GPUTerrainTiledFlow({
        grid: rowGrid,
        tiles: [
          {
            column: 0,
            row: 0,
            width: 4,
            height: 1,
            owned: [0, 0, 4, 1],
            flowDirections,
            runoff: input([1, 1, 1, 1], 'float32'),
            streams,
            output: {
              accumulation: output(4, 'float32').view,
              status: tiledStatus.view,
              reachIds: output(4, 'uint32').view
            }
          }
        ]
      })
    );
    const destroy = run();
    try {
      // No stream edge joins two resolved stream cells.
      expect((await statistics.read())[4]).toBe(0);
      // Cell 0 drains into an unresolved cell (2), the two sentinels are unresolved (2), cell 3 is
      // a complete profile (1).
      expect(await profileStatus.read()).toEqual([2, 2, 2, 1]);
      // Tiled flow: the sentinel cells are invalid (0) and everything below them is unresolved (2).
      expect(await tiledStatus.read()).toEqual([1, 0, 0, 2]);
    } finally {
      destroy();
    }
  });
});

describe('stream network vertex positions (terrain-hydrology-03)', () => {
  it('positions relative to the grid origin are exact where absolute float32 ones are 0.2 cell off', async () => {
    const {graph, input, output, run} = await setup();
    // 1 m cells at UTM magnitudes, one row flowing east. Column 1234 is the report's example.
    const width = 1300;
    const originX = 4500000.3;
    const originY = 5200000.7;
    const utmGrid: GPURasterMetadata = {
      width,
      height: 1,
      affine: [1, 0, originX, 0, -1, originY],
      pixelInterpretation: 'area'
    };
    const flowDirections = input(
      Array.from({length: width}, (_, column) => (column === width - 1 ? 0 : 1)),
      'uint32'
    );
    const streams = input(Array(width).fill(1), 'uint32');
    const absolute = createStreamNetworkOutput(output, 1, width);
    const relative = createStreamNetworkOutput(output, 1, width);
    graph.add(
      new GPUTerrainStreamNetwork({
        id: 'absolute',
        grid: utmGrid,
        flowDirections,
        streams,
        output: getStreamNetworkOutputViews(absolute)
      })
    );
    const relativeNetwork = new GPUTerrainStreamNetwork({
      id: 'relative',
      grid: utmGrid,
      flowDirections,
      streams,
      localOrigin: [originX, originY],
      output: getStreamNetworkOutputViews(relative)
    });
    graph.add(relativeNetwork);
    const destroy = run();
    try {
      expect(await relative.reachCount.read()).toEqual([1]);
      expect(await relative.vertexCount.read()).toEqual([width]);
      expect(relativeNetwork.localOrigin).toEqual([originX, originY]);
      const relativePositions = await relative.positions.read();
      const absolutePositions = await absolute.positions.read();
      let worstRelative = 0;
      let worstAbsolute = 0;
      for (let column = 0; column < width; column++) {
        // Float64 cell centre: the affine applied to (column + 0.5, 0.5).
        const exactX = originX + column + 0.5;
        const exactY = originY - 0.5;
        worstRelative = Math.max(
          worstRelative,
          Math.abs(relativePositions[2 * column] + relativeNetwork.localOrigin[0] - exactX),
          Math.abs(relativePositions[2 * column + 1] + relativeNetwork.localOrigin[1] - exactY)
        );
        worstAbsolute = Math.max(
          worstAbsolute,
          Math.abs(absolutePositions[2 * column] - exactX),
          Math.abs(absolutePositions[2 * column + 1] - exactY)
        );
      }
      // Relative: column + 0.5 and -0.5 are exact float32 values.
      expect(worstRelative).toBeLessThan(1e-6);
      // Absolute (the default): no better than the float32 spacing of 0.5 m at 4.5e6, which is the
      // documented precision, and visibly worse than a tenth of a cell.
      expect(worstAbsolute).toBeGreaterThan(0.1);
      expect(worstAbsolute).toBeLessThanOrEqual(0.25 + 1e-9);
      // The report's vertex: exact 4,501,234.8, absolute float32 4,501,235.0.
      expect(absolutePositions[2 * 1234]).toBe(4501235);
      expect(relativePositions[2 * 1234]).toBe(1234.5);
    } finally {
      destroy();
    }
  });
});
