// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it, vi} from 'vitest';
import {GPUTerrainBreach} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-breach';
import {GPUTerrainBranchingFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-branching-flow';
import {GPUTerrainTiledFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow';
import {computeTerrainBranchingFlowOnCPU} from '../../../src/gpu-terrain/terrain-processing/cpu-terrain-branching-flow-operation';
import {
  createTerrainReviewFixture,
  createTerrainReviewTiles,
  createTerrainReviewEmptyView,
  createTerrainReviewPackedDispatch
} from './terrain-review-fixture';
import {requireTestDevice} from '../../utils/require-test-device';

const GRID = {
  width: 1,
  height: 1,
  affine: [1, 0, 0, 0, 1, 0] as const,
  pixelInterpretation: 'area' as const
};

it('rejects a physical breach input/output alias imported once at construction', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createTerrainReviewFixture(device);
  try {
    const elevation = fixture.input('float32', [5]);
    expect(
      () =>
        new GPUTerrainBreach({
          grid: GRID,
          elevation: elevation.view,
          pitCells: fixture.input('uint32', [0]).view,
          maximumSteps: 1,
          maximumCutDepth: 1,
          elevationDecrement: 0.25,
          output: {
            elevation: fixture.graph.createDataView(elevation.view.buffer, {
              format: 'float32',
              length: 1
            }),
            cutDepth: fixture.output('float32', 1).view,
            modificationMask: fixture.output('uint32', 1).view,
            pathCells: fixture.output('uint32', 2).view,
            pathLengths: fixture.output('uint32', 1).view,
            pitStatus: fixture.output('uint32', 1).view,
            unresolvedCount: fixture.output('uint32', 1).view
          }
        })
    ).toThrow(/terrain-breach.*input and output.*overlap/);
  } finally {
    fixture.destroy();
  }
});

for (const kind of ['tiled', 'catchment'] as const) {
  it(`rejects mixed storage before graph mutation and executes aligned writable packing (${kind})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createTerrainReviewFixture(device);
    let compiled: ReturnType<typeof fixture.run> | undefined;
    try {
      for (const byteOffset of [4, 256]) {
        const scenario = createTerrainReviewPackedDispatch(fixture, kind, byteOffset, 'mixed');
        const resources = vi.spyOn(fixture.graph, 'createTransientBuffer');
        const declarations = vi.spyOn(fixture.graph, 'add');
        expect(scenario.create).toThrow(/input and output aligned binding usage scopes/);
        expect(resources).not.toHaveBeenCalled();
        expect(declarations).not.toHaveBeenCalled();
        resources.mockRestore();
        declarations.mockRestore();
      }
      const legal = createTerrainReviewPackedDispatch(fixture, kind, 256, 'writable');
      fixture.graph.add(legal.create());
      compiled = fixture.run();
      expect(await legal.read()).toEqual({
        values: kind === 'tiled' ? [3] : [0, 7, 0],
        status: [1],
        invalidCount: [0]
      });
    } finally {
      compiled?.destroy();
      fixture.destroy();
    }
  });
}

for (const execution of ['resident', 'auto'] as const) {
  it(`reuses shared stream/output storage across separate dispatches (${execution})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createTerrainReviewFixture(device);
    let compiled: ReturnType<typeof fixture.run> | undefined;
    try {
      const scenario = createTerrainReviewPackedDispatch(fixture, 'tiled', 256, 'mixed', execution);
      fixture.graph.add(scenario.create());
      compiled = fixture.run();
      expect(await scenario.read()).toEqual({values: [3], status: [1], invalidCount: [0]});
    } finally {
      compiled?.destroy();
      fixture.destroy();
    }
  });
}

it('executes burn-only breach with genuine independent zero-byte imported pit and path buffers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createTerrainReviewFixture(device);
  const empty = ['pits', 'paths', 'lengths', 'pit-status'].map(() =>
    device.createBuffer({byteLength: 0, usage: Buffer.COPY_SRC})
  );
  let compiled: ReturnType<typeof fixture.run> | undefined;
  try {
    const elevation = fixture.output('float32', 3);
    const cutDepth = fixture.output('float32', 3);
    const mask = fixture.output('uint32', 3);
    const unresolved = fixture.output('uint32', 1);
    fixture.graph.add(
      new GPUTerrainBreach({
        grid: {...GRID, width: 3},
        elevation: fixture.input('float32', [5, NaN, -3]).view,
        burnDepth: fixture.input('float32', [1, 0, 2]).view,
        pitCells: createTerrainReviewEmptyView(fixture, empty[0], 'pits'),
        maximumSteps: 0xffffffff,
        maximumCutDepth: 1,
        elevationDecrement: 0.25,
        output: {
          elevation: elevation.view,
          cutDepth: cutDepth.view,
          modificationMask: mask.view,
          pathCells: createTerrainReviewEmptyView(fixture, empty[1], 'paths'),
          pathLengths: createTerrainReviewEmptyView(fixture, empty[2], 'lengths'),
          pitStatus: createTerrainReviewEmptyView(fixture, empty[3], 'pit-status'),
          unresolvedCount: unresolved.view
        }
      })
    );
    compiled = fixture.run();
    expect(Array.from(await elevation.read())).toEqual([4, NaN, -5]);
    expect(Array.from(await cutDepth.read())).toEqual([1, NaN, 2]);
    expect(Array.from(await mask.read())).toEqual([2, 0, 2]);
    expect(Array.from(await unresolved.read())).toEqual([0]);
  } finally {
    compiled?.destroy();
    fixture.destroy();
    empty.forEach(buffer => buffer.destroy());
  }
});

for (const execution of ['resident', 'tiled'] as const) {
  it(`writes adjacent tile output and aggregate status ranges safely (${execution})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createTerrainReviewFixture(device);
    let compiled: ReturnType<typeof fixture.run> | undefined;
    try {
      const packed = fixture.output('float32', 2);
      const accumulation = fixture.graph.createDataView(packed.view.buffer, {
        format: 'float32',
        length: 1
      });
      const invalidCount = fixture.graph.createDataView(packed.view.buffer, {
        format: 'uint32',
        length: 1,
        byteOffset: 4
      });
      const status = fixture.output('uint32', 1);
      const reach = fixture.output('uint32', 1);
      fixture.graph.add(
        new GPUTerrainTiledFlow({
          grid: GRID,
          execution,
          maximumIterations: 8,
          tiles: [
            {
              column: 0,
              row: 0,
              width: 1,
              height: 1,
              owned: [0, 0, 1, 1],
              flowDirections: fixture.input('uint32', [0]).view,
              streams: fixture.input('uint32', [2]).view,
              runoff: fixture.input('float32', [3]).view,
              output: {accumulation, status: status.view, reachIds: reach.view}
            }
          ],
          output: {status: {invalidCount}}
        })
      );
      compiled = fixture.run();
      expect(Array.from(await packed.read())).toEqual([3, 0]);
      expect(Array.from(await status.read())).toEqual([1]);
      expect(Array.from(await reach.read())).toEqual([0]);
    } finally {
      compiled?.destroy();
      fixture.destroy();
    }
  });
}

it('returns zero for a one-cell distance outlet whose unused affine determinant overflows float32', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createTerrainReviewFixture(device);
  let compiled: ReturnType<typeof fixture.run> | undefined;
  try {
    const values = fixture.output('float32', 1);
    const status = fixture.output('uint32', 1);
    const unresolved = fixture.output('uint32', 1);
    fixture.graph.add(
      new GPUTerrainBranchingFlow({
        grid: {...GRID, affine: [1e20, 0, 0, 0, 1e20, 0]},
        fractions: fixture.input('float32', Array(8).fill(0)).view,
        values: fixture.input('float32', [0]).view,
        mode: 'expected-distance',
        maximumIterations: 4,
        output: {values: values.view, status: status.view, unresolvedCount: unresolved.view}
      })
    );
    compiled = fixture.run();
    expect(Array.from(await values.read())).toEqual([0]);
    expect(Array.from(await status.read())).toEqual([1]);
    expect(Array.from(await unresolved.read())).toEqual([0]);
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
});

for (const scene of [
  'unequal-255-256-257',
  'snake',
  'invalid-and-cycle',
  'dynamic-range'
] as const) {
  it(`reconciles D8 boundaries against independent values and the resident route: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createTerrainReviewFixture(device);
    const width = scene === 'unequal-255-256-257' ? 768 : scene === 'snake' ? 8 : 6;
    const height = scene === 'unequal-255-256-257' ? 1 : scene === 'snake' ? 4 : 3;
    const count = width * height;
    const directions = Uint32Array.from({length: count}, (_, cell) =>
      (cell % width) + 1 < width ? 1 : 0
    );
    const runoff = new Float32Array(count).fill(1);
    const streams = new Uint32Array(count).fill(2);
    const expected = Float32Array.from({length: count}, (_, cell) => (cell % width) + 1);
    const expectedStates = new Uint32Array(count).fill(1);
    const expectedReaches = Uint32Array.from(
      {length: count},
      (_, cell) => Math.floor(cell / width) * width
    );
    if (scene === 'snake')
      for (let row = 0; row < height; row++)
        for (let column = 0; column < width; column++) {
          const cell = row * width + column;
          const reverse = row % 2 === 1;
          const last = reverse ? column === 0 : column === width - 1;
          directions[cell] = last ? (row + 1 < height ? 4 : 0) : reverse ? 16 : 1;
          expected[cell] = row * width + (reverse ? width - column : column + 1);
          expectedReaches[cell] = 0;
        }
    if (scene === 'invalid-and-cycle') {
      runoff[1] = NaN;
      directions[width + 2] = 16;
      expectedStates[1] = 0;
      expected[1] = NaN;
      expectedReaches[1] = 0xffffffff;
      for (let cell = 2; cell < width; cell++) {
        expectedStates[cell] = 2;
        expected[cell] = NaN;
        expectedReaches[cell] = 0xffffffff;
      }
      for (let cell = width + 1; cell < width + 3; cell++) {
        expectedStates[cell] = 2;
        expected[cell] = NaN;
        expectedReaches[cell] = 0xffffffff;
      }
      for (let cell = width + 3; cell < width * 2; cell++) {
        expected[cell] = cell - width - 2;
        expectedReaches[cell] = width + 3;
      }
      streams.fill(0, width * 2);
      expectedReaches.fill(0xffffffff, width * 2);
    }
    if (scene === 'dynamic-range') {
      for (let cell = 0; cell < count; cell++) {
        const row = Math.floor(cell / width);
        const injection = row === 0 ? Math.fround(1e30) : row === 1 ? 1 : Math.fround(1e-20);
        runoff[cell] = injection;
        expected[cell] = Math.fround(((cell % width) + 1) * injection);
      }
    }
    const fractions = new Float32Array(count * 8);
    for (let cell = 0; cell < count; cell++)
      if (directions[cell] > 0) fractions[cell * 8 + Math.log2(directions[cell])] = 1;
    const grid = {...GRID, width, height};
    const oracle = computeTerrainBranchingFlowOnCPU(
      {fractions, values: runoff},
      {grid, mode: 'transport'}
    );
    expect(Array.from(oracle.values)).toEqual(Array.from(expected));
    expect(Array.from(oracle.status)).toEqual(Array.from(expectedStates));
    const widths =
      scene === 'unequal-255-256-257'
        ? [255, 256, 257]
        : [Math.floor(width / 2), width - Math.floor(width / 2)];
    const captures = (['resident', 'tiled'] as const).map(execution => {
      const tiles = createTerrainReviewTiles(
        fixture,
        width,
        height,
        widths,
        widths.map((_, index) => index),
        directions,
        runoff,
        streams
      );
      const invalidCount = fixture.output('uint32', 1);
      const converged = fixture.output('uint32', 1);
      const limit = fixture.output('uint32', 1);
      fixture.graph.add(
        new GPUTerrainTiledFlow({
          id: execution,
          grid,
          execution,
          maximumIterations: 64,
          tiles: tiles.map(capture => capture.tile).reverse(),
          output: {
            status: {
              invalidCount: invalidCount.view,
              converged: converged.view,
              iterationLimitReached: limit.view
            }
          }
        })
      );
      return {tiles, invalidCount, converged, limit};
    });
    let compiled: ReturnType<typeof fixture.run> | undefined;
    try {
      compiled = fixture.run();
      for (const capture of captures) {
        for (const tileCapture of capture.tiles) {
          const values = await tileCapture.accumulation.read();
          const states = await tileCapture.status.read();
          const reaches = await tileCapture.reachIds.read();
          const {tile} = tileCapture;
          const [ownedColumn, ownedRow, ownedWidth, ownedHeight] = tile.owned;
          for (let index = 0; index < values.length; index++) {
            const column = index % tile.width;
            const row = Math.floor(index / tile.width);
            if (
              column >= ownedColumn &&
              column < ownedColumn + ownedWidth &&
              row >= ownedRow &&
              row < ownedRow + ownedHeight
            ) {
              const cell =
                (row - ownedRow) * width + tileCapture.globalColumn + column - ownedColumn;
              if (scene === 'dynamic-range')
                expect(Math.abs(values[index] - expected[cell])).toBeLessThanOrEqual(
                  Math.abs(expected[cell]) * 2e-5
                );
              else expect(values[index]).toBe(expected[cell]);
              expect(states[index]).toBe(expectedStates[cell]);
              expect(reaches[index]).toBe(expectedReaches[cell]);
            } else {
              expect(values[index]).toBeNaN();
              expect(states[index]).toBe(0);
              expect(reaches[index]).toBe(0xffffffff);
            }
          }
        }
        expect(Array.from(await capture.invalidCount.read())).toEqual([
          scene === 'invalid-and-cycle' ? 1 : 0
        ]);
        expect(Array.from(await capture.converged.read())).toEqual([
          scene === 'invalid-and-cycle' ? 0 : 1
        ]);
        expect(Array.from(await capture.limit.read())).toEqual([0]);
      }
    } finally {
      compiled?.destroy();
      fixture.destroy();
    }
  });
}

it('automatically reconciles 4196352 cells above the old resident fraction ceiling', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createTerrainReviewFixture(device);
  const width = 2049;
  const height = 2048;
  const count = width * height;
  const directions = Uint32Array.from({length: count}, (_, cell) =>
    (cell % width) + 1 < width ? 1 : 0
  );
  const captures = createTerrainReviewTiles(
    fixture,
    width,
    height,
    [410, 410, 410, 410, 409],
    [0, 0, 0, 0, 0],
    directions,
    new Float32Array(count).fill(1),
    new Uint32Array(count).fill(2)
  );
  const converged = fixture.output('uint32', 1);
  let compiled: ReturnType<typeof fixture.run> | undefined;
  try {
    fixture.graph.add(
      new GPUTerrainTiledFlow({
        grid: {...GRID, width, height},
        tiles: captures.map(capture => capture.tile),
        output: {status: {converged: converged.view}}
      })
    );
    compiled = fixture.run();
    for (const capture of captures) {
      const values = await capture.accumulation.read();
      const states = await capture.status.read();
      const reaches = await capture.reachIds.read();
      // Check every cell with a closed form, without millions of assertion objects.
      const mismatch = values.findIndex(
        (value, index) =>
          value !== (index % capture.tile.width) + capture.globalColumn + 1 ||
          states[index] !== 1 ||
          reaches[index] !== Math.floor(index / capture.tile.width) * width
      );
      expect(mismatch).toBe(-1);
    }
    expect(Array.from(await converged.read())).toEqual([1]);
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
}, 120000);

it('reconciles diagonal flow across an irregular two-row-band tile partition', async () => {
  const {createTerrainReviewRectangles} = await import('./terrain-review-fixture');
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createTerrainReviewFixture(device);
  const width = 5;
  const height = 5;
  const count = width * height;
  const directions = Uint32Array.from({length: count}, (_, cell) => {
    const column = cell % width;
    const row = Math.floor(cell / width);
    return column < 4 && row < 4 ? 2 : column < 4 ? 1 : row < 4 ? 4 : 0;
  });
  const expected = Float32Array.from({length: count}, (_, cell) => {
    const column = cell % width;
    const row = Math.floor(cell / width);
    return row === 4 && column === 4
      ? 25
      : row === 4
        ? ((column + 1) * (column + 2)) / 2
        : column === 4
          ? ((row + 1) * (row + 2)) / 2
          : Math.min(row, column) + 1;
  });
  const expectedReaches = Uint32Array.from({length: count}, (_, cell) => {
    const column = cell % width;
    const row = Math.floor(cell / width);
    const length = Math.min(row, column);
    return row === 4 || column === 4 ? cell : (row - length) * width + column - length;
  });
  const captures = createTerrainReviewRectangles(
    fixture,
    width,
    [
      [0, 0, 2, 2],
      [2, 0, 3, 2],
      [0, 2, 3, 3],
      [3, 2, 2, 3]
    ],
    [0, 1, 2, 1],
    directions,
    new Float32Array(count).fill(1),
    new Uint32Array(count).fill(2)
  );
  let compiled: ReturnType<typeof fixture.run> | undefined;
  try {
    fixture.graph.add(
      new GPUTerrainTiledFlow({
        grid: {...GRID, width, height},
        execution: 'tiled',
        tiles: captures.map(capture => capture.tile).reverse()
      })
    );
    compiled = fixture.run();
    for (const capture of captures) {
      const values = await capture.accumulation.read();
      const states = await capture.status.read();
      const reaches = await capture.reachIds.read();
      const {tile} = capture;
      const [column, row, ownedWidth, ownedHeight] = tile.owned;
      for (let local = 0; local < values.length; local++) {
        const localColumn = local % tile.width;
        const localRow = Math.floor(local / tile.width);
        if (
          localColumn >= column &&
          localColumn < column + ownedWidth &&
          localRow >= row &&
          localRow < row + ownedHeight
        ) {
          const cell =
            (capture.globalRow + localRow - row) * width +
            capture.globalColumn +
            localColumn -
            column;
          expect(values[local]).toBe(expected[cell]);
          expect(states[local]).toBe(1);
          expect(reaches[local]).toBe(expectedReaches[cell]);
        } else {
          expect(values[local]).toBeNaN();
          expect(states[local]).toBe(0);
          expect(reaches[local]).toBe(0xffffffff);
        }
      }
    }
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
});

it('reuses the same compiled tiled route after runoff changes, resetting every output and status port', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createTerrainReviewFixture(device);
  const width = 6;
  const height = 2;
  const count = width * height;
  const directions = Uint32Array.from({length: count}, (_, cell) =>
    cell % width < width - 1 ? 1 : 0
  );
  const captures = createTerrainReviewTiles(
    fixture,
    width,
    height,
    [2, 4],
    [1, 2],
    directions,
    new Float32Array(count).fill(1),
    new Uint32Array(count).fill(2)
  );
  const invalidCount = fixture.output('uint32', 1);
  const converged = fixture.output('uint32', 1);
  const iterationLimitReached = fixture.output('uint32', 1);
  let compiled: ReturnType<typeof fixture.run> | undefined;
  try {
    fixture.graph.add(
      new GPUTerrainTiledFlow({
        grid: {...GRID, width, height},
        execution: 'tiled',
        tiles: captures.map(capture => capture.tile),
        output: {
          status: {
            invalidCount: invalidCount.view,
            converged: converged.view,
            iterationLimitReached: iterationLimitReached.view
          }
        }
      })
    );
    const runs = [
      new Float32Array(count).fill(1),
      Float32Array.from({length: count}, (_, cell) => (cell === 1 ? NaN : 2)),
      new Float32Array(count).fill(3)
    ];
    for (const [run, runoff] of runs.entries()) {
      for (const capture of captures) capture.writeRunoff(runoff);
      if (compiled) fixture.submit(compiled);
      else compiled = fixture.run();
      for (const capture of captures) {
        const values = await capture.accumulation.read();
        const states = await capture.status.read();
        const reaches = await capture.reachIds.read();
        const [ownedColumn, ownedRow, ownedWidth, ownedHeight] = capture.tile.owned;
        for (let index = 0; index < values.length; index++) {
          const column = index % capture.tile.width;
          const row = Math.floor(index / capture.tile.width);
          const owned =
            column >= ownedColumn &&
            column < ownedColumn + ownedWidth &&
            row >= ownedRow &&
            row < ownedRow + ownedHeight;
          const globalColumn = capture.globalColumn + column - ownedColumn;
          const globalRow = capture.globalRow + row - ownedRow;
          const invalid = owned && run === 1 && globalRow === 0 && globalColumn === 1;
          const blocked = owned && run === 1 && globalRow === 0 && globalColumn > 1;
          const state = owned ? (invalid ? 0 : blocked ? 2 : 1) : 0;
          expect(states[index]).toBe(state);
          if (state === 1) expect(values[index]).toBe((globalColumn + 1) * (run + 1));
          else expect(values[index]).toBeNaN();
          expect(reaches[index]).toBe(state === 1 ? globalRow * width : 0xffffffff);
        }
      }
      expect(Array.from(await invalidCount.read())).toEqual([run === 1 ? 1 : 0]);
      expect(Array.from(await converged.read())).toEqual([run === 1 ? 0 : 1]);
      expect(Array.from(await iterationLimitReached.read())).toEqual([0]);
    }
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
});

for (const widths of [
  [6, 2],
  [3, 5]
]) {
  it(`preserves upstream trees and invalid donors around a ${widths[0] === 6 ? 'within-tile' : 'cross-tile'} cycle`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createTerrainReviewFixture(device);
    const width = 8;
    const height = 4;
    const count = width * height;
    const directions = new Uint32Array(count);
    for (const cell of [8, 9, 10]) directions[cell] = 1;
    directions[11] = 16;
    directions[0] = 2;
    directions[1] = 2;
    directions[16] = 64;
    directions[18] = 64;
    const runoff = new Float32Array(count).fill(1);
    runoff[1] = NaN;
    const streams = new Uint32Array(count).fill(2);
    streams[18] = 0xffffffff;
    // Closed form: 16 -> 8 -> 9 -> (10 <-> 11), with 0 -> 9 and invalid 1/18 -> 10.
    const expectedValues = new Float32Array(count).fill(1);
    expectedValues[8] = 2;
    expectedValues[9] = 4;
    const expectedStates = new Uint32Array(count).fill(1);
    const expectedReaches = Uint32Array.from({length: count}, (_, cell) => cell);
    expectedReaches[8] = 16;
    for (const cell of [1, 18, 10, 11]) {
      expectedValues[cell] = NaN;
      expectedStates[cell] = cell === 1 || cell === 18 ? 0 : 2;
      expectedReaches[cell] = 0xffffffff;
    }
    const captures = createTerrainReviewTiles(
      fixture,
      width,
      height,
      widths,
      [1, 2],
      directions,
      runoff,
      streams
    );
    const invalidCount = fixture.output('uint32', 1);
    const converged = fixture.output('uint32', 1);
    const iterationLimitReached = fixture.output('uint32', 1);
    let compiled: ReturnType<typeof fixture.run> | undefined;
    try {
      fixture.graph.add(
        new GPUTerrainTiledFlow({
          grid: {...GRID, width, height},
          execution: 'tiled',
          tiles: captures.map(capture => capture.tile),
          output: {
            status: {
              invalidCount: invalidCount.view,
              converged: converged.view,
              iterationLimitReached: iterationLimitReached.view
            }
          }
        })
      );
      compiled = fixture.run();
      for (const capture of captures) {
        const values = await capture.accumulation.read();
        const states = await capture.status.read();
        const reaches = await capture.reachIds.read();
        const [ownedColumn, ownedRow, ownedWidth, ownedHeight] = capture.tile.owned;
        for (let index = 0; index < values.length; index++) {
          const column = index % capture.tile.width;
          const row = Math.floor(index / capture.tile.width);
          const owned =
            column >= ownedColumn &&
            column < ownedColumn + ownedWidth &&
            row >= ownedRow &&
            row < ownedRow + ownedHeight;
          const cell =
            (capture.globalRow + row - ownedRow) * width +
            capture.globalColumn +
            column -
            ownedColumn;
          if (owned) {
            expect(values[index]).toBe(expectedValues[cell]);
            expect(states[index]).toBe(expectedStates[cell]);
            expect(reaches[index]).toBe(expectedReaches[cell]);
          } else {
            expect(values[index]).toBeNaN();
            expect(states[index]).toBe(0);
            expect(reaches[index]).toBe(0xffffffff);
          }
        }
      }
      expect(Array.from(await invalidCount.read())).toEqual([2]);
      expect(Array.from(await converged.read())).toEqual([0]);
      expect(Array.from(await iterationLimitReached.read())).toEqual([0]);
    } finally {
      compiled?.destroy();
      fixture.destroy();
    }
  });
}

it('searches nonempty breach windows clamped at both corners and centred in the interior', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createTerrainReviewFixture(device);
  const source = new Float32Array(81).fill(9);
  for (const cell of [0, 40, 80]) source[cell] = 2;
  for (const cell of [1, 31, 79]) source[cell] = 0;
  const elevation = fixture.output('float32', 81);
  const cutDepth = fixture.output('float32', 81);
  const modificationMask = fixture.output('uint32', 81);
  const pathCells = fixture.output('uint32', 9);
  const pathLengths = fixture.output('uint32', 3);
  const pitStatus = fixture.output('uint32', 3);
  const unresolvedCount = fixture.output('uint32', 1);
  let compiled: ReturnType<typeof fixture.run> | undefined;
  try {
    fixture.graph.add(
      new GPUTerrainBreach({
        grid: {...GRID, width: 9, height: 9},
        elevation: fixture.input('float32', source).view,
        pitCells: fixture.input('uint32', [0, 40, 80]).view,
        maximumSteps: 2,
        maximumCutDepth: 1,
        elevationDecrement: 0.5,
        output: {
          elevation: elevation.view,
          cutDepth: cutDepth.view,
          modificationMask: modificationMask.view,
          pathCells: pathCells.view,
          pathLengths: pathLengths.view,
          pitStatus: pitStatus.view,
          unresolvedCount: unresolvedCount.view
        }
      })
    );
    compiled = fixture.run();
    expect(Array.from(await elevation.read())).toEqual(Array.from(source));
    expect(Array.from(await cutDepth.read())).toEqual(Array(81).fill(0));
    expect(Array.from(await modificationMask.read())).toEqual(Array(81).fill(0));
    expect(Array.from(await pathCells.read())).toEqual([
      0, 1, 0xffffffff, 40, 31, 0xffffffff, 80, 79, 0xffffffff
    ]);
    expect(Array.from(await pathLengths.read())).toEqual([2, 2, 2]);
    expect(Array.from(await pitStatus.read())).toEqual([1, 1, 1]);
    expect(Array.from(await unresolvedCount.read())).toEqual([0]);
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
});
