// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUTerrainFlowDistance} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow-distance';
import {GPUTerrainStreamNetwork} from '../../../src/gpu-terrain/hydrology/gpu-terrain-stream-network';
import {
  getRasterGridGeoreference,
  getRasterGroundCellSize
} from '../../../src/analysis-core/contracts/raster-grid';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';

async function setup() {
  const candidateDevice = await getWebGPUTestDevice();
  requireTestDevice(candidateDevice);
  const device = candidateDevice;
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createOutputBuffer>[] = [];
  let identifier = 0;
  function input<Format extends 'uint32' | 'float32' | 'float32x2'>(
    values: number[],
    format: Format
  ): GraphDataView<Format> {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
    );
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `input-${identifier++}`,
      buffer,
      format,
      values.length / (format === 'float32x2' ? 2 : 1)
    );
  }
  function output<Format extends 'uint32' | 'float32' | 'float32x2' | 'uint32x2' | 'uint32x3'>(
    length: number,
    format: Format
  ) {
    const components = format.endsWith('x2') ? 2 : format.endsWith('x3') ? 3 : 1;
    const buffer = createOutputBuffer(device, length * components);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `output-${identifier++}`, buffer, format, length)
    };
  }
  return {
    graph,
    input,
    output,
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
function grid(width: number, height: number): GPURasterMetadata {
  return {width, height, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'};
}

it('flow distance rejects malformed directions and unresolved stream masks throughout a path', async () => {
  const {graph, input, output, run} = await setup();
  const results = [];
  for (const [caseIndex, testCase] of [
    {directions: [1, 3, 0], streams: [0, 0, 0]},
    {directions: [1, 64, 0], streams: [0, 0, 0]},
    {directions: [1, 1, 0], streams: [0, 0xfffffffe, 0]}
  ].entries()) {
    const distance = output(3, 'float32');
    const validity = output(3, 'uint32');
    const terminals = output(3, 'uint32');
    results.push({distance, validity, terminals});
    graph.add(
      new GPUTerrainFlowDistance({
        id: `invalid-distance-${caseIndex}`,
        grid: grid(3, 1),
        flowDirections: input(testCase.directions, 'uint32'),
        streams: input(testCase.streams, 'uint32'),
        output: {distance: distance.view, validity: validity.view, terminalCells: terminals.view}
      })
    );
  }
  const destroy = run();
  try {
    for (const result of results) {
      expect(Array.from(await readUint32(result.validity.buffer, 3))).toEqual([0, 0, 1]);
      expect(Array.from(await readUint32(result.terminals.buffer, 3))).toEqual([
        0xfffffffe, 0xfffffffe, 2
      ]);
      const values = Array.from(await readFloat32(result.distance.buffer, 3));
      expect(values[0]).toBeNaN();
      expect(values[1]).toBeNaN();
      expect(values[2]).toBe(0);
    }
  } finally {
    destroy();
  }
});

it('stream network rejects malformed directions and unresolved stream masks without publishing reaches', async () => {
  const {graph, input, output, run} = await setup();
  const results = [];
  for (const [caseIndex, testCase] of [
    // Cells 0 and 1 have invalid downstream paths. Cell 2 has no possible donor.
    {directions: [1, 3, 0], streams: [1, 1, 1], unresolved: 2},
    {directions: [1, 64, 0], streams: [1, 1, 1], unresolved: 2},
    {directions: [1, NO_CELL, 0], streams: [1, 1, 1], unresolved: 2},
    // Cell 1 can donate to either neighbor when its direction is unresolved.
    {directions: [1, UNRESOLVED_DIRECTION, 0], streams: [1, 1, 1], unresolved: 3},
    // Cell 1 and the cell draining to it are unresolved. Cell 2 may receive a donor from cell 1.
    {directions: [1, 1, 0], streams: [1, NO_CELL, 1], unresolved: 3},
    {directions: [1, 1, 0], streams: [1, UNRESOLVED_DIRECTION, 1], unresolved: 3}
  ].entries()) {
    const reachCount = output(1, 'uint32');
    const unresolvedCount = output(1, 'uint32');
    results.push({reachCount, unresolvedCount, expectedUnresolved: testCase.unresolved});
    graph.add(
      new GPUTerrainStreamNetwork({
        id: `invalid-network-${caseIndex}`,
        grid: grid(3, 1),
        flowDirections: input(testCase.directions, 'uint32'),
        streams: input(testCase.streams, 'uint32'),
        output: {
          sourceCells: output(3, 'uint32').view,
          targetCells: output(3, 'uint32').view,
          vertexOffsets: output(4, 'uint32').view,
          positions: output(6, 'float32x2').view,
          lengths: output(3, 'float32').view,
          reachCount: reachCount.view,
          vertexCount: output(1, 'uint32').view,
          status: {overflow: output(1, 'uint32').view},
          unresolvedCount: unresolvedCount.view
        }
      })
    );
  }
  const destroy = run();
  try {
    for (const result of results) {
      expect(Array.from(await readUint32(result.reachCount.buffer, 1))).toEqual([0]);
      expect(Array.from(await readUint32(result.unresolvedCount.buffer, 1))).toEqual([
        result.expectedUnresolved
      ]);
    }
  } finally {
    destroy();
  }
});

it('stream reach length is independent of a large affine translation', async () => {
  const {graph, input, output, run} = await setup();
  const lengths = output(1, 'float32');
  const reachCount = output(1, 'uint32');
  graph.add(
    new GPUTerrainStreamNetwork({
      grid: {...grid(3, 1), affine: [0.25, 0, 1e8, 0, 0.25, 1e8]},
      flowDirections: input([1, 1, 0], 'uint32'),
      streams: input([1, 1, 1], 'uint32'),
      output: {
        sourceCells: output(1, 'uint32').view,
        targetCells: output(1, 'uint32').view,
        vertexOffsets: output(2, 'uint32').view,
        positions: output(3, 'float32x2').view,
        lengths: lengths.view,
        reachCount: reachCount.view,
        vertexCount: output(1, 'uint32').view,
        status: {overflow: output(1, 'uint32').view},
        unresolvedCount: output(1, 'uint32').view
      }
    })
  );
  const destroy = run();
  try {
    expect(Array.from(await readUint32(reachCount.buffer, 1))).toEqual([1]);
    expect(Array.from(await readFloat32(lengths.buffer, 1))).toEqual([0.5]);
  } finally {
    destroy();
  }
});

const NO_CELL = 0xffffffff;
const UNRESOLVED_DIRECTION = 0xfffffffe;
const COLUMN_OFFSETS = [1, 1, 0, -1, -1, -1, 0, 1];
const ROW_OFFSETS = [0, 1, 1, 1, 0, -1, -1, -1];

/** Deterministic pseudo-random sequence so the failing case is reproducible. */
function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function getReceiver(cell: number, code: number, width: number, height: number): number {
  if (code === 0 || code > 128 || (code & (code - 1)) !== 0) return NO_CELL;
  const direction = Math.log2(code);
  const column = (cell % width) + COLUMN_OFFSETS[direction];
  const row = Math.floor(cell / width) + ROW_OFFSETS[direction];
  if (column < 0 || row < 0 || column >= width || row >= height) return NO_CELL;
  return row * width + column;
}

/** Mostly eastward and southward flow with outlets, so paths are long, plus cycles and invalid codes. */
function createRandomDirections(width: number, height: number, seed: number): number[] {
  const random = createRandom(seed);
  return Array.from({length: width * height}, () => {
    const draw = random();
    if (draw < 0.03) return 0;
    if (draw < 0.04) return 3;
    if (draw < 0.05) return NO_CELL;
    if (draw < 0.15) return 1 << Math.floor(random() * 8);
    return [1, 2, 4][Math.floor(random() * 3)];
  });
}

/** The serial per-cell walk that GPUTerrainFlowDistance used before pointer jumping. */
function walkFlowDistance(
  cell: number,
  directions: number[],
  streams: number[] | undefined,
  costs: number[] | undefined,
  width: number,
  height: number,
  maximumSteps: number
) {
  if (directions[cell] === NO_CELL) return {distance: NaN, valid: 0, terminal: NO_CELL};
  let current = cell;
  let total = 0;
  for (let step = 0; step < maximumSteps; step++) {
    const code = directions[current];
    const receiver = getReceiver(current, code, width, height);
    if (code !== 0 && receiver === NO_CELL) break;
    if (streams) {
      if (streams[current] === NO_CELL || streams[current] === UNRESOLVED_DIRECTION) break;
      if (streams[current] !== 0) return {distance: total, valid: 1, terminal: current};
    }
    if (receiver === NO_CELL) return {distance: total, valid: 1, terminal: current};
    const cost = costs
      ? Math.fround(costs[current])
      : Math.hypot(
          (receiver % width) - (current % width),
          Math.floor(receiver / width) - Math.floor(current / width)
        );
    if (!Number.isFinite(cost) || cost < 0) break;
    total += cost;
    current = receiver;
  }
  return {distance: NaN, valid: 0, terminal: UNRESOLVED_DIRECTION};
}

it('flow distance by pointer jumping matches the serial path walk on random grids', async () => {
  const width = 24;
  const height = 20;
  const count = width * height;
  const directions = createRandomDirections(width, height, 7);
  const random = createRandom(11);
  const costs = Array.from({length: count}, () => {
    const draw = random();
    return draw < 0.01 ? NaN : draw < 0.02 ? -1 : 0.5 + 1.5 * random();
  });
  const streams = Array.from({length: count}, () => {
    const draw = random();
    return draw < 0.01 ? NO_CELL : draw < 0.02 ? UNRESOLVED_DIRECTION : draw < 0.1 ? 1 : 0;
  });
  const {graph, input, output, run} = await setup();
  const cases = [
    {name: 'costs', streams: undefined, costs, maximumSteps: undefined},
    {name: 'streams', streams, costs: undefined, maximumSteps: undefined},
    {name: 'bounded', streams: undefined, costs, maximumSteps: 5},
    {name: 'plain', streams: undefined, costs: undefined, maximumSteps: undefined}
  ].map(testCase => {
    const result = {
      distance: output(count, 'float32'),
      validity: output(count, 'uint32'),
      terminals: output(count, 'uint32')
    };
    const contributor = new GPUTerrainFlowDistance({
      id: `random-${testCase.name}`,
      grid: grid(width, height),
      flowDirections: input(directions, 'uint32'),
      ...(testCase.streams ? {streams: input(testCase.streams, 'uint32')} : {}),
      ...(testCase.costs ? {transitionCosts: input(testCase.costs, 'float32')} : {}),
      ...(testCase.maximumSteps ? {maximumSteps: testCase.maximumSteps} : {}),
      output: {
        distance: result.distance.view,
        validity: result.validity.view,
        terminalCells: result.terminals.view
      }
    });
    graph.add(contributor);
    return {...testCase, result};
  });
  const destroy = run();
  try {
    for (const testCase of cases) {
      const distance = Array.from(await readFloat32(testCase.result.distance.buffer, count));
      const validity = Array.from(await readUint32(testCase.result.validity.buffer, count));
      const terminals = Array.from(await readUint32(testCase.result.terminals.buffer, count));
      let validCount = 0;
      for (let cell = 0; cell < count; cell++) {
        const expected = walkFlowDistance(
          cell,
          directions,
          testCase.streams,
          testCase.costs,
          width,
          height,
          testCase.maximumSteps ?? count
        );
        expect(validity[cell], `${testCase.name} validity of ${cell}`).toBe(expected.valid);
        expect(terminals[cell], `${testCase.name} terminal of ${cell}`).toBe(expected.terminal);
        if (expected.valid) {
          validCount++;
          expect(
            Math.abs(distance[cell] - expected.distance),
            `${testCase.name} distance of ${cell}`
          ).toBeLessThanOrEqual(1e-4 * Math.max(1, expected.distance));
        } else {
          expect(distance[cell]).toBeNaN();
        }
      }
      expect(validCount, `${testCase.name} has resolved paths`).toBeGreaterThan(10);
    }
  } finally {
    destroy();
  }
});

it('flow distance resolves a path of several hundred cells in logarithmic rounds', async () => {
  const width = 300;
  const {graph, input, output, run} = await setup();
  const distance = output(width, 'float32');
  const validity = output(width, 'uint32');
  const terminals = output(width, 'uint32');
  const contributor = new GPUTerrainFlowDistance({
    grid: {...grid(width, 1), affine: [0.5, 0, 0, 0, 0.5, 0]},
    flowDirections: input([...Array(width - 1).fill(1), 0], 'uint32'),
    output: {distance: distance.view, validity: validity.view, terminalCells: terminals.view}
  });
  graph.add(contributor);
  const destroy = run();
  try {
    const values = Array.from(await readFloat32(distance.buffer, width));
    for (let cell = 0; cell < width; cell++) {
      expect(values[cell]).toBeCloseTo((width - 1 - cell) * 0.5, 3);
    }
    expect(Array.from(await readUint32(validity.buffer, width)).every(value => value === 1)).toBe(
      true
    );
    expect(
      Array.from(await readUint32(terminals.buffer, width)).every(value => value === width - 1)
    ).toBe(true);
  } finally {
    destroy();
  }
});

/** Exhaustively classify donors, then trace each candidate stream path one cell at a time. */
function traceStreamNetwork(
  directions: number[],
  streams: number[],
  width: number,
  height: number,
  maximumSteps: number
) {
  const count = width * height;
  const isStream = (cell: number) => streams[cell] !== 0;
  const hasUnknownMembership = (cell: number) =>
    streams[cell] === NO_CELL || streams[cell] === UNRESOLVED_DIRECTION;
  const receivers = directions.map((code, cell) => getReceiver(cell, code, width, height));
  const donorCounts = directions.map((_, cell) => {
    let donors = 0;
    for (let donor = 0; donor < count; donor++) {
      if (!isStream(donor) || donor === cell) continue;
      if (receivers[donor] === cell) {
        if (hasUnknownMembership(donor)) return undefined;
        donors++;
      } else if (
        directions[donor] === UNRESOLVED_DIRECTION &&
        Math.abs((donor % width) - (cell % width)) <= 1 &&
        Math.abs(Math.floor(donor / width) - Math.floor(cell / width)) <= 1
      ) {
        return undefined;
      }
    }
    return donors;
  });
  const vertices = new Array<number>(count).fill(0);
  let unresolved = 0;
  for (let index = 0; index < count; index++) {
    if (!isStream(index)) continue;
    let current = index;
    let complete = false;
    for (let step = 0; step < maximumSteps; step++) {
      if (hasUnknownMembership(current) || donorCounts[current] === undefined) break;
      const receiver = receivers[current];
      if (directions[current] !== 0 && receiver === NO_CELL) break;
      if (receiver === NO_CELL || !isStream(receiver)) {
        complete = true;
        break;
      }
      current = receiver;
    }
    unresolved += complete ? 0 : 1;
    if (complete && donorCounts[index] !== 1) {
      current = index;
      let reachVertices = 1;
      for (let step = 0; step < maximumSteps; step++) {
        const receiver = receivers[current];
        if (receiver === NO_CELL || !isStream(receiver)) break;
        current = receiver;
        reachVertices++;
        if (donorCounts[current] !== 1) break;
      }
      vertices[index] = reachVertices < 2 ? 0 : reachVertices;
    }
  }
  return {vertices, unresolved};
}

it('stream network by pointer jumping and compaction matches the serial trace on random grids', async () => {
  const width = 22;
  const height = 18;
  const count = width * height;
  const directions = createRandomDirections(width, height, 23);
  const random = createRandom(5);
  const streams = Array.from({length: count}, () => {
    const draw = random();
    return draw < 0.01 ? NO_CELL : draw < 0.75 ? 1 : 0;
  });
  const {graph, input, output, run} = await setup();
  const reference = traceStreamNetwork(directions, streams, width, height, count);
  const reaches = reference.vertices.flatMap((vertices, cell) =>
    vertices > 0 ? [{cell, vertices}] : []
  );
  expect(reaches.length, 'the grid has reaches').toBeGreaterThan(5);
  const totalVertices = reaches.reduce((total, reach) => total + reach.vertices, 0);
  const createCase = (name: string, reachCapacity: number, vertexCapacity: number) => {
    const result = {
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
    const contributor = new GPUTerrainStreamNetwork({
      id: name,
      grid: grid(width, height),
      flowDirections: input(directions, 'uint32'),
      streams: input(streams, 'uint32'),
      output: {
        sourceCells: result.sourceCells.view,
        targetCells: result.targetCells.view,
        vertexOffsets: result.vertexOffsets.view,
        positions: result.positions.view,
        lengths: result.lengths.view,
        reachCount: result.reachCount.view,
        vertexCount: result.vertexCount.view,
        status: {overflow: result.overflow.view},
        unresolvedCount: result.unresolvedCount.view
      }
    });
    graph.add(contributor);
    return result;
  };
  const full = createCase('full', reaches.length + 3, totalVertices + 5);
  // Room for the first four reaches only, and for fewer vertices than that.
  const keptReaches = 4;
  const keptVertices = reaches
    .slice(0, keptReaches)
    .reduce((total, reach) => total + reach.vertices, 0);
  const small = createCase('small', keptReaches, keptVertices + 1);
  const destroy = run();
  try {
    expect(Array.from(await readUint32(full.unresolvedCount.buffer, 1))).toEqual([
      reference.unresolved
    ]);
    expect(Array.from(await readUint32(full.reachCount.buffer, 1))).toEqual([reaches.length]);
    expect(Array.from(await readUint32(full.vertexCount.buffer, 1))).toEqual([totalVertices]);
    expect(Array.from(await readUint32(full.overflow.buffer, 1))).toEqual([0]);
    expect(Array.from(await readUint32(full.sourceCells.buffer, reaches.length))).toEqual(
      reaches.map(reach => reach.cell)
    );
    let offset = 0;
    const offsets = [0, ...reaches.map(reach => (offset += reach.vertices))];
    expect(Array.from(await readUint32(full.vertexOffsets.buffer, reaches.length + 1))).toEqual(
      offsets
    );
    const targets = Array.from(await readUint32(full.targetCells.buffer, reaches.length));
    const lengths = Array.from(await readFloat32(full.lengths.buffer, reaches.length));
    for (const [reachIndex, reach] of reaches.entries()) {
      let current = reach.cell;
      let length = 0;
      for (let vertex = 1; vertex < reach.vertices; vertex++) {
        const receiver = getReceiver(current, directions[current], width, height);
        length += Math.hypot(
          (receiver % width) - (current % width),
          Math.floor(receiver / width) - Math.floor(current / width)
        );
        current = receiver;
      }
      expect(targets[reachIndex]).toBe(current);
      expect(lengths[reachIndex]).toBeCloseTo(length, 4);
    }
    // Overflow keeps the reaches that fit in order, reports it, and still counts every unresolved cell.
    expect(Array.from(await readUint32(small.overflow.buffer, 1))).toEqual([1]);
    expect(Array.from(await readUint32(small.unresolvedCount.buffer, 1))).toEqual([
      reference.unresolved
    ]);
    expect(Array.from(await readUint32(small.reachCount.buffer, 1))).toEqual([keptReaches]);
    expect(Array.from(await readUint32(small.vertexCount.buffer, 1))).toEqual([keptVertices]);
    expect(Array.from(await readUint32(small.sourceCells.buffer, keptReaches))).toEqual(
      reaches.slice(0, keptReaches).map(reach => reach.cell)
    );
  } finally {
    destroy();
  }
});

it('flow distance and stream network lengths follow cellSizeMode', async () => {
  const {graph, input, output, run} = await setup();
  const geographicGrid: GPURasterMetadata = {
    width: 3,
    height: 2,
    affine: [0.001, 0, 10, 0, -0.001, 50],
    pixelInterpretation: 'area'
  };
  const distance = output(6, 'float32');
  const validity = output(6, 'uint32');
  const terminals = output(6, 'uint32');
  graph.add(
    new GPUTerrainFlowDistance({
      id: 'geographic-distance',
      grid: geographicGrid,
      cellSizeMode: 'geographic',
      // Row 0 flows east then south in the last column; row 1 flows east.
      flowDirections: input([1, 1, 4, 1, 1, 0], 'uint32'),
      output: {distance: distance.view, validity: validity.view, terminalCells: terminals.view}
    })
  );
  const lengths = output(1, 'float32');
  graph.add(
    new GPUTerrainStreamNetwork({
      id: 'geographic-network',
      grid: geographicGrid,
      cellSizeMode: 'geographic',
      flowDirections: input([1, 1, 4, 1, 1, 0], 'uint32'),
      streams: input([1, 1, 1, 0, 0, 0], 'uint32'),
      output: {
        sourceCells: output(1, 'uint32').view,
        targetCells: output(1, 'uint32').view,
        vertexOffsets: output(2, 'uint32').view,
        positions: output(3, 'float32x2').view,
        lengths: lengths.view,
        reachCount: output(1, 'uint32').view,
        vertexCount: output(1, 'uint32').view,
        status: {overflow: output(1, 'uint32').view},
        unresolvedCount: output(1, 'uint32').view
      }
    })
  );
  const destroy = run();
  try {
    const georeference = getRasterGridGeoreference(geographicGrid);
    const [rowZeroWidth] = getRasterGroundCellSize('geographic', georeference, 0.5, 2);
    const [rowOneWidth] = getRasterGroundCellSize('geographic', georeference, 1.5, 2);
    const [, midpointHeight] = getRasterGroundCellSize('geographic', georeference, 1, 2);
    const values = Array.from(await readFloat32(distance.buffer, 6));
    expect(values[4] / rowOneWidth).toBeCloseTo(1, 4);
    expect(values[2] / midpointHeight).toBeCloseTo(1, 4);
    expect(values[0] / (2 * rowZeroWidth + midpointHeight)).toBeCloseTo(1, 3);
    // The mask ends at cell 2, so the reach is the two east moves of row 0.
    const reachLength = (await readFloat32(lengths.buffer, 1))[0];
    expect(reachLength / (2 * rowZeroWidth)).toBeCloseTo(1, 3);
  } finally {
    destroy();
  }
});
