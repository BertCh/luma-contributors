// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  addRasterResistanceConnectivityRecipe,
  getRasterResistanceConnectivityParameterValues
} from '../../../src/gpu-spatial-analysis/landscape-ecology/index';
import {
  getRasterGridGeoreference,
  getRasterGroundCellSize,
  type GPURasterCellSizeMode
} from '../../../src/analysis-core/contracts/raster-grid';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from '../recipes/recipe-harness';

type GridCase = {
  name: string;
  grid: GPURasterMetadata;
  resistance: number[];
  validity: number[];
  neighborhood: 4 | 8;
  cellSizeMode?: GPURasterCellSizeMode;
  source: number;
  sink: number;
};

/** One undirected reference edge: its two cells (lower index first) and its conductance. */
type ReferenceEdge = {cells: [number, number]; conductance: number};

type Reference = {
  /** Keyed by {@link getPairKey}. */
  conductances: Map<string, ReferenceEdge>;
  potentials: number[];
  vertexCurrents: number[];
  reachable: boolean;
  invalidCells: number;
  invalidEdges: number;
};

/** Order-free key of a cell pair. */
function getPairKey(first: number, second: number): string {
  return first < second ? `${first}-${second}` : `${second}-${first}`;
}

/**
 * Binary64 reference edges, built without the implementation's slot table: every pair of cells is
 * compared, and two cells are neighbours when their rows and columns each differ by at most one
 * (8 neighbours) or when exactly one of them differs, by one (4 neighbours). Conductance and
 * no-data rules are the documented ones; a conductance that is not finite in float32 is invalid.
 */
function buildReferenceEdges(test: GridCase): Map<string, ReferenceEdge> {
  const {width, height, affine} = test.grid;
  const cellCount = width * height;
  const mode = test.cellSizeMode ?? 'uniform';
  const georeference = getRasterGridGeoreference(test.grid);
  const hasData = (cell: number) =>
    test.validity[cell] !== 0 && Number.isFinite(test.resistance[cell]!);
  const edges = new Map<string, ReferenceEdge>();
  for (let first = 0; first < cellCount; first++) {
    for (let second = first + 1; second < cellCount; second++) {
      const firstRow = Math.floor(first / width);
      const secondRow = Math.floor(second / width);
      const deltaColumn = (second % width) - (first % width);
      const deltaRow = secondRow - firstRow;
      const isNeighbor =
        test.neighborhood === 4
          ? Math.abs(deltaColumn) + Math.abs(deltaRow) === 1
          : Math.max(Math.abs(deltaColumn), Math.abs(deltaRow)) === 1;
      if (!isNeighbor) {
        continue;
      }
      const cells: [number, number] = [first, second];
      if (!hasData(first) || !hasData(second)) {
        edges.set(getPairKey(first, second), {cells, conductance: 0});
        continue;
      }
      const firstResistance = test.resistance[first]!;
      const secondResistance = test.resistance[second]!;
      if (!(firstResistance > 0 && secondResistance > 0)) {
        edges.set(getPairKey(first, second), {cells, conductance: NaN});
        continue;
      }
      let length: number;
      if (mode === 'uniform') {
        length = Math.hypot(
          affine[0] * deltaColumn + affine[1] * deltaRow,
          affine[3] * deltaColumn + affine[4] * deltaRow
        );
      } else {
        // Ground cell size at the midpoint between the two cell centres.
        const [groundX, groundY] = getRasterGroundCellSize(
          mode,
          georeference,
          (firstRow + secondRow) / 2 + 0.5,
          height
        );
        length = Math.hypot(groundX * deltaColumn, groundY * deltaRow);
      }
      const conductance = 2 / (firstResistance + secondResistance) / length;
      edges.set(getPairKey(first, second), {
        cells,
        conductance: Number.isFinite(Math.fround(conductance)) ? conductance : NaN
      });
    }
  }
  return edges;
}

/**
 * Cells with data that are invalid: a zero or negative resistance, or a positive resistance on an
 * edge to another positive resistance whose conductance is not finite in float32.
 */
function getReferenceInvalidCells(test: GridCase, edges: Map<string, ReferenceEdge>): number {
  const invalid = new Set<number>();
  test.resistance.forEach((value, cell) => {
    if (test.validity[cell] !== 0 && Number.isFinite(value) && !(value > 0)) invalid.add(cell);
  });
  for (const {cells, conductance} of edges.values()) {
    const [first, second] = cells;
    if (Number.isNaN(conductance) && test.resistance[first]! > 0 && test.resistance[second]! > 0) {
      invalid.add(first);
      invalid.add(second);
    }
  }
  return invalid.size;
}

/** Dense binary64 grounded-Laplacian solve on the sink's component (Gaussian elimination). */
function solveReference(test: GridCase): Reference {
  const cellCount = test.grid.width * test.grid.height;
  const conductances = buildReferenceEdges(test);
  const invalidCells = getReferenceInvalidCells(test, conductances);
  const invalidEdges = [...conductances.values()].filter(edge =>
    Number.isNaN(edge.conductance)
  ).length;
  const missing = new Array<number>(cellCount).fill(NaN);
  if (invalidEdges > 0) {
    return {
      conductances,
      potentials: missing,
      vertexCurrents: missing,
      reachable: false,
      invalidCells,
      invalidEdges
    };
  }
  const neighbors: [number, number][][] = Array.from({length: cellCount}, () => []);
  for (const {
    cells: [first, second],
    conductance
  } of conductances.values()) {
    if (conductance > 0) {
      neighbors[first]!.push([second, conductance]);
      neighbors[second]!.push([first, conductance]);
    }
  }
  const component = new Set([test.sink]);
  const queue = [test.sink];
  while (queue.length > 0) {
    for (const [other] of neighbors[queue.pop()!]!) {
      if (!component.has(other)) {
        component.add(other);
        queue.push(other);
      }
    }
  }
  if (!component.has(test.source)) {
    return {
      conductances,
      potentials: missing,
      vertexCurrents: missing,
      reachable: false,
      invalidCells,
      invalidEdges
    };
  }
  const unknowns = [...component].filter(cell => cell !== test.sink).sort((a, b) => a - b);
  const position = new Map(unknowns.map((cell, index) => [cell, index]));
  const size = unknowns.length;
  const matrix = unknowns.map(() => new Array<number>(size + 1).fill(0));
  unknowns.forEach((cell, row) => {
    for (const [other, conductance] of neighbors[cell]!) {
      matrix[row]![row]! += conductance;
      const column = position.get(other);
      if (column !== undefined) matrix[row]![column]! -= conductance;
    }
    matrix[row]![size] = cell === test.source ? 1 : 0;
  });
  for (let pivot = 0; pivot < size; pivot++) {
    let best = pivot;
    for (let row = pivot + 1; row < size; row++)
      if (Math.abs(matrix[row]![pivot]!) > Math.abs(matrix[best]![pivot]!)) best = row;
    [matrix[pivot], matrix[best]] = [matrix[best]!, matrix[pivot]!];
    for (let row = pivot + 1; row < size; row++) {
      const factor = matrix[row]![pivot]! / matrix[pivot]![pivot]!;
      for (let column = pivot; column <= size; column++)
        matrix[row]![column]! -= factor * matrix[pivot]![column]!;
    }
  }
  const solution = new Array<number>(size).fill(0);
  for (let row = size - 1; row >= 0; row--) {
    let sum = matrix[row]![size]!;
    for (let column = row + 1; column < size; column++)
      sum -= matrix[row]![column]! * solution[column]!;
    solution[row] = sum / matrix[row]![row]!;
  }
  const potentials = missing.slice();
  potentials[test.sink] = 0;
  unknowns.forEach((cell, index) => {
    potentials[cell] = solution[index]!;
  });
  const vertexCurrents = missing.slice();
  for (const cell of component) {
    let total = cell === test.source || cell === test.sink ? 1 : 0;
    for (const [other, conductance] of neighbors[cell]!)
      total += Math.abs(conductance * (potentials[cell]! - potentials[other]!));
    vertexCurrents[cell] = total / 2;
  }
  return {conductances, potentials, vertexCurrents, reachable: true, invalidCells, invalidEdges};
}

function copyView(
  fixture: RecipeTestFixture,
  name: string,
  source: GraphDataView<'uint32'>
): {buffer: ReturnType<RecipeTestFixture['output']>['buffer']} {
  const output = fixture.output(name, 'uint32', 1);
  fixture.graph.add(
    createWGSLKernelNode(fixture.graph, {
      id: name,
      operation: 'raster-resistance-readback',
      invocationCount: 1,
      bindings: [
        {name: 'source', view: source, type: 'u32', access: 'read'},
        {name: 'output', view: output.view, type: 'u32', access: 'read_write'}
      ],
      body: 'output[outputOffset] = source[sourceOffset];'
    })
  );
  return output;
}

async function readEdges(
  fixture: RecipeTestFixture,
  edges: ReturnType<RecipeTestFixture['output']>,
  edgeCount: number
): Promise<[number, number][]> {
  const words = await fixture.readUint32(edges, edgeCount * 2);
  return Array.from({length: edgeCount}, (_, slot) => [words[slot * 2]!, words[slot * 2 + 1]!]);
}

/**
 * Expects the GPU edge list to be the reference adjacency: every neighbouring pair of cells appears
 * exactly once (the sparse solve takes each undirected edge once), every other slot is a self edge
 * of zero conductance, and each conductance matches the reference.
 */
function expectEdgesMatchReference(
  actual: {edges: [number, number][]; conductances: number[]},
  reference: Map<string, ReferenceEdge>
): void {
  const seen = new Map<string, number>();
  actual.edges.forEach(([first, second], slot) => {
    const conductance = actual.conductances[slot]!;
    if (first === second) {
      expect(conductance, `self edge slot ${slot}`).toBe(0);
      return;
    }
    const key = getPairKey(first, second);
    const edge = reference.get(key);
    expect(edge, `slot ${slot} joins non-neighbours ${key}`).toBeDefined();
    seen.set(key, (seen.get(key) ?? 0) + 1);
    const expected = edge!.conductance;
    if (Number.isNaN(expected)) {
      expect(conductance, `slot ${slot}`).toBeNaN();
    } else {
      expect(Math.abs(conductance - expected), `slot ${slot}`).toBeLessThanOrEqual(
        1e-6 * Math.max(1, Math.abs(expected))
      );
    }
  });
  expect([...seen.entries()].filter(([, count]) => count !== 1)).toEqual([]);
  expect([...reference.keys()].filter(key => !seen.has(key))).toEqual([]);
}

async function runCase(test: GridCase) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, test.name);
  try {
    const cellCount = test.grid.width * test.grid.height;
    const edgeCount = cellCount * (test.neighborhood === 4 ? 2 : 4);
    const potentials = fixture.output('potentials', 'float32', cellCount);
    const vertexCurrents = fixture.output('vertex-currents', 'float32', cellCount);
    const diagnostics = fixture.output('diagnostics', 'float32x4', 1);
    const solverStatus = fixture.output('solver-status', 'uint32x4', 1);
    const conductances = fixture.output('conductances', 'float32', edgeCount);
    const edges = fixture.output('edges', 'uint32x2', edgeCount);
    const invalidCellCount = fixture.output('invalid-cells', 'uint32', 1);
    const result = addRasterResistanceConnectivityRecipe(fixture.graph, {
      grid: test.grid,
      resistance: {
        id: 'resistance',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: fixture.input(
            'resistance',
            Float32Array.from(test.resistance),
            'float32',
            cellCount
          )
        },
        validity: fixture.input('validity', Uint32Array.from(test.validity), 'uint32', cellCount)
      },
      parameters: fixture.parameters(
        'parameters',
        'float32',
        getRasterResistanceConnectivityParameterValues(test.grid)
      ),
      neighborhood: test.neighborhood,
      cellSizeMode: test.cellSizeMode,
      source: test.source,
      sink: test.sink,
      maximumIterations: 1024,
      tolerance: 1e-5,
      output: {
        potentials: potentials.view,
        vertexCurrents: vertexCurrents.view,
        diagnostics: diagnostics.view,
        solverStatus: solverStatus.view,
        conductances: conductances.view,
        edges: edges.view,
        status: {invalidCellCount: invalidCellCount.view}
      }
    });
    const solve = result.status.stages.find(stage => stage.stage === 'solve')!.status;
    const limitReached = copyView(fixture, 'limit-reached', solve.iterationLimitReached!);
    fixture.run();
    return {
      result,
      potentials: await fixture.readFloat32(potentials, cellCount),
      vertexCurrents: await fixture.readFloat32(vertexCurrents, cellCount),
      diagnostics: await fixture.readFloat32(diagnostics, 4),
      solverStatus: await fixture.readUint32(solverStatus, 4),
      conductances: await fixture.readFloat32(conductances, edgeCount),
      edges: await readEdges(fixture, edges, edgeCount),
      invalidCells: (await fixture.readUint32(invalidCellCount, 1))[0]!,
      limitReached: (await fixture.readUint32(limitReached, 1))[0]!
    };
  } finally {
    fixture.destroy();
  }
}

function createGrid(width: number, height: number, affine?: GPURasterMetadata['affine']) {
  return {
    width,
    height,
    affine: affine ?? ([1, 0, 0, 0, -1, 0] as const),
    pixelInterpretation: 'area' as const
  };
}

/** A 7 x 6 grid with a no-data wall in column 3 that leaves a one-cell gap at `gapRow`. */
function createWall(gapRow: number | undefined): Pick<GridCase, 'resistance' | 'validity'> {
  const resistance = Array.from({length: 42}, (_, cell) => 1 + (cell % 5) * 0.25);
  const validity = resistance.map((_, cell) =>
    cell % 7 === 3 && Math.floor(cell / 7) !== gapRow ? 0 : 1
  );
  return {resistance, validity};
}

const CASES: GridCase[] = [
  ...([4, 8] as const).map(neighborhood => ({
    name: `uniform-${neighborhood}`,
    grid: createGrid(6, 6),
    resistance: new Array<number>(36).fill(1),
    validity: new Array<number>(36).fill(1),
    neighborhood,
    source: 0,
    sink: 35
  })),
  ...([4, 8] as const).map(neighborhood => ({
    name: `barrier-with-gap-${neighborhood}`,
    grid: createGrid(7, 6, [2, 0, 10, 0, -1, 5]),
    ...createWall(4),
    neighborhood,
    source: 7,
    sink: 13
  })),
  ...([4, 8] as const).map(neighborhood => {
    // Holes: zero validity, a NaN and a +Infinity resistance are all no-data.
    const resistance = Array.from({length: 40}, (_, cell) => 0.5 + ((cell * 7) % 11) / 4);
    resistance[12] = NaN;
    resistance[27] = Infinity;
    const validity = resistance.map((_, cell) => ([9, 21, 22, 30].includes(cell) ? 0 : 1));
    return {
      name: `no-data-holes-${neighborhood}`,
      grid: createGrid(8, 5, [0.8, 0.3, 0, -0.2, -1.1, 0]),
      resistance,
      validity,
      neighborhood,
      source: 1,
      sink: 38
    };
  }),
  {
    name: 'geographic-8',
    grid: createGrid(5, 4, [0.01, 0, 10, 0, -0.01, 60]),
    resistance: Array.from({length: 20}, (_, cell) => 1 + (cell % 3)),
    validity: new Array<number>(20).fill(1),
    neighborhood: 8,
    cellSizeMode: 'geographic',
    source: 0,
    sink: 19
  }
];

it.each(CASES)('$name matches a binary64 circuit solve', async test => {
  const actual = await runCase(test);
  const expected = solveReference(test);
  expect(actual.solverStatus[0], 'converged').toBe(1);
  expect(actual.solverStatus[2], 'reachable').toBe(1);
  expect(actual.solverStatus[3], 'invalid edges').toBe(0);
  expect(actual.invalidCells).toBe(0);
  expect(actual.limitReached).toBe(0);
  expectEdgesMatchReference(actual, expected.conductances);
  const scale = Math.max(...expected.potentials.filter(Number.isFinite));
  actual.potentials.forEach((value, cell) => {
    const reference = expected.potentials[cell]!;
    if (Number.isNaN(reference)) expect(value, `cell ${cell}`).toBeNaN();
    else expect(Math.abs(value - reference), `cell ${cell}`).toBeLessThanOrEqual(1e-4 * scale);
  });
  actual.vertexCurrents.forEach((value, cell) => {
    const reference = expected.vertexCurrents[cell]!;
    if (Number.isNaN(reference)) expect(value, `cell ${cell}`).toBeNaN();
    else expect(Math.abs(value - reference), `cell ${cell}`).toBeLessThanOrEqual(1e-3);
  });
  const resistance = expected.potentials[test.source]!;
  expect(Math.abs(actual.diagnostics[0]! - resistance)).toBeLessThanOrEqual(1e-4 * resistance);
  expect(Math.abs(actual.diagnostics[2]! - resistance)).toBeLessThanOrEqual(1e-3 * resistance);
});

it('the barrier gap carries the whole current, and closing it disconnects the sink', async () => {
  const open = await runCase({
    name: 'gap-current',
    grid: createGrid(7, 6),
    ...createWall(4),
    neighborhood: 4,
    source: 7,
    sink: 13
  });
  // Unit current crosses the wall only through the gap cell (row 4, column 3).
  expect(Math.abs(open.vertexCurrents[4 * 7 + 3]! - 1)).toBeLessThanOrEqual(1e-4);
  const closed = await runCase({
    name: 'closed-wall',
    grid: createGrid(7, 6),
    ...createWall(undefined),
    neighborhood: 8,
    source: 7,
    sink: 13
  });
  expect(closed.solverStatus).toEqual([0, 0, 0, 0]);
  expect(closed.diagnostics[0]).toBeNaN();
  expect(closed.potentials.every(Number.isNaN)).toBe(true);
});

it('zero and negative resistance with data are counted and block the solve', async () => {
  const resistance = new Array<number>(16).fill(1);
  resistance[5] = 0;
  resistance[10] = -2;
  const test: GridCase = {
    name: 'invalid-resistance',
    grid: createGrid(4, 4),
    resistance,
    validity: new Array<number>(16).fill(1),
    neighborhood: 4,
    source: 0,
    sink: 15
  };
  const actual = await runCase(test);
  const expected = solveReference(test);
  expect(actual.invalidCells).toBe(2);
  expect(expected.invalidEdges).toBe(8);
  expect(actual.solverStatus).toEqual([0, 0, 0, expected.invalidEdges]);
  expect(actual.conductances.filter(Number.isNaN).length).toBe(expected.invalidEdges);
  expect(actual.potentials.every(Number.isNaN)).toBe(true);
  expect(actual.diagnostics[0]).toBeNaN();
});

it.each(
  ([4, 8] as const).flatMap(neighborhood =>
    (
      [
        [5, 3, [1, 0, 0, 0, -1, 0]],
        [3, 5, [2, 0, 0, 0, -0.5, 0]],
        [6, 1, [0.5, 0, 0, 0, -3, 0]],
        [1, 6, [1, 0.25, 0, 0, -2, 0]]
      ] as const
    ).map(([width, height, affine]) => ({
      name: `adjacency-${width}x${height}-${neighborhood}`,
      grid: createGrid(width, height, affine),
      resistance: Array.from({length: width * height}, (_, cell) => 1 + (cell % 4) * 0.5),
      validity: new Array<number>(width * height).fill(1),
      neighborhood,
      source: 0,
      sink: width * height - 1
    }))
  )
)('$name lists every neighbouring pair exactly once', async test => {
  // Non-square grids (5 x 3, 3 x 5, one row, one column) and non-square cells (2 x 0.5,
  // 0.5 x 3, sheared). The reference adjacency compares every pair of cells, so a wrong slot
  // table (a missing, repeated or wrapped direction) fails here.
  const actual = await runCase(test);
  const expected = solveReference(test);
  expect(expected.conductances.size).toBeGreaterThan(0);
  expectEdgesMatchReference(actual, expected.conductances);
  expect(actual.solverStatus[3], 'invalid edges').toBe(0);
});

it.each([
  {neighborhood: 4 as const, tiny: [5, 6]},
  {neighborhood: 4 as const, tiny: [5, 9]},
  {neighborhood: 8 as const, tiny: [5, 10]},
  {neighborhood: 8 as const, tiny: [6, 9]}
])('a conductance that overflows float32 counts both cells as invalid ($neighborhood, $tiny)', async ({
  neighborhood,
  tiny
}) => {
  // Regression: two neighbouring resistances of 1e-35 on 1e-4 cells give a conductance near 1e39,
  // which overflows float32. The edge was invalid (NaN) and blocked the solve, yet
  // invalidCellCount stayed 0.
  const resistance = new Array<number>(16).fill(1);
  for (const cell of tiny) resistance[cell] = 1e-35;
  const test: GridCase = {
    name: `overflow-${neighborhood}-${tiny.join('-')}`,
    grid: createGrid(4, 4, [1e-4, 0, 0, 0, -1e-4, 0]),
    resistance,
    validity: new Array<number>(16).fill(1),
    neighborhood,
    source: 0,
    sink: 15
  };
  const actual = await runCase(test);
  const expected = solveReference(test);
  expect(expected.invalidEdges).toBe(1);
  expect(expected.invalidCells).toBe(2);
  expect(actual.solverStatus).toEqual([0, 0, 0, 1]);
  expect(actual.invalidCells).toBe(2);
  expectEdgesMatchReference(actual, expected.conductances);
});

// Hand-computed circuits. `solveReference` above rebuilds the conductance formula, so it cannot
// pin the convention; these cases do. Resistance is per unit ground length: an edge between cells
// `a` and `b` is a resistor of `mean(R(a), R(b)) * edgeLength`.
type HandCase = {
  test: GridCase;
  /** Conductance of each neighbouring pair, keyed by {@link getPairKey}. */
  conductances: Record<string, number>;
  potentials: number[];
  vertexCurrents: number[];
  effectiveResistance: number;
};

const HAND_CASES: HandCase[] = [
  {
    // Two cells with R = 1, 30 apart: one resistor of 1 * 30. (Circuitscape, which does not
    // divide by the cell size, gives 1.)
    test: {
      name: 'hand-two-cells',
      grid: createGrid(2, 1, [30, 0, 0, 0, -30, 0]),
      resistance: [1, 1],
      validity: [1, 1],
      neighborhood: 4,
      source: 0,
      sink: 1
    },
    conductances: {'0-1': 1 / 30},
    potentials: [30, 0],
    vertexCurrents: [1, 1],
    effectiveResistance: 30
  },
  {
    // Three cells in series with R = 1, 3, 5, 2 apart: resistors mean(1, 3) * 2 = 4 and
    // mean(3, 5) * 2 = 8, 12 in total. Unit current gives potentials 12, 8, 0.
    test: {
      name: 'hand-three-cells-in-series',
      grid: createGrid(3, 1, [2, 0, 0, 0, -2, 0]),
      resistance: [1, 3, 5],
      validity: [1, 1, 1],
      neighborhood: 8,
      source: 0,
      sink: 2
    },
    conductances: {'0-1': 0.25, '1-2': 0.125},
    potentials: [12, 8, 0],
    vertexCurrents: [1, 1, 1],
    effectiveResistance: 12
  },
  {
    // 2 x 2 unit cells with R = 1, eight neighbours, corner to opposite corner. The two side
    // paths are 2 each (conductance 1 together), the diagonal is sqrt(2) long (conductance
    // 1 / sqrt(2)); the other diagonal joins two cells at the same potential and carries
    // nothing. Effective resistance 1 / (1 + 1 / sqrt(2)) = 2 - sqrt(2). Each side cell sits
    // halfway and carries half of the side current (2 - sqrt(2)) / 2.
    test: {
      name: 'hand-square-with-diagonals',
      grid: createGrid(2, 2),
      resistance: [1, 1, 1, 1],
      validity: [1, 1, 1, 1],
      neighborhood: 8,
      source: 0,
      sink: 3
    },
    conductances: {
      '0-1': 1,
      '0-2': 1,
      '1-3': 1,
      '2-3': 1,
      '0-3': Math.SQRT1_2,
      '1-2': Math.SQRT1_2
    },
    potentials: [2 - Math.SQRT2, 1 - Math.SQRT1_2, 1 - Math.SQRT1_2, 0],
    vertexCurrents: [1, 1 - Math.SQRT1_2, 1 - Math.SQRT1_2, 1],
    effectiveResistance: 2 - Math.SQRT2
  }
];

it.each(HAND_CASES)('$test.name matches the circuit solved by hand', async hand => {
  const actual = await runCase(hand.test);
  expect(actual.solverStatus.slice(0, 1), 'converged').toEqual([1]);
  expect(actual.solverStatus[3], 'invalid edges').toBe(0);
  const seen = new Set<string>();
  actual.edges.forEach(([first, second], slot) => {
    if (first === second) return;
    const key = getPairKey(first, second);
    seen.add(key);
    expect(hand.conductances[key], `slot ${slot} joins ${key}`).toBeDefined();
    expect(actual.conductances[slot]!, `conductance ${key}`).toBeCloseTo(
      hand.conductances[key]!,
      6
    );
  });
  expect([...seen].sort()).toEqual(Object.keys(hand.conductances).sort());
  const scale = hand.effectiveResistance;
  hand.potentials.forEach((expected, cell) => {
    expect(Math.abs(actual.potentials[cell]! - expected), `potential ${cell}`).toBeLessThanOrEqual(
      1e-4 * scale
    );
  });
  hand.vertexCurrents.forEach((expected, cell) => {
    expect(
      Math.abs(actual.vertexCurrents[cell]! - expected),
      `current ${cell}`
    ).toBeLessThanOrEqual(1e-4);
  });
  expect(Math.abs(actual.diagnostics[0]! - scale)).toBeLessThanOrEqual(1e-4 * scale);
});
