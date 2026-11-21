// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// CORE-102: the shared point-neighbor grid writer must file a point in the cell the 3x3 reader
// walks. These cases are float32 transcriptions (Math.fround at every f32 operation) of the WGSL
// of `readLattice`, `getCellColumn` (spatial clustering) and `getGridUnitPosition`, and of the
// published `getCoordinate` on the unit domain the index takes. The source checks below keep the
// transcription tied to the WGSL text. The headless twin is
// test/gpu-spatial-analysis/spatial-clustering/spatial-clustering-projected-origin.spec.ts.

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {POINT_NEIGHBOR_GRID_UNIT_POSITION_WGSL} from '../../src/analysis-core/kernels/point-neighbor-grid-index';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  getGPUDBSCANParameterValues,
  GPUDBSCAN
} from '../../src/gpu-spatial-analysis/spatial-clustering/index';
import {getSpatialClusteringSharedWGSL} from '../../src/gpu-spatial-analysis/spatial-clustering/spatial-clustering-kernels';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

const toFloat32 = Math.fround;
const CELL_MARGIN = 1.0009765625;

type Lattice = {
  minimumX: number;
  maximumX: number;
  cellWidth: number;
  columns: number;
  gridColumns: number;
};

/** `readLattice` of spatial clustering, planar, one axis. */
function readLattice(minimumX: number, maximumX: number, epsilon: number, gridColumns: number) {
  const width = toFloat32(maximumX - minimumX);
  const cellStep = toFloat32(epsilon * CELL_MARGIN);
  const cellWidth = Math.max(toFloat32(width / gridColumns), cellStep);
  const columns = Math.min(gridColumns - 1, Math.floor(toFloat32(width / cellWidth))) + 1;
  return {minimumX, maximumX, cellWidth, columns, gridColumns};
}

/** Reader: `getCellColumn` of spatial clustering. */
function getCellColumn(lattice: Lattice, x: number): number {
  return Math.min(
    Math.floor(toFloat32(toFloat32(x - lattice.minimumX) / lattice.cellWidth)),
    lattice.columns - 1
  );
}

/** Published `getCoordinate(unit, 0.0, 1.0, size)`. */
function getUnitCoordinate(unit: number, size: number): number {
  if (unit <= 0) return 0;
  if (unit >= 1) return size - 1;
  return Math.min(Math.floor(toFloat32(unit * size)), size - 1);
}

/** The writer before CORE-102: the point rescaled by an extended lattice width. */
function getPreviousUnitPosition(lattice: Lattice, x: number): number {
  const extended = toFloat32(lattice.minimumX + toFloat32(lattice.cellWidth * lattice.gridColumns));
  const width = toFloat32(Math.max(lattice.maximumX, extended) - lattice.minimumX);
  return toFloat32(toFloat32(x - lattice.minimumX) / width);
}

const scratch = new Float32Array(1);
const scratchBits = new Uint32Array(scratch.buffer);
function moveFloat(value: number, units: number): number {
  scratch[0] = value;
  scratchBits[0] += units;
  return scratch[0];
}

/** `getGridCellUnitCoordinate`, the writer after CORE-102. */
function getGridCellUnitCoordinate(cell: number, size: number): number {
  let unit = toFloat32(toFloat32(cell + 0.5) / size);
  for (let attempt = 0; attempt < 4; attempt++) {
    const mapped = Math.floor(toFloat32(unit * size));
    if (mapped < cell) {
      unit = moveFloat(unit, 1);
    } else if (mapped > cell) {
      unit = moveFloat(unit, -1);
    }
  }
  return unit;
}

function getWriterCell(lattice: Lattice, x: number): number {
  return getUnitCoordinate(
    getGridCellUnitCoordinate(getCellColumn(lattice, x), lattice.gridColumns),
    lattice.gridColumns
  );
}

it('CORE-102 transcription matches the WGSL of the writer and the spatial-clustering reader', () => {
  const writer = POINT_NEIGHBOR_GRID_UNIT_POSITION_WGSL.replace(/\s+/g, ' ');
  expect(writer).toContain('var unit = (f32(cell) + 0.5) / f32(size);');
  expect(writer).toContain('let mapped = u32(floor(unit * f32(size)));');
  expect(writer).toContain('getGridCellUnitCoordinate(getCellColumn(lattice, x), COLUMNS)');
  expect(writer).toContain('getGridCellUnitCoordinate(getCellRow(lattice, y), ROWS)');
  // No second cell definition: the writer reads no lattice extent of its own.
  expect(writer).not.toMatch(/cellWidth|minimumX|maximumX/);
  const reader = getSpatialClusteringSharedWGSL(8, 1).replace(/\s+/g, ' ');
  expect(reader).toContain(
    'return min(u32(floor((x - lattice.minimumX) / lattice.cellWidth)), lattice.columns - 1u);'
  );
  expect(reader).toContain('lattice.cellWidth = max(width / f32(COLUMNS), cellStepX);');
  expect(reader).toContain(
    'lattice.columns = min(COLUMNS - 1u, u32(floor(width / lattice.cellWidth))) + 1u;'
  );
});

it('CORE-102 two points one epsilon apart at a projected origin share the reader window', () => {
  // Bounds [1e6, 1e6 + 28], epsilon 4, 8 columns: cellWidth = 4 * (1 + 2^-10) = 4.00390625 and
  // 7 active columns. Exact cells: floor(4 / 4.00390625) = 0, floor(8 / 4.00390625) = 1, so the
  // 3x3 window of either point holds the other (distance exactly epsilon).
  const lattice = readLattice(1000000, 1000028, 4, 8);
  expect(lattice.cellWidth).toBe(4.00390625);
  expect(lattice.columns).toBe(7);
  const points = [1000004, 1000008];
  expect(points.map(x => getCellColumn(lattice, x))).toEqual([0, 1]);
  // Before the fix the writer rounded 1e6 + 32.03125 to 1e6 + 32 and filed the points in
  // columns [1, 2]: the window of the first point (columns 0-1) missed the second.
  expect(points.map(x => getUnitCoordinate(getPreviousUnitPosition(lattice, x), 8))).toEqual([
    1, 2
  ]);
  expect(points.map(x => getWriterCell(lattice, x))).toEqual([0, 1]);
});

it('CORE-102 a point on the maximum bound is filed inside the active columns', () => {
  // Bounds [8490376, 8490393], epsilon f32(1.0175639...), 20 columns: 17 active columns, the
  // last is 16. The previous writer filed the maximum-bound point in column 17, which no window
  // reads, so the point did not even find itself.
  const epsilon = toFloat32(1.0175639390945435);
  const lattice = readLattice(8490376, 8490393, epsilon, 20);
  expect(lattice.columns).toBe(17);
  expect(getUnitCoordinate(getPreviousUnitPosition(lattice, 8490393), 20)).toBe(17);
  expect(getCellColumn(lattice, 8490393)).toBe(16);
  expect(getWriterCell(lattice, 8490393)).toBe(16);
  expect(getWriterCell(lattice, 8490392)).toBe(getCellColumn(lattice, 8490392));
});

it('CORE-102 the writer files every sampled point in its reader cell', () => {
  // Deterministic sample over origins 1e0..1e8 (both signs), epsilons 1..1e4 and up to 64
  // columns, with points on, and one float either side of, the reader cell edges.
  let seed = 0x2545f491;
  const random = () => {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return (seed >>> 0) / 4294967296;
  };
  let checked = 0;
  for (let trial = 0; trial < 20000; trial++) {
    const origin = toFloat32((random() < 0.5 ? -1 : 1) * 10 ** (8 * random()));
    const epsilon = toFloat32(10 ** (4 * random()));
    const gridColumns = 1 + Math.floor(random() * 64);
    const maximumX = toFloat32(origin + epsilon * (0.3 + random() * gridColumns * 1.3));
    if (!(maximumX > origin)) continue;
    const lattice = readLattice(origin, maximumX, epsilon, gridColumns);
    const edge = toFloat32(origin + Math.floor(random() * lattice.columns) * lattice.cellWidth);
    for (const x of [edge, moveFloat(edge, 1), moveFloat(edge, -1), maximumX, origin]) {
      if (x < origin || x > maximumX) continue;
      const cell = getCellColumn(lattice, x);
      expect(cell).toBeGreaterThanOrEqual(0);
      expect(getWriterCell(lattice, x)).toBe(cell);
      checked++;
    }
  }
  expect(checked).toBeGreaterThan(50000);
});

it('CORE-102 the interpreted spatial-clustering grid-positions kernel writes the reader cells', () => {
  // Runs the generated WGSL of the writer (wgsl_reflect interpreter), then the published
  // coordinate mapping. The unit written is the float32 cell centre (cell + 0.5) / columns.
  for (const testCase of [
    {
      positions: [1000004, 0.5, 1000008, 0.5],
      bounds: [1000000, 0, 1000028, 1] as const,
      epsilon: 4,
      gridSize: [8, 1] as const,
      cells: [0, 1]
    },
    {
      positions: [8490392, 0.5, 8490393, 0.5],
      bounds: [8490376, 0, 8490393, 1] as const,
      epsilon: toFloat32(1.0175639390945435),
      gridSize: [20, 1] as const,
      cells: [15, 16]
    }
  ]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    recordWGSLKernelNodes(graph);
    const positions = createTransientView(graph, 'positions', 'float32x2', 2);
    const parameters = createTransientView(graph, 'parameters', 'float32', 8);
    const nodes = new GPUDBSCAN({
      positions,
      parameters,
      gridSize: testCase.gridSize,
      output: {labels: createTransientView(graph, 'labels', 'uint32', 2)}
    }).getCommandNodes(graph);
    const node = nodes.find(candidate => candidate.id === 'spatial-clustering-grid-positions');
    expect(node).toBeDefined();
    const memory = createCPUGraphMemory();
    memory.write(positions, Float32Array.from(testCase.positions));
    memory.write(
      parameters,
      getGPUDBSCANParameterValues({
        bounds: testCase.bounds,
        epsilon: testCase.epsilon,
        minimumSamples: 2
      })
    );
    executeCommandNodesOnCPU([node!], memory);
    const gridPositions = getWGSLKernelNodeRecord(
      graph,
      'spatial-clustering-grid-positions'
    )!.bindings.find(binding => binding.name === 'gridPositions')!.view;
    const units = Array.from(memory.readFloat32(gridPositions, 4));
    expect(units.every(Number.isFinite)).toBe(true);
    const columns = testCase.gridSize[0];
    expect([units[0], units[2]].map(unit => getUnitCoordinate(unit, columns))).toEqual(
      testCase.cells
    );
    expect([units[0], units[2]]).toEqual(
      testCase.cells.map(cell => toFloat32((cell + 0.5) / columns))
    );
    expect([units[1], units[3]]).toEqual([0.5, 0.5]);
  }
});

it('CORE-102 the cell encoding is exact on axes up to 2^23 cells', () => {
  for (const size of [1, 2, 3, 7, 1000, 65535, 1 << 20, 3000017, (1 << 23) - 1, 1 << 23]) {
    for (const cell of [0, 1, Math.floor(size / 3), Math.floor(size / 2), size - 2, size - 1]) {
      if (cell < 0 || cell >= size) continue;
      const unit = getGridCellUnitCoordinate(cell, size);
      expect(unit).toBeGreaterThan(0);
      expect(unit).toBeLessThan(1);
      expect(getUnitCoordinate(unit, size)).toBe(cell);
    }
  }
});
