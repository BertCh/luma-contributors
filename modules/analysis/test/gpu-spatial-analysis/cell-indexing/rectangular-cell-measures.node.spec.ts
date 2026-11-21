// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPUCellMeasures,
  type GPUCellMeasuresOutput
} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-measures';
import {GPUCellMeasuresOperation} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-measures-operation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {computeCellMeasuresOnCPU} from '../../../src/gpu-spatial-analysis/cell-indexing/cpu-cell-measures-operation';
import {GPU_EARTH_AUTHALIC_RADIUS_METERS} from '../../../src/analysis-core/contracts/earth';
import {quadbinTileToCell} from '../cell-aggregation/cell-aggregation-oracle';
import {getAngularDistance, s2CellFromFaceIJ, s2CellToBoundary} from './cell-indexing-oracle';

function packKeys(keys: bigint[]): Uint32Array {
  return Uint32Array.from(keys.flatMap(key => [Number(key & 0xffffffffn), Number(key >> 32n)]));
}

it('S2 asymmetric cell edges follow consecutive geometry boundary slots', () => {
  const key = 0x0300000000000000n;
  const boundary = s2CellToBoundary(key);
  const expectedEdges = boundary.map(
    (first, edge) =>
      getAngularDistance(first, boundary[(edge + 1) % 4]) * GPU_EARTH_AUTHALIC_RADIUS_METERS
  );
  // All four sides differ, so a reversed traversal cannot pass as it does on root faces.
  expect(new Set(expectedEdges.map(Math.round)).size).toBe(4);
  for (const wordOrder of ['little-endian', 'high-low'] as const)
    for (const maximumEdgeCount of [4, 10]) {
      const cells = packKeys([key]);
      if (wordOrder === 'high-low') cells.reverse();
      const result = computeCellMeasuresOnCPU(cells, {family: 's2', wordOrder, maximumEdgeCount});
      expectedEdges.forEach((expected, edge) =>
        expect(Math.abs(result.edgeLengths[edge] / expected - 1), `edge ${edge}`).toBeLessThan(1e-7)
      );
      expect(Array.from(result.edgeLengths.slice(4))).toEqual(
        new Array(maximumEdgeCount - 4).fill(0)
      );
      expect(
        Math.abs(result.perimeters[0] / expectedEdges.reduce((sum, edge) => sum + edge, 0) - 1)
      ).toBeLessThan(1e-7);
    }
});

it('S2 face cells partition the sphere and have four equal cube-face edges', () => {
  const keys = Array.from({length: 6}, (_, face) => s2CellFromFaceIJ(face, 0, 0, 0));
  const result = computeCellMeasuresOnCPU(packKeys(keys), {family: 's2', maximumEdgeCount: 4});
  const radius = GPU_EARTH_AUTHALIC_RADIUS_METERS;
  keys.forEach((_, row) => {
    expect(Math.abs(result.areas[row] / ((4 * Math.PI * radius ** 2) / 6) - 1)).toBeLessThan(1e-7);
    for (let edge = 0; edge < 4; edge++)
      expect(
        Math.abs(result.edgeLengths[row * 4 + edge] / (Math.acos(1 / 3) * radius) - 1)
      ).toBeLessThan(1e-7);
    expect(result.edgeCounts[row]).toBe(4);
    expect(result.pentagons[row]).toBe(0);
  });
});

it('S2 leaf areas agree with an independent gnomonic Jacobian bound at levels 26–30', () => {
  const radius = GPU_EARTH_AUTHALIC_RADIUS_METERS;
  for (const level of [26, 27, 28, 29, 30]) {
    const size = 2 ** level;
    const firstCoordinate = Math.floor(size * 0.73);
    const secondCoordinate = Math.floor(size * 0.18);
    const key = s2CellFromFaceIJ(
      4,
      firstCoordinate * 2 ** (30 - level),
      secondCoordinate * 2 ** (30 - level),
      level
    );
    const result = computeCellMeasuresOnCPU(packKeys([key]), {family: 's2'});
    const transform = (surface: number) =>
      surface >= 0.5 ? (4 * surface ** 2 - 1) / 3 : (1 - 4 * (1 - surface) ** 2) / 3;
    const firstSurface = (firstCoordinate + 0.5) / size;
    const secondSurface = (secondCoordinate + 0.5) / size;
    const firstDelta = ((8 / 3) * Math.max(firstSurface, 1 - firstSurface)) / size;
    const secondDelta = ((8 / 3) * Math.max(secondSurface, 1 - secondSurface)) / size;
    // Midpoint density integrates this tiny rectangle to relative O(2^-level).
    const expectedArea =
      (radius ** 2 * firstDelta * secondDelta) /
      (1 + transform(firstSurface) ** 2 + transform(secondSurface) ** 2) ** 1.5;
    expect(Math.abs(result.areas[0] / expectedArea - 1), `level ${level}`).toBeLessThan(6e-7);
  }
});

it('Quadbin measures the full Mercator band and preserves north/east/south/west edge order', () => {
  const radius = GPU_EARTH_AUTHALIC_RADIUS_METERS;
  const root = computeCellMeasuresOnCPU(packKeys([quadbinTileToCell(0, 0, 0)]), {
    family: 'quadbin',
    maximumEdgeCount: 4
  });
  expect(
    Math.abs(root.areas[0] / (4 * Math.PI * radius ** 2 * Math.tanh(Math.PI)) - 1)
  ).toBeLessThan(1e-7);
  const result = computeCellMeasuresOnCPU(packKeys([quadbinTileToCell(1, 0, 2)]), {
    family: 'quadbin',
    maximumEdgeCount: 6
  });
  const north = Math.atan(Math.sinh(Math.PI));
  const south = Math.atan(Math.sinh(Math.PI / 2));
  const expected = [
    (Math.PI / 2) * Math.cos(north),
    north - south,
    (Math.PI / 2) * Math.cos(south),
    north - south
  ].map(value => value * radius);
  expected.forEach((edge, index) =>
    expect(Math.abs(result.edgeLengths[index] / edge - 1)).toBeLessThan(1e-7)
  );
  expect(Array.from(result.edgeLengths.slice(4))).toEqual([0, 0]);
  expect(result.edgeCounts[0]).toBe(4);
  expect(result.pentagons[0]).toBe(0);
});

it.each([
  's2',
  'quadbin'
] as const)('%s preserves invalid zeros, empty input, padding, units and word order', family => {
  const key = family === 's2' ? s2CellFromFaceIJ(1, 0, 0, 2) : quadbinTileToCell(3, 2, 2);
  const keys = [key, 0n, 2n, 0xffffffffffffffffn];
  const meters = computeCellMeasuresOnCPU(packKeys(keys), {family, maximumEdgeCount: 4});
  const highLow = packKeys(keys).map((_, index, words) => words[index ^ 1]);
  const kilometers = computeCellMeasuresOnCPU(highLow, {
    family,
    wordOrder: 'high-low',
    units: 'kilometers',
    maximumEdgeCount: 10
  });
  expect(Math.abs(meters.areas[0] / kilometers.areas[0] / 1e6 - 1)).toBeLessThan(2e-7);
  expect(Math.abs(meters.perimeters[0] / kilometers.perimeters[0] / 1000 - 1)).toBeLessThan(2e-7);
  expect(Array.from(kilometers.edgeLengths.slice(4))).toEqual(new Array(36).fill(0));
  for (const values of [meters.areas, meters.perimeters, meters.edgeCounts, meters.pentagons])
    expect(Array.from(values.slice(1))).toEqual([0, 0, 0]);
  expect(computeCellMeasuresOnCPU(new Uint32Array(), {family}).areas.length).toBe(0);
});

it.each([
  's2',
  'quadbin'
] as const)('%s contributor validates its stride and assembles every optional output', family => {
  const device = createNullWebGPUDevice();
  try {
    let serial = 0;
    const graph = new GPUCommandGraph(device);
    const view = <Format extends 'float32' | 'uint32' | 'uint32x2'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `measure-${serial++}`, format, length);
    const cells = view('uint32x2', 3);
    const outputs = {
      areas: view('float32', 3),
      perimeters: view('float32', 3),
      edgeLengths: view('float32', 30),
      edgeCounts: view('uint32', 3),
      pentagons: view('uint32', 3)
    };
    const outputSelections: GPUCellMeasuresOutput[] = [
      outputs,
      {areas: outputs.areas},
      {perimeters: outputs.perimeters},
      {edgeLengths: outputs.edgeLengths},
      {edgeCounts: outputs.edgeCounts},
      {pentagons: outputs.pentagons}
    ];
    for (const [outputIndex, output] of outputSelections.entries()) {
      // Each declaration now owns its chunk-parameter resource in this shared graph.
      const contributor = new GPUCellMeasures({
        id: `cell-measures-${family}-${outputIndex}`,
        family,
        cells,
        maximumEdgeCount: 10,
        output
      });
      expect(contributor.maximumEdgeCount).toBe(10);
      expect(contributor.measures.lengthModel).toBe(
        family === 's2' ? 'great-circle' : 'sphere-parallel-meridian'
      );
      expect(contributor.measures.areaModel).toBe(
        family === 's2' ? 'sphere-polygon' : 'sphere-rectangle'
      );
      const nodes = contributor.getCommandNodes(graph);
      for (const node of nodes) graph.add(node);
      const kernels = getGPUCommandNodeKernelSources(nodes);
      expect(kernels).toHaveLength(1);
      for (const kernel of kernels) {
        expect(() => new WgslReflect(kernel.source)).not.toThrow();
        expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
      }
    }
    expect(
      new GPUCellMeasures({family, cells, output: {areas: outputs.areas}}).maximumEdgeCount
    ).toBe(4);
    for (const maximumEdgeCount of [0, 3, 4.5, 11, Infinity])
      expect(
        () => new GPUCellMeasures({family, cells, maximumEdgeCount, output: {areas: outputs.areas}})
      ).toThrow(/cell-measures maximumEdgeCount/);
    expect(
      () => new GPUCellMeasures({family, cells, output: {edgeLengths: view('float32', 11)}})
    ).toThrow(/cell-measures output.edgeLengths/);
    const operation = new GPUCellMeasuresOperation({
      family,
      rowCount: 3,
      cells: 'cells',
      output: {edgeLengths: 'edges'}
    });
    expect(operation.outputs['edgeLengths'].length).toBe(12);
    expect(() =>
      constructNetworkOperationContributor(operation, {cells: {format: 'uint32x2', length: 3}})
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('Quadbin zoom-26 measures agree with the local Mercator Jacobian at both poles and the equator', () => {
  const resolution = 26;
  const size = 2 ** resolution;
  const radius = GPU_EARTH_AUTHALIC_RADIUS_METERS;
  for (const row of [0, 1234567, size / 2 - 1, size / 2, size - 1]) {
    const result = computeCellMeasuresOnCPU(packKeys([quadbinTileToCell(123, row, resolution)]), {
      family: 'quadbin'
    });
    const step = (2 * Math.PI) / size;
    const latitude = Math.atan(Math.sinh(Math.PI * (1 - (2 * (row + 0.5)) / size)));
    // The midpoint Jacobian gives area and edge lengths with O(step) relative error here.
    const expectedLength = radius * step * Math.cos(latitude);
    const expectedArea = expectedLength ** 2;
    expect(Math.abs(result.areas[0] / expectedArea - 1)).toBeLessThan(2e-7);
    for (const edge of result.edgeLengths)
      expect(Math.abs(edge / expectedLength - 1)).toBeLessThan(2e-7);
  }
});
