// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPURasterFractionalCoverage,
  getGPURasterFractionalCoverageParameterValues
} from '../../../src/raster-analysis/gpu-raster-fractional-coverage';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('R8 bounds translated decimal slivers and reports their tuple capacity on both routes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const searchStructure of ['bounds', 'brute-force'] as const) {
    for (const capacity of [2, 1]) {
      const rig = createParityGraph(device);
      const grid = {
        width: 133,
        height: 1,
        affine: [0.3, 0, 0, 0, 1, 0] as const,
        pixelInterpretation: 'area' as const
      };
      const output = {
        zones: rig.output('zones', capacity, 'uint32'),
        cells: rig.output('cells', capacity, 'uint32'),
        fractions: rig.output('fractions', capacity, 'float32'),
        count: rig.output('count', 1, 'uint32'),
        required: rig.output('required', 1, 'uint32'),
        overflow: rig.output('overflow', 1, 'uint32'),
        valid: rig.output('valid', 1, 'uint32')
      };
      rig.graph.add(
        new GPURasterFractionalCoverage({
          grid,
          searchStructure,
          candidateCapacity: 8,
          parameters: rig.input(
            'parameters',
            Array.from(getGPURasterFractionalCoverageParameterValues(grid)),
            'float32'
          ).view,
          zones: {
            positions: rig.input('positions', [39.3, 0, 39.6, 0, 39.6, 1, 39.3, 1], 'float32x2')
              .view,
            featureOffsets: rig.input('features', [0, 1], 'uint32').view,
            polygonOffsets: rig.input('polygons', [0, 1], 'uint32').view,
            ringOffsets: rig.input('rings', [0, 4], 'uint32').view
          },
          output: {
            zoneIds: output.zones.view,
            cellIds: output.cells.view,
            fractions: output.fractions.view,
            zoneValidity: output.valid.view,
            status: {
              count: output.count.view,
              requiredCount: output.required.view,
              overflow: output.overflow.view
            }
          }
        })
      );
      const compiled = rig.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const [count] = await readUint32(output.count.buffer, 1);
        const [required] = await readUint32(output.required.buffer, 1);
        console.info(
          `rt-zonal R8 ${searchStructure}, capacity ${capacity}: requiredCount=${required}`
        );
        expect(required).toBeGreaterThanOrEqual(1);
        expect(required).toBeLessThanOrEqual(2);
        expect(count).toBe(Math.min(capacity, required));
        expect(await readUint32(output.overflow.buffer, 1)).toEqual([Number(required > capacity)]);
        expect(await readUint32(output.valid.buffer, 1)).toEqual([1]);
        const cells = await readUint32(output.cells.buffer, count);
        const fractions = await readFloat32(output.fractions.buffer, count);
        if (capacity === 2) {
          expect(cells).toContain(131);
          let total = 0;
          for (let tuple = 0; tuple < count; tuple++) {
            expect([130, 131]).toContain(cells[tuple]);
            expect(Number.isFinite(fractions[tuple])).toBe(true);
            expect(fractions[tuple]).toBeGreaterThan(0);
            if (cells[tuple] === 130) expect(fractions[tuple]).toBeLessThan(3e-5);
            else expect(fractions[tuple]).toBeGreaterThan(0.99997);
            total += fractions[tuple];
          }
          expect(Math.abs(total - 1)).toBeLessThan(3e-5);
        }
      } finally {
        compiled.destroy();
        rig.destroy();
      }
    }
  }
});

it('fractional coverage preserves subcell areas far from the raster origin', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = createParityGraph(device);
  const width = 512;
  const capacity = 1024;
  // Binary-exact fractions isolate clipping arithmetic from source-coordinate quantization.
  // The same rectangle near the origin and near row/column 460 must have the same support.
  const rectangles = [
    [4.203125, 4.296875, 24.453125, 24.796875],
    [460.203125, 460.296875, 480.453125, 480.796875]
  ];
  const positions = rectangles.flatMap(([left, top, right, bottom]) => [
    left,
    top,
    right,
    top,
    right,
    bottom,
    left,
    bottom
  ]);
  const grid = {
    width,
    height: width,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const output = {
    zoneIds: rig.output('zone-ids', capacity, 'uint32'),
    cellIds: rig.output('cell-ids', capacity, 'uint32'),
    fractions: rig.output('fractions', capacity, 'float32'),
    count: rig.output('count', 1, 'uint32'),
    requiredCount: rig.output('required-count', 1, 'uint32'),
    overflow: rig.output('overflow', 1, 'uint32'),
    zoneValidity: rig.output('zone-validity', 2, 'uint32')
  };
  rig.graph.add(
    new GPURasterFractionalCoverage({
      grid,
      zones: {
        positions: rig.input('positions', positions, 'float32x2').view,
        featureOffsets: rig.input('feature-offsets', [0, 1, 2], 'uint32').view,
        polygonOffsets: rig.input('polygon-offsets', [0, 1, 2], 'uint32').view,
        ringOffsets: rig.input('ring-offsets', [0, 4, 8], 'uint32').view
      },
      parameters: rig.input(
        'parameters',
        Array.from(getGPURasterFractionalCoverageParameterValues(grid)),
        'float32'
      ).view,
      output: {
        zoneIds: output.zoneIds.view,
        cellIds: output.cellIds.view,
        fractions: output.fractions.view,
        status: {
          count: output.count.view,
          requiredCount: output.requiredCount.view,
          overflow: output.overflow.view
        },
        zoneValidity: output.zoneValidity.view
      }
    })
  );
  const compiled = await rig.graph.compileAsync();
  try {
    submitGraph(device, compiled, undefined);
    const [count] = await readUint32(output.count.buffer, 1);
    expect(await readUint32(output.overflow.buffer, 1)).toEqual([0]);
    expect(await readUint32(output.zoneValidity.buffer, 2)).toEqual([1, 1]);
    const zoneIds = await readUint32(output.zoneIds.buffer, count);
    const cellIds = await readUint32(output.cellIds.buffer, count);
    const fractions = await readFloat32(output.fractions.buffer, count);
    const areas = [0, 0];
    const meaningfulCounts = [0, 0];
    const distinctCells = [new Set<number>(), new Set<number>()];
    // 2e-5 cell area is well below the smallest analytic boundary overlap (~0.06).
    // Raw positive tuple counts may include representational slivers and are not an oracle.
    const fractionTolerance = 2e-5;
    for (let tuple = 0; tuple < count; tuple++) {
      const zone = zoneIds[tuple];
      expect(zone).toBeLessThan(2);
      const cell = cellIds[tuple];
      expect(distinctCells[zone].has(cell)).toBe(false);
      distinctCells[zone].add(cell);
      const column = cell % width;
      const row = Math.floor(cell / width);
      const [left, top, right, bottom] = rectangles[zone];
      const expected =
        Math.max(0, Math.min(column + 1, right) - Math.max(column, left)) *
        Math.max(0, Math.min(row + 1, bottom) - Math.max(row, top));
      expect(
        Math.abs(fractions[tuple] - expected),
        `zone ${zone}, cell ${cell}`
      ).toBeLessThanOrEqual(fractionTolerance);
      areas[zone] += fractions[tuple];
      if (fractions[tuple] > fractionTolerance) meaningfulCounts[zone]++;
    }
    expect(meaningfulCounts).toEqual([441, 441]);
    for (const area of areas) expect(Math.abs(area - 20.25 * 20.5)).toBeLessThan(0.002);
    expect(Math.abs(areas[0] - areas[1])).toBeLessThan(0.002);
  } finally {
    compiled.destroy();
    rig.destroy();
  }
}, 120000);

it('both coverage routes retain translated quarter cells and decimal cell boundaries', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const searchStructure of ['bounds', 'brute-force'] as const) {
    for (const scenario of [
      {column: 0, scale: 1, decimal: false, point: false},
      {column: 2 ** 20, scale: 1, decimal: false, point: false},
      {column: 2 ** 20, scale: 2, decimal: false, point: false},
      {column: 3, scale: 0.3, decimal: true, point: false},
      {column: 2, scale: 0.3, decimal: false, point: true}
    ]) {
      const {column, scale, decimal, point} = scenario;
      const rig = createParityGraph(device);
      const grid = {
        width: column + 1,
        height: decimal ? 2 : 1,
        affine: [scale, 0, 0, 0, decimal ? 0.7 : 1, 0] as const,
        pixelInterpretation: point ? ('point' as const) : ('area' as const)
      };
      const output = {
        zoneIds: rig.output('zones', 1, 'uint32'),
        cellIds: rig.output('cells', 1, 'uint32'),
        fractions: rig.output('fractions', 1, 'float32'),
        count: rig.output('count', 1, 'uint32'),
        requiredCount: rig.output('required', 1, 'uint32'),
        overflow: rig.output('overflow', 1, 'uint32'),
        candidateOverflow: rig.output('candidate-overflow', 1, 'uint32'),
        zoneValidity: rig.output('validity', 1, 'uint32')
      };
      rig.graph.add(
        new GPURasterFractionalCoverage({
          grid,
          searchStructure,
          candidateCapacity: 8,
          zones: {
            positions: rig.input(
              'positions',
              point
                ? [0.45, -0.5, 0.75, -0.5, 0.75, 0.5, 0.45, 0.5]
                : decimal
                  ? [0.9, 0.7, 1.2, 0.7, 1.2, 1.4, 0.9, 1.4]
                  : [
                      scale * (column + 0.125),
                      0,
                      scale * (column + 0.375),
                      0,
                      scale * (column + 0.375),
                      1,
                      scale * (column + 0.125),
                      1
                    ],
              'float32x2'
            ).view,
            featureOffsets: rig.input('features', [0, 1], 'uint32').view,
            polygonOffsets: rig.input('polygons', [0, 1], 'uint32').view,
            ringOffsets: rig.input('rings', [0, 4], 'uint32').view
          },
          parameters: rig.input(
            'parameters',
            Array.from(getGPURasterFractionalCoverageParameterValues(grid)),
            'float32'
          ).view,
          output: {
            zoneIds: output.zoneIds.view,
            cellIds: output.cellIds.view,
            fractions: output.fractions.view,
            zoneValidity: output.zoneValidity.view,
            status: {
              count: output.count.view,
              requiredCount: output.requiredCount.view,
              overflow: output.overflow.view,
              candidateOverflow: output.candidateOverflow.view
            }
          }
        })
      );
      const compiled = await rig.graph.compileAsync();
      try {
        submitGraph(device, compiled, undefined);
        expect(await readUint32(output.count.buffer, 1)).toEqual([1]);
        expect(await readUint32(output.requiredCount.buffer, 1)).toEqual([1]);
        expect(await readUint32(output.overflow.buffer, 1)).toEqual([0]);
        expect(await readUint32(output.candidateOverflow.buffer, 1)).toEqual([0]);
        expect(await readUint32(output.zoneValidity.buffer, 1)).toEqual([1]);
        expect(await readUint32(output.cellIds.buffer, 1)).toEqual([decimal ? 7 : column]);
        expect(await readUint32(output.zoneIds.buffer, 1)).toEqual([0]);
        expect(await readFloat32(output.fractions.buffer, 1)).toEqual([
          decimal || point ? 1 : 0.25
        ]);
      } finally {
        compiled.destroy();
        rig.destroy();
      }
    }
  }
}, 120000);
