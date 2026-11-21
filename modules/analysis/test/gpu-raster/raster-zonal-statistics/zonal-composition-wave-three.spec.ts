// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  GPURasterZonalStatistics,
  getGPURasterZonalStatisticsParameterValues
} from '../../../src/raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics';
import {GPUPolygonRasterization} from '../../../src/raster-analysis/polygon-rasterization/gpu-polygon-rasterization';
import {getGPUPolygonRasterizationParameterValues} from '../../../src/raster-analysis/polygon-rasterization/polygon-rasterization-parameters';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';

it('CMP-09 distinguishes unfinished cost values and polygon loss from complete unreachable values', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [polygonLoss, costLoss, independentLoss, costs] of [
    [0, 0, 0, [Infinity, Infinity]],
    [0, 1, 0, [NaN, NaN]],
    [1, 0, 0, [0, 1]],
    [0, 0, 1, [0, 1]],
    [0, 0, 0, [0, 1]]
  ] as const) {
    const rig = createParityGraph(device);
    const mean = rig.output('mean', 1, 'float32');
    const counts = rig.output('counts', 1, 'uint32');
    const incomplete = rig.output('incomplete', 1, 'uint32');
    rig.graph.add(
      new GPURasterZonalStatistics({
        grid: {width: 2, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
        zones: rig.input('zones', [0, 0], 'uint32').view,
        zoneCapacity: 1,
        values: {
          id: 'costs',
          format: 'float32',
          storage: {kind: 'buffer', values: rig.input('costs', [...costs], 'float32').view}
        },
        producerOverflow: rig.input('polygon-overflow', [polygonLoss], 'uint32').view,
        incompleteSources: [
          rig.input('iteration-limit', [costLoss], 'uint32').view,
          rig.input('independent-loss', [independentLoss], 'uint32').view
        ],
        output: {means: mean.view, valueCounts: counts.view, status: {incomplete: incomplete.view}}
      })
    );
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const lost = Boolean(polygonLoss || costLoss || independentLoss);
      expect(await readUint32(incomplete.buffer, 1)).toEqual([Number(lost)]);
      expect(await readUint32(counts.buffer, 1)).toEqual([
        lost ? 0xffffffff : Number.isFinite(costs[0]) ? 2 : 0
      ]);
      const [value] = await readFloat32(mean.buffer, 1);
      if (lost || !Number.isFinite(costs[0])) expect(Number.isNaN(value)).toBe(true);
      else {
        expect(Number.isFinite(value)).toBe(true);
        expect(value).toBe(0.5);
      }
    } finally {
      compiled.destroy();
      rig.destroy();
    }
  }
});

it('CMP-10 rasterized polygon row zero remains a real zonal result', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = createParityGraph(device);
  const grid = {
    width: 2,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const zones = rig.output('zones', 2, 'uint32');
  const mean = rig.output('mean', 1, 'float32');
  const counts = rig.output('counts', 1, 'uint32');
  const overflow = rig.output('overflow', 1, 'uint32');
  rig.graph.add(
    new GPUPolygonRasterization({
      grid,
      crossingCapacity: 8,
      parameters: rig.input(
        'parameters',
        Array.from(getGPUPolygonRasterizationParameterValues(grid)),
        'float32'
      ).view,
      positions: rig.input('positions', [0, 0, 1, 0, 1, 1, 0, 1], 'float32x2').view,
      featureOffsets: rig.input('features', [0, 1], 'uint32').view,
      polygonOffsets: rig.input('polygons', [0, 1], 'uint32').view,
      ringOffsets: rig.input('rings', [0, 4], 'uint32').view,
      output: {
        zones: zones.view,
        status: {overflow: rig.output('polygon-overflow', 1, 'uint32').view}
      }
    })
  );
  rig.graph.add(
    new GPURasterZonalStatistics({
      grid,
      zones: zones.view,
      zoneCapacity: 1,
      values: {
        id: 'values',
        format: 'float32',
        storage: {kind: 'buffer', values: rig.input('values', [7, 42], 'float32').view}
      },
      output: {means: mean.view, valueCounts: counts.view, status: {overflow: overflow.view}}
    })
  );
  const compiled = rig.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(zones.buffer, 2)).toEqual([0, 0xffffffff]);
    const [value] = await readFloat32(mean.buffer, 1);
    expect(Number.isFinite(value)).toBe(true);
    expect(value).toBe(7);
    expect(await readUint32(counts.buffer, 1)).toEqual([1]);
    expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});

for (const sumOrder of ['sorted', 'atomic'] as const) {
  for (const cellSizeMode of ['uniform', 'geographic'] as const) {
    it(`URT-03 population and sample variance with unequal latitude rows (${sumOrder}, ${cellSizeMode})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = createParityGraph(device);
      const grid = {
        width: 1,
        height: 2,
        affine: [1, 0, 0, 0, -30, 60] as const,
        pixelInterpretation: 'area' as const
      };
      const sample = rig.output('sample', 1, 'float32');
      const population = rig.output('population', 1, 'float32');
      rig.graph.add(
        new GPURasterZonalStatistics({
          grid,
          sumOrder,
          cellSizeMode,
          zoneCapacity: 1,
          zones: rig.input('zones', [0, 0], 'uint32').view,
          parameters:
            cellSizeMode === 'geographic'
              ? rig.input(
                  'parameters',
                  Array.from(getGPURasterZonalStatisticsParameterValues(grid)),
                  'float32'
                ).view
              : undefined,
          values: {
            id: 'values',
            format: 'float32',
            storage: {kind: 'buffer', values: rig.input('values', [1, 3], 'float32').view}
          },
          output: {variances: sample.view, populationVariances: population.view}
        })
      );
      const compiled = rig.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        // NumPy reference: areas R²*pi/180*(sin(60)-sin(30)), R²*pi/180*sin(30),
        // np.cov([1,3], aweights=areas, ddof=0/1) = .9760677434251697 / 2.
        for (const [output, expected] of [
          [sample, 2],
          [population, cellSizeMode === 'uniform' ? 1 : 0.9760677434251697]
        ] as const) {
          const [value] = await readFloat32(output.buffer, 1);
          expect(Number.isFinite(value)).toBe(true);
          expect(value).toBeCloseTo(expected, 5);
        }
      } finally {
        compiled.destroy();
        rig.destroy();
      }
    });
  }
}

it('B3 inherited calibration and nodata produce a calibrated zonal mean', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = createParityGraph(device);
  const mean = rig.output('mean', 1, 'float32');
  const counts = rig.output('counts', 1, 'uint32');
  const values = Object.assign(
    Object.create({
      get scale() {
        return 3;
      },
      get noDataValue() {
        return -9999;
      }
    }),
    {
      id: 'values',
      format: 'float32' as const,
      storage: {
        kind: 'buffer' as const,
        values: rig.input('raw-values', [-9999, 2], 'float32').view
      }
    }
  );
  const output = Object.create({
    get means() {
      return mean.view;
    },
    get valueCounts() {
      return counts.view;
    }
  });
  rig.graph.add(
    new GPURasterZonalStatistics({
      grid: {width: 2, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      zones: rig.input('zones', [0, 0], 'uint32').view,
      zoneCapacity: 1,
      values,
      output
    })
  );
  const compiled = rig.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const [value] = await readFloat32(mean.buffer, 1);
    expect(Number.isFinite(value)).toBe(true);
    expect(value).toBe(6); // Discard raw -9999; calibrate the remaining 2 * 3.
    expect(await readUint32(counts.buffer, 1)).toEqual([1]);
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});
