// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPURasterZonalStatisticsParameterValues,
  getRasterZonalCellArea,
  GPURasterZonalStatistics
} from '../../../src/raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('R1 preserves legacy areas with zero and nonzero padding', () => {
  const expected = getRasterZonalCellArea([1, 1, 60, 59], 0, 1, 1);
  for (const sentinel of [0, 13]) {
    expect(getRasterZonalCellArea([1, 1, 60, 59, sentinel, 17, 23, 29], 0, 1, 1)).toBe(expected);
  }
});

it('R2 preserves the area of a shifted thin polar window', () => {
  const grid: GPURasterMetadata = {
    width: 1,
    height: 3,
    affine: [1, 0, 0, 0, -1e-6, 90],
    pixelInterpretation: 'area'
  };
  const crop = {
    ...grid,
    height: 1,
    affine: [1, 0, 0, 0, -1e-6, 89.999998] as GPURasterMetadata['affine']
  };
  const expected = 0.0005394961;
  for (const [metadata, row] of [
    [grid, 2],
    [crop, 0]
  ] as const) {
    expect(
      getRasterZonalCellArea(
        getGPURasterZonalStatisticsParameterValues(metadata),
        row,
        metadata.height,
        2
      ) / expected
    ).toBeCloseTo(1, 6);
  }
});

it('preserves thin geographic cell area in either row direction and across windows', () => {
  for (const rowStep of [-0.000001, 0.000001]) {
    const grid: GPURasterMetadata = {
      width: 1,
      height: 1,
      affine: [0.000001, 0, 0, 0, rowStep, 60],
      pixelInterpretation: 'area'
    };
    // Independent midpoint identity for sin(top) - sin(bottom), stable for thin cells.
    const radians = Math.PI / 180;
    const expected =
      6371007.180918475 ** 2 *
      (0.000001 * radians) *
      2 *
      Math.cos((60 + rowStep / 2) * radians) *
      Math.sin((Math.abs(rowStep) * radians) / 2);
    const windowArea = getRasterZonalCellArea(
      getGPURasterZonalStatisticsParameterValues(grid),
      0,
      1,
      2
    );
    const fullArea = getRasterZonalCellArea(
      getGPURasterZonalStatisticsParameterValues({...grid, height: 256}),
      0,
      256,
      2
    );
    expect(Number.isFinite(windowArea)).toBe(true);
    expect(windowArea / expected).toBeCloseTo(1, 7);
    expect(fullArea).toBe(windowArea);
  }
});

it('RT-101 default props decode the default packer output with the signed row step', () => {
  // A one-row window at 60 degrees N, 1e-6 degrees square. Expected from /usr/bin/python3:
  // R^2 dLon 2 cos(midLatitude) sin(|dLat| / 2) with R = 6371007.180918475 m, dLon = dLat = 1e-6
  // degrees, midLatitude = 60 - 0.5e-6 degrees: 0.00618216988533045 m2. The legacy decoding gives
  // 0 because the float32 north and south edges round equal.
  const grid: GPURasterMetadata = {
    width: 1,
    height: 1,
    affine: [1e-6, 0, 10, 0, -1e-6, 60],
    pixelInterpretation: 'area'
  };
  const settings = getGPURasterZonalStatisticsParameterValues(grid);
  const area = getRasterZonalCellArea(settings, 0, 1);
  expect(Number.isFinite(area)).toBe(true);
  expect(area / 0.00618216988533045).toBeCloseTo(1, 6);

  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = {
    grid,
    zones: createTransientView(graph, 'zones', 'uint32', 1),
    values: {
      id: 'values',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: createTransientView(graph, 'values', 'float32', 1)}
    },
    zoneCapacity: 1,
    cellSizeMode: 'geographic' as const,
    parameters: createTransientView(graph, 'parameters', 'float32', settings.length),
    output: {weightSums: createTransientView(graph, 'weights', 'float32', 1)}
  };
  const sources = getGPUCommandNodeKernelSources(
    new GPURasterZonalStatistics(props).getCommandNodes(graph)
  );
  expect(
    sources.some(({source}) => source.includes('const ZONAL_PARAMETER_LAYOUT_VERSION: u32 = 2u;'))
  ).toBe(true);

  // Without an explicit version, a view that is not the seven-word packer output is rejected
  // instead of being decoded under a layout it may not have.
  for (const length of [4, 8]) {
    const shortGraph = new GPUCommandGraph(createNullWebGPUDevice());
    expect(
      () =>
        new GPURasterZonalStatistics({
          ...props,
          zones: createTransientView(shortGraph, 'zones', 'uint32', 1),
          values: {
            ...props.values,
            storage: {
              kind: 'buffer' as const,
              values: createTransientView(shortGraph, 'values', 'float32', 1)
            }
          },
          parameters: createTransientView(shortGraph, 'parameters', 'float32', length),
          output: {weightSums: createTransientView(shortGraph, 'weights', 'float32', 1)}
        })
    ).toThrow(/parameterLayoutVersion/);
    expect(() => getRasterZonalCellArea(new Array(length).fill(1), 0, 1)).toThrow(
      /parameterLayoutVersion/
    );
  }
});

it('selects the signed-step shader layout and rejects non-axis-aligned geographic grids', () => {
  for (const length of [4, 7]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const props = {
      grid: {
        width: 1,
        height: 1,
        affine: [1, 0, 0, 0, -1, 60],
        pixelInterpretation: 'area'
      } as GPURasterMetadata,
      zones: createTransientView(graph, 'zones', 'uint32', 1),
      values: {
        id: 'values',
        format: 'float32' as const,
        storage: {
          kind: 'buffer' as const,
          values: createTransientView(graph, 'values', 'float32', 1)
        }
      },
      zoneCapacity: 1,
      cellSizeMode: 'geographic' as const,
      parameterLayoutVersion: length === 7 ? (2 as const) : (1 as const),
      parameters: createTransientView(graph, 'parameters', 'float32', length),
      output: {weightSums: createTransientView(graph, 'weights', 'float32', 1)}
    };
    const sources = getGPUCommandNodeKernelSources(
      new GPURasterZonalStatistics(props).getCommandNodes(graph)
    );
    expect(
      sources.some(({source}) =>
        source.includes(`const ZONAL_PARAMETER_LAYOUT_VERSION: u32 = ${length === 7 ? 2 : 1}u;`)
      )
    ).toBe(true);
    const rotated = {...props.grid, affine: [1, 0.1, 0, 0, -1, 60] as GPURasterMetadata['affine']};
    expect(() => new GPURasterZonalStatistics({...props, grid: rotated})).toThrow(/axis-aligned/);
    expect(() => getGPURasterZonalStatisticsParameterValues(rotated)).toThrow(/axis-aligned/);
  }
});
