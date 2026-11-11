// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPURasterFractionalCoverageParameterValues,
  GPURasterFractionalCoverage
} from '../../src/raster-analysis/gpu-raster-fractional-coverage';
import {GPURasterFractionalCategoricalStatistics} from '../../src/raster-analysis/gpu-raster-fractional-categorical-statistics';
import {GPURasterFractionalZonalDistribution} from '../../src/raster-analysis/gpu-raster-fractional-zonal-distribution';
import {getRasterRowAreaScale} from '../../src/raster-analysis/raster-grid/raster-cell-area-scale';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

// A 2 x 2 grid of one-degree cells whose rows are centred at 60.5 and 59.5 degrees north; the zone
// is the whole grid, so every coverage fraction is 1 and each area is a sum of ground cell areas.
const GEOGRAPHIC_GRID: GPURasterMetadata = {
  width: 2,
  height: 2,
  affine: [1, 0, 10, 0, -1, 61],
  pixelInterpretation: 'area'
};

async function runCellSizeMode(
  mode: 'uniform' | 'geographic' | 'web-mercator' | 'explicit-areas',
  options: {withParameters?: boolean; parameterGrid?: GPURasterMetadata} = {}
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: `fractional-cell-size-${mode}`});
  const buffers: Buffer[] = [];
  const parameterBuffers: GPUParameterBuffer[] = [];
  let identifier = 0;
  const input = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    values: number[],
    format: Format
  ): GraphDataView<Format> => {
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
  };
  const output = <Format extends 'uint32' | 'float32'>(length: number, format: Format) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `output-${identifier++}`, buffer, format, length)
    };
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'georeference',
    format: 'float32',
    length: 8
  });
  parameterBuffer.write(getGPURasterFractionalCoverageParameterValues(GEOGRAPHIC_GRID));
  const aggregatorParameters = options.withParameters
    ? (() => {
        const buffer = new GPUParameterBuffer(device, {
          id: 'aggregator-georeference',
          format: 'float32',
          length: 8
        });
        buffer.write(
          getGPURasterFractionalCoverageParameterValues(options.parameterGrid ?? GEOGRAPHIC_GRID)
        );
        parameterBuffers.push(buffer);
        return buffer.importToGraph(graph);
      })()
    : undefined;
  const parameters = parameterBuffer.importToGraph(graph);
  const coverage = {
    zoneIds: output(4, 'uint32').view,
    cellIds: output(4, 'uint32').view,
    fractions: output(4, 'float32').view,
    status: {
      count: output(1, 'uint32').view,
      requiredCount: output(1, 'uint32').view,
      overflow: output(1, 'uint32').view
    },
    zoneValidity: output(1, 'uint32').view
  };
  graph.add(
    new GPURasterFractionalCoverage({
      grid: GEOGRAPHIC_GRID,
      parameters,
      zones: {
        positions: input([10, 59, 12, 59, 12, 61, 10, 61], 'float32x2'),
        featureOffsets: input([0, 1], 'uint32'),
        polygonOffsets: input([0, 1], 'uint32'),
        ringOffsets: input([0, 4], 'uint32')
      },
      output: coverage
    })
  );
  const cellSizeMode = mode === 'explicit-areas' ? undefined : mode;
  const categorical = {
    classAreas: output(2, 'float32'),
    coverageSums: output(2, 'float32'),
    proportions: output(2, 'float32'),
    coveredArea: output(1, 'float32'),
    validArea: output(1, 'float32'),
    missingArea: output(1, 'float32'),
    majority: output(1, 'uint32'),
    minority: output(1, 'uint32'),
    validity: output(1, 'uint32')
  };
  graph.add(
    new GPURasterFractionalCategoricalStatistics({
      id: 'categorical',
      grid: GEOGRAPHIC_GRID,
      input: {
        id: 'categories',
        format: 'uint32',
        storage: {kind: 'buffer', values: input([0, 1, 0, 1], 'uint32')}
      },
      zoneCount: 1,
      categoryCapacity: 2,
      coverage,
      cellSizeMode,
      parameters: aggregatorParameters,
      output: {
        coverageSums: categorical.coverageSums.view,
        classAreas: categorical.classAreas.view,
        proportions: categorical.proportions.view,
        coveredArea: categorical.coveredArea.view,
        validArea: categorical.validArea.view,
        missingArea: categorical.missingArea.view,
        majority: categorical.majority.view,
        minority: categorical.minority.view,
        validity: categorical.validity.view
      }
    })
  );
  const rowScales = [0, 1].map(row => getRasterRowAreaScale(GEOGRAPHIC_GRID, 'geographic', row));
  const distribution = {
    means: output(1, 'float32'),
    populationVariances: output(1, 'float32'),
    sampleVariances: output(1, 'float32'),
    effectiveSampleSizes: output(1, 'float32'),
    weightSums: output(1, 'float32'),
    squaredWeightSums: output(1, 'float32'),
    coveredArea: output(1, 'float32'),
    validArea: output(1, 'float32'),
    quantiles: output(1, 'float32'),
    quantileValidity: output(1, 'uint32'),
    validity: output(1, 'uint32'),
    status: output(1, 'uint32')
  };
  graph.add(
    new GPURasterFractionalZonalDistribution({
      id: 'distribution',
      grid: GEOGRAPHIC_GRID,
      input: {
        id: 'distribution-values',
        format: 'float32',
        storage: {kind: 'buffer', values: input([1, 2, 3, 4], 'float32')}
      },
      zoneCount: 1,
      coverage,
      cellSizeMode,
      parameters: aggregatorParameters,
      cellAreas:
        mode === 'explicit-areas'
          ? input([rowScales[0], rowScales[0], rowScales[1], rowScales[1]], 'float32')
          : undefined,
      probabilities: input([0.5], 'float32'),
      output: {
        means: distribution.means.view,
        populationVariances: distribution.populationVariances.view,
        sampleVariances: distribution.sampleVariances.view,
        effectiveSampleSizes: distribution.effectiveSampleSizes.view,
        weightSums: distribution.weightSums.view,
        squaredWeightSums: distribution.squaredWeightSums.view,
        coveredArea: distribution.coveredArea.view,
        validArea: distribution.validArea.view,
        quantiles: distribution.quantiles.view,
        quantileValidity: distribution.quantileValidity.view,
        validity: distribution.validity.view,
        status: distribution.status.view
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const read = async (buffer: Buffer, length: number) =>
      Array.from(await readFloat32(buffer, length));
    return {
      classAreas: await read(categorical.classAreas.buffer, 2),
      categoricalCovered: await read(categorical.coveredArea.buffer, 1),
      proportions: await read(categorical.proportions.buffer, 2),
      distributionCovered: await read(distribution.coveredArea.buffer, 1),
      distributionMean: await read(distribution.means.buffer, 1),
      rowScales
    };
  } finally {
    compiled.destroy();
    parameterBuffer.destroy();
    for (const buffer of parameterBuffers) {
      buffer.destroy();
    }
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

const expectRelative = (actual: number[], expected: number[]) => {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    expect(Math.abs(actual[index] - expected[index])).toBeLessThanOrEqual(
      2e-5 * Math.max(1, Math.abs(expected[index]))
    );
  }
};

it('fractional aggregators keep planar areas by default and use ground areas with cellSizeMode', async () => {
  const uniform = await runCellSizeMode('uniform');
  // One-degree cells: category 0 is cells 0 and 2, category 1 is cells 1 and 3.
  expectRelative(uniform.classAreas, [2, 2]);
  expectRelative(uniform.categoricalCovered, [4]);
  expectRelative(uniform.distributionCovered, [4]);

  const geographic = await runCellSizeMode('geographic');
  const [north, south] = geographic.rowScales;
  // Square meters per square degree at 60.5 and 59.5 degrees north is about 6.2e9.
  expect(north).toBeGreaterThan(5.9e9);
  expect(north).toBeLessThan(6.5e9);
  expect(north).toBeLessThan(south);
  expectRelative(geographic.classAreas, [north + south, north + south]);
  expectRelative(geographic.categoricalCovered, [2 * (north + south)]);
  expectRelative(geographic.proportions, [0.5, 0.5]);
  expectRelative(geographic.distributionCovered, [2 * (north + south)]);

  // The same ground areas passed as explicit per-cell areas give the same distribution result.
  const explicit = await runCellSizeMode('explicit-areas');
  expectRelative(explicit.distributionCovered, geographic.distributionCovered);
  expectRelative(explicit.distributionMean, geographic.distributionMean);
});

// Independent reference, not the library's scale formula: geodesic areas of the two one-degree
// cells (lon 10..11, lat 60..61 and 59..60) on the WGS84 ellipsoid, edges densified along their
// parallels and meridians. From /usr/bin/python3 (NumPy 2.0.2, pyproj 3.6.1):
//   g = Geod(ellps='WGS84'); a, _ = g.polygon_area_perimeter(lons, lats)  # ring every 0.01 degree
const GEODESIC_NORTH_CELL_AREA = 6123140858.974197;
const GEODESIC_SOUTH_CELL_AREA = 6309805649.383578;

it('fractional aggregator ground areas match geodesic cell areas', async () => {
  // The scale is taken at the row center, so the match is to about 1e-5 for one-degree cells.
  for (const withParameters of [false, true]) {
    const geographic = await runCellSizeMode('geographic', {withParameters});
    const total = GEODESIC_NORTH_CELL_AREA + GEODESIC_SOUTH_CELL_AREA;
    for (const covered of [geographic.categoricalCovered[0], geographic.distributionCovered[0]]) {
      expect(Math.abs(covered / (2 * total) - 1)).toBeLessThan(5e-5);
    }
    expect(Math.abs(geographic.classAreas[0] / total - 1)).toBeLessThan(5e-5);
  }
});

it('fractional aggregators read the row edges from parameters on every encoding', async () => {
  // Compiled for rows at 60.5 and 59.5 degrees north; the parameters describe the same grid
  // 30 degrees further south, so the ground area must follow the parameters, not the compile-time
  // affine. Fails when the row edges are baked into the shader.
  const moved: GPURasterMetadata = {...GEOGRAPHIC_GRID, affine: [1, 0, 10, 0, -1, 31]};
  const baked = await runCellSizeMode('geographic', {withParameters: true});
  const parameterized = await runCellSizeMode('geographic', {
    withParameters: true,
    parameterGrid: moved
  });
  const [northScale, southScale] = [0, 1].map(row =>
    getRasterRowAreaScale(moved, 'geographic', row)
  );
  // Cells at 30 degrees are about 1.6 times larger on the ground than at 60 degrees.
  expect(parameterized.categoricalCovered[0]).toBeGreaterThan(baked.categoricalCovered[0] * 1.1);
  expectRelative(parameterized.classAreas, [northScale + southScale, northScale + southScale]);
  expectRelative(parameterized.categoricalCovered, [2 * (northScale + southScale)]);
  expectRelative(parameterized.distributionCovered, [2 * (northScale + southScale)]);
});
