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
  GPURasterFractionalCoverage,
  GPURasterFractionalZonalStatistics
} from '../../src/raster-analysis/gpu-raster-fractional-coverage';
import {GPURasterFractionalCategoricalStatistics} from '../../src/raster-analysis/gpu-raster-fractional-categorical-statistics';
import {GPURasterFractionalZonalDistribution} from '../../src/raster-analysis/gpu-raster-fractional-zonal-distribution';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {createParityGraph} from '../oracles/parity-wp2-geodesic-harness';

// raster-fields-02. The three fractional statistics contributors used to bake
// abs(det(grid.affine)) into their shaders while GPURasterFractionalCoverage reads the affine per
// frame, so repacking the coverage parameters for another cell size left every area column on the
// old cell area.
//
// The zone is the rectangle x in [0, 20], y in [0, 10]: 200 square units whatever the grid. Frame
// one is a 2 x 2 grid of 10-unit cells over [0, 20]^2 (the zone covers the two cells of row 1).
// Frame two is the same compiled graph repacked for 20-unit cells over [0, 40]^2 (the zone covers
// half of cell 2). The expected numbers are areas of rectangles, computed by hand.
const COMPILED_GRID: GPURasterMetadata = {
  width: 2,
  height: 2,
  affine: [10, 0, 0, 0, -10, 20],
  pixelInterpretation: 'area'
};
const REPACKED_GRID: GPURasterMetadata = {
  width: 2,
  height: 2,
  affine: [20, 0, 0, 0, -20, 40],
  pixelInterpretation: 'area'
};

it('fractional statistics read the cell area from the per-frame georeference (raster-fields-02)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'fractional-georeference'});
  const buffers: Buffer[] = [];
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
      grid: COMPILED_GRID,
      parameters,
      zones: {
        positions: input([0, 0, 20, 0, 20, 10, 0, 10], 'float32x2'),
        featureOffsets: input([0, 1], 'uint32'),
        polygonOffsets: input([0, 1], 'uint32'),
        ringOffsets: input([0, 4], 'uint32')
      },
      output: coverage
    })
  );
  const density = {
    totals: output(1, 'float32'),
    means: output(1, 'float32'),
    coveredArea: output(1, 'float32'),
    validArea: output(1, 'float32'),
    validity: output(1, 'uint32')
  };
  graph.add(
    new GPURasterFractionalZonalStatistics({
      grid: COMPILED_GRID,
      input: {
        id: 'density',
        format: 'float32',
        storage: {kind: 'buffer', values: input([1, 2, 3, 4], 'float32')}
      },
      zoneCount: 1,
      coverage,
      quantity: 'density',
      parameters,
      output: {
        totals: density.totals.view,
        means: density.means.view,
        coveredArea: density.coveredArea.view,
        validArea: density.validArea.view,
        validity: density.validity.view
      }
    })
  );
  const categorical = {
    coverageSums: output(2, 'float32'),
    classAreas: output(2, 'float32'),
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
      grid: COMPILED_GRID,
      input: {
        id: 'categories',
        format: 'uint32',
        storage: {kind: 'buffer', values: input([0, 1, 0, 1], 'uint32')}
      },
      zoneCount: 1,
      categoryCapacity: 2,
      coverage,
      parameters,
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
      grid: COMPILED_GRID,
      input: {
        id: 'distribution-values',
        format: 'float32',
        storage: {kind: 'buffer', values: input([1, 2, 3, 4], 'float32')}
      },
      zoneCount: 1,
      coverage,
      parameters,
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
  // The coverage fraction divides by the float32 determinant, so areas carry a few ulps.
  const expectClose = (actual: number[], expected: number[]) => {
    expect(actual.length).toBe(expected.length);
    for (let index = 0; index < expected.length; index++) {
      expect(Math.abs(actual[index] - expected[index])).toBeLessThanOrEqual(1e-3);
    }
  };
  const read = async (buffer: Buffer, length: number) =>
    Array.from(await readFloat32(buffer, length));
  try {
    // Frame one: 10-unit cells. The zone is cells 2 and 3 (row 1), fraction 1 each.
    parameterBuffer.write(getGPURasterFractionalCoverageParameterValues(COMPILED_GRID));
    submitGraph(device, compiled, undefined);
    expectClose(await read(density.coveredArea.buffer, 1), [200]);
    expectClose(await read(density.validArea.buffer, 1), [200]);
    // Densities 3 and 4 over 100 square units each.
    expectClose(await read(density.totals.buffer, 1), [700]);
    expectClose(await read(categorical.classAreas.buffer, 2), [100, 100]);
    expectClose(await read(categorical.coveredArea.buffer, 1), [200]);
    expectClose(await read(categorical.validArea.buffer, 1), [200]);
    expectClose(await read(distribution.coveredArea.buffer, 1), [200]);
    expectClose(await read(distribution.validArea.buffer, 1), [200]);

    // Frame two: the same graph repacked for 20-unit cells. The zone is half of cell 2.
    parameterBuffer.write(getGPURasterFractionalCoverageParameterValues(REPACKED_GRID));
    submitGraph(device, compiled, undefined);
    // The old literal gave 0.5 * 100 = 50 for every area and 150 for the density total.
    expectClose(await read(density.coveredArea.buffer, 1), [200]);
    expectClose(await read(density.validArea.buffer, 1), [200]);
    // Density 3 over half of a 400 square unit cell.
    expectClose(await read(density.totals.buffer, 1), [600]);
    expectClose(await read(categorical.classAreas.buffer, 2), [200, 0]);
    expectClose(await read(categorical.coveredArea.buffer, 1), [200]);
    expectClose(await read(categorical.validArea.buffer, 1), [200]);
    expectClose(await read(categorical.missingArea.buffer, 1), [0]);
    expectClose(await read(distribution.coveredArea.buffer, 1), [200]);
    expectClose(await read(distribution.validArea.buffer, 1), [200]);
  } finally {
    compiled.destroy();
    parameterBuffer.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
});

for (const cellSizeMode of ['geographic', 'web-mercator'] as const) {
  it(`UNI-03 origin-relative categorical areas retain absolute latitude (${cellSizeMode})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = createParityGraph(device);
    const originY = cellSizeMode === 'geographic' ? 45 : 4600000;
    const grid: GPURasterMetadata = {
      width: 1,
      height: 1,
      affine: [1, 0, 0, 0, 1, originY],
      pixelInterpretation: 'area'
    };
    const coveredArea = rig.output('covered-area', 1, 'float32');
    rig.graph.add(
      new GPURasterFractionalCategoricalStatistics({
        grid,
        cellSizeMode,
        zoneCount: 1,
        categoryCapacity: 1,
        parameters: rig.input(
          'parameters',
          Array.from(
            getGPURasterFractionalCoverageParameterValues(grid, {localOrigin: [0, originY]})
          ),
          'float32'
        ).view,
        input: {
          id: 'categories',
          format: 'uint32',
          storage: {kind: 'buffer', values: rig.input('categories', [0], 'uint32').view}
        },
        coverage: {
          zoneIds: rig.input('zones', [0], 'uint32').view,
          cellIds: rig.input('cells', [0], 'uint32').view,
          fractions: rig.input('fractions', [0.5], 'float32').view,
          zoneValidity: rig.input('zone-validity', [1], 'uint32').view,
          status: {
            count: rig.input('count', [1], 'uint32').view,
            requiredCount: rig.input('required', [1], 'uint32').view,
            overflow: rig.input('overflow', [0], 'uint32').view
          }
        },
        output: {
          coverageSums: rig.output('coverage-sums', 1, 'float32').view,
          classAreas: rig.output('areas', 1, 'float32').view,
          proportions: rig.output('proportions', 1, 'float32').view,
          coveredArea: coveredArea.view,
          validArea: rig.output('valid-area', 1, 'float32').view,
          missingArea: rig.output('missing-area', 1, 'float32').view,
          majority: rig.output('majority', 1, 'uint32').view,
          minority: rig.output('minority', 1, 'uint32').view,
          validity: rig.output('validity', 1, 'uint32').view
        }
      })
    );
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      // Independent WGS84: f=1/298.257223563, e²=f(2-f), N=a/sqrt(1-e²sin²φ),
      // M=a(1-e²)/(1-e²sin²φ)^1.5. Half-cell midpoint area is .5*N*M*cosφ*(pi/180)^2
      // for degrees, .5*N*M*cos²φ/a² for Mercator, φ=atan(sinh(4600000.5/a)).
      // /usr/bin/python3 math gives these literals (a=6378137).
      const expected = cellSizeMode === 'geographic' ? 4343304122.976737 : 0.30875252328196756;
      const [area] = await readFloat32(coveredArea.buffer, 1);
      expect(Number.isFinite(area)).toBe(true);
      expect(area / expected).toBeCloseTo(1, 5);
    } finally {
      compiled.destroy();
      rig.destroy();
    }
  });
}
