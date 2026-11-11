// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {
  GPU_WGS84_FLATTENING,
  GPU_WGS84_SEMI_MAJOR_AXIS_METERS
} from '../../../src/analysis-core/contracts/earth';
import {
  computeRasterFractionalZonalDistributionOnCPU,
  CPURasterFractionalZonalDistributionOperation
} from '../../../src/raster-analysis/raster-zonal-statistics/cpu-raster-fractional-zonal-distribution-operation';
import {GPURasterFractionalZonalDistributionOperation} from '../../../src/raster-analysis/raster-zonal-statistics/gpu-raster-fractional-zonal-distribution-operation';
import {constructRasterOperationContributor} from '../../utils/raster-operation-test-utils';
import {
  createDistributionInput,
  createDistributionGraphInputs,
  createUnitDistributionInput,
  DISTRIBUTION_PROPS
} from './raster-fractional-zonal-distribution-operation-fixture';

it('matches hand-computed reliability moments and equal-weight distinct-value interpolation', () => {
  // Weights 1,2,1 at 0,10,20: W=4, W2=6, M2=200, denominator=2.5.
  const result = computeRasterFractionalZonalDistributionOnCPU(
    createDistributionInput(),
    DISTRIBUTION_PROPS
  );
  expect(result.means[0]).toBe(10);
  expect(result.populationVariances[0]).toBe(50);
  expect(result.sampleVariances[0]).toBe(80);
  expect(result.effectiveSampleSizes[0]).toBe(Math.fround(8 / 3));
  expect(result.weightSums[0]).toBe(4);
  expect(result.squaredWeightSums[0]).toBe(6);
  expect(result.coveredArea[0]).toBe(8);
  expect(result.validArea[0]).toBe(8);
  expect(Array.from(result.quantiles)).toEqual([0, 5, 10, 15, 20, NaN]);
  expect(Array.from(result.quantileValidity)).toEqual([1, 1, 1, 1, 1, 0]);
  expect(result.validity[0]).toBe(1);
  expect(result.status[0]).toBe(0);
  const inverse = computeRasterFractionalZonalDistributionOnCPU(createDistributionInput(), {
    ...DISTRIBUTION_PROPS,
    quantileMethod: 'inverse-cdf'
  });
  expect(Array.from(inverse.quantiles)).toEqual([0, 0, 10, 10, 20, NaN]);
});

it('publishes missing/support statuses, dynamic determinant area and invalid tuple flags', () => {
  const empty = computeRasterFractionalZonalDistributionOnCPU(
    createDistributionInput(true),
    DISTRIBUTION_PROPS
  );
  expect(empty.status[0]).toBe(12);
  expect(empty.validity[0]).toBe(0);
  expect(empty.means[0]).toBeNaN();
  expect(empty.weightSums[0]).toBe(0);
  expect(empty.effectiveSampleSizes[0]).toBe(0);
  const input = createDistributionInput();
  input.cellAreas = undefined;
  input.parameters = new Float32Array([0, 0, 2, 0, 0, -2, -4, 0]);
  expect(
    computeRasterFractionalZonalDistributionOnCPU(input, DISTRIBUTION_PROPS).coveredArea[0]
  ).toBe(8);
  input.coverage.overflow[0] = 1;
  const incomplete = computeRasterFractionalZonalDistributionOnCPU(input, DISTRIBUTION_PROPS);
  expect(incomplete.status[0]).toBe(1);
  expect(incomplete.means[0]).toBeNaN();
  expect(incomplete.coveredArea[0]).toBeNaN();
  input.coverage.overflow[0] = 0;
  input.coverage.cellIds[1] = 0;
  expect(
    computeRasterFractionalZonalDistributionOnCPU(input, DISTRIBUTION_PROPS).status[0] & 2
  ).toBe(2);
  input.coverage.zoneIds[1] = 9;
  expect(
    computeRasterFractionalZonalDistributionOnCPU(input, DISTRIBUTION_PROPS).status[0] & 2
  ).toBe(2);
});

it('checks construction-time limits of both classes and pure input lengths', () => {
  const inputs = Object.fromEntries(
    Object.entries(createDistributionGraphInputs()).map(([name, value]) => [
      name,
      {format: value.format, length: value.values.length}
    ])
  );
  for (const Operation of [
    CPURasterFractionalZonalDistributionOperation,
    GPURasterFractionalZonalDistributionOperation
  ])
    expect(() => new Operation({...DISTRIBUTION_PROPS, probabilityCount: 0})).toThrow(
      /^distribution /
    );
  expect(
    () =>
      new CPURasterFractionalZonalDistributionOperation({
        ...DISTRIBUTION_PROPS,
        quantileMethod: 'invalid' as never
      })
  ).toThrow(/^distribution /);
  expect(() =>
    constructRasterOperationContributor(
      new GPURasterFractionalZonalDistributionOperation({
        ...DISTRIBUTION_PROPS,
        quantileMethod: 'invalid' as never
      }),
      inputs
    )
  ).toThrow(/^distribution /);
  expect(() =>
    constructRasterOperationContributor(
      new GPURasterFractionalZonalDistributionOperation(DISTRIBUTION_PROPS),
      inputs
    )
  ).not.toThrow();
  expect(
    () =>
      new CPURasterFractionalZonalDistributionOperation({
        ...DISTRIBUTION_PROPS,
        probabilityCount: 33
      })
  ).not.toThrow();
  const input = createDistributionInput();
  input.coverage.cellIds = new Uint32Array(2);
  expect(() => computeRasterFractionalZonalDistributionOnCPU(input, DISTRIBUTION_PROPS)).toThrow(
    /coverage.cellIds/
  );
  const anotherInput = createDistributionInput();
  anotherInput.observationWeights = new Float32Array(2);
  expect(() =>
    computeRasterFractionalZonalDistributionOnCPU(anotherInput, DISTRIBUTION_PROPS)
  ).toThrow(/observationWeights/);
});

it('executes the CPU callback with independent input/output validity ports and renamed outputs', async () => {
  const graphInputs = createDistributionGraphInputs();
  const inputs = Object.fromEntries(
    Object.entries(graphInputs).map(([name, input]) => [
      name,
      {
        format: input.format,
        length: input.values.length,
        chunks: [{values: input.values, length: input.values.length}],
        pack: () => input.values
      }
    ])
  );
  const props = {
    ...DISTRIBUTION_PROPS,
    output: {...DISTRIBUTION_PROPS.output, means: 'renamedMeans', validity: 'outputValidity'}
  };
  const result = await new CPURasterFractionalZonalDistributionOperation(props).execute({inputs});
  expect(result['renamedMeans']).toEqual(new Float32Array([10]));
  expect(result['outputValidity']).toEqual(new Uint32Array([1]));
  expect(Object.keys(result).sort()).toEqual(Object.values(props.output).sort());
});

it('excludes the rounded float32 nodata value from every distribution output', () => {
  const input = createUnitDistributionInput(
    new Float32Array([0.1, 2]),
    new Float32Array([0, 0.5, 1])
  );
  input.input.noDataValue = 0.1;
  const result = computeRasterFractionalZonalDistributionOnCPU(input, {
    ...DISTRIBUTION_PROPS,
    grid: makeGPURasterMetadata({width: 2, height: 1})
  });
  // Only value 2 survives: W=W2=1, population variance zero, sample variance undefined.
  expect(result).toEqual({
    means: new Float32Array([2]),
    populationVariances: new Float32Array([0]),
    sampleVariances: new Float32Array([NaN]),
    effectiveSampleSizes: new Float32Array([1]),
    weightSums: new Float32Array([1]),
    squaredWeightSums: new Float32Array([1]),
    coveredArea: new Float32Array([2]),
    validArea: new Float32Array([1]),
    quantiles: new Float32Array([2, 2, 2]),
    quantileValidity: new Uint32Array([1, 1, 1]),
    validity: new Uint32Array([1]),
    status: new Uint32Array([8])
  });
});

it('rejects trailing band samples and validity flags for a one-pixel distribution', () => {
  const options = {...DISTRIBUTION_PROPS, grid: makeGPURasterMetadata({width: 1, height: 1})};
  for (const oversized of ['values', 'validity'] as const) {
    const input = createUnitDistributionInput(new Float32Array([3]), new Float32Array([0.5]));
    if (oversized === 'values') input.input.values = new Float32Array([3, 100]);
    else input.input.validity = new Uint32Array([1, 1]);
    expect(() => computeRasterFractionalZonalDistributionOnCPU(input, options)).toThrow(
      oversized === 'values'
        ? /input.values must contain exactly one sample per pixel/
        : /input.validity must contain exactly one flag per pixel/
    );
  }
});

it('uses asymmetric weights in exactextract quantile interpolation', () => {
  const input = createUnitDistributionInput(new Float32Array([0, 10, 20]), new Float32Array([0.5]));
  input.observationWeights = new Float32Array([1, 1, 2]);
  // Distinct-value knot positions divided by 8 are 0, 3/8, 1. At p=1/2 the
  // interpolation fraction between 10 and 20 is (1/2-3/8)/(1-3/8)=1/5, giving 12.
  const result = computeRasterFractionalZonalDistributionOnCPU(input, DISTRIBUTION_PROPS);
  expect(Array.from(result.quantiles)).toEqual([12]);
  expect(Array.from(result.quantileValidity)).toEqual([1]);
  expect(Array.from(result.status)).toEqual([0]);
});

it('computes geographic and web-mercator ground area at the equator', () => {
  const input = createUnitDistributionInput(new Float32Array([3]), new Float32Array([0.5]));
  input.cellAreas = undefined;
  const grid = {
    ...DISTRIBUTION_PROPS.grid,
    width: 1,
    height: 1,
    affine: [1, 0, 0, 0, -1, 0.5] as const
  };
  // The row edges are read from parameters when it is given (origin y in slot 1, row step y in
  // slot 5), so it is packed for this grid: one cell centred on the equator.
  input.parameters = new Float32Array([0, 0.5, 1, 0, 0, -1, -1, 0]);
  // At latitude zero N=a and M=a(1-e^2). Per square degree multiply N*M by
  // (pi/180)^2; per square Mercator meter divide N*M by a^2.
  const equatorialAreaScale = 1 - GPU_WGS84_FLATTENING * (2 - GPU_WGS84_FLATTENING);
  for (const cellSizeMode of ['geographic', 'web-mercator'] as const) {
    const expectedArea =
      cellSizeMode === 'geographic'
        ? ((GPU_WGS84_SEMI_MAJOR_AXIS_METERS * Math.PI) / 180) ** 2 * equatorialAreaScale
        : equatorialAreaScale;
    const result = computeRasterFractionalZonalDistributionOnCPU(input, {
      ...DISTRIBUTION_PROPS,
      grid,
      cellSizeMode
    });
    expect(Math.abs(result.coveredArea[0] - expectedArea)).toBeLessThanOrEqual(expectedArea * 1e-6);
    expect(result.validArea[0]).toBe(result.coveredArea[0]);
    expect(result.means[0]).toBe(3);
    expect(result.quantiles[0]).toBe(3);
    expect(result.status[0]).toBe(8);
    // Without parameters the row edges come from grid.affine: the same cell, the same area.
    const fromAffine = computeRasterFractionalZonalDistributionOnCPU(
      {...input, parameters: undefined},
      {...DISTRIBUTION_PROPS, grid, cellSizeMode}
    );
    expect(fromAffine.coveredArea[0]).toBe(result.coveredArea[0]);
    // A georeference packed for a cell half a degree further south gives a different area per
    // square degree (the radii of curvature change with latitude), so parameters, not the
    // affine, decide. Half a Mercator meter is below float32 resolution of the area, so only
    // the geographic mode can show it.
    if (cellSizeMode === 'geographic') {
      const moved = computeRasterFractionalZonalDistributionOnCPU(
        {...input, parameters: new Float32Array([0, 0, 1, 0, 0, -1, -1, 0])},
        {...DISTRIBUTION_PROPS, grid, cellSizeMode}
      );
      expect(moved.coveredArea[0]).not.toBe(result.coveredArea[0]);
    }
  }
});

it('validates cellSizeMode and its grid and manual-area constraints in both constructors', () => {
  const inputs = Object.fromEntries(
    Object.entries(createDistributionGraphInputs()).map(([name, input]) => [
      name,
      {format: input.format, length: input.values.length}
    ])
  );
  const constructGPU = (props: typeof DISTRIBUTION_PROPS) =>
    constructRasterOperationContributor(
      new GPURasterFractionalZonalDistributionOperation(props),
      inputs
    );
  expect(
    () =>
      new CPURasterFractionalZonalDistributionOperation({
        ...DISTRIBUTION_PROPS,
        cellSizeMode: 'uniform'
      })
  ).not.toThrow();
  expect(() => constructGPU({...DISTRIBUTION_PROPS, cellSizeMode: 'uniform'})).not.toThrow();
  const defaultResult = computeRasterFractionalZonalDistributionOnCPU(
    createDistributionInput(),
    DISTRIBUTION_PROPS
  );
  expect(
    computeRasterFractionalZonalDistributionOnCPU(createDistributionInput(), {
      ...DISTRIBUTION_PROPS,
      cellSizeMode: 'uniform'
    })
  ).toEqual(defaultResult);
  for (const cellSizeMode of ['geographic', 'web-mercator'] as const) {
    const props = {...DISTRIBUTION_PROPS, cellSizeMode};
    expect(() => new CPURasterFractionalZonalDistributionOperation(props)).toThrow(
      /^distribution cellSizeMode must be uniform when cellAreas/
    );
    expect(() => constructGPU(props)).toThrow(
      /^distribution cellSizeMode must be uniform when cellAreas/
    );
    const withoutAreas = {...props, cellAreas: undefined};
    expect(() => new CPURasterFractionalZonalDistributionOperation(withoutAreas)).not.toThrow();
    expect(() => constructGPU(withoutAreas)).not.toThrow();
    const rotated = {
      ...withoutAreas,
      grid: {...props.grid, affine: [1, 0.25, 0, 0, -1, 0] as const}
    };
    expect(() => new CPURasterFractionalZonalDistributionOperation(rotated)).toThrow(
      /axis-aligned/
    );
    expect(() => constructGPU(rotated)).toThrow(/axis-aligned/);
    expect(() =>
      computeRasterFractionalZonalDistributionOnCPU(createDistributionInput(), props)
    ).toThrow(/cellSizeMode/);
  }
  const unknown = {...DISTRIBUTION_PROPS, cellSizeMode: 'invalid' as never};
  expect(() => new CPURasterFractionalZonalDistributionOperation(unknown)).toThrow(
    /^distribution cellSizeMode/
  );
  expect(() => constructGPU(unknown)).toThrow(/^distribution cellSizeMode/);
});
