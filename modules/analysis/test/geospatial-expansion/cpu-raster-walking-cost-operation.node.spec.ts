// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeRasterWalkingCostOnCPU,
  CPURasterWalkingCostOperation
} from '../../src/raster-analysis/cost-distance/cpu-raster-walking-cost-operation';
import {GPURasterWalkingCostOperation} from '../../src/raster-analysis/cost-distance/gpu-raster-walking-cost-operation';
import {getGPURasterWalkingCostParameterValues} from '../../src/raster-analysis/gpu-raster-walking-cost';
import {GPU_WGS84_SEMI_MAJOR_AXIS_METERS} from '../../src/analysis-core/contracts/earth';
import {constructRasterOperationContributor} from '../utils/raster-operation-test-utils';
import {
  createWalkingInput,
  createWalkingProps,
  createWalkingGraphInputs,
  createSingletonWalkingProps,
  createSingletonWalkingInput
} from './raster-walking-cost-operation-fixture';
import fixtures from './raster-walking-cost-operation-fixtures.json';

it('matches independent all-pairs shortest paths for directed Tobler and GRASS models with two sources', () => {
  for (const fixture of fixtures.cases) {
    const props = createWalkingProps(fixture.kind as 'grass' | 'tobler');
    const result = computeRasterWalkingCostOnCPU(createWalkingInput(props), props);
    expect(Array.from(result.costs)).toEqual(fixture.expected);
    expect(result.validity.every(value => value === 1)).toBe(true);
    expect(Array.from(result.converged)).toEqual([1]);
    expect(Array.from(result.iterationLimitReached)).toEqual([0]);
    expect(Array.from(result.invalidCount)).toEqual([0]);
  }
});

it('matches a flat equatorial closed form in geographic and Mercator modes and reads repacked affine', () => {
  for (const cellSizeMode of ['uniform', 'geographic', 'web-mercator'] as const) {
    const props = createWalkingProps('grass');
    props.grid = {
      width: 3,
      height: 1,
      affine: [0.001, 0, 0, 0, -0.002, 0.001],
      pixelInterpretation: 'area'
    };
    props.cellSizeMode = cellSizeMode;
    // Changing the compile-time coefficient numbers cannot change the per-frame model.
    props.model = {kind: 'grass', coefficients: [999, 999, 999, 999], downhillSlopeThreshold: -99};
    const parameters = getGPURasterWalkingCostParameterValues(
      {
        model: {kind: 'grass', coefficients: [2, 3, -0.5, -2], downhillSlopeThreshold: -0.25},
        frictionWeight: 0
      },
      props.grid
    );
    const result = computeRasterWalkingCostOnCPU(
      {
        elevation: {id: 'elevation', format: 'float32', values: new Float32Array(3)},
        friction: {id: 'friction', format: 'float32', values: new Float32Array(3)},
        sources: Uint32Array.of(1, 0, 0),
        parameters
      },
      props
    );
    const distance =
      Math.fround(0.001) *
      (cellSizeMode === 'geographic' ? (Math.PI * GPU_WGS84_SEMI_MAJOR_AXIS_METERS) / 180 : 1);
    expect(Array.from(result.costs)).toEqual([
      0,
      Math.fround(2 * distance),
      Math.fround(4 * distance)
    ]);
    expect(result.iterationCount[0]).toBe(3);
  }
});

it('distinguishes barriers, unreached cells, missing sources and a mismatched packed model', () => {
  const props = createWalkingProps('grass');
  const input = createWalkingInput(props);
  input.elevation.validity![2] = 0;
  input.friction.values[3] = -1;
  input.elevation.noDataValue = 0.1;
  input.elevation.values[4] = Math.fround(0.1);
  const result = computeRasterWalkingCostOnCPU(input, props);
  for (const cell of [2, 3, 4]) {
    expect(result.validity[cell]).toBe(0);
    expect(result.costs[cell]).toBeNaN();
  }
  input.sources.fill(0);
  const unreached = computeRasterWalkingCostOnCPU(input, props);
  expect(unreached.costs[1]).toBe(Infinity);
  expect(unreached.iterationCount[0]).toBe(1);
  input.sources[0] = 1;
  input.parameters = getGPURasterWalkingCostParameterValues({model: {kind: 'tobler'}}, props.grid);
  const mismatch = computeRasterWalkingCostOnCPU(input, props);
  expect(mismatch.invalidCount[0]).toBe(1);
  expect(mismatch.costs[0]).toBe(0);
  expect(mismatch.costs[1]).toBe(Infinity);
});

it('matches independent pyproj ground distances away from the equator in both ground modes', () => {
  for (const fixture of fixtures.groundCases) {
    const props = createWalkingProps('grass');
    props.grid = {
      width: 3,
      height: 1,
      affine: fixture.affine as [number, number, number, number, number, number],
      pixelInterpretation: 'area'
    };
    props.cellSizeMode = fixture.mode as 'geographic' | 'web-mercator';
    const result = computeRasterWalkingCostOnCPU(
      {
        elevation: {id: 'elevation', format: 'float32', values: new Float32Array(3)},
        friction: {id: 'friction', format: 'float32', values: new Float32Array(3)},
        sources: Uint32Array.of(1, 0, 0),
        parameters: getGPURasterWalkingCostParameterValues(
          {model: props.model, frictionWeight: 0},
          props.grid
        )
      },
      props
    );
    for (let cell = 1; cell < 3; cell++)
      expect(Math.abs(result.costs[cell] - fixture.expected[cell])).toBeLessThanOrEqual(
        3e-6 * fixture.expected[cell]
      );
    expect(result.iterationCount[0]).toBe(3);
  }
});

it('validates pure arrays and the construction-time limits of both classes', () => {
  const props = createWalkingProps('grass');
  const input = createWalkingInput(props);
  const descriptors = Object.fromEntries(
    Object.entries(createWalkingGraphInputs(input)).map(([port, value]) => [
      port,
      {format: value.format, length: value.values.length}
    ])
  );
  expect(() => new CPURasterWalkingCostOperation({...props, maximumIterations: 0})).toThrow(
    /^walking .*maximumIterations/
  );
  expect(() =>
    constructRasterOperationContributor(
      new GPURasterWalkingCostOperation({...props, maximumIterations: 0}),
      descriptors
    )
  ).toThrow(/^walking .*maximumIterations/);
  expect(
    () => new CPURasterWalkingCostOperation({...props, maximumIterations: 30000})
  ).not.toThrow();
  expect(() =>
    constructRasterOperationContributor(
      new GPURasterWalkingCostOperation({...props, maximumIterations: 30000}),
      descriptors
    )
  ).toThrow(/^walking /);
  for (const Operation of [CPURasterWalkingCostOperation, GPURasterWalkingCostOperation])
    expect(() => new Operation({...props, grid: {...props.grid, width: 0}})).toThrow(/^walking /);
  expect(() =>
    constructRasterOperationContributor(new GPURasterWalkingCostOperation(props), descriptors)
  ).not.toThrow();
  expect(() =>
    constructRasterOperationContributor(new GPURasterWalkingCostOperation(props), {
      ...descriptors,
      parameters: {format: 'float32', length: 15}
    })
  ).toThrow(/^walking .*parameters/);
  expect(() =>
    computeRasterWalkingCostOnCPU({...input, sources: new Uint32Array(19)}, props)
  ).toThrow(/sources/);
  expect(() =>
    computeRasterWalkingCostOnCPU({...input, parameters: new Float32Array(15)}, props)
  ).toThrow(/parameters/);
  input.friction.validity = new Uint32Array(21);
  expect(() => computeRasterWalkingCostOnCPU(input, props)).toThrow(/input.validity/);
});

it('maps both band validities independently from output validity in the CPU callback', async () => {
  const props = createWalkingProps('grass');
  const inputs = Object.fromEntries(
    Object.entries(createWalkingGraphInputs(createWalkingInput(props))).map(([port, input]) => [
      port,
      {
        format: input.format,
        length: input.values.length,
        chunks: [{values: input.values, length: input.values.length}],
        pack: () => input.values
      }
    ])
  );
  const result = await new CPURasterWalkingCostOperation({...props, maximumIterations: 1}).execute({
    inputs
  });
  expect(Array.from(result['costs'] as Float32Array)).toEqual(fixtures.cases[0].expected);
  expect(Object.keys(result).sort()).toEqual([
    'converged',
    'costs',
    'invalid',
    'limit',
    'rounds',
    'validity'
  ]);
});

it('ignores surplus source rows on a unit grid without creating a source from the tail', () => {
  const props = createSingletonWalkingProps();
  for (const sources of [Uint32Array.of(1, 0), Uint32Array.of(0, 1)]) {
    const input = createSingletonWalkingInput(sources);
    const result = computeRasterWalkingCostOnCPU(input, props);
    // The active cell starts at zero only when its own source word is nonzero; no moves exist.
    expect(result.costs[0] === (sources[0] !== 0 ? 0 : Infinity)).toBe(true);
    expect(result.costs.length).toBe(1);
    expect(result.validity).toEqual(Uint32Array.of(1));
    expect(result.converged).toEqual(Uint32Array.of(1));
    expect(result.iterationCount).toEqual(Uint32Array.of(1));
    expect(result.iterationLimitReached).toEqual(Uint32Array.of(0));
    expect(result.invalidCount).toEqual(Uint32Array.of(0));
    const descriptors = Object.fromEntries(
      Object.entries(createWalkingGraphInputs(input)).map(([port, value]) => [
        port,
        {format: value.format, length: value.values.length}
      ])
    );
    expect(() =>
      constructRasterOperationContributor(new GPURasterWalkingCostOperation(props), descriptors)
    ).not.toThrow();
  }
  expect(() =>
    computeRasterWalkingCostOnCPU(createSingletonWalkingInput(new Uint32Array(0)), props)
  ).toThrow(/sources/);
});
