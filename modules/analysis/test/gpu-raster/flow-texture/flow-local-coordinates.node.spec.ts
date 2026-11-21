// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {expectCoordinateStreamline} from './flow-coordinate-fixtures';
import {getGPUParticleAdvectionParameterValues} from '../../../src/raster-analysis/particle-advection/particle-advection-parameters';
import {getGPULagrangianFlowMapParameterValues} from '../../../src/raster-analysis/lagrangian-diagnostics/gpu-lagrangian-flow-map';
import {
  getGPUStreamlinesParameterValues,
  getGPUStreamlinesUint32ParameterValues
} from '../../../src/raster-analysis/flow-texture/streamlines-parameters';
import {computeStreamlinesOnCPU} from '../../../src/raster-analysis/flow-texture/cpu-streamlines-operation';

const localOrigin = [10_000_000, 10_000_000] as const;
const fieldPlacement = {origin: [10_000_000, 10_000_000], cellSize: [100, 100]} as const;

it('FLT-07 rebases planar placements and explicit/default particle spawn bounds before float32', () => {
  const particle = getGPUParticleAdvectionParameterValues(
    {fieldPlacement, localOrigin, timeStep: 0.25},
    [1, 1]
  );
  expect(Array.from(particle.subarray(0, 4))).toEqual([0, 0, 100, 100]);
  expect(Array.from(particle.subarray(8))).toEqual([0, 0, 100, 100]);
  const explicit = getGPUParticleAdvectionParameterValues(
    {
      fieldPlacement,
      localOrigin,
      timeStep: 0.25,
      spawnBounds: [10_000_000.25, 10_000_000.5, 10_000_001.25, 10_000_001.5]
    },
    [1, 1]
  );
  expect(Array.from(explicit.subarray(8))).toEqual([0.25, 0.5, 1.25, 1.5]);
  const flow = getGPULagrangianFlowMapParameterValues({
    fieldPlacement,
    localOrigin,
    startTime: 0,
    timeStep: 0.25,
    sliceStartTime: 0,
    sliceEndTime: 1
  });
  expect(Array.from(flow.subarray(0, 4))).toEqual([0, 0, 100, 100]);
  // Constant velocity 1: four quarter-second steps from local zero travel exactly one metre.
  let position = 0;
  for (let frame = 0; frame < 4; frame++) position = Math.fround(position + particle[4]);
  expect(position + localOrigin[0]).toBe(10_000_001);
});

it('FLT-07 translated streamlines retain quarter-metre steps in their published local frame', () => {
  const result = computeStreamlinesOnCPU(
    {
      velocities: Float32Array.of(1, 0),
      parameters: getGPUStreamlinesParameterValues({
        fieldPlacement,
        gridPlacement: fieldPlacement,
        localOrigin,
        stepLength: 0.25
      }),
      uint32Parameters: getGPUStreamlinesUint32ParameterValues({seed: 8})
    },
    {
      fieldWidth: 1,
      fieldHeight: 1,
      gridWidth: 1,
      gridHeight: 1,
      seedColumns: 1,
      seedRows: 1,
      stepsPerDirection: 8,
      lineCapacity: 1,
      pointCapacity: 17
    }
  );
  expect(result.count[0]).toBe(1);
  expect(result.positionCount[0]).toBe(17);
  expectCoordinateStreamline(result.positions);
});

it('FLT-10 LIC differences the two grid origins before narrowing, retaining its twelve-word layout', async () => {
  const {getGPULineIntegralConvolutionParameterValues} = await import(
    '../../../src/raster-analysis/flow-texture/line-integral-convolution-parameters'
  );
  for (const origin of [0, 10_000_000]) {
    const values = getGPULineIntegralConvolutionParameterValues({
      fieldPlacement: {origin: [origin + 0.25, origin - 0.5], cellSize: [1, -1]},
      outputPlacement: {origin: [origin, origin], cellSize: [1, -1]},
      localOrigin: [origin, origin],
      stepLength: 0.25
    });
    expect(Array.from(values)).toEqual([0.25, -0.5, 1, -1, 0, 0, 1, -1, 0.25, 0, 0, 0]);
  }
});

it('FLT-03 rebases absolute instants and preserves duration; rejects accidentally aliased slices', () => {
  const settings = {
    fieldPlacement: {origin: [0, 0], cellSize: [100, 100]} as const,
    startTime: 1_700_000_000,
    timeStep: 10,
    sliceStartTime: 1_700_000_000,
    sliceEndTime: 1_700_000_010
  };
  expect(
    Array.from(
      getGPULagrangianFlowMapParameterValues({...settings, timeOrigin: settings.sliceStartTime})
    )
  ).toEqual([0, 0, 100, 100, 0, 10, 0, 10]);
  expect(() => getGPULagrangianFlowMapParameterValues(settings)).toThrow(
    /GPULagrangianFlowMap.*sliceStartTime.*sliceEndTime.*timeOrigin/
  );
  expect(() =>
    getGPULagrangianFlowMapParameterValues({...settings, timeOrigin: -1_700_000_000})
  ).toThrow(/timeOrigin/);
  expect(
    Array.from(
      getGPULagrangianFlowMapParameterValues({
        ...settings,
        sliceEndTime: settings.sliceStartTime
      }).subarray(6)
    )
  ).toEqual([1_700_000_000, 1_700_000_000]);
});

it('PCK-11 rejects a nonfinite encoded reciprocal instead of a successful empty vertical line', () => {
  const settings = {
    fieldPlacement: {origin: [0, 0], cellSize: [1, 100]} as const,
    gridPlacement: {origin: [0, 0], cellSize: [1, 100]} as const,
    stepLength: 0.25
  };
  const options = {
    fieldWidth: 1,
    fieldHeight: 1,
    gridWidth: 1,
    gridHeight: 1,
    seedColumns: 1,
    seedRows: 1,
    stepsPerDirection: 1,
    lineCapacity: 1,
    pointCapacity: 3
  };
  const uint32Parameters = getGPUStreamlinesUint32ParameterValues({seed: 8, minimumPoints: 1});
  const parameters = getGPUStreamlinesParameterValues(settings);
  const control = computeStreamlinesOnCPU(
    {velocities: Float32Array.of(0, 1), parameters, uint32Parameters},
    options
  );
  expect(Array.from(control.count)).toEqual([1]);
  expect(Array.from(control.positionCount)).toEqual([3]);
  expect(() =>
    getGPUStreamlinesParameterValues({
      ...settings,
      gridPlacement: {origin: [0, 0], cellSize: [1e-40, 100]}
    })
  ).toThrow(/GPUStreamlines.*reciprocal.*finite.*received/);
  // A resident packet can bypass the packer: the CPU form must reject its invalid word too.
  parameters[2] = parameters[6] = 1e-40;
  parameters[8] = Infinity;
  expect(() =>
    computeStreamlinesOnCPU(
      {velocities: Float32Array.of(0, 1), parameters, uint32Parameters},
      options
    )
  ).toThrow(/computeStreamlinesOnCPU.*reciprocal/);
});

it('OUT-09 repeats the terminal offset through every unused CSR row', () => {
  const result = computeStreamlinesOnCPU(
    {
      velocities: Float32Array.of(3, 0),
      parameters: getGPUStreamlinesParameterValues({
        fieldPlacement: {origin: [0, 0], cellSize: [10, 10]},
        gridPlacement: {origin: [0, 0], cellSize: [10, 10]},
        stepLength: 0.125
      }),
      uint32Parameters: getGPUStreamlinesUint32ParameterValues({seed: 7})
    },
    {
      fieldWidth: 1,
      fieldHeight: 1,
      gridWidth: 1,
      gridHeight: 1,
      seedColumns: 4,
      seedRows: 1,
      stepsPerDirection: 2,
      lineCapacity: 4,
      pointCapacity: 20
    }
  );
  // All four seeds compete for one occupancy cell. The single surviving line has 2+1+2 points.
  expect(result.count[0]).toBe(1);
  expect(result.positionCount[0]).toBe(5);
  expect(Array.from(result.lineOffsets)).toEqual([0, 5, 5, 5, 5]);
});

it('NAM-04 rejects each retired placement prop and preserves signed placement words', async () => {
  const {getGPULineIntegralConvolutionParameterValues} = await import(
    '../../../src/raster-analysis/flow-texture/line-integral-convolution-parameters'
  );
  const fieldPlacement = {origin: [100, 220], cellSize: [10, -10]} as const;
  const gridPlacement = {origin: [100, 200], cellSize: [10, 10]} as const;
  const times = {startTime: 0, timeStep: 0.25, sliceStartTime: 0, sliceEndTime: 1};
  expect(Array.from(getGPULagrangianFlowMapParameterValues({...times, fieldPlacement}))).toEqual([
    100, 220, 10, -10, 0, 0.25, 0, 1
  ]);
  const retiredField = {fieldPlacement, fieldExtent: [100, 220, 10, -10]};
  for (const pack of [
    () => getGPULagrangianFlowMapParameterValues({...times, ...retiredField}),
    () => getGPUParticleAdvectionParameterValues({...retiredField, timeStep: 1}, [2, 2]),
    () =>
      getGPULineIntegralConvolutionParameterValues({
        ...retiredField,
        outputPlacement: fieldPlacement
      }),
    () => getGPUStreamlinesParameterValues({...retiredField, gridPlacement, stepLength: 1})
  ]) {
    expect(pack).toThrow(/fieldExtent.*removed.*use fieldPlacement/);
  }
  const retiredGrid = {
    fieldPlacement,
    gridPlacement,
    gridExtent: [100, 200, 10, 10],
    stepLength: 1
  };
  expect(() => getGPUStreamlinesParameterValues(retiredGrid)).toThrow(
    /gridExtent.*removed.*use gridPlacement/
  );
  const retiredOutput = {
    fieldPlacement,
    outputPlacement: fieldPlacement,
    outputExtent: [100, 220, 10, -10]
  };
  expect(() => getGPULineIntegralConvolutionParameterValues(retiredOutput)).toThrow(
    /outputExtent.*removed.*use outputPlacement/
  );
});

it('DEF-15 uint32 packers accept zero arguments with byte-identical defaults', async () => {
  const {getGPULineIntegralConvolutionUint32ParameterValues} = await import(
    '../../../src/raster-analysis/flow-texture/line-integral-convolution-parameters'
  );
  expect(new Uint8Array(getGPULineIntegralConvolutionUint32ParameterValues().buffer)).toEqual(
    new Uint8Array(getGPULineIntegralConvolutionUint32ParameterValues({}).buffer)
  );
  expect(new Uint8Array(getGPUStreamlinesUint32ParameterValues().buffer)).toEqual(
    new Uint8Array(getGPUStreamlinesUint32ParameterValues({}).buffer)
  );
});
