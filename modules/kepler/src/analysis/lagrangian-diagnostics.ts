// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  GPULagrangianFlowMap,
  GPUFiniteTimeLyapunovExponent,
  getGPULagrangianFlowMapParameterValues,
  getGPUFiniteTimeLyapunovExponentParameterValues
} from '@luma-contributors/analysis/raster-analysis';
import {withCatalogGpuSession} from './catalog-gpu-session';
import {getCatalogNumberProblem} from './catalog-parameter-constraints';

/** Cell-centered planar velocity field in metres/second; row zero south. Host owns reprojection. */
export type KeplerLagrangianField = {
  width: number;
  height: number;
  extent: readonly [number, number, number, number];
  velocitiesStart: Float32Array;
  velocitiesEnd?: Float32Array;
};

/** Integrates a regular inset seed lattice, then computes FTLE in the same metre frame. */
export async function runKeplerLagrangianDiagnostics(
  device: Device,
  field: KeplerLagrangianField,
  options: {
    timeStep: number;
    stepCount: number;
    /** Absolute integration start in the slice time unit; defaults to absolute 0 for forward integration or -duration for backward integration. */
    startTime?: number;
    /** Absolute slice times (seconds for metre/second fields), rebased internally before float32 packing. */
    sliceStartTime?: number;
    /** Absolute end of the second slice; defaults to Math.abs(stepCount * timeStep). */
    sliceEndTime?: number;
  }
) {
  const {width, height, extent} = field;
  const duration = options.stepCount * options.timeStep;
  const sliceStartTime = options.sliceStartTime ?? 0;
  const sliceEndTime = options.sliceEndTime ?? Math.abs(duration);
  if (
    getCatalogNumberProblem(
      {nonzero: true, nonzeroFloat32Scale: options.stepCount},
      options.timeStep
    ) !== null ||
    !Number.isInteger(options.stepCount) ||
    options.stepCount < 1 ||
    options.stepCount > 8192 ||
    !Number.isFinite(duration) ||
    Math.fround(duration) === 0 ||
    !(sliceEndTime > sliceStartTime)
  )
    throw new Error(
      'Flow diagnostics need a finite nonzero time step, 1–8192 integration steps and increasing slice times'
    );
  if (
    ![width, height].every(value => Number.isInteger(value) && value >= 4 && value <= 512) ||
    field.velocitiesStart.length !== width * height * 2 ||
    (field.velocitiesEnd && field.velocitiesEnd.length !== width * height * 2) ||
    !extent.every(Number.isFinite) ||
    extent[2] <= 0 ||
    extent[3] <= 0
  )
    throw new Error(
      'Lagrangian diagnostics require a 4–512 cell axis and two velocity components per cell, with positive metre spacing'
    );
  const seedWidth = width - 2;
  const seedHeight = height - 2;
  const count = seedWidth * seedHeight;
  const seeds = new Float32Array(count * 2);
  for (let row = 0; row < seedHeight; row++)
    for (let column = 0; column < seedWidth; column++) {
      const index = row * seedWidth + column;
      seeds[index * 2] = extent[0] + (column + 1.5) * extent[2];
      seeds[index * 2 + 1] = extent[1] + (row + 1.5) * extent[3];
    }
  return withCatalogGpuSession(device, 'kepler-lagrangian', async session => {
    const positions = session.output('positions', 'float32x2', count);
    const flowValidity = session.output('flow-validity', 'uint32', count);
    const ftle = session.output('ftle', 'float32', count);
    const validity = session.output('validity', 'uint32', count);
    const startTime = options.startTime ?? (duration < 0 ? -duration : 0);
    session.graph.add(
      new GPULagrangianFlowMap({
        fieldWidth: width,
        fieldHeight: height,
        stepCount: options.stepCount,
        seeds: session.input('seeds', seeds, 'float32x2'),
        velocitiesStart: session.input('velocity-start', field.velocitiesStart, 'float32x2'),
        velocitiesEnd: session.input(
          'velocity-end',
          field.velocitiesEnd ?? field.velocitiesStart,
          'float32x2'
        ),
        parameters: session.input(
          'flow-parameters',
          getGPULagrangianFlowMapParameterValues({
            fieldPlacement: {origin: [extent[0], extent[1]], cellSize: [extent[2], extent[3]]},
            startTime,
            timeStep: options.timeStep,
            sliceStartTime,
            sliceEndTime,
            timeOrigin: sliceStartTime
          }),
          'float32'
        ),
        output: {positions: positions.view, validity: flowValidity.view}
      })
    );
    session.graph.add(
      new GPUFiniteTimeLyapunovExponent({
        seedWidth,
        seedHeight,
        positions: positions.view,
        validity: flowValidity.view,
        parameters: session.input(
          'ftle-parameters',
          getGPUFiniteTimeLyapunovExponentParameterValues({
            seedSpacingX: extent[2],
            seedSpacingY: extent[3],
            integrationTime: duration
          }),
          'float32'
        ),
        output: {ftle: ftle.view, validity: validity.view}
      })
    );
    session.run();
    return {
      width: seedWidth,
      height: seedHeight,
      seeds,
      positions: await session.readFloat32(positions),
      flowValidity: await session.readUint32(flowValidity),
      ftle: await session.readFloat32(ftle),
      validity: await session.readUint32(validity),
      contributors: ['GPULagrangianFlowMap', 'GPUFiniteTimeLyapunovExponent']
    };
  });
}
