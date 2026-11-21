// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {describe, expect, it} from 'vitest';
import {
  GPUScalarFieldRidges,
  getGPUScalarFieldRidgesParameterValues,
  GPULagrangianFlowMap,
  getGPULagrangianFlowMapParameterValues,
  GPUFiniteTimeLyapunovExponent,
  getGPUFiniteTimeLyapunovExponentParameterValues
} from '../../../src/raster-analysis/lagrangian-diagnostics/index';
import fixture from '../../oracles/fixtures/flow-field-extensions-mob06-numpy.json';
import {GraphHarness} from '../../gpu-spatial-analysis/trajectory-staypoints/movement-spec-utils';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

describe('MOB-06 scalar ridges (NumPy eigensolve oracle)', () => {
  for (const reference of fixture.ridgeCases) {
    it(reference.name, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = new GraphHarness(device, `ridge-${reference.name}`);
      const count = reference.width * reference.height;
      const grid: GPURasterMetadata = {
        width: reference.width,
        height: reference.height,
        affine: [reference.spacing[0], 0, 0, 0, reference.spacing[1], 0],
        pixelInterpretation: 'point'
      };
      const values = harness.input(
        'values',
        Float32Array.from(reference.values, value => value ?? NaN),
        'float32',
        count
      );
      const inputValidity = harness.input(
        'input-validity',
        Uint32Array.from(reference.inputValidity),
        'uint32',
        count
      );
      const parameters = harness.input(
        'parameters',
        getGPUScalarFieldRidgesParameterValues(reference, grid),
        'float32',
        8
      );
      const ridges = harness.output('ridges', 'uint32', count);
      const validity = harness.output('validity', 'uint32', count);
      const strength = harness.output('strength', 'float32', count);
      harness.graph.add(
        new GPUScalarFieldRidges({
          grid,
          input: {
            id: 'density',
            format: 'float32',
            storage: {kind: 'buffer', values},
            validity: inputValidity,
            scale: reference.scale,
            offset: reference.offset
          },
          parameters,
          output: {ridges: ridges.view, validity: validity.view, strength: strength.view}
        })
      );
      const actual = await harness.run(async () => ({
        ridges: await readUint32(ridges.buffer, count),
        validity: await readUint32(validity.buffer, count),
        strength: await readFloat32(strength.buffer, count)
      }));
      expect(actual.ridges).toEqual(reference.ridges);
      expect(actual.validity).toEqual(reference.validity);
      actual.strength.forEach((value, index) => {
        const expected = reference.strength[index];
        if (expected === 0) {
          expect(value).toBe(0);
        } else {
          // Relative error is essential: an absolute tolerance would let zero strength pass
          // for the small-curvature regressions (2^-26 and about 1e-25).
          expect(Math.abs(value / expected - 1)).toBeLessThan(2e-5);
        }
      });
    });
  }

  it.each([
    'geographic',
    'web-mercator'
  ] as const)('uses row-local ground scales in %s mode', async cellSizeMode => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new GraphHarness(device, `ridge-ground-${cellSizeMode}`);
    // At the equator one degree is ~111319.49 m east-west; a 111319.49-m Mercator cell has
    // the same ground width. f = 20 - column^2 has downward curvature 2 / width^2.
    const cellWidth = cellSizeMode === 'geographic' ? 1 : 111319.49079327357;
    const grid: GPURasterMetadata = {
      width: 5,
      height: 3,
      affine: [cellWidth, 0, 0, 0, 1, -1],
      pixelInterpretation: 'point'
    };
    const values = harness.input(
      'values',
      Float32Array.from(Array.from({length: 15}, (_, index) => 20 - ((index % 5) - 2) ** 2)),
      'float32',
      15
    );
    const parameters = harness.input(
      'parameters',
      getGPUScalarFieldRidgesParameterValues({}, grid),
      'float32',
      8
    );
    const ridges = harness.output('ridges', 'uint32', 15);
    const validity = harness.output('validity', 'uint32', 15);
    const strength = harness.output('strength', 'float32', 15);
    harness.graph.add(
      new GPUScalarFieldRidges({
        grid,
        cellSizeMode,
        spatialContext:
          cellSizeMode === 'geographic'
            ? {coordinateSpace: 'longitude-latitude', metric: 'none'}
            : undefined,
        input: {id: 'density', format: 'float32', storage: {kind: 'buffer', values}},
        parameters,
        output: {ridges: ridges.view, validity: validity.view, strength: strength.view}
      })
    );
    const actual = await harness.run(async () => ({
      ridges: await readUint32(ridges.buffer, 15),
      strength: await readFloat32(strength.buffer, 15)
    }));
    expect(actual.ridges).toEqual([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]);
    expect(Math.abs(actual.strength[7] - 2 / 111319.49079327357 ** 2)).toBeLessThan(1e-15);
  });
});

describe('MOB-06 geographic flow (NumPy analytic projection oracle)', () => {
  for (const reference of fixture.flowCases) {
    it(`integrates ${reference.timeStep > 0 ? 'forward' : 'backward'} with regions, validity and padded dispatch`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = new GraphHarness(device, `geographic-flow-${reference.timeStep}`);
      const count = reference.seeds.length / 2;
      const seeds = harness.input('seeds', Float32Array.from(reference.seeds), 'float32x2', count);
      const velocities = (factor: number) =>
        Float32Array.from(
          Array.from({length: 128}, (_, index) => (index % 2 ? 100 : 300) * factor)
        );
      const start = harness.input('start', velocities(1), 'float32x2', 64);
      const end = harness.input('end', velocities(3), 'float32x2', 64);
      const regions = harness.input(
        'regions',
        Uint32Array.from(Array.from({length: 64}, (_, index) => (index % 8 < 4 ? 0 : 1))),
        'uint32',
        64
      );
      const parameters = harness.input(
        'parameters',
        getGPULagrangianFlowMapParameterValues({
          fieldPlacement: {origin: [0, 40], cellSize: [1, 1]},
          startTime: reference.startTime,
          timeStep: reference.timeStep,
          sliceStartTime: 0,
          sliceEndTime: 200
        }),
        'float32',
        8
      );
      const positions = harness.output('positions', 'float32x2', count);
      const validity = harness.output('validity', 'uint32', count);
      const classes = harness.output('classes', 'uint32', 2 * count);
      const residence = harness.output('residence', 'float32', count);
      harness.graph.add(
        new GPULagrangianFlowMap({
          seeds,
          velocitiesStart: start,
          velocitiesEnd: end,
          fieldWidth: 8,
          fieldHeight: 8,
          stepCount: reference.stepCount,
          parameters,
          regions,
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'},
          output: {
            positions: positions.view,
            validity: validity.view,
            regionClasses: classes.view,
            residenceTimes: residence.view
          }
        })
      );
      const actual = await harness.run(async () => ({
        positions: await readFloat32(positions.buffer, 2 * count),
        validity: await readUint32(validity.buffer, count),
        classes: await readUint32(classes.buffer, 2 * count),
        residence: await readFloat32(residence.buffer, count)
      }));
      expect(actual.validity).toEqual(reference.validity);
      expect(actual.classes).toEqual(reference.regionClasses);
      expect(actual.residence).toEqual(reference.residence);
      actual.positions.forEach((value, index) => {
        const expected = reference.positions[index];
        if (expected === null) {
          expect(Number.isNaN(value)).toBe(true);
        } else {
          expect(Math.abs(value - expected)).toBeLessThan(3e-5);
        }
      });
    });
  }
});

it('MOB-06 FTLE projects geographic positions and rejects missing centers and projection domains', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new GraphHarness(device, 'geographic-ftle');
  const width = 7;
  const count = width * width;
  // A regular grid of projected seeds with a linear final stretch of 2 along x. The exact
  // FTLE for T=1 is log(2); inverse Mercator here is an independent closed-form oracle.
  const positions = Float32Array.from(
    Array.from({length: 2 * count}, (_, index) => {
      const particle = Math.floor(index / 2);
      const column = (particle % width) - 3;
      const row = Math.floor(particle / width) - 3;
      return index % 2
        ? (Math.atan(Math.sinh((row * 1000) / 6378137)) * 180) / Math.PI
        : (((2 * column * 1000) / 6378137) * 180) / Math.PI;
    })
  );
  // A masked center and an invalid projection domain exercise both invalidity routes.
  const flags = new Uint32Array(count).fill(1);
  flags[3 * width + 3] = 0;
  positions[1] = 86;
  const positionView = harness.input('positions', positions, 'float32x2', count);
  const inputValidity = harness.input('flags', flags, 'uint32', count);
  const parameters = harness.input(
    'parameters',
    getGPUFiniteTimeLyapunovExponentParameterValues({
      seedSpacingX: 1000,
      seedSpacingY: 1000,
      integrationTime: 1
    }),
    'float32',
    4
  );
  const ftle = harness.output('ftle', 'float32', count);
  const validity = harness.output('validity', 'uint32', count);
  harness.graph.add(
    new GPUFiniteTimeLyapunovExponent({
      positions: positionView,
      validity: inputValidity,
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'},
      seedWidth: width,
      seedHeight: width,
      parameters,
      output: {ftle: ftle.view, validity: validity.view}
    })
  );
  const actual = await harness.run(async () => ({
    ftle: await readFloat32(ftle.buffer, count),
    validity: await readUint32(validity.buffer, count)
  }));
  for (let row = 0; row < width; row++) {
    for (let column = 0; column < width; column++) {
      const index = row * width + column;
      const stencil = [
        index,
        row * width + Math.max(0, column - 1),
        row * width + Math.min(width - 1, column + 1),
        Math.max(0, row - 1) * width + column,
        Math.min(width - 1, row + 1) * width + column
      ];
      const expectedValid = !stencil.some(particle => particle === 0 || particle === 3 * width + 3);
      expect(actual.validity[index]).toBe(Number(expectedValid));
      if (expectedValid) {
        expect(Math.abs(actual.ftle[index] - Math.log(2))).toBeLessThan(2e-5);
      } else {
        expect(Number.isNaN(actual.ftle[index])).toBe(true);
      }
    }
  }
});
