// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeSolarPositionOnCPU,
  CPUSolarPositionOperation
} from '../../../src/gpu-terrain/terrain-illumination/cpu-solar-position-operation';
import {
  GPUSolarPositionOperation,
  type SolarPositionOperationProps
} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-position-operation';
import {getGPUSolarPositionParameterValues} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-position';

const PARAMETERS = getGPUSolarPositionParameterValues({
  timestamp: Date.UTC(2003, 9, 17, 19, 30, 30)
});

it('keeps a north-seam azimuth inside the output range after float32 rounding', () => {
  const result = computeSolarPositionOnCPU(
    new Float32Array([0.48121488094329834, -30]),
    getGPUSolarPositionParameterValues({timestamp: Date.UTC(2024, 5, 21, 12)}),
    {refraction: false}
  );
  // Near local noon the June sun is north of this southern location: azimuth 0.
  // Declination is approximately +23.44 degrees, giving altitude 90 - 30 - 23.44.
  expect(result.azimuth[0]).toBe(0);
  expect(result.azimuth[0]).toBeGreaterThanOrEqual(0);
  expect(result.azimuth[0]).toBeLessThan(360);
  expect(result.altitude[0]).toBeCloseTo(36.56, 2);
  expect(result.daylight[0]).toBe(1);
});

it('agrees with the NREL SPA published example within NOAA model accuracy', () => {
  // NREL/TP-560-34302, Appendix A: https://www.nrel.gov/docs/fy08osti/34302.pdf
  // SPA report example, 2003-10-17 12:30:30 MST, longitude -105.1786, latitude 39.742476:
  // azimuth 194.34024°, zenith 50.11162° (pressure 820 mbar, temperature 11°C).
  // NOAA omits parallax and uses standard refraction; allow 0.02° model/atmosphere error.
  const result = computeSolarPositionOnCPU(new Float32Array([-105.1786, 39.742476]), PARAMETERS);
  expect(Math.abs(result.azimuth[0] - 194.34024)).toBeLessThan(0.02);
  expect(Math.abs(result.altitude[0] - (90 - 50.11162))).toBeLessThan(0.02);
  expect(Array.from(result.daylight)).toEqual([1]);
});

it('obeys antipodal geometric altitude symmetry and strict daylight thresholds', () => {
  const positions = new Float32Array([20, 35, 200, -35]);
  const result = computeSolarPositionOnCPU(positions, PARAMETERS, {refraction: false});
  expect(Math.abs(result.altitude[0] + result.altitude[1])).toBeLessThan(1e-5);
  const allDay = computeSolarPositionOnCPU(
    positions,
    new Float32Array([PARAMETERS[0], PARAMETERS[1], -90, 0])
  );
  const noDay = computeSolarPositionOnCPU(
    positions,
    new Float32Array([PARAMETERS[0], PARAMETERS[1], 90, 0])
  );
  expect(Array.from(allDay.daylight)).toEqual([1, 1]);
  expect(Array.from(noDay.daylight)).toEqual([0, 0]);
  const invalid = computeSolarPositionOnCPU(
    new Float32Array([NaN, 0, 0, 91, Infinity, 0]),
    PARAMETERS
  );
  expect(invalid.azimuth.every(Number.isNaN)).toBe(true);
  expect(invalid.altitude.every(Number.isNaN)).toBe(true);
  expect(Array.from(invalid.daylight)).toEqual([0, 0, 0]);
});

it('validates array shapes and both operation declarations at construction', () => {
  const props: SolarPositionOperationProps = {
    id: 'solar-validation',
    positions: 'positions',
    parameters: 'parameters',
    rowCount: 1,
    output: {altitude: 'altitude'}
  };
  for (const Operation of [CPUSolarPositionOperation, GPUSolarPositionOperation]) {
    expect(() => new Operation({...props, rowCount: 0})).toThrow(/^solar-validation/);
    expect(() => new Operation({...props, output: {altitude: 'positions'}})).toThrow(
      /^solar-validation/
    );
  }
  expect(() => Reflect.construct(CPUSolarPositionOperation, [{...props, output: {}}])).toThrow(
    /solar-validation.*output/
  );
  expect(() => computeSolarPositionOnCPU(new Float32Array(1), PARAMETERS)).toThrow(/positions/);
  expect(() => computeSolarPositionOnCPU(new Float32Array(2), new Float32Array(3))).toThrow(
    /parameters/
  );
});
