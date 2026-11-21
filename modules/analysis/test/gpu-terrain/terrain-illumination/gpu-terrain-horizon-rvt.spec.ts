// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUTerrainHorizonParameterValues,
  GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH,
  GPUTerrainHorizon,
  type GPUTerrainHorizonAlgorithm
} from '../../../src/gpu-terrain/terrain-illumination/gpu-terrain-horizon';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import skyViewFixture from '../../oracles/fixtures/terrain-sky-view-rvt.json';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// GPUTerrainHorizon against unmodified rvt-py (rvt.vis.sky_view_factor): the fixture is written by
// `test/oracles/terrain-sky-view-rvt.py`. It pins the anisotropic sky-view direction convention
// (review finding terrain-surface-10) and the sky-view factor and openness reductions. Nothing
// here models the kernel: every expected number comes from rvt.

type AnisotropicReference = {
  /** rvt `asvf_dir` / `a_main_direction`. */
  mainDirection: number;
  /** rvt `a_poly_level`. */
  level: number;
  /** rvt `a_min_weight`. */
  minimumWeight: number;
  /** rvt `asvf` on the interior pixels, row-major. */
  interior: number[];
};
type SkyViewCase = {
  name: string;
  elevationModel: string;
  directionCount: number;
  /** North-up raster, row 0 in the north. */
  elevation: number[];
  skyViewFactor: number[];
  positiveOpenness: number[];
  negativeOpenness: number[];
  anisotropic: AnisotropicReference[];
};
type SkyViewFixture = {
  versions: Record<string, string>;
  width: number;
  height: number;
  resolution: number;
  maximumRadius: number;
  /** Rows and columns `[first, end)` farther than `maximumRadius` from every edge. */
  interior: {first: number; end: number};
  /** Per wall: the rvt main directions that give the lowest and highest centre `asvf`. */
  conventionProbe: {
    wallSide: string;
    compassAzimuthOfWall: number;
    heaviestMainDirection: number;
    lightestMainDirection: number;
  }[];
  cases: SkyViewCase[];
};

const FIXTURE = decodeOracleFixture<SkyViewFixture>(skyViewFixture);
const {width: WIDTH, height: HEIGHT} = FIXTURE;
const PIXEL_COUNT = WIDTH * HEIGHT;
const INTERIOR_SIZE = FIXTURE.interior.end - FIXTURE.interior.first;
const CENTER_PIXEL = Math.floor(HEIGHT / 2) * WIDTH + Math.floor(WIDTH / 2);
/** Row 0 is the northern row, as in the fixture: `affine[4] < 0`. */
const NORTH_UP_GRID = makeGPURasterMetadata({
  width: WIDTH,
  height: HEIGHT,
  cellSize: FIXTURE.resolution
});
/** Row 0 is the southern row: `affine[4] > 0`. */
const ROWS_NORTH_GRID = makeGPURasterMetadata({
  width: WIDTH,
  height: HEIGHT,
  cellSize: FIXTURE.resolution,
  rowDirection: 'north'
});

/** Sky-view factors are compared in float32 units of a value near 1. */
const FACTOR_TOLERANCE = 2e-6;
/** Openness in degrees: float32 `atan` and a sum of up to 8 angles near 90. */
const OPENNESS_TOLERANCE_DEGREES = 2e-4;

type SkyViewResult = {
  skyViewFactor: number[];
  positiveOpenness: number[];
  negativeOpenness: number[];
  /** One whole-grid array per entry of the case's `anisotropic` list. */
  anisotropic: number[][];
};

function getCase(name: string): SkyViewCase {
  const found = FIXTURE.cases.find(entry => entry.name === name);
  if (!found) {
    throw new Error(`terrain-sky-view-rvt.json has no case ${name}`);
  }
  return found;
}

/** Reverses the row order of a row-major grid. */
function flipRows(values: ArrayLike<number>): number[] {
  const flipped = new Array<number>(PIXEL_COUNT);
  for (let row = 0; row < HEIGHT; row++) {
    for (let column = 0; column < WIDTH; column++) {
      flipped[(HEIGHT - 1 - row) * WIDTH + column] = values[row * WIDTH + column];
    }
  }
  return flipped;
}

/** Row-major values of the pixels farther than the search radius from every edge. */
function cropInterior(values: ArrayLike<number>): number[] {
  const {first, end} = FIXTURE.interior;
  const interior: number[] = [];
  for (let row = first; row < end; row++) {
    for (let column = first; column < end; column++) {
      interior.push(values[row * WIDTH + column]);
    }
  }
  return interior;
}

function getMaximumDifference(actual: ArrayLike<number>, expected: ArrayLike<number>): number {
  expect(actual.length).toBe(expected.length);
  let maximum = 0;
  for (let index = 0; index < expected.length; index++) {
    const difference = Math.abs(actual[index] - expected[index]);
    // A NaN on either side must not pass as agreement.
    maximum = Number.isNaN(difference) ? Infinity : Math.max(maximum, difference);
  }
  return maximum;
}

/**
 * Runs the contributor with rvt's settings: z factor 1, no curvature, one-pixel steps, the
 * fixture's radius and cell size. With `rowsNorth` the samples are stored south row first on a
 * rows-north grid and the outputs are flipped back, so results are always north row first.
 * `rowDirection` is never passed: north comes from the grid.
 */
async function runSkyView(
  device: Device,
  fixtureCase: SkyViewCase,
  algorithm: GPUTerrainHorizonAlgorithm,
  rowsNorth: boolean = false
): Promise<SkyViewResult> {
  const grid = rowsNorth ? ROWS_NORTH_GRID : NORTH_UP_GRID;
  const orient = (values: ArrayLike<number>): number[] =>
    rowsNorth ? flipRows(values) : Array.from(values);
  const graph = new GPUCommandGraph(device, {id: `rvt-sky-view-${fixtureCase.name}-${algorithm}`});
  const elevationBuffer = createInputBuffer(
    device,
    Float32Array.from(orient(fixtureCase.elevation))
  );
  const outputs = {
    skyViewFactor: createOutputBuffer(device, PIXEL_COUNT),
    positiveOpenness: createOutputBuffer(device, PIXEL_COUNT),
    negativeOpenness: createOutputBuffer(device, PIXEL_COUNT),
    anisotropic: createOutputBuffer(device, PIXEL_COUNT)
  };
  const settings = new GPUParameterBuffer(device, {
    id: 'horizon-parameters',
    format: 'float32',
    length: GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH
  });
  graph.add(
    new GPUTerrainHorizon({
      grid,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', PIXEL_COUNT)
        }
      },
      parameters: settings.importToGraph(graph),
      directionCount: fixtureCase.directionCount,
      maximumRadius: FIXTURE.maximumRadius,
      algorithm,
      output: {
        skyViewFactor: importGraphBuffer(
          graph,
          'sky-view-factor',
          outputs.skyViewFactor,
          'float32',
          PIXEL_COUNT
        ),
        positiveOpenness: importGraphBuffer(
          graph,
          'positive-openness',
          outputs.positiveOpenness,
          'float32',
          PIXEL_COUNT
        ),
        negativeOpenness: importGraphBuffer(
          graph,
          'negative-openness',
          outputs.negativeOpenness,
          'float32',
          PIXEL_COUNT
        ),
        anisotropicSkyViewFactor: importGraphBuffer(
          graph,
          'anisotropic',
          outputs.anisotropic,
          'float32',
          PIXEL_COUNT
        )
      }
    })
  );
  const compiled = graph.compile();
  const anisotropic: number[][] = [];
  // The anisotropy settings are per-frame parameters: one compiled graph serves them all.
  for (const reference of fixtureCase.anisotropic) {
    settings.write(
      getGPUTerrainHorizonParameterValues(
        {
          rvtDirectionDegrees: reference.mainDirection,
          anisotropyLevel: reference.level,
          anisotropyMinimumWeight: reference.minimumWeight
        },
        grid
      )
    );
    submitGraph(device, compiled, undefined);
    anisotropic.push(orient(await readFloat32(outputs.anisotropic, PIXEL_COUNT)));
  }
  const result: SkyViewResult = {
    skyViewFactor: orient(await readFloat32(outputs.skyViewFactor, PIXEL_COUNT)),
    positiveOpenness: orient(await readFloat32(outputs.positiveOpenness, PIXEL_COUNT)),
    negativeOpenness: orient(await readFloat32(outputs.negativeOpenness, PIXEL_COUNT)),
    anisotropic
  };
  compiled.destroy();
  settings.destroy();
  elevationBuffer.destroy();
  for (const buffer of Object.values(outputs)) {
    buffer.destroy();
  }
  return result;
}

/** Largest interior difference from rvt per output. */
function compareInterior(
  actual: SkyViewResult,
  fixtureCase: SkyViewCase
): {
  skyViewFactor: number;
  positiveOpenness: number;
  negativeOpenness: number;
  anisotropic: number;
} {
  return {
    skyViewFactor: getMaximumDifference(
      cropInterior(actual.skyViewFactor),
      cropInterior(fixtureCase.skyViewFactor)
    ),
    positiveOpenness: getMaximumDifference(
      cropInterior(actual.positiveOpenness),
      cropInterior(fixtureCase.positiveOpenness)
    ),
    negativeOpenness: getMaximumDifference(
      cropInterior(actual.negativeOpenness),
      cropInterior(fixtureCase.negativeOpenness)
    ),
    anisotropic: Math.max(
      ...fixtureCase.anisotropic.map((reference, index) =>
        getMaximumDifference(cropInterior(actual.anisotropic[index]), reference.interior)
      )
    )
  };
}

function expectInteriorEqualsReference(
  actual: SkyViewResult,
  fixtureCase: SkyViewCase,
  label: string
): void {
  const difference = compareInterior(actual, fixtureCase);
  expect(difference.skyViewFactor, `${label} skyViewFactor`).toBeLessThan(FACTOR_TOLERANCE);
  expect(difference.anisotropic, `${label} anisotropicSkyViewFactor`).toBeLessThan(
    FACTOR_TOLERANCE
  );
  expect(difference.positiveOpenness, `${label} positiveOpenness`).toBeLessThan(
    OPENNESS_TOLERANCE_DEGREES
  );
  expect(difference.negativeOpenness, `${label} negativeOpenness`).toBeLessThan(
    OPENNESS_TOLERANCE_DEGREES
  );
}

/**
 * Cases where rvt and the kernel read the same samples, so the comparison is exact up to float32:
 * rvt samples whole pixels on the axes and on the diagonals. The march reads the same pixels on
 * the axes (4 directions) and anywhere on a plane (bilinear is exact there); the sweep reads pixel
 * centres on the diagonals too.
 */
const SAME_SAMPLE_CASES: {name: string; algorithms: GPUTerrainHorizonAlgorithm[]}[] = [
  ...['north', 'east', 'south', 'west', 'north-east'].flatMap(side => [
    {name: `wall-${side}-4`, algorithms: ['march', 'sweep'] as GPUTerrainHorizonAlgorithm[]},
    {name: `wall-${side}-8`, algorithms: ['sweep'] as GPUTerrainHorizonAlgorithm[]},
    {name: `ramp-${side}-8`, algorithms: ['march', 'sweep'] as GPUTerrainHorizonAlgorithm[]}
  ]),
  {name: 'rough-4', algorithms: ['march', 'sweep']},
  {name: 'rough-8', algorithms: ['sweep']}
];

it('rvt fixture records the versions that made it and rvt direction angles counterclockwise from north', () => {
  expect(FIXTURE.versions['rvt-py']).toBe('2.2.1');
  // The wall's compass azimuth t is weighted most by rvt main direction (360 - t) mod 360.
  expect(
    FIXTURE.conventionProbe.map(probe => [probe.wallSide, probe.heaviestMainDirection])
  ).toEqual([
    ['north', 0],
    ['west', 90],
    ['south', 180],
    ['east', 270],
    ['north-east', 315],
    ['north-north-east', 337.5]
  ]);
  for (const probe of FIXTURE.conventionProbe) {
    expect(probe.heaviestMainDirection).toBe((360 - probe.compassAzimuthOfWall) % 360);
    expect(probe.lightestMainDirection).toBe((probe.heaviestMainDirection + 180) % 360);
  }
});

it('GPUTerrainHorizon anisotropic sky-view factor weights the same side of the sky as rvt', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const probe of FIXTURE.conventionProbe) {
    // Eight directions for the compass and diagonal walls, sixteen (the default of both rvt and
    // the contributor) for the wall at compass azimuth 22.5.
    const fixtureCase = getCase(
      `wall-${probe.wallSide}-${probe.wallSide === 'north-north-east' ? 16 : 8}`
    );
    for (const algorithm of ['march', 'sweep'] as const) {
      const actual = await runSkyView(device, fixtureCase, algorithm);
      // Among rvt's low preset (level 4), the main direction that darkens the centre pixel most
      // is the one that looks at the wall, and the opposite one darkens it least. Off the axes
      // the march and rvt sample differently, so only the ranking is compared here;
      // `SAME_SAMPLE_CASES` compares values.
      const centreValues = fixtureCase.anisotropic
        .map((reference, index) => ({
          mainDirection: reference.mainDirection,
          level: reference.level,
          value: actual.anisotropic[index][CENTER_PIXEL]
        }))
        .filter(entry => entry.level === 4)
        .sort((first, second) => first.value - second.value);
      const label = `${fixtureCase.name} ${algorithm}`;
      expect(centreValues[0].mainDirection, label).toBe(probe.heaviestMainDirection);
      expect(centreValues[centreValues.length - 1].mainDirection, label).toBe(
        probe.lightestMainDirection
      );
      // The wall must matter: the centre value moves by more than 0.08 across main directions.
      expect(centreValues[centreValues.length - 1].value - centreValues[0].value).toBeGreaterThan(
        0.08
      );
    }
  }
});

it('GPUTerrainHorizon sky-view factor, anisotropic sky-view factor and openness equal rvt where the samples coincide', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const {name, algorithms} of SAME_SAMPLE_CASES) {
    const fixtureCase = getCase(name);
    for (const algorithm of algorithms) {
      const actual = await runSkyView(device, fixtureCase, algorithm);
      expectInteriorEqualsReference(actual, fixtureCase, `${name} ${algorithm}`);
    }
  }
});

it('GPUTerrainHorizon equals rvt on a rows-north grid with rowDirection left out', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The same terrain stored south row first. A kernel that took north from the row index instead
  // of the grid would weight the mirrored side: rvt direction 315 would act as 225.
  for (const {name, algorithm} of [
    {name: 'wall-north-east-4', algorithm: 'march'},
    {name: 'wall-north-4', algorithm: 'march'},
    {name: 'rough-8', algorithm: 'sweep'}
  ] as const) {
    const fixtureCase = getCase(name);
    const actual = await runSkyView(device, fixtureCase, algorithm, true);
    expectInteriorEqualsReference(actual, fixtureCase, `${name} ${algorithm} rows-north`);
  }
});

it('GPUTerrainHorizon march equals the closed form on a plane at 16 directions, where rvt follows a rounded staircase', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A plane of gradient 0.5 rising toward compass azimuth 45 is its own horizon:
  // tan(h) = 0.5 * cos(t - 45 degrees) at any distance, so the sky-view factor is
  // 1 - mean(sin(max(h, 0))) with sin(atan(x)) = x / sqrt(1 + x^2), and openness is 90 because
  // h is antisymmetric.
  const fixtureCase = getCase('ramp-north-east-16');
  let sineSum = 0;
  for (let sector = 0; sector < 16; sector++) {
    const tangent = 0.5 * Math.cos(((sector * 22.5 - 45) * Math.PI) / 180);
    sineSum += Math.max(tangent, 0) / Math.hypot(1, tangent);
  }
  const expectedSkyViewFactor = 1 - sineSum / 16;
  const interiorCount = INTERIOR_SIZE * INTERIOR_SIZE;
  const closedForm = {
    skyViewFactor: new Array<number>(interiorCount).fill(expectedSkyViewFactor),
    openness: new Array<number>(interiorCount).fill(90)
  };
  const march = await runSkyView(device, fixtureCase, 'march');
  expect(
    getMaximumDifference(cropInterior(march.skyViewFactor), closedForm.skyViewFactor)
  ).toBeLessThan(FACTOR_TOLERANCE);
  expect(
    getMaximumDifference(cropInterior(march.positiveOpenness), closedForm.openness)
  ).toBeLessThan(OPENNESS_TOLERANCE_DEGREES);
  expect(
    getMaximumDifference(cropInterior(march.negativeOpenness), closedForm.openness)
  ).toBeLessThan(OPENNESS_TOLERANCE_DEGREES);
  // rvt rounds each of the eight directions between the axes and diagonals to whole pixels and
  // keeps the steepest of them, so on this plane it reads a lower sky-view factor and openness
  // than the closed form. This is the size of the sampling difference from rvt, not a defect.
  const referenceSkyViewFactorDifference = getMaximumDifference(
    cropInterior(fixtureCase.skyViewFactor),
    closedForm.skyViewFactor
  );
  expect(referenceSkyViewFactorDifference).toBeGreaterThan(0.02);
  expect(referenceSkyViewFactorDifference).toBeLessThan(0.023);
  const referenceOpennessDifference = getMaximumDifference(
    cropInterior(fixtureCase.positiveOpenness),
    closedForm.openness
  );
  expect(referenceOpennessDifference).toBeGreaterThan(3.2);
  expect(referenceOpennessDifference).toBeLessThan(3.4);
  // The sweep also reads whole pixels (a digital line, distances along the ideal ray): measured
  // 0.019 and 2.58 degrees from the closed form here, inside rvt's 0.0217 and 3.32 degrees.
  const sweep = await runSkyView(device, fixtureCase, 'sweep');
  expect(
    getMaximumDifference(cropInterior(sweep.skyViewFactor), closedForm.skyViewFactor)
  ).toBeLessThan(referenceSkyViewFactorDifference);
  expect(
    getMaximumDifference(cropInterior(sweep.positiveOpenness), closedForm.openness)
  ).toBeLessThan(referenceOpennessDifference);
});
