// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {runInNewContext} from 'node:vm';
import {expect, it} from 'vitest';
import {calibrateRasterOperationBand} from '../../../src/raster-analysis/raster-grid/raster-operation-band';
import {computeRasterWatershedSegmentationOnCPU} from '../../../src/raster-analysis/raster-watershed/cpu-raster-watershed-segmentation-operation';
import {computeRasterWalkingCostOnCPU} from '../../../src/raster-analysis/cost-distance/cpu-raster-walking-cost-operation';
import {getGPURasterWalkingCostParameterValues} from '../../../src/raster-analysis/gpu-raster-walking-cost';
import {computeRasterFillHolesOnCPU} from '../../../src/raster-analysis/raster-morphology/cpu-raster-fill-holes-operation';
import {computeRasterReconstructionOnCPU} from '../../../src/raster-analysis/raster-morphology/cpu-raster-reconstruction-operation';
import {computeRasterFractionalZonalStatisticsOnCPU} from '../../../src/raster-analysis/raster-zonal-statistics/cpu-raster-fractional-zonal-statistics-operation';

const grid = {
  width: 1,
  height: 1,
  affine: [1, 0, 0, 0, -1, 1] as const,
  pixelInterpretation: 'area' as const
};
const band = {id: 'sample', format: 'float32' as const, values: Float32Array.of(7)};
const coverage = {
  zoneIds: Uint32Array.of(0),
  cellIds: Uint32Array.of(0),
  fractions: Float32Array.of(1),
  zoneValidity: Uint32Array.of(1),
  count: Uint32Array.of(1),
  requiredCount: Uint32Array.of(1),
  overflow: Uint32Array.of(0),
  candidateOverflow: Uint32Array.of(0)
};

it('B4 rejects float validity before NaN can admit a calibrated sample', () => {
  const input = {...band, validity: Float32Array.of(NaN) as unknown as Uint32Array};
  expect(() => calibrateRasterOperationBand(input, 1)).toThrow(/validity.*Uint32Array/);
  // A usable one-cell band has calibrated value 7, including foreign-realm flags.
  expect([
    ...calibrateRasterOperationBand({...band, validity: runInNewContext('Uint32Array.of(1)')}, 1)
  ]).toEqual([7]);
});

it('B4 rejects wrong raw watershed, walking and morphology element brands', () => {
  const wrongMask = Float32Array.of(0.5) as unknown as Uint32Array;
  expect(() => computeRasterWatershedSegmentationOnCPU(band, wrongMask, {grid})).toThrow(
    /markers.*Uint32Array/
  );
  expect(() => computeRasterFillHolesOnCPU(wrongMask, {grid})).toThrow(/input.*Uint32Array/);
  expect(() =>
    computeRasterReconstructionOnCPU(
      Uint32Array.of(2) as unknown as Float32Array,
      Float32Array.of(2),
      {grid}
    )
  ).toThrow(/marker.*Float32Array/);
  expect(() =>
    computeRasterReconstructionOnCPU(
      Float32Array.of(2),
      Uint32Array.of(2) as unknown as Float32Array,
      {grid}
    )
  ).toThrow(/mask.*Float32Array/);
  const input = {
    elevation: band,
    friction: {...band, values: Float32Array.of(1)},
    sources: Uint32Array.of(1),
    parameters: getGPURasterWalkingCostParameterValues({}, grid)
  };
  expect(() => computeRasterWalkingCostOnCPU({...input, sources: wrongMask}, {grid})).toThrow(
    /sources.*Uint32Array/
  );
  expect(() =>
    computeRasterWalkingCostOnCPU(
      {...input, parameters: new Uint32Array(16) as unknown as Float32Array},
      {grid}
    )
  ).toThrow(/parameters.*Float32Array/);
});

for (const name of Object.keys(coverage) as (keyof typeof coverage)[]) {
  it(`B4 checks fractional coverage.${name} before aggregating`, () => {
    const invalid = {...coverage};
    Reflect.set(invalid, name, name === 'fractions' ? Uint32Array.of(1) : Float32Array.of(1));
    expect(() =>
      computeRasterFractionalZonalStatisticsOnCPU(
        {input: band, coverage: invalid},
        {grid, zoneCount: 1, quantity: 'extensive'}
      )
    ).toThrow(
      new RegExp(`coverage.${name}.*${name === 'fractions' ? 'Float32Array' : 'Uint32Array'}`)
    );
  });
}

it('B4 accepts corresponding foreign-realm raw arrays with literal one-cell results', () => {
  const integer = () => runInNewContext('Uint32Array.of(1)') as Uint32Array;
  const floating = () => runInNewContext('Float32Array.of(2)') as Float32Array;
  expect([...computeRasterWatershedSegmentationOnCPU(band, integer(), {grid}).labels]).toEqual([1]);
  expect([...computeRasterFillHolesOnCPU(integer(), {grid}).values]).toEqual([1]);
  expect([...computeRasterReconstructionOnCPU(floating(), floating(), {grid}).values]).toEqual([2]);
  const parameters = getGPURasterWalkingCostParameterValues({}, grid);
  const foreignParameters = runInNewContext(`Float32Array.of(${[...parameters]})`) as Float32Array;
  expect([
    ...computeRasterWalkingCostOnCPU(
      {
        elevation: band,
        friction: {...band, values: Float32Array.of(1)},
        sources: integer(),
        parameters: foreignParameters
      },
      {grid}
    ).costs
  ]).toEqual([0]);
  const foreignCoverage = Object.fromEntries(
    Object.entries(coverage).map(([name, values]) => [
      name,
      runInNewContext(`${values.constructor.name}.of(${[...values]})`)
    ])
  ) as typeof coverage;
  expect([
    ...computeRasterFractionalZonalStatisticsOnCPU(
      {input: band, coverage: foreignCoverage},
      {grid, zoneCount: 1, quantity: 'extensive'}
    ).totals
  ]).toEqual([7]);
});

for (const name of ['parameters', 'cellAreas'] as const) {
  it(`B4 rejects wrong ${name} scalar arrays`, () => {
    const input = {input: band, coverage, [name]: new Uint32Array(name === 'parameters' ? 8 : 1)};
    expect(() =>
      computeRasterFractionalZonalStatisticsOnCPU(input as never, {
        grid,
        zoneCount: 1,
        quantity: 'extensive'
      })
    ).toThrow(new RegExp(`${name}.*Float32Array`));
  });
}
