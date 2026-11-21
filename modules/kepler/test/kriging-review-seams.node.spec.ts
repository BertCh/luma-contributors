// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {NullDevice} from '@luma.gl/core/test-utils';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {
  getKrigingSheetSettings,
  makeKrigingSheetDefaults
} from '../../../examples/deck/kepler-compute/src/analysis-kriging-panel';
import {getAnalysisFormInitialization} from '../../../examples/deck/kepler-compute/src/analysis-form-initialization';
import {
  readFloat32Vector,
  suggestKrigingVariogram
} from '../../../examples/deck/kepler-compute/src/kriging-diagnostics';
import {
  parseKeplerKrigingSettings,
  restoreKeplerAnalysisRecipe,
  serializeKeplerAnalysisRecipe,
  type KeplerAnalysisRecipe
} from '../src/recipes/analysis-recipe';

const common = {
  longitudeColumn: 'longitude',
  latitudeColumn: 'latitude',
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters',
  longitudeSearch: 'wrapped',
  grid: {bounds: [-179.99, 10, -179.98, 11], width: 2, height: 2},
  neighborCount: 4
};
const kriging = parseKeplerKrigingSettings({
  ...common,
  searchRadiusMeters: 30000,
  minimumNeighbors: 3,
  variogram: {model: 'exponential', nugget: 0.1, sill: 1, rangeMeters: 30000}
});
const initialization = getAnalysisFormInitialization(undefined, ['longitude', 'latitude', 'value']);

test('restored wrapped kriging sheet retains its longitude choice when applying the draft', () => {
  const recipe: KeplerAnalysisRecipe = {
    schemaVersion: 1,
    algorithmVersion: 1,
    id: 'wrapped-kriging',
    source: {datasetId: 'stations', revision: 1},
    population: 'all',
    backendPolicy: 'require-gpu',
    operation: {kind: 'kriging-points', column: 'value', ...kriging}
  };
  const restored = restoreKeplerAnalysisRecipe(serializeKeplerAnalysisRecipe(recipe), {
    resolveDatasetRevision: () => 1
  });
  if (restored.operation.kind !== 'kriging-points') throw new Error('Expected kriging');
  const defaults = makeKrigingSheetDefaults(restored.operation, initialization);
  const applied = getKrigingSheetSettings(defaults);
  expect(applied.longitudeSearch).toBe('wrapped');
  expect(applied.grid).toEqual(kriging.grid);
  expect(
    serializeKeplerAnalysisRecipe({
      ...recipe,
      operation: {
        kind: 'kriging-points',
        column: 'value',
        ...applied
      }
    })
  ).toBe(serializeKeplerAnalysisRecipe(recipe));
});

test('wrapped WLS suggestions refuse clipped empirical pair discovery while the translated control fits all 15 pairs', async () => {
  const device = new NullDevice({});
  const buffers: ReturnType<NullDevice['createBuffer']>[] = [];
  const vector = (values: number[]) => {
    const data = Float32Array.from(values);
    const buffer = device.createBuffer({data});
    buffers.push(buffer);
    vi.spyOn(buffer, 'readAsync').mockResolvedValue(new Uint8Array(data.buffer));
    return new GPUVector({
      type: 'data',
      name: 'fit-source',
      data: [new GPUData({format: 'float32', length: values.length, buffer})]
    });
  };
  try {
    const latitude = vector([0, 0, 0, 0, 0, 0]);
    const values = vector([1, 2, 4, 8, 16, 32]);
    const longitude = vector([179.91, 179.96, 179.99, -179.99, -179.96, -179.91]);
    expect(Array.from(await readFloat32Vector(values))).toEqual([1, 2, 4, 8, 16, 32]);
    // The farthest equatorial separation is .18 degrees = 20015.1 m < 30000 m.
    // Six sites therefore require C(6,2)=15 pairs; a clipped search used only 6.
    await expect(
      suggestKrigingVariogram({longitude, latitude, values}, kriging, {
        maximumLagMeters: 30000
      })
    ).rejects.toThrow(/wrapped.*manual variogram/);
    const control = await suggestKrigingVariogram(
      {longitude: vector([-0.09, -0.04, -0.01, 0.01, 0.04, 0.09]), latitude, values},
      {...kriging, longitudeSearch: 'clipped'},
      {maximumLagMeters: 30000}
    );
    expect(control.accumulatedPairCount).toBe(15);
    expect(control.exhaustive).toBe(true);
    expect(Number.isFinite(control.fit.rangeMeters)).toBe(true);
  } finally {
    buffers.forEach(buffer => buffer.destroy());
    device.destroy();
  }
});
