// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getCatalogBinding, runCatalogOperationAsync} from '../src/analysis/catalog-bindings';
import {getKeplerOperationDescriptor} from '../src/descriptors/operation-descriptors';
import {describeKeplerCatalogResult} from '../src/presentation/result-report';
import {profileKeplerColumnTable} from '../src/profile/dataset-profile';
import {buildAnalysisSet, runLocalMoran} from '../src/analysis/catalog-platform3-b-statistics';

const reviewPoints = {
  type: 'FeatureCollection' as const,
  features: Array.from({length: 6}, (_, index) => ({
    geometry: {type: 'Point' as const, coordinates: [index, 0]},
    properties: {value: index ** 2}
  }))
};
const reviewParameters = {
  valueProperty: 'value',
  coordinateSystem: 'planar' as const,
  neighborhood: 'knn' as const,
  neighborCount: 2,
  distanceBand: 10,
  rowStandardize: true,
  permutations: 19,
  seed: 42,
  significance: 0.1,
  multipleTesting: 'none' as const
};

test('local Moran classifies the rational GPU floor inclusively and preserves actual p-value provenance', () => {
  const set = buildAnalysisSet(reviewPoints, 'value', reviewParameters, true);
  const precomputed = {
    set,
    statistic: new Float64Array(6).fill(1),
    spatialLag: new Float64Array(6).fill(1),
    pValue: new Float32Array(6).fill(2 / 20),
    pValueExceedances: new Uint32Array(6),
    pValueConstruction: 'doubled-minimum-tail' as const
  };
  const atFloor = runLocalMoran(reviewPoints, reviewParameters, precomputed);
  expect(atFloor.tables[0].columns['p_value']).toEqual(new Array(6).fill(0.1));
  expect(atFloor.tables[0].columns['cluster']).not.toContain('not-significant');
  expect(
    runLocalMoran(reviewPoints, {...reviewParameters, significance: 0.099}, precomputed).tables[0]
      .columns['cluster']
  ).toEqual(new Array(6).fill('not-significant'));
});

test('local Moran provenance follows the generating branch, leaving imported conventions unknown', () => {
  const set = buildAnalysisSet(reviewPoints, 'value', reviewParameters, true);
  const values = {
    set,
    statistic: new Float64Array(6).fill(100),
    spatialLag: new Float64Array(6).fill(1)
  };
  const cpu = runLocalMoran(reviewPoints, reviewParameters, {
    ...values,
    pValueConstruction: 'doubled-minimum-tail'
  });
  expect(cpu.summary['pValueConstruction']).toBe('upper-tail-count');
  expect(cpu.tables[0].columns['p_value']).toEqual(new Array(6).fill(0.05));
  const imported = runLocalMoran(reviewPoints, reviewParameters, {
    ...values,
    pValue: new Float64Array(6).fill(0.2)
  });
  expect(imported.summary['pValueConstruction']).toBe('unknown');
});

test('retained GPU Moran reports recover dispatched construction and suppress unknown resolution', async () => {
  const binding = getCatalogBinding('local-moran-statistics')!;
  const result = await runCatalogOperationAsync(
    binding,
    {
      valueProperty: 'value',
      coordinateSystem: 'planar',
      neighborhood: 'knn',
      neighborCount: 2,
      permutations: 19,
      seed: 42,
      significance: 0.025,
      multipleTesting: 'none'
    },
    {features: reviewPoints},
    {engine: 'cpu'}
  );
  const descriptor = getKeplerOperationDescriptor(binding.id)!;
  const summary = {...result.summary};
  delete summary['pValueConstruction'];
  const report = (knownDispatch: boolean) =>
    describeKeplerCatalogResult(
      {
        ...result,
        engine: 'gpu',
        summary,
        dispatchedContributors: knownDispatch ? ['GPULocalMoran', 'GPULocalPermutationTest'] : []
      },
      binding,
      {
        ...descriptor,
        outputs: descriptor.outputs.map(output => ({
          ...output,
          ...(output.presentation
            ? {presentation: {...output.presentation, classEncoding: 'catalog-text' as const}}
            : {})
        }))
      }
    );
  const known = report(true);
  expect(known.cautions.find(caution => caution.id === 'pseudo-p-values')?.text).toContain(
    'below 0.1'
  );
  expect(known.cautions.find(caution => caution.id === 'pseudo-p-values')?.text).toContain('79');
  const unknown = report(false);
  expect(unknown.cautions.find(caution => caution.id === 'pseudo-p-values')?.text).not.toMatch(
    /1\s*\/|below 0\.05|39 permutations/
  );
  expect(unknown.cautions.some(caution => caution.id === 'significance-not-attainable')).toBe(
    false
  );
});

test('local Moran CPU fallback carries its single-count construction into the report', async () => {
  const binding = getCatalogBinding('local-moran-statistics')!;
  const result = await runCatalogOperationAsync(
    binding,
    {
      valueProperty: 'value',
      coordinateSystem: 'planar',
      neighborhood: 'knn',
      neighborCount: 2,
      permutations: 19,
      seed: 42,
      significance: 0.025,
      multipleTesting: 'none'
    },
    {
      features: {
        type: 'FeatureCollection',
        features: Array.from({length: 6}, (_, index) => ({
          geometry: {type: 'Point', coordinates: [index, 0]},
          properties: {value: index ** 2}
        }))
      }
    },
    {engine: 'cpu'}
  );
  expect(result.summary['pValueConstruction']).toBe('upper-tail-count');
  const descriptor = getKeplerOperationDescriptor(binding.id)!;
  const report = (pValueConstruction: string) =>
    describeKeplerCatalogResult(
      {...result, summary: {...result.summary, pValueConstruction}},
      binding,
      {
        ...descriptor,
        outputs: descriptor.outputs.map(output => ({
          ...output,
          ...(output.presentation
            ? {presentation: {...output.presentation, classEncoding: 'catalog-text' as const}}
            : {})
        }))
      }
    );
  // One absolute-statistic exceedance count has floor 1/20; doubled min-tail has floor 2/20.
  expect(
    report('upper-tail-count').cautions.find(caution => caution.id === 'pseudo-p-values')?.text
  ).toContain('below 0.05');
  expect(
    report('doubled-minimum-tail').cautions.find(caution => caution.id === 'pseudo-p-values')?.text
  ).toContain('below 0.1');
});

test('point-column spacing counts rejected source rows exactly once, independently of sampling', () => {
  const profile = profileKeplerColumnTable(
    {
      rowCount: 3,
      columns: [
        {name: 'x', values: [8, NaN, -3]},
        {name: 'y', values: [3, 6, 4]}
      ]
    },
    {
      datasetId: 'planar',
      label: 'planar',
      pointColumns: {longitude: 'x', latitude: 'y'},
      coordinateSystem: 'planar',
      maximumSampleSize: 2
    }
  );
  expect(profile.rowCount).toBe(3);
  expect(profile.spacing).toMatchObject({
    pointCount: 2,
    invalidCount: 1,
    sampleSize: 2,
    sampled: false
  });
  const sampled = profileKeplerColumnTable(
    {
      rowCount: 4,
      columns: [
        {name: 'x', values: [8, NaN, -3, 4]},
        {name: 'y', values: [3, 6, 4, 5]}
      ]
    },
    {
      datasetId: 'sampled',
      label: 'sampled',
      pointColumns: {longitude: 'x', latitude: 'y'},
      coordinateSystem: 'planar',
      maximumSampleSize: 2
    }
  );
  expect(sampled.rowCount).toBe(4);
  expect(sampled.spacing).toMatchObject({
    pointCount: 3,
    invalidCount: 1,
    sampleSize: 2,
    sampled: true
  });
});
