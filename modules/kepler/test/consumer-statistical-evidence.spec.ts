// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {requireTestDevice} from '../../analysis/src/testing/require-test-device';
import {getCatalogBinding, runCatalogOperationAsync} from '../src/analysis/catalog-bindings';
import {getKeplerOperationDescriptor} from '../src/descriptors/operation-descriptors';
import {describeKeplerCatalogResult} from '../src/presentation/result-report';

test('local Moran reports the executed GPU or CPU fallback construction at 19 draws', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const binding = getCatalogBinding('local-moran-statistics')!;
  const descriptor = getKeplerOperationDescriptor(binding.id)!;
  const points = {
    type: 'FeatureCollection' as const,
    features: Array.from({length: 35}, (_, index) => ({
      geometry: {type: 'Point' as const, coordinates: [index, 0]},
      properties: {value: index ** 2}
    }))
  };
  // The GPU permutation route supports at most 32 neighbors. A request for 34 executes the CPU
  // fallback, which counts absolute-statistic exceedances once rather than doubling min tails.
  for (const neighborCount of [2, 34]) {
    const result = await runCatalogOperationAsync(
      binding,
      {
        valueProperty: 'value',
        coordinateSystem: 'planar',
        neighborhood: 'knn',
        neighborCount,
        permutations: 19,
        seed: 42,
        significance: 0.025,
        multipleTesting: 'none'
      },
      {features: points},
      {device}
    );
    expect(result.engine).toBe(neighborCount === 2 ? 'gpu' : 'cpu-reference');
    expect(result.summary['pValueConstruction']).toBe(
      neighborCount === 2 ? 'doubled-minimum-tail' : 'upper-tail-count'
    );
    if (neighborCount === 2) {
      // The host table reconstructs exact rational p-values from the executed producer's
      // integer exceedances. Float32 0.1 (0.10000000149011612) must not survive this boundary.
      const table = result.tables.find(table => table.id === 'local-moran')!;
      for (const value of table.columns['p_value']) {
        expect(typeof value).toBe('number');
        expect(Number.isFinite(value)).toBe(true);
        expect(value).toBe(Math.round((value as number) * 10) / 10);
      }
    }
    const report = describeKeplerCatalogResult(result, binding, {
      ...descriptor,
      outputs: descriptor.outputs.map(output => ({
        ...output,
        ...(output.presentation
          ? {presentation: {...output.presentation, classEncoding: 'catalog-text' as const}}
          : {})
      }))
    });
    expect(report.cautions.find(caution => caution.id === 'pseudo-p-values')?.text).toContain(
      neighborCount === 2 ? 'below 0.1' : 'below 0.05'
    );
  }
});
