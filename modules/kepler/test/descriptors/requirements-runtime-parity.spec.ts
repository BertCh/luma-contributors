// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {requireTestDevice} from '../../../analysis/test/utils/require-test-device';
import {getCatalogBinding, runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {evaluateKeplerOperationApplicability} from '../../src/descriptors/applicability';
import {resolveKeplerOperationDescriptor} from '../../src/descriptors/operation-descriptors';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';

// The negative local G fixture must reach the real shader's invalid-count readback. The other
// two fixtures exercise the GPU-only adapters' own guards with a real device supplied, so a
// missing-device error cannot stand in for the expected data rejection.
const GPU_REJECTIONS = [
  {
    bindingId: 'global-join-counts-getis-ord',
    defect: 'three features',
    values: [1, 2, 3],
    requirement: 'minimum-rows',
    reason: /3.*4/,
    remedy: /at least four/i,
    runtimeError: /Too few features or links/
  },
  {
    bindingId: 'local-lisa-variants',
    defect: 'constant values',
    values: [7, 7, 7, 7],
    requirement: 'varying-values',
    reason: /values must differ/,
    remedy: /values differ/i,
    runtimeError: /Too few or constant values/
  },
  {
    bindingId: 'local-lisa-variants',
    defect: 'negative value for local Getis-Ord G*',
    values: [-1, 2, 3, 4],
    requirement: 'non-negative-values',
    reason: /no value may be below 0/i,
    remedy: /remove or shift|bivariate Moran/i,
    runtimeError: /Getis-Ord local G requires non-negative values/
  }
] as const;

for (const fixture of GPU_REJECTIONS) {
  test(`AX-13 ${fixture.bindingId}: preflight and GPU reject ${fixture.defect}`, async () => {
    const binding = getCatalogBinding(fixture.bindingId)!;
    const {descriptor, source} = resolveKeplerOperationDescriptor(binding);
    const collection: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: fixture.values.map((value, index) => ({
        type: 'Feature',
        geometry: {type: 'Point', coordinates: [index % 2, Math.floor(index / 2)]},
        properties: {value}
      }))
    };
    const parameters = {
      coordinateSystem: 'planar',
      valueProperty: 'value',
      neighborhood: 'knn',
      neighborCount: 2,
      ...(binding.id === 'local-lisa-variants'
        ? {variant: 'local-getis-ord-g-star', permutations: 9}
        : {})
    };
    const profile = profileKeplerFeatureCollection(collection, {
      datasetId: 'features',
      label: 'features',
      coordinateSystem: 'planar',
      declarations: {rowSemantics: 'objects', measurement: {value: 'amount'}}
    });
    // Preflight has only the profile and settings, and runs before acquiring the GPU device.
    const assessment = evaluateKeplerOperationApplicability(binding, descriptor, [profile], {
      executionEngine: 'gpu',
      inputAssignments: {features: profile.datasetId},
      fieldBindings: {valueProperty: 'value'},
      overrides: parameters
    });
    const report = JSON.stringify({bindingId: binding.id, defect: fixture.defect, assessment});
    expect(source, report).toBe('hand-written');
    expect(binding.engine, report).toBe('gpu');
    expect(binding.run, report).toBeUndefined();
    expect(assessment.status, report).toBe('blocked');
    expect(assessment.requirements, report).toContainEqual(
      expect.objectContaining({
        id: fixture.requirement,
        input: 'features',
        state: 'failed',
        severity: 'block',
        evidence: 'profile',
        message: expect.stringMatching(fixture.reason),
        remedy: expect.stringMatching(fixture.remedy)
      })
    );
    expect(
      assessment.missing.some(
        entry => fixture.reason.test(entry.what) && fixture.remedy.test(entry.remedy ?? '')
      ),
      report
    ).toBe(true);
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // Invoke the real runtime independently of the blocked applicability result. GPU-only
    // bindings cannot silently fall back to a CPU oracle; the expected error must propagate.
    await expect(
      runCatalogOperationAsync(binding, parameters, {features: collection}, {device}),
      report
    ).rejects.toThrow(fixture.runtimeError);
  });
}
