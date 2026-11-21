// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as networkModule from '@luma-contributors/analysis/gpu-network';
import * as terrainModule from '@luma-contributors/analysis/gpu-terrain';
import * as dataframeModule from '@luma-contributors/analysis/dataframe-analysis';
import * as rasterModule from '@luma-contributors/analysis/raster-analysis';
import * as analysisModule from '@luma-contributors/analysis/gpu-spatial-analysis';
import {validateGPUAnalysisResultEncoding} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {describe, expect, it} from 'vitest';
import type {CatalogRunResult} from '../../src/analysis/catalog-binding-types';
import {
  getCatalogBinding,
  getCatalogDefaultParameters,
  KEPLER_CATALOG_BINDINGS,
  normalizeCatalogParameters
} from '../../src/analysis/catalog-bindings';
import {KEPLER_CURATED_BINDINGS} from '../../src/curated/bindings/index';
import {evaluateKeplerOperationApplicability} from '../../src/descriptors/applicability';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import {CURATED_OPERATION_DESCRIPTORS} from '../../src/descriptors/descriptors-curated';
import {validateKeplerOperationDescriptor} from '../../src/descriptors/validate-operation-descriptor';
import {
  CORRELOGRAM_LATTICE,
  DBSCAN_PLANAR,
  KRIGING_POINTS,
  RATE_LATTICE,
  REGRESSION_LATTICE
} from './curated-bindings-oracle-data';
import {makeLattice, makePoints, runCuratedBinding} from './curated-bindings-test-support';

const EXPECTED_IDS = [
  'curated-rate-cluster-map',
  'curated-excess-risk',
  'curated-clusters-and-footprints',
  'curated-regression-with-diagnostics',
  'curated-kriging-with-diagnostics',
  'curated-contiguity-correlogram',
  'curated-emerging-hot-spots',
  'curated-dasymetric-redistribution',
  'curated-track-quality',
  'curated-landscape-metrics',
  'curated-drainage-network',
  'curated-surface-preparation',
  'curated-zonal-exposure',
  'curated-regional-raster-change',
  'curated-network-disruption',
  'curated-cell-flow-topology',
  'curated-drive-time-bands',
  'curated-fleet-dwell'
];

describe('KEPLER_CURATED_BINDINGS', () => {
  it('lists the curated bindings, registers each in the catalog, and keeps ids unique across the catalog', () => {
    expect(KEPLER_CURATED_BINDINGS.map(binding => binding.id)).toEqual(EXPECTED_IDS);
    for (const binding of KEPLER_CURATED_BINDINGS)
      expect(getCatalogBinding(binding.id)).toBe(binding);
    const ids = KEPLER_CATALOG_BINDINGS.map(binding => binding.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('declares the CPU reference engine only, with honest declarations', () => {
    for (const binding of KEPLER_CURATED_BINDINGS.filter(
      binding => binding.engine === 'cpu-reference'
    )) {
      expect(binding.engine, binding.id).toBe('cpu-reference');
      expect(binding.runGpu).toBeUndefined();
      expect(binding.run).toBeTypeOf('function');
      expect(binding.gpuContributorNames).toEqual([]);
      expect(binding.capabilityIds.length).toBeGreaterThan(0);
      expect(binding.contributorNames.length).toBeGreaterThan(0);
      expect(binding.declaredBehavior.length).toBeGreaterThanOrEqual(4);
      expect(binding.limitation).toMatch(
        /CPU reference engine only; the GPU path .* is future work/
      );
      expect(binding.outputs.every(output => output.identityColumns.length > 0)).toBe(true);
      expect(binding.inputs.length).toBeGreaterThan(0);
      expect(() => getCatalogDefaultParameters(binding)).not.toThrow();
      // The defaults normalise unchanged, except that column parameters are empty until filled.
      expect(() =>
        normalizeCatalogParameters(binding, getCatalogDefaultParameters(binding))
      ).not.toThrow();
    }
  });

  it('names contributors that the analysis package exports', () => {
    const exports = {
      ...analysisModule,
      ...dataframeModule,
      ...rasterModule,
      ...networkModule,
      ...terrainModule
    } as Record<string, unknown>;
    const missing: string[] = [];
    for (const binding of KEPLER_CURATED_BINDINGS)
      for (const name of binding.contributorNames)
        if (typeof exports[name] !== 'function') missing.push(`${binding.id}:${name}`);
    expect(missing).toEqual([]);
  });
});

describe('CURATED_OPERATION_DESCRIPTORS', () => {
  it('has one descriptor per binding, in order, and each validates against its binding', () => {
    expect(CURATED_OPERATION_DESCRIPTORS.map(descriptor => descriptor.bindingId)).toEqual(
      EXPECTED_IDS.filter(
        id =>
          ![
            'curated-emerging-hot-spots',
            'curated-dasymetric-redistribution',
            'curated-track-quality',
            'curated-landscape-metrics',
            'curated-fleet-dwell',
            'curated-drainage-network',
            'curated-surface-preparation',
            'curated-zonal-exposure',
            'curated-regional-raster-change',
            'curated-network-disruption',
            'curated-cell-flow-topology',
            'curated-drive-time-bands'
          ].includes(id)
      )
    );
    for (const descriptor of CURATED_OPERATION_DESCRIPTORS) {
      const binding = getCatalogBinding(descriptor.bindingId)!;
      expect(validateKeplerOperationDescriptor(descriptor, binding)).toEqual([]);
    }
  });

  it('fails validation when a descriptor is made wrong', () => {
    const descriptor = CURATED_OPERATION_DESCRIPTORS[0];
    const binding = getCatalogBinding(descriptor.bindingId)!;
    expect(validateKeplerOperationDescriptor({...descriptor, action: ''}, binding)).not.toEqual([]);
    expect(
      validateKeplerOperationDescriptor(
        {...descriptor, parameters: descriptor.parameters.slice(1)},
        binding
      )
    ).not.toEqual([]);
    expect(validateKeplerOperationDescriptor({...descriptor, outputs: []}, binding)).not.toEqual(
      []
    );
  });

  it('names presentation columns that the real tables have, with an encoding the real values follow', () => {
    const results: Record<string, CatalogRunResult> = {
      'curated-rate-cluster-map': runCuratedBinding(
        'curated-rate-cluster-map',
        {
          casesProperty: 'cases',
          populationProperty: 'population',
          coordinateSystem: 'planar',
          permutations: 99,
          significance: 0.3
        },
        {
          areas: makeLattice(5, 5, index => ({
            cases: RATE_LATTICE.cases[index],
            population: RATE_LATTICE.population[index]
          }))
        }
      ),
      'curated-excess-risk': runCuratedBinding(
        'curated-excess-risk',
        {casesProperty: 'cases', populationProperty: 'population', coordinateSystem: 'planar'},
        {
          areas: makeLattice(5, 5, index => ({
            cases: RATE_LATTICE.cases[index],
            population: RATE_LATTICE.population[index]
          }))
        }
      ),
      'curated-clusters-and-footprints': runCuratedBinding(
        'curated-clusters-and-footprints',
        {
          coordinateSystem: 'planar',
          epsilon: DBSCAN_PLANAR.epsilon,
          minimumSamples: DBSCAN_PLANAR.minimumSamples
        },
        {
          points: makePoints(
            DBSCAN_PLANAR.points.map(p => p[0]),
            DBSCAN_PLANAR.points.map(p => p[1])
          )
        }
      ),
      'curated-regression-with-diagnostics': runCuratedBinding(
        'curated-regression-with-diagnostics',
        {
          responseProperty: 'y',
          predictorProperties: 'x1,x2',
          coordinateSystem: 'planar',
          permutations: 99
        },
        {
          areas: makeLattice(6, 6, index => ({
            y: REGRESSION_LATTICE.y[index],
            x1: REGRESSION_LATTICE.x1[index],
            x2: REGRESSION_LATTICE.x2[index]
          }))
        }
      ),
      'curated-kriging-with-diagnostics': runCuratedBinding(
        'curated-kriging-with-diagnostics',
        {
          valueProperty: 'z',
          coordinateSystem: 'planar',
          cellSize: 200,
          variogram: 'stated',
          model: 'spherical',
          range: 600,
          sill: 4
        },
        {
          points: makePoints(KRIGING_POINTS.x, KRIGING_POINTS.y, index => ({
            z: KRIGING_POINTS.z[index]
          }))
        }
      ),
      'curated-contiguity-correlogram': runCuratedBinding(
        'curated-contiguity-correlogram',
        {valueProperty: 'v', coordinateSystem: 'planar', permutations: 99, maximumOrder: 3},
        {areas: makeLattice(8, 8, index => ({v: CORRELOGRAM_LATTICE.values[index]}))}
      )
    };
    for (const descriptor of CURATED_OPERATION_DESCRIPTORS) {
      const result = results[descriptor.bindingId];
      for (const output of descriptor.outputs) {
        const table = result.tables.find(candidate => candidate.id === output.tableId);
        expect(table, `${descriptor.bindingId} ${output.tableId}`).toBeDefined();
        if (!output.presentation) continue;
        const column = table!.columns[output.presentation.column];
        expect(
          column,
          `${descriptor.bindingId} column ${output.presentation.column}`
        ).toBeDefined();
        if (output.presentation.scale === 'class-scheme')
          expect(
            validateGPUAnalysisResultEncoding(
              output.presentation.classScheme!,
              output.presentation.classEncoding!,
              column
            ),
            `${descriptor.bindingId} ${output.presentation.column}`
          ).toBe(1);
      }
    }
  });
});

it('describes conditional kriging constancy, geometry provenance and regression row counts', () => {
  const kriging = CURATED_OPERATION_DESCRIPTORS.find(
    descriptor => descriptor.bindingId === 'curated-kriging-with-diagnostics'
  )!;
  expect(
    kriging.requirements.find(requirement => requirement.id === 'varying-values')?.activeWhen
  ).toEqual({parameter: 'variogram', equals: ['fit']});
  const regression = CURATED_OPERATION_DESCRIPTORS.find(
    descriptor => descriptor.bindingId === 'curated-regression-with-diagnostics'
  )!;
  expect(
    regression.requirements.find(
      requirement => requirement.id === 'minimum-rows' && requirement.severity === 'block'
    )?.remedy
  ).toContain('predictors plus four');
  expect(regression.outputs.find(output => output.tableId === 'residuals')?.kindFrom).toBe('areas');
  for (const descriptor of CURATED_OPERATION_DESCRIPTORS) {
    expect(descriptor.overlappingInputs ?? []).toEqual([]);
    for (const output of descriptor.outputs) {
      if (output.tableId === 'rate-clusters' || output.tableId === 'excess-risk')
        expect(output.kindFrom).toBe('areas');
      if (output.tableId === 'points' || output.tableId === 'cross-validation')
        expect(output.kindFrom).toBe('points');
    }
  }
});

it('explains inference differences and incomplete isotropic kriging screens without a guarantee', () => {
  const rate = CURATED_OPERATION_DESCRIPTORS.find(
    descriptor => descriptor.bindingId === 'curated-rate-cluster-map'
  )!;
  const rateBinding = KEPLER_CURATED_BINDINGS.find(binding => binding.id === rate.bindingId)!;
  expect(rate.notes!.join(' ')).toMatch(
    /differ materially on skewed data; either may give the smaller p-value/
  );
  expect(rateBinding.declaredBehavior.join(' ')).toMatch(
    /differ materially on skewed data; either may give the smaller p-value/
  );
  expect(rateBinding.declaredBehavior.join(' ')).toContain('Sidak over more than 51 tested areas');
  const kriging = CURATED_OPERATION_DESCRIPTORS.find(
    descriptor => descriptor.bindingId === 'curated-kriging-with-diagnostics'
  )!;
  expect(kriging.notes!.join(' ')).toMatch(/incomplete isotropic screening checks/);
  expect(kriging.notes!.join(' ')).toContain(
    'absence of a warning does not establish a good or identified fit'
  );
  expect(kriging.notes!.join(' ')).toContain('nugget can mask anisotropy');
  expect(kriging.notes!.join(' ')).toContain(
    'variance uses the squared units specified by the variogram'
  );
  expect(kriging.notes!.join(' ')).toContain(
    'calibration depends on how well that variogram describes the data'
  );
});

it('blocks constant measurements only for a fitted variogram, while a stated variogram executes', () => {
  const descriptor = CURATED_OPERATION_DESCRIPTORS.find(
    candidate => candidate.bindingId === 'curated-kriging-with-diagnostics'
  )!;
  const binding = getCatalogBinding(descriptor.bindingId)!;
  const points = makePoints([0, 1, 2, 3, 0, 1, 2, 3], [0, 0, 0, 0, 1, 1, 1, 1], () => ({value: 7}));
  const profile = profileKeplerFeatureCollection(points, {
    datasetId: 'constant',
    label: 'Constant',
    coordinateSystem: 'planar'
  });
  for (const variogram of ['fit', 'stated']) {
    const applicability = evaluateKeplerOperationApplicability(binding, descriptor, [profile], {
      fieldBindings: {valueProperty: 'value'},
      overrides: {variogram, model: 'spherical', cellSize: 1, sill: 3, range: 2}
    });
    const requirement = applicability.requirements.find(
      candidate => candidate.id === 'varying-values'
    );
    if (variogram === 'fit') expect(requirement?.state).toBe('failed');
    else {
      expect(requirement?.state).not.toBe('failed');
      expect(applicability.status).not.toBe('blocked');
      const result = runCuratedBinding(
        binding.id,
        {
          coordinateSystem: 'planar',
          valueProperty: 'value',
          variogram,
          model: 'spherical',
          cellSize: 1,
          sill: 3,
          range: 2
        },
        {points}
      );
      for (const value of result.tables.find(table => table.id === 'kriging-cells')!.columns[
        'value'
      ])
        expect(Number(value)).toBeCloseTo(7, 12);
    }
  }
});
