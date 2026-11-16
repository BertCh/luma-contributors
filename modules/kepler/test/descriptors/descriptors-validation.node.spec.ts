// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  getGPUVecchiaLogLikelihoodParameterValues,
  getGPUVecchiaSpaceTimeLogLikelihoodParameterValues,
  getGPUMGTWRParameterValues,
  type GPUVecchiaCovariance
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {CatalogParameterValue} from '../../src/analysis/catalog-binding-types';
import {
  getCatalogBinding,
  getCatalogDefaultParameters,
  normalizeCatalogParameters
} from '../../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {parseCatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {VALIDATION_BINDINGS} from '../../src/analysis/catalog-validation-bindings';
import {makeValidationResult} from '../../src/analysis/catalog-validation-common';
import {makeValidationSpaceTimeCovariance} from '../../src/analysis/catalog-validation-vecchia';
import {getCatalogSampleInputs} from '../../../../examples/deck/kepler-compute/src/analysis-catalog-input';
import {evaluateKeplerOperationApplicability} from '../../src/descriptors/applicability';
import {resolveKeplerOperationParameters} from '../../src/descriptors/default-parameters';
import {VALIDATION_OPERATION_DESCRIPTORS} from '../../src/descriptors/index';
import {resolveKeplerOperationDescriptor} from '../../src/descriptors/operation-descriptors';
import {validateKeplerOperationDescriptor} from '../../src/descriptors/validate-operation-descriptor';
import {planKeplerResultPresentation} from '../../src/presentation/result-presentation';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import type {KeplerDatasetProfile} from '../../src/profile/dataset-profile-types';

const VALIDATION_IDS = [
  'regression-gtwr',
  'regression-mgtwr',
  'validation-vecchia-likelihood',
  'validation-vecchia-fisher',
  'validation-vecchia-space-time',
  'interpolation-space-time-kriging',
  'validation-area-of-applicability',
  'validation-conformal',
  'validation-geo-conformal',
  'validation-gaussian-scores',
  'validation-ensemble-scores',
  'validation-fdr'
];

function getDescriptor(bindingId: string) {
  return VALIDATION_OPERATION_DESCRIPTORS.find(descriptor => descriptor.bindingId === bindingId)!;
}

function getSampleProfiles(bindingId: string) {
  const samples = getCatalogSampleInputs(bindingId);
  return Object.fromEntries(
    getCatalogBinding(bindingId)!.inputs.map(input => [
      input.name,
      profileKeplerFeatureCollection(parseCatalogFeatureCollection(samples[input.name]), {
        datasetId: input.name,
        label: input.name,
        coordinateSystem: 'planar',
        declarations: {coordinateModel: 'planar', rowSemantics: 'surface-samples'}
      })
    ])
  );
}

function assess(
  bindingId: string,
  profilesByInput: Record<string, KeplerDatasetProfile>,
  fieldBindings: Record<string, string | readonly string[]> = {},
  overrides: Record<string, CatalogParameterValue> = {}
) {
  const binding = getCatalogBinding(bindingId)!;
  return evaluateKeplerOperationApplicability(
    binding,
    getDescriptor(bindingId),
    Object.values(profilesByInput),
    {
      executionEngine: 'gpu',
      inputAssignments: Object.fromEntries(Object.keys(profilesByInput).map(name => [name, name])),
      overrides: {coordinateSystem: 'planar', ...overrides},
      fieldBindings
    }
  );
}

describe('validation descriptor registry and binding contract', () => {
  it('registers exactly the twelve audited bindings by hand', () => {
    expect(VALIDATION_OPERATION_DESCRIPTORS.map(descriptor => descriptor.bindingId).sort()).toEqual(
      [...VALIDATION_IDS].sort()
    );
    expect(VALIDATION_BINDINGS.map(binding => binding.id).sort()).toEqual(
      [...VALIDATION_IDS].sort()
    );
    for (const binding of VALIDATION_BINDINGS) {
      const resolved = resolveKeplerOperationDescriptor(binding);
      expect(resolved.source, binding.id).toBe('hand-written');
      expect(resolved.descriptor).toBe(getDescriptor(binding.id));
      expect(validateKeplerOperationDescriptor(resolved.descriptor, binding), binding.id).toEqual(
        []
      );
    }
  });

  for (const bindingId of VALIDATION_IDS) {
    it(`${bindingId}: maps every column string and resolves normalized settings`, () => {
      const binding = getCatalogBinding(bindingId)!;
      const descriptor = getDescriptor(bindingId);
      const fieldNames = new Set(
        descriptor.inputs.flatMap(input => input.fields.map(field => field.parameter))
      );
      expect(
        binding.parameters
          .filter(parameter => parameter.kind === 'string')
          .map(parameter => parameter.name)
          .sort()
      ).toEqual([...fieldNames].sort());
      expect(descriptor.parameters.every(parameter => !fieldNames.has(parameter.name))).toBe(true);
      expect(
        descriptor.parameters.every(parameter => ['live', 'rebuild'].includes(parameter.apply))
      ).toBe(true);
      expect(descriptor.outputs.map(output => output.tableId)).toEqual(
        binding.outputs.map(output => output.tableId)
      );
      expect(descriptor.outputs.every(output => Boolean(output.interpretation))).toBe(true);
      expect(
        descriptor.outputs
          .filter(output => output.kind !== 'summary')
          .every(output => Boolean(output.presentation))
      ).toBe(true);
      const fieldBindings = Object.fromEntries(
        binding.parameters
          .filter(parameter => parameter.kind === 'string')
          .map(parameter => [parameter.name, parameter.defaultValue as string])
      );
      for (const profiles of [{}, getSampleProfiles(bindingId)]) {
        const resolved = resolveKeplerOperationParameters(binding, descriptor, profiles, {
          fieldBindings
        });
        expect(normalizeCatalogParameters(binding, resolved.parameters)).toEqual(
          resolved.parameters
        );
        expect([...resolved.basic, ...resolved.advanced].sort()).toEqual(
          binding.parameters.map(parameter => parameter.name).sort()
        );
      }
    });

    it(`${bindingId}: refuses wrong geometry and over-capacity inputs before execution`, () => {
      const profiles = getSampleProfiles(bindingId);
      const firstInput = Object.keys(profiles)[0];
      const profile = profiles[firstInput];
      for (const incompatible of [
        {...profile, kind: 'polygons' as const, catalogGeometry: 'polygon' as const},
        {...profile, rowCount: 2049},
        {...profile, vertexCount: 2049}
      ]) {
        const assessment = assess(bindingId, {...profiles, [firstInput]: incompatible});
        expect(assessment.status, JSON.stringify(assessment.missing)).toBe('blocked');
        expect(
          assessment.missing.some(missing => missing.what.length > 0 && Boolean(missing.remedy))
        ).toBe(true);
      }
    });
  }
});

describe('validation roles keep training, calibration and query data separate', () => {
  function getFieldParameters(bindingId: string, input: string) {
    return getDescriptor(bindingId)
      .inputs.find(candidate => candidate.name === input)!
      .fields.map(field => field.parameter);
  }

  it('queries need neither observed responses nor observation error variances', () => {
    expect(getFieldParameters('interpolation-space-time-kriging', 'points')).toEqual([
      'valueProperty',
      'errorVarianceProperty',
      'timeProperty'
    ]);
    expect(getFieldParameters('interpolation-space-time-kriging', 'queries')).toEqual([
      'timeProperty'
    ]);
    for (const bindingId of ['validation-conformal', 'validation-geo-conformal']) {
      expect(getFieldParameters(bindingId, 'calibration')).toEqual([
        'observationProperty',
        'predictionProperty',
        'scaleProperty'
      ]);
      expect(getFieldParameters(bindingId, 'queries')).toEqual([
        'predictionProperty',
        'scaleProperty'
      ]);
      const descriptor = getDescriptor(bindingId);
      expect(
        descriptor.inputs.every(
          input => input.fields.find(field => field.role === 'scale')?.required === false
        )
      ).toBe(true);
      expect(descriptor.notes!.join(' ')).toMatch(/held out from model training/);
      expect(descriptor.overlappingInputs).toBeUndefined();
    }
  });

  it('requires the same predictor list on training and query inputs', () => {
    const descriptor = getDescriptor('validation-area-of-applicability');
    expect(
      descriptor.inputs.map(input => [
        input.name,
        input.fields[0].parameter,
        input.fields[0].multiple
      ])
    ).toEqual([
      ['training', 'predictorProperties', {minimum: 1, maximum: 16}],
      ['queries', 'predictorProperties', {minimum: 1, maximum: 16}]
    ]);
    const profiles = getSampleProfiles(descriptor.bindingId);
    const predictorName = getCatalogDefaultParameters(getCatalogBinding(descriptor.bindingId)!)[
      'predictorProperties'
    ] as string;
    const missingQuery = {
      ...profiles['queries'],
      columns: profiles['queries'].columns.filter(column => column.name !== predictorName)
    };
    const assessment = assess(
      descriptor.bindingId,
      {...profiles, queries: missingQuery},
      {predictorProperties: predictorName}
    );
    expect(assessment.status).toBe('blocked');
    expect(assessment.missing.map(missing => missing.what).join(' ')).toContain(predictorName);
  });

  it.each([
    'regression-gtwr',
    'regression-mgtwr',
    'validation-vecchia-space-time',
    'interpolation-space-time-kriging'
  ])('%s accepts numeric seconds rather than promising date conversion', bindingId => {
    for (const input of getDescriptor(bindingId).inputs) {
      const time = input.fields.find(field => field.role === 'time')!;
      expect(time.accepts).toEqual(['number', 'integer']);
      expect(time.required).toBe(true);
    }
    const notes = getDescriptor(bindingId).notes!.join(' ');
    expect(notes).toContain('subtracted before Float32 rounding');
    expect(notes).toContain('not milliseconds or date strings');
  });

  it('gives explicit predictor/member multiplicities and uncertainty roles', () => {
    for (const bindingId of ['regression-gtwr', 'regression-mgtwr']) {
      expect(
        getDescriptor(bindingId).inputs[0].fields.find(field => field.role === 'predictors')!
          .multiple
      ).toEqual({minimum: 1, maximum: 11});
    }
    const ensemble = getDescriptor('validation-ensemble-scores');
    expect(ensemble.inputs[0].fields.find(field => field.role === 'members')!.multiple).toEqual({
      minimum: 1,
      maximum: 64
    });
    expect(ensemble.notes!.join(' ')).toContain(
      'Fair CRPS and energy scores need at least two members'
    );
    expect(getFieldParameters('validation-gaussian-scores', 'points')).toEqual([
      'observationProperty',
      'meanProperty',
      'varianceProperty'
    ]);
    expect(getFieldParameters('validation-fdr', 'points')).toEqual(['pValueProperty']);
  });

  it('requires remapping duplicate and overlong explicitly mapped predictor lists', () => {
    for (const bindingId of [
      'regression-gtwr',
      'regression-mgtwr',
      'validation-area-of-applicability'
    ]) {
      const profiles = getSampleProfiles(bindingId);
      for (const names of [
        ['predictor', 'predictor'],
        Array.from({length: 17}, (_, index) => `predictor${index}`)
      ]) {
        const assessment = assess(bindingId, profiles, {predictorProperties: names});
        expect(assessment.status).not.toBe('ready');
        expect(
          assessment.missing.some(
            missing => missing.what.includes('distinct columns') && Boolean(missing.remedy)
          )
        ).toBe(true);
      }
    }
  });
});

describe('validation defaults, apply policies and honest interpretation', () => {
  const profile: KeplerDatasetProfile = {
    datasetId: 'observations',
    label: 'Observations',
    kind: 'points',
    catalogGeometry: 'point',
    geometryColumns: [],
    rowCount: 4,
    vertexCount: 4,
    coordinates: {model: 'planar', source: 'declared'},
    extent: [0, 0, 30, 40],
    columns: [
      {
        name: 'observed',
        type: 'number',
        rowCount: 4,
        nullCount: 0,
        minimum: 0,
        maximum: 6,
        mean: 3,
        standardDeviation: 2
      }
    ]
  };

  it('derives scalar spatial scales and sill from the observation input, preserving overrides', () => {
    // The extent diagonal is 50; one third rounds to two significant digits (17). Sill = 2².
    for (const bindingId of [
      'validation-vecchia-likelihood',
      'validation-vecchia-fisher',
      'validation-vecchia-space-time',
      'interpolation-space-time-kriging'
    ]) {
      const binding = getCatalogBinding(bindingId)!;
      const descriptor = getDescriptor(bindingId);
      const resolved = resolveKeplerOperationParameters(
        binding,
        descriptor,
        {points: profile},
        {fieldBindings: {valueProperty: 'observed'}, overrides: {coordinateSystem: 'planar'}}
      );
      expect(resolved.parameters['range']).toBe(17);
      expect(resolved.parameters['sill']).toBe(4);
      expect(
        resolved.resolutions.find(resolution => resolution.parameter === 'range')!.source
      ).toBe('data');
      expect(resolved.resolutions.find(resolution => resolution.parameter === 'sill')!.source).toBe(
        'data'
      );
      const overridden = resolveKeplerOperationParameters(
        binding,
        descriptor,
        {points: profile},
        {overrides: {range: 250, sill: 9}}
      );
      expect(overridden.parameters['range']).toBe(250);
      expect(overridden.parameters['sill']).toBe(9);
    }
    for (const bindingId of ['regression-gtwr', 'regression-mgtwr', 'validation-geo-conformal']) {
      const input = bindingId === 'validation-geo-conformal' ? 'calibration' : 'points';
      const resolved = resolveKeplerOperationParameters(
        getCatalogBinding(bindingId)!,
        getDescriptor(bindingId),
        {[input]: profile},
        {overrides: {coordinateSystem: 'planar'}}
      );
      expect(resolved.parameters['bandwidth']).toBe(17);
    }
  });

  it('hides optimizer budgets and known means unless their controllers activate them', () => {
    for (const bindingId of [
      'validation-vecchia-likelihood',
      'validation-vecchia-fisher',
      'validation-vecchia-space-time'
    ]) {
      const binding = getCatalogBinding(bindingId)!;
      const descriptor = getDescriptor(bindingId);
      expect(resolveKeplerOperationParameters(binding, descriptor, {}).inactive).toContain(
        'maximumIterations'
      );
      expect(
        resolveKeplerOperationParameters(binding, descriptor, {}, {overrides: {fit: true}}).inactive
      ).not.toContain('maximumIterations');
    }
    const bindingId = 'interpolation-space-time-kriging';
    expect(
      resolveKeplerOperationParameters(getCatalogBinding(bindingId)!, getDescriptor(bindingId), {})
        .inactive
    ).toContain('mean');
    expect(
      resolveKeplerOperationParameters(
        getCatalogBinding(bindingId)!,
        getDescriptor(bindingId),
        {},
        {overrides: {mode: 'simple'}}
      ).inactive
    ).not.toContain('mean');
  });

  it('distinguishes packed runtime parameters from prepared inputs and shader/allocation settings', () => {
    const apply = (bindingId: string, name: string) =>
      getDescriptor(bindingId).parameters.find(parameter => parameter.name === name)!.apply;
    expect(apply('regression-gtwr', 'bandwidth')).toBe('live');
    expect(apply('regression-gtwr', 'timeScale')).toBe('live');
    expect(apply('regression-gtwr', 'kernel')).toBe('rebuild');
    expect(apply('regression-mgtwr', 'iterationCount')).toBe('rebuild');
    expect(apply('validation-geo-conformal', 'bandwidth')).toBe('rebuild');
    expect(apply('validation-geo-conformal', 'miscoverage')).toBe('live');
    expect(apply('validation-vecchia-space-time', 'anisotropy')).toBe('rebuild');
    expect(apply('interpolation-space-time-kriging', 'anisotropy')).toBe('live');
    expect(apply('validation-vecchia-space-time', 'mean')).toBe('rebuild');
    expect(apply('validation-fdr', 'alpha')).toBe('live');
    expect(apply('validation-fdr', 'method')).toBe('rebuild');
  });

  it('states uncertainty assumptions and diagnostic/status meanings for every method', () => {
    const notes = (bindingId: string) => getDescriptor(bindingId).notes!.join(' ');
    expect(notes('interpolation-space-time-kriging')).toContain(
      'latent conditional model variance'
    );
    expect(notes('interpolation-space-time-kriging')).toContain(
      'uncertainty in fitted covariance parameters'
    );
    expect(notes('validation-area-of-applicability')).toContain(
      'not calibrated predictive accuracy'
    );
    expect(notes('validation-conformal')).toContain('exchangeability for marginal coverage');
    expect(notes('validation-geo-conformal')).toContain(
      'does not provide a distribution-free spatial coverage guarantee'
    );
    expect(notes('validation-gaussian-scores')).toContain(
      'squared response units, not standard deviation'
    );
    expect(notes('validation-ensemble-scores')).toContain('row order defines the pair list');
    expect(notes('validation-fdr')).toContain('complete testing family');
    expect(notes('validation-fdr')).toContain('Invalid rows are untested');
    for (const descriptor of VALIDATION_OPERATION_DESCRIPTORS) {
      expect(descriptor.prominence).toBe(3);
      expect(descriptor.notes!.join(' ')).toMatch(/diagnostics/i);
    }
  });

  it('plans a diverging residual map and significance classes from published numeric columns', () => {
    const collection: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: [
        {geometry: {type: 'Point', coordinates: [0, 0]}, properties: {}},
        {geometry: {type: 'Point', coordinates: [1, 1]}, properties: {}}
      ]
    };
    for (const [bindingId, columns] of [
      ['regression-gtwr', {residual: [-2, 4]}],
      ['validation-fdr', {significant: [0, 1]}]
    ] as const) {
      const output = getDescriptor(bindingId).outputs[0];
      const table = makeValidationResult(bindingId, bindingId, collection, columns, {}, [])
        .tables[0];
      const plan = planKeplerResultPresentation(table, output)!;
      expect(plan.column).toBe(output.presentation!.column);
      expect(plan.rowCount).toBe(2);
      expect(plan.missingCount).toBe(0);
      if (bindingId === 'regression-gtwr') expect(plan.scale).toBe('diverging');
      else {
        expect(plan.classScheme).toBe('significance-flag');
        expect(plan.unrecognized ?? 0).toBe(0);
      }
    }
  });
});

describe('validation review regressions', () => {
  function makeProfile(datasetId: string, rows: Record<string, number>[]): KeplerDatasetProfile {
    return profileKeplerFeatureCollection(
      {
        type: 'FeatureCollection',
        features: rows.map((properties, index) => ({
          geometry: {type: 'Point', coordinates: [index * 10, index % 2]},
          properties
        }))
      },
      {
        datasetId,
        label: datasetId,
        coordinateSystem: 'planar',
        declarations: {coordinateModel: 'planar', rowSemantics: 'surface-samples'}
      }
    );
  }

  it.each([
    'validation-conformal',
    'validation-geo-conformal'
  ])('%s leaves unit scale in place until a scale is explicitly mapped', bindingId => {
    const profiles = {
      calibration: makeProfile(
        'calibration',
        [1, 2, 3].map(value => ({observed: value, predicted: value + 1, temperature: value + 20}))
      ),
      queries: makeProfile(
        'queries',
        [1, 2, 3].map(value => ({predicted: value, temperature: value + 20}))
      )
    };
    const fields = {observationProperty: 'observed', predictionProperty: 'predicted'};
    const assessment = assess(bindingId, profiles, fields);
    expect(assessment.status, JSON.stringify(assessment.missing)).toBe('ready');
    expect(assessment.parameters['scaleProperty']).toBe('');
    expect(assessment.fieldBindings['scaleProperty']).toBeUndefined();
    const mapped = assess(bindingId, profiles, {...fields, scaleProperty: 'temperature'});
    expect(mapped.status, JSON.stringify(mapped.missing)).toBe('ready');
    expect(mapped.parameters['scaleProperty']).toBe('temperature');
  });

  it.each([
    'validation-vecchia-likelihood',
    'validation-vecchia-fisher',
    'validation-vecchia-space-time',
    'interpolation-space-time-kriging'
  ])('%s keeps zero observation error until error variance is explicitly mapped', bindingId => {
    const profiles = {
      points: makeProfile(
        'points',
        [1, 2, 3].map(value => ({
          observed: value,
          predicted: value + 1,
          temperature: value + 20,
          time: value
        }))
      ),
      ...(bindingId === 'interpolation-space-time-kriging'
        ? {queries: makeProfile('queries', [{time: 1}, {time: 2}])}
        : {})
    };
    const fields = {
      valueProperty: 'observed',
      ...(bindingId.includes('space-time') ? {timeProperty: 'time'} : {})
    };
    const assessment = assess(bindingId, profiles, fields);
    expect(assessment.status, JSON.stringify(assessment.missing)).toBe('ready');
    expect(assessment.parameters['errorVarianceProperty']).toBe('');
    expect(assessment.fieldBindings['errorVarianceProperty']).toBeUndefined();
    const mapped = assess(bindingId, profiles, {...fields, errorVarianceProperty: 'predicted'});
    expect(mapped.parameters['errorVarianceProperty']).toBe('predicted');
  });

  it.each([
    {
      adjusted: [0.8, null, null],
      flags: [0, 0, 0],
      excluded: [1, 2],
      nonsignificant: 1,
      significant: 0
    },
    {adjusted: [null, null], flags: [0, 0], excluded: [0, 1], nonsignificant: 0, significant: 0},
    {
      adjusted: [0.01, 0.8, null],
      flags: [1, 0, 0],
      excluded: [2],
      nonsignificant: 1,
      significant: 1
    }
  ])('FDR excludes untested rows from significance classes: $adjusted', ({
    adjusted,
    flags,
    excluded,
    nonsignificant,
    significant
  }) => {
    const collection: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: adjusted.map((_, index) => ({geometry: {type: 'Point', coordinates: [index, 0]}}))
    };
    const result = makeValidationResult(
      'validation-fdr',
      'FDR',
      collection,
      {
        adjusted_pvalue: adjusted.map(value => value ?? NaN),
        significant: flags
      },
      {},
      []
    );
    const table = result.tables[0];
    const plan = planKeplerResultPresentation(table, getDescriptor('validation-fdr').outputs[0])!;
    expect(plan.excludedRows).toEqual(excluded);
    expect(plan.legend.find(entry => entry.value === 0)!.count).toBe(nonsignificant);
    expect(plan.legend.find(entry => entry.value === 1)!.count).toBe(significant);
    expect(plan.legend.find(entry => entry.label === 'Untested')!.count).toBe(excluded.length);
    expect(plan.missingCount).toBe(0);
    expect(plan.unrecognized).toBe(0);
    expect(plan.legend.reduce((count, entry) => count + entry.count, 0)).toBe(adjusted.length);
    expect(table.columns['significant']).toEqual(flags);
  });

  function packCovariance(bindingId: string, parameters: Record<string, CatalogParameterValue>) {
    const covariance: GPUVecchiaCovariance = {
      model: parameters['model'] as GPUVecchiaCovariance['model'],
      sill: parameters['sill'] as number,
      nugget: parameters['nugget'] as number,
      range: parameters['range'] as number
    };
    return bindingId === 'validation-vecchia-space-time'
      ? getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(
          makeValidationSpaceTimeCovariance(covariance, parameters['anisotropy'] as number)
        )
      : getGPUVecchiaLogLikelihoodParameterValues(covariance);
  }

  it.each([
    'validation-vecchia-likelihood',
    'validation-vecchia-fisher',
    'validation-vecchia-space-time'
  ])('%s resolves admissible Matern defaults and blocks incompatible explicit nuggets', bindingId => {
    const points = makeProfile(
      'points',
      [0, 10, 20].map((observed, time) => ({observed, time}))
    );
    const fields = {
      valueProperty: 'observed',
      ...(bindingId.endsWith('space-time') ? {timeProperty: 'time'} : {})
    };
    const assessment = assess(bindingId, {points}, fields, {model: 'matern-3/2'});
    expect(assessment.status, JSON.stringify(assessment.missing)).toBe('ready');
    expect(assessment.parameters['sill']).toBe(100);
    expect(assessment.parameters['nugget']).toBeGreaterThanOrEqual(0.1);
    expect(() => packCovariance(bindingId, assessment.parameters)).not.toThrow();
    expect(
      assessment.resolutions.find(resolution => resolution.parameter === 'nugget')!.reason
    ).toMatch(/floor/i);
    const invalid = assess(bindingId, {points}, fields, {model: 'matern-3/2', nugget: 0.05});
    expect(invalid.parameters['nugget']).toBe(0.05);
    expect(invalid.status).toBe('blocked');
    expect(
      invalid.missing.some(missing => missing.what.includes('nugget') && Boolean(missing.remedy))
    ).toBe(true);
    expect(() => packCovariance(bindingId, invalid.parameters)).toThrow(/nugget/);
    const valid = assess(bindingId, {points}, fields, {model: 'matern-3/2', nugget: 0.5});
    expect(valid.status).toBe('ready');
    expect(valid.parameters['nugget']).toBe(0.5);
    expect(() => packCovariance(bindingId, valid.parameters)).not.toThrow();
    const exponential = assess(bindingId, {points}, fields, {model: 'exponential', nugget: 0});
    expect(exponential.status).toBe('ready');
    expect(exponential.parameters['nugget']).toBe(0);
    const spatialFloor = assess(bindingId, {points}, fields, {model: 'matern-3/2', nugget: 0.1});
    expect(spatialFloor.status).toBe(bindingId.endsWith('space-time') ? 'blocked' : 'ready');
    if (bindingId.endsWith('space-time')) {
      expect(() => packCovariance(bindingId, spatialFloor.parameters)).toThrow(/nugget/);
    } else {
      expect(() => packCovariance(bindingId, spatialFloor.parameters)).not.toThrow();
    }
  });

  it.each([
    1, 2, 11
  ])('MGTWR generates one component per coefficient with %s predictors', predictorCount => {
    const points = makeProfile(
      'points',
      Array.from({length: 16}, (_, row) => ({
        observed: row * 2,
        time: row,
        ...Object.fromEntries(
          Array.from({length: predictorCount}, (_, column) => [
            `predictor${column}`,
            Math.sin(row + column)
          ])
        )
      }))
    );
    const fields = {
      responseProperty: 'observed',
      timeProperty: 'time',
      predictorProperties: Array.from({length: predictorCount}, (_, column) => `predictor${column}`)
    };
    const assessment = assess('regression-mgtwr', {points}, fields, {timeScale: 3});
    expect(assessment.status, JSON.stringify(assessment.missing)).toBe('ready');
    const parameters = assessment.parameters;
    expect(parameters['componentBandwidths']).toEqual(
      Array(predictorCount + 1).fill(parameters['bandwidth'])
    );
    expect(parameters['componentTimeScales']).toEqual(Array(predictorCount + 1).fill(3));
    const bandwidths = parameters['componentBandwidths'] as readonly number[];
    const timeScales = parameters['componentTimeScales'] as readonly number[];
    expect(() =>
      getGPUMGTWRParameterValues({
        initial: {bandwidth: parameters['bandwidth'] as number, timeScale: 3},
        components: bandwidths.map((bandwidth, index) => ({
          bandwidth,
          timeScale: timeScales[index]
        }))
      })
    ).not.toThrow();
    for (const name of ['componentBandwidths', 'componentTimeScales']) {
      const invalid = assess('regression-mgtwr', {points}, fields, {[name]: [2]});
      expect(invalid.status).toBe('blocked');
      expect(invalid.parameters[name]).toEqual([2]);
      expect(
        invalid.missing.some(missing => missing.what.includes(name) && Boolean(missing.remedy))
      ).toBe(true);
    }
    const explicit = assess('regression-mgtwr', {points}, fields, {
      componentBandwidths: Array(predictorCount + 1).fill(200),
      componentTimeScales: Array(predictorCount + 1).fill(4)
    });
    expect(explicit.status).toBe('ready');
    expect(explicit.parameters['componentBandwidths']).toEqual(Array(predictorCount + 1).fill(200));
    expect(explicit.parameters['componentTimeScales']).toEqual(Array(predictorCount + 1).fill(4));
  });

  it.each([
    {members: ['member1'], fair: false, status: 'ready'},
    {members: ['member1'], fair: true, status: 'blocked'},
    {members: ['member1', 'member2'], fair: true, status: 'ready'}
  ])('ensemble $members with fair=$fair is $status', ({members, fair, status}) => {
    const points = makeProfile(
      'points',
      [1, 2, 3].map(value => ({observed: value, member1: value + 1, member2: value - 1}))
    );
    const assessment = assess(
      'validation-ensemble-scores',
      {points},
      {observationProperty: 'observed', memberProperties: members},
      {fair}
    );
    expect(assessment.status, JSON.stringify(assessment.missing)).toBe(status);
    if (fair && members.length === 1) {
      expect(
        assessment.missing.some(
          missing => /fair.*2|2.*fair/i.test(missing.what) && Boolean(missing.remedy)
        )
      ).toBe(true);
    }
  });
});
