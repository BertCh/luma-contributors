// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it, vi} from 'vitest';
import {
  GPU_ANALYSIS_REQUIREMENT_IDS,
  matchGPUAnalysisFieldRole
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {
  getCatalogBinding,
  KEPLER_CATALOG_BINDINGS,
  normalizeCatalogParameters
} from '../../src/analysis/catalog-bindings';
import {SPACETIME_BINDINGS} from '../../src/analysis/catalog-spacetime-bindings';
import {
  readSpacetimePoints,
  readSpacetimeSeries
} from '../../src/analysis/catalog-spacetime-common';
import type {CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import type {CatalogParameterValue} from '../../src/analysis/catalog-binding-types';
import {SPACETIME_OPERATION_DESCRIPTORS} from '../../src/descriptors/descriptors-spacetime';
import {SPACETIME_OPERATION_DESCRIPTORS as PUBLIC_SPACETIME_DESCRIPTORS} from '../../src/descriptors/index';
import {
  getKeplerOperationDescriptor,
  getKeplerOperationDescriptorCoverage,
  resolveKeplerOperationDescriptor
} from '../../src/descriptors/operation-descriptors';
import {validateKeplerOperationDescriptor} from '../../src/descriptors/validate-operation-descriptor';
import {resolveKeplerOperationParameters} from '../../src/descriptors/default-parameters';
import {evaluateKeplerOperationApplicability} from '../../src/descriptors/applicability';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import type {KeplerDatasetProfile} from '../../src/profile/dataset-profile-types';
import {SPACETIME_TEST_POINTS} from '../catalog-spacetime-fixture';

const BINDING_IDS = [
  'space-time-kernel-density',
  'space-time-relative-risk',
  'adaptive-bandwidth-kernel-density',
  'space-time-hawkes',
  'expectation-based-space-time-scan',
  'space-time-cross-correlation',
  'space-time-local-moran',
  'jacquez-test',
  'local-knox',
  'near-repeat-knox',
  'space-time-diggle-test',
  'space-time-k-function',
  'pettitt-test',
  'location-change-points'
];
const SERIES_IDS = [
  'expectation-based-space-time-scan',
  'space-time-cross-correlation',
  'space-time-local-moran',
  'pettitt-test',
  'location-change-points'
];
const FIELD_BINDINGS = {
  timeProperty: 'time',
  locationProperty: 'location',
  valueProperty: 'value',
  secondValueProperty: 'second',
  baselineProperty: 'baseline',
  caseProperty: 'case'
};

function getDescriptor(bindingId: string) {
  return getKeplerOperationDescriptor(bindingId)!;
}

function makeProfile(collection: CatalogFeatureCollection = SPACETIME_TEST_POINTS) {
  return profileKeplerFeatureCollection(collection, {
    datasetId: 'space-time',
    label: 'Synthetic location series',
    coordinateSystem: 'planar',
    declarations: {
      coordinateModel: 'planar',
      rowSemantics: 'events',
      measurement: {
        location: 'identifier',
        value: 'count',
        second: 'amount',
        baseline: 'count',
        case: 'nominal'
      }
    }
  });
}

function makeSeriesCollection(
  locationCount: number,
  periodCount: number
): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: Array.from({length: locationCount}, (_, location) =>
      Array.from({length: periodCount}, (_, period) => ({
        geometry: {type: 'Point' as const, coordinates: [location, 0]},
        properties: {
          location: `location-${location}`,
          time: 1700000000000 + period * 60000,
          value: period + 1,
          second: period * 2,
          baseline: 1,
          case: period % 2
        }
      }))
    ).flat()
  };
}

function evaluate(
  bindingId: string,
  profile: KeplerDatasetProfile,
  overrides: Record<string, CatalogParameterValue> = {}
) {
  return evaluateKeplerOperationApplicability(
    getCatalogBinding(bindingId)!,
    getDescriptor(bindingId),
    [profile],
    {overrides, executionEngine: 'gpu'}
  );
}

async function expectRuntimePreflight(
  bindingId: string,
  collection: CatalogFeatureCollection,
  parameters: Readonly<Record<string, CatalogParameterValue>>,
  rejection?: RegExp
) {
  // Valid fixtures deliberately stop before any GPU allocation; rejected fixtures never reach it.
  const allocationReached = new Error('Space-time preflight passed: GPU allocation reached');
  const createBuffer = vi.fn(() => {
    throw allocationReached;
  });
  await expect(
    getCatalogBinding(bindingId)!.runGpu!(
      {device: {type: 'webgpu', createBuffer} as never},
      {points: collection},
      parameters
    )
  ).rejects.toThrow(rejection ?? allocationReached);
  if (rejection) expect(createBuffer).not.toHaveBeenCalled();
  else expect(createBuffer).toHaveBeenCalledOnce();
}

describe('space-time registration and integrated requirement design', () => {
  it('registers exactly fourteen hand-written descriptors through the public barrel', () => {
    expect(PUBLIC_SPACETIME_DESCRIPTORS).toBe(SPACETIME_OPERATION_DESCRIPTORS);
    expect(SPACETIME_OPERATION_DESCRIPTORS.map(descriptor => descriptor.bindingId)).toEqual(
      BINDING_IDS
    );
    expect(SPACETIME_BINDINGS.map(binding => binding.id)).toEqual(BINDING_IDS);
    for (const binding of SPACETIME_BINDINGS) {
      const descriptor = getDescriptor(binding.id);
      expect(resolveKeplerOperationDescriptor(binding).source).toBe('hand-written');
      expect(validateKeplerOperationDescriptor(descriptor, binding), binding.id).toEqual([]);
      expect(
        descriptor.inputs.flatMap(input => input.fields.map(field => field.parameter)).sort()
      ).toEqual(
        binding.parameters
          .filter(parameter => parameter.kind === 'string')
          .map(parameter => parameter.name)
          .sort()
      );
      for (const output of binding.outputs.filter(output => output.geometry !== 'none'))
        expect(
          descriptor.outputs.find(candidate => candidate.tableId === output.tableId)?.presentation,
          `${binding.id}.${output.tableId}`
        ).toBeDefined();
      expect(descriptor.outputs.every(output => output.interpretation && output.next?.length)).toBe(
        true
      );
    }
  });

  it('preserves every landed family and leaves no binding derived', () => {
    const coverage = getKeplerOperationDescriptorCoverage(KEPLER_CATALOG_BINDINGS);
    expect(coverage.validationIssues).toEqual([]);
    expect(coverage.orphanedDescriptorIds).toEqual([]);
    expect(coverage.derivedBindingIds).toEqual([]);
    for (const prefix of ['pointcloud-', 'validation-', 'raster-embedding-', 'trajectory-']) {
      expect(
        coverage.handWrittenBindingIds.some(bindingId => bindingId.startsWith(prefix)),
        prefix
      ).toBe(true);
    }
    expect(getDescriptor('colocation-quotient')).toBeDefined();
  });

  it('consolidates maximum bounds in numeric-range, including every point-cloud classification user', () => {
    expect(GPU_ANALYSIS_REQUIREMENT_IDS).toContain('numeric-range');
    expect(GPU_ANALYSIS_REQUIREMENT_IDS).not.toContain('maximum-value');
    for (const bindingId of [
      'pointcloud-ground',
      'pointcloud-smrf',
      'pointcloud-surfaces',
      'pointcloud-height'
    ]) {
      const requirements = getDescriptor(bindingId).requirements.filter(
        requirement => requirement.role === 'classification' && requirement.id === 'numeric-range'
      );
      expect(requirements, bindingId).toEqual([
        expect.objectContaining({id: 'numeric-range', minimum: 0, maximum: 255, integer: true})
      ]);
    }
  });

  it('validates parameter references and numeric-range metadata', () => {
    const binding = getCatalogBinding('space-time-cross-correlation')!;
    const descriptor = getDescriptor(binding.id);
    for (const requirement of [
      {
        input: 'points',
        id: 'minimum-rows' as const,
        severity: 'block' as const,
        limitFromParameter: {parameter: 'missing'}
      },
      {
        input: 'points',
        id: 'numeric-range' as const,
        severity: 'block' as const,
        parameter: 'lag',
        minimum: 3,
        maximum: 1
      },
      {
        input: 'points',
        id: 'numeric-range' as const,
        severity: 'block' as const,
        role: 'time',
        parameter: 'lag',
        minimum: 0
      }
    ])
      expect(
        validateKeplerOperationDescriptor(
          {...descriptor, requirements: [...descriptor.requirements, requirement]},
          binding
        ).length
      ).toBeGreaterThan(0);
  });
});

describe('space-time roles, time encodings and defaults', () => {
  it('reads numeric epochs and declines string or generic time storage', () => {
    const common = {name: 'time', rowCount: 48, nullCount: 0};
    for (const descriptor of SPACETIME_OPERATION_DESCRIPTORS) {
      const time = descriptor.inputs[0].fields.find(field => field.role === 'time')!;
      expect(time).toMatchObject({parameter: 'timeProperty', required: true});
      expect(
        matchGPUAnalysisFieldRole(time, [
          {...common, type: 'integer'},
          {...common, name: 'seconds', type: 'number'}
        ])
      ).toHaveLength(2);
      expect(
        matchGPUAnalysisFieldRole(time, [
          {...common, type: 'time'},
          {...common, type: 'category'},
          {...common, type: 'boolean'}
        ])
      ).toEqual([]);
      expect(descriptor.notes?.join(' ')).toContain(
        'ISO strings and Date objects must be converted'
      );
    }
  });

  it('derives equivalent defaults in seconds from epoch milliseconds and seconds', () => {
    const seconds = structuredClone(SPACETIME_TEST_POINTS);
    for (const feature of seconds.features)
      feature.properties!['time'] = Number(feature.properties!['time']) / 1000;
    const common = {coordinateSystem: 'planar', timeProperty: 'time'};
    expect(readSpacetimePoints(seconds, {...common, timeUnit: 'seconds'}).times).toEqual(
      readSpacetimePoints(SPACETIME_TEST_POINTS, {...common, timeUnit: 'milliseconds'}).times
    );
    for (const binding of SPACETIME_BINDINGS) {
      const descriptor = getDescriptor(binding.id);
      for (const parameter of descriptor.parameters.filter(
        parameter =>
          parameter.defaultPolicy.from === 'data' && parameter.defaultPolicy.rule === 'column-range'
      )) {
        const milliseconds = resolveKeplerOperationParameters(
          binding,
          descriptor,
          {points: makeProfile()},
          {fieldBindings: {timeProperty: 'time'}}
        );
        const resolvedSeconds = resolveKeplerOperationParameters(
          binding,
          descriptor,
          {points: makeProfile(seconds)},
          {fieldBindings: {timeProperty: 'time'}, overrides: {timeUnit: 'seconds'}}
        );
        // Eight one-minute periods span 420 seconds; the stated starting scale is one tenth.
        expect(milliseconds.parameters[parameter.name], `${binding.id}.${parameter.name}`).toBe(42);
        expect(resolvedSeconds.parameters[parameter.name]).toBe(42);
        expect(
          milliseconds.resolutions.find(resolution => resolution.parameter === parameter.name)
            ?.source
        ).toBe('data');
        expect(normalizeCatalogParameters(binding, milliseconds.parameters)).toEqual(
          milliseconds.parameters
        );
      }
    }
  });

  it('preserves explicit temporal choices and falls back visibly without profile evidence', () => {
    const binding = getCatalogBinding('space-time-kernel-density')!;
    const descriptor = getDescriptor(binding.id);
    const resolved = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {points: makeProfile()},
      {fieldBindings: {timeProperty: 'time'}, overrides: {temporalBandwidth: 7}}
    );
    expect(resolved.parameters['temporalBandwidth']).toBe(7);
    expect(
      resolved.resolutions.find(resolution => resolution.parameter === 'temporalBandwidth')?.source
    ).toBe('override');
    const fallback = resolveKeplerOperationParameters(binding, descriptor, {});
    for (const name of ['spatialBandwidth', 'temporalBandwidth']) {
      expect(fallback.resolutions.find(resolution => resolution.parameter === name)?.source).toBe(
        'fallback'
      );
      expect(fallback.parameters[name]).toBe(
        binding.parameters.find(parameter => parameter.name === name)?.defaultValue
      );
    }
  });

  it('scales spatial defaults with coordinates even for repeated locations and declares application cost', () => {
    const scaled = structuredClone(SPACETIME_TEST_POINTS);
    for (const feature of scaled.features)
      feature.geometry!.coordinates = (feature.geometry!.coordinates as number[]).map(
        value => value * 10
      );
    for (const binding of SPACETIME_BINDINGS) {
      const descriptor = getDescriptor(binding.id);
      const fieldBindings = Object.fromEntries(
        descriptor.inputs[0].fields.map(field => [
          field.parameter,
          FIELD_BINDINGS[field.parameter as keyof typeof FIELD_BINDINGS]
        ])
      );
      const base = resolveKeplerOperationParameters(
        binding,
        descriptor,
        {points: makeProfile()},
        {fieldBindings}
      );
      const larger = resolveKeplerOperationParameters(
        binding,
        descriptor,
        {points: makeProfile(scaled)},
        {fieldBindings}
      );
      for (const parameter of descriptor.parameters.filter(
        parameter =>
          parameter.defaultPolicy.from === 'data' && parameter.defaultPolicy.rule !== 'column-range'
      )) {
        expect(
          base.resolutions.find(resolution => resolution.parameter === parameter.name)?.source,
          `${binding.id}.${parameter.name}`
        ).toBe('data');
        expect(
          Number(larger.parameters[parameter.name]) / Number(base.parameters[parameter.name])
        ).toBeCloseTo(10, 8);
      }
      if (binding.parameters.some(parameter => parameter.name === 'permutations'))
        expect(base.parameters['permutations']).toBe(999);
      for (const name of [
        'gridSize',
        'sliceCount',
        'classCount',
        'bandCount',
        'maximumIterations',
        'maximumParents',
        'cost',
        'timeUnit',
        'coordinateSystem'
      ]) {
        const parameter = descriptor.parameters.find(parameter => parameter.name === name);
        if (parameter) expect(parameter.apply, `${binding.id}.${name}`).toBe('rebuild');
      }
    }
    expect(
      getDescriptor('adaptive-bandwidth-kernel-density').parameters.find(
        parameter => parameter.name === 'sensitivity'
      )?.apply
    ).toBe('live');
    for (const name of ['penalty', 'minimumSegmentLength'])
      expect(
        getDescriptor('location-change-points').parameters.find(
          parameter => parameter.name === name
        )?.apply
      ).toBe('live');
  });

  it('records complete-series, count, baseline and second-series roles', () => {
    for (const bindingId of SERIES_IDS) {
      const descriptor = getDescriptor(bindingId);
      expect(descriptor.inputs[0].fields.find(field => field.role === 'location')).toMatchObject({
        parameter: 'locationProperty',
        required: false
      });
      for (const text of [
        'complete location/time rectangle',
        'equally spaced periods',
        'fixed Point coordinates'
      ])
        expect(descriptor.notes?.join(' ')).toContain(text);
    }
    const scan = getDescriptor('expectation-based-space-time-scan');
    expect(scan.inputs[0].fields.find(field => field.role === 'value')?.measurement).toEqual([
      'count'
    ]);
    expect(scan.inputs[0].fields.find(field => field.role === 'baseline')).toMatchObject({
      parameter: 'baselineProperty',
      measurement: ['count'],
      required: true
    });
    expect(scan.notes?.join(' ')).toContain('not a population denominator');
    expect(
      getDescriptor('space-time-cross-correlation').inputs[0].fields.find(
        field => field.role === 'second-value'
      )?.parameter
    ).toBe('secondValueProperty');
  });
});

describe('space-time preflight regressions', () => {
  it('accepts the valid fixture and blocks row, vertex and missing-time violations', () => {
    for (const binding of SPACETIME_BINDINGS) {
      expect(evaluate(binding.id, makeProfile()).status, binding.id).not.toBe('blocked');
      const overflow = evaluate(binding.id, {...makeProfile(), rowCount: 513, vertexCount: 513});
      expect(overflow.status).toBe('blocked');
      for (const requirementId of ['maximum-rows', 'maximum-vertices'])
        expect(overflow.requirements).toContainEqual(
          expect.objectContaining({id: requirementId, state: 'failed'})
        );
      const missing = {
        ...makeProfile(),
        columns: makeProfile().columns.map(column =>
          column.name === 'time' ? {...column, nullCount: 1} : column
        )
      };
      expect(evaluate(binding.id, missing).requirements).toContainEqual(
        expect.objectContaining({id: 'no-missing-values', role: 'time', state: 'failed'})
      );
    }
  });

  it('checks 129 periods, 65 locations and the Moran location minimum from real series profiles', () => {
    for (const bindingId of SERIES_IDS) {
      for (const [role, collection] of [
        ['time', makeSeriesCollection(1, 129)],
        ['location', makeSeriesCollection(65, 2)]
      ] as const) {
        const result = evaluate(bindingId, makeProfile(collection));
        expect(result.status, bindingId).toBe('blocked');
        expect(result.requirements).toContainEqual(
          expect.objectContaining({id: 'maximum-distinct-values', role, state: 'failed'})
        );
      }
    }
    expect(
      evaluate('space-time-local-moran', makeProfile(makeSeriesCollection(3, 2))).requirements
    ).toContainEqual(
      expect.objectContaining({id: 'minimum-distinct-values', role: 'location', state: 'failed'})
    );
  });

  it('prefers location IDs without forced mappings and preserves blank coordinate grouping', () => {
    const automatic = evaluate('pettitt-test', makeProfile());
    expect(automatic.status).toBe('ready');
    expect(automatic.parameters['locationProperty']).toBe('location');
    expect(
      readSpacetimeSeries(
        readSpacetimePoints(SPACETIME_TEST_POINTS, automatic.parameters),
        automatic.parameters
      ).cellCount
    ).toBe(6);
    for (const fieldBindings of [undefined, {locationProperty: ''}]) {
      const grouped = evaluateKeplerOperationApplicability(
        getCatalogBinding('pettitt-test')!,
        getDescriptor('pettitt-test'),
        [makeProfile()],
        {overrides: fieldBindings ? {} : {locationProperty: ''}, fieldBindings}
      );
      expect(grouped.status).toBe('ready');
      expect(grouped.parameters['locationProperty']).toBe('');
      expect(grouped.fieldBindings['locationProperty']).toBe('');
      expect(grouped.filledRoles.some(field => field.role === 'location')).toBe(false);
      expect(grouped.requirements.some(requirement => requirement.role === 'location')).toBe(false);
      expect(
        readSpacetimeSeries(
          readSpacetimePoints(SPACETIME_TEST_POINTS, grouped.parameters),
          grouped.parameters
        ).cellCount
      ).toBe(6);
    }
    const numericLocations = structuredClone(SPACETIME_TEST_POINTS);
    numericLocations.features.forEach((feature, row) => {
      feature.properties!['location'] = Math.floor(row / 8);
    });
    expect(
      evaluate('pettitt-test', makeProfile(numericLocations)).parameters['locationProperty']
    ).toBe('location');
    const coordinatesOnly = structuredClone(SPACETIME_TEST_POINTS);
    for (const feature of coordinatesOnly.features) delete feature.properties!['location'];
    expect(
      evaluate('pettitt-test', makeProfile(coordinatesOnly)).parameters['locationProperty']
    ).toBe('');
  });

  it('checks lag, neighbor-count and segmentation relationships against runtime before allocation', async () => {
    const threeEvents = {
      ...SPACETIME_TEST_POINTS,
      features: SPACETIME_TEST_POINTS.features.slice(0, 3)
    };
    for (const [bindingId, collection, overrides, rejection] of [
      ['space-time-cross-correlation', makeSeriesCollection(6, 4), {}, /four overlapping periods/],
      ['space-time-cross-correlation', makeSeriesCollection(6, 4), {lag: 0}, undefined],
      [
        'space-time-cross-correlation',
        makeSeriesCollection(6, 6),
        {lag: -3},
        /four overlapping periods/
      ],
      ['space-time-cross-correlation', makeSeriesCollection(6, 6), {lag: -2}, undefined],
      ['jacquez-test', threeEvents, {}, /below event count/],
      ['jacquez-test', threeEvents, {neighborCount: 2}, undefined],
      ['space-time-local-moran', SPACETIME_TEST_POINTS, {lag: 8}, /lag below periodCount/],
      ['space-time-local-moran', SPACETIME_TEST_POINTS, {lag: 7}, undefined],
      [
        'location-change-points',
        SPACETIME_TEST_POINTS,
        {cost: 'slope'},
        /minimumSegmentLength must be 3/
      ],
      [
        'location-change-points',
        SPACETIME_TEST_POINTS,
        {cost: 'slope', minimumSegmentLength: 3},
        undefined
      ],
      [
        'location-change-points',
        SPACETIME_TEST_POINTS,
        {cost: 'variance', minimumSegmentLength: 1},
        /minimumSegmentLength must be 2/
      ],
      [
        'location-change-points',
        SPACETIME_TEST_POINTS,
        {cost: 'mean-variance', minimumSegmentLength: 1},
        /minimumSegmentLength must be 2/
      ],
      [
        'location-change-points',
        SPACETIME_TEST_POINTS,
        {minimumSegmentLength: 9},
        /minimumSegmentLength/
      ],
      ['location-change-points', SPACETIME_TEST_POINTS, {minimumSegmentLength: 8}, undefined]
    ] as const) {
      const result = evaluate(bindingId, makeProfile(collection), overrides);
      expect(result.status, `${bindingId} ${JSON.stringify(overrides)}`).toBe(
        rejection ? 'blocked' : 'ready'
      );
      if (rejection)
        expect(result.missing.some(missing => (missing.remedy?.length ?? 0) > 0)).toBe(true);
      await expectRuntimePreflight(bindingId, collection, result.parameters, rejection);
    }
  });

  it('checks scan count boundaries and fractions without switching the observed-count column', async () => {
    for (const value of [0, 16777215, -1, 16777216, 16777214.75]) {
      const collection = structuredClone(SPACETIME_TEST_POINTS);
      collection.features[0].properties!['value'] = value;
      const result = evaluate('expectation-based-space-time-scan', makeProfile(collection));
      expect(result.parameters['valueProperty']).toBe('value');
      const valid = Number.isInteger(value) && value >= 0 && value <= 16777215;
      expect(result.status, String(value)).toBe(valid ? 'ready' : 'blocked');
      await expectRuntimePreflight(
        'expectation-based-space-time-scan',
        collection,
        result.parameters,
        valid ? undefined : /integer counts/
      );
    }
  });

  it('requires finite positive scan baselines and nonnegative Poisson segmentation values', () => {
    for (const value of [0, -1, null]) {
      const collection = structuredClone(SPACETIME_TEST_POINTS);
      collection.features[0].properties!['baseline'] = value;
      expect(
        evaluate('expectation-based-space-time-scan', makeProfile(collection), {
          baselineProperty: 'baseline'
        }).status
      ).toBe('blocked');
    }
    const collection = structuredClone(SPACETIME_TEST_POINTS);
    collection.features[0].properties!['value'] = -1;
    expect(
      evaluate('location-change-points', makeProfile(collection), {cost: 'poisson'}).requirements
    ).toContainEqual(expect.objectContaining({id: 'non-negative-values', state: 'failed'}));
    expect(
      evaluate('location-change-points', makeProfile(collection), {cost: 'mean'}).requirements.some(
        requirement => requirement.id === 'non-negative-values'
      )
    ).toBe(false);
  });

  it('requires literal binary 0/1 and both case/control populations', async () => {
    for (const [first, second, rejection] of [
      [0, 1, undefined],
      [2, 3, /binary 0\/1/],
      [0, 0.5, /binary 0\/1/],
      [0.25, 0.75, /binary 0\/1/],
      [-1, 0, /binary 0\/1/],
      [0, 0, /both cases and controls/],
      [1, 1, /both cases and controls/]
    ] as const) {
      const collection = structuredClone(SPACETIME_TEST_POINTS);
      collection.features.forEach((feature, row) => {
        feature.properties!['case'] = row % 2 ? first : second;
      });
      const result = evaluate('space-time-relative-risk', makeProfile(collection));
      expect(result.status, `${first}/${second}`).toBe(rejection ? 'blocked' : 'ready');
      if (first !== second && rejection)
        expect(result.requirements).toContainEqual(
          expect.objectContaining({id: 'numeric-range', role: 'case', state: 'failed'})
        );
      await expectRuntimePreflight(
        'space-time-relative-risk',
        collection,
        result.parameters,
        rejection
      );
    }
  });
});

describe('space-time interpretation', () => {
  it('names actual output columns without colouring p-values as discoveries', () => {
    const expected = {
      'space-time-kernel-density': {grid: 'value'},
      'space-time-relative-risk': {grid: 'value'},
      'adaptive-bandwidth-kernel-density': {grid: 'value', points: 'bandwidth'},
      'space-time-hawkes': {points: 'background_probability'},
      'expectation-based-space-time-scan': {points: 'cluster_member'},
      'space-time-cross-correlation': {points: 'correlation'},
      'space-time-local-moran': {points: 'observed'},
      'local-knox': {points: 'observed'},
      'pettitt-test': {points: 'change_time'},
      'location-change-points': {points: 'change_count'}
    };
    for (const [bindingId, tables] of Object.entries(expected))
      for (const [tableId, column] of Object.entries(tables))
        expect(
          getDescriptor(bindingId).outputs.find(output => output.tableId === tableId)?.presentation
            ?.column
        ).toBe(column);
    for (const descriptor of SPACETIME_OPERATION_DESCRIPTORS)
      expect(
        descriptor.outputs.every(output => !output.presentation?.column.match(/p_value|pseudoP/))
      ).toBe(true);
    expect(getDescriptor('space-time-relative-risk').outputs[0].presentation).toMatchObject({
      scale: 'diverging',
      center: 1
    });
  });

  it('carries uncorrected inference cautions and distinguishes global or descriptive results', () => {
    for (const bindingId of [
      'space-time-cross-correlation',
      'space-time-local-moran',
      'local-knox',
      'near-repeat-knox',
      'pettitt-test'
    ]) {
      expect(
        getDescriptor(bindingId).outputs.flatMap(output => output.interpretation?.cautions ?? [])
      ).toContain('multiple-testing-uncorrected');
      expect(getDescriptor(bindingId).notes?.join(' ')).toContain('no correction setting');
    }
    expect(getDescriptor('space-time-local-moran').notes?.join(' ')).toContain(
      'Only latest-period source rows'
    );
    expect(getDescriptor('space-time-diggle-test').notes?.join(' ')).toContain(
      'one global time-permutation test'
    );
    for (const bindingId of ['space-time-k-function', 'location-change-points'])
      expect(getDescriptor(bindingId).notes?.join(' ')).toContain('no p-value');
    expect(getDescriptor('adaptive-bandwidth-kernel-density').notes?.join(' ')).toContain(
      'is not used by the spatial contributor'
    );
  });

  it('keeps model-fit and density-denominator caveats method-specific', () => {
    expect(
      getDescriptor('space-time-hawkes').outputs.flatMap(
        output => output.interpretation?.cautions ?? []
      )
    ).not.toContain('poor-model-fit');
    expect(getDescriptor('space-time-hawkes').notes?.join(' ')).toContain(
      'Inspect converged, iterationLimitReached, candidateOverflow and approximation'
    );
    expect(getDescriptor('space-time-relative-risk').notes?.join(' ')).toContain(
      'Very small control density makes the ratio unstable'
    );
    expect(
      getDescriptor('expectation-based-space-time-scan').outputs.flatMap(
        output => output.interpretation?.cautions ?? []
      )
    ).not.toContain('pseudo-p-values');
    expect(getDescriptor('expectation-based-space-time-scan').notes?.join(' ')).toContain(
      'Poisson Monte Carlo replicates'
    );
  });
});
