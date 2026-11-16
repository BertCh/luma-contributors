// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {getNetworkCategoricalSampleInputs} from '../../../../examples/deck/kepler-compute/src/analysis-catalog-network-categorical-samples';
import {NETWORK_CATEGORICAL_BINDINGS} from '../../src/analysis/catalog-network-categorical-bindings';
import {prepareCategoryWeights} from '../../src/analysis/catalog-network-categorical-categories';
import {readCategories, readNumbers} from '../../src/analysis/catalog-network-categorical-common';
import {prepareFlowMatrix} from '../../src/analysis/catalog-network-categorical-flows';
import {prepareDatumPositions} from '../../src/analysis/catalog-network-categorical-datum';
import {getCatalogBinding, normalizeCatalogParameters} from '../../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {evaluateKeplerOperationApplicability} from '../../src/descriptors/applicability';
import {resolveKeplerOperationParameters} from '../../src/descriptors/default-parameters';
import {NETWORK_CATEGORICAL_OPERATION_DESCRIPTORS} from '../../src/descriptors/descriptors-network-categorical';
import {
  getKeplerOperationDescriptor,
  resolveKeplerOperationDescriptor
} from '../../src/descriptors/operation-descriptors';
import {validateKeplerOperationDescriptor} from '../../src/descriptors/validate-operation-descriptor';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';

const EXPECTED_IDENTIFIERS = [
  'kernel-flow-smoothing',
  'radiation-flow-model',
  'map-matched-edge-volumes',
  'network-potential-path-area',
  'network-travel-time-percentiles',
  'colocation-quotient',
  'local-category-dependence',
  'spatial-entropy',
  'category-pair-counts',
  'geographical-detector',
  'redcap-single-linkage',
  'healpix-point-to-cell',
  'healpix-cell-aggregation',
  'datum-grid-transform',
  'sinkhorn-divergence'
];
const CATEGORY_IDENTIFIERS = [
  'colocation-quotient',
  'local-category-dependence',
  'spatial-entropy',
  'category-pair-counts',
  'geographical-detector'
];

function makePoints(
  properties: Record<string, unknown>[],
  positions?: number[][]
): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: properties.map((property, index) => ({
      id: `zone-${index}`,
      geometry: {type: 'Point', coordinates: positions?.[index] ?? [index * 10, 0]},
      properties: property
    }))
  };
}

function assessPoints(
  identifier: string,
  points: CatalogFeatureCollection,
  fieldBindings: Record<string, string>,
  overrides: Record<string, string | number | boolean> = {}
) {
  return evaluateKeplerOperationApplicability(
    getCatalogBinding(identifier)!,
    getKeplerOperationDescriptor(identifier)!,
    [
      profileKeplerFeatureCollection(points, {
        datasetId: 'points',
        label: 'Points',
        coordinateSystem: 'planar'
      })
    ],
    {inputAssignments: {points: 'points'}, fieldBindings, overrides, executionEngine: 'gpu'}
  );
}

function getSampleInputs(identifier: string): Record<string, CatalogFeatureCollection> {
  return Object.fromEntries(
    Object.entries(getNetworkCategoricalSampleInputs(identifier)!).map(([name, text]) => [
      name,
      JSON.parse(text) as CatalogFeatureCollection
    ])
  );
}

describe('AX-80 network/categorical descriptors', () => {
  it('registers exactly the 15 bindings with complete roles, policies and output guidance', () => {
    expect(
      NETWORK_CATEGORICAL_OPERATION_DESCRIPTORS.map(descriptor => descriptor.bindingId).sort()
    ).toEqual(EXPECTED_IDENTIFIERS.slice().sort());
    for (const binding of NETWORK_CATEGORICAL_BINDINGS) {
      const descriptor = getKeplerOperationDescriptor(binding.id)!;
      expect(resolveKeplerOperationDescriptor(binding).source, binding.id).toBe('hand-written');
      expect(validateKeplerOperationDescriptor(descriptor, binding), binding.id).toEqual([]);
      expect(descriptor.notes?.length, binding.id).toBeGreaterThan(0);
      for (const output of binding.outputs) {
        const metadata = descriptor.outputs.find(
          candidate => candidate.tableId === output.tableId
        )!;
        expect(metadata.interpretation, `${binding.id}.${output.tableId}`).toBeDefined();
        expect(metadata.next?.length, binding.id).toBeGreaterThan(0);
        if (output.geometry !== 'none') expect(metadata.presentation, binding.id).toBeDefined();
      }
      const resolved = resolveKeplerOperationParameters(binding, descriptor, {});
      expect(normalizeCatalogParameters(binding, resolved.parameters), binding.id).toEqual(
        resolved.parameters
      );
      expect(
        descriptor.parameters.every(parameter => ['live', 'rebuild'].includes(parameter.apply))
      ).toBe(true);
    }
  });

  it.each(
    EXPECTED_IDENTIFIERS
  )('%s accepts its real UI sample with explicit column mapping', identifier => {
    const binding = getCatalogBinding(identifier)!;
    const descriptor = getKeplerOperationDescriptor(identifier)!;
    const inputs = getSampleInputs(identifier);
    const profiles = Object.entries(inputs).map(([name, collection]) =>
      profileKeplerFeatureCollection(collection, {
        datasetId: name,
        label: name,
        coordinateSystem: 'EPSG:4326'
      })
    );
    const fieldBindings = Object.fromEntries(
      descriptor.inputs.flatMap(input =>
        input.fields.flatMap(field => {
          const property = binding.parameters.find(
            parameter => parameter.name === field.parameter
          )!.defaultValue;
          return typeof property === 'string' &&
            property &&
            inputs[input.name].features.some(feature => property in (feature.properties ?? {}))
            ? [[field.parameter, property]]
            : [];
        })
      )
    );
    const assessment = evaluateKeplerOperationApplicability(binding, descriptor, profiles, {
      inputAssignments: Object.fromEntries(Object.keys(inputs).map(name => [name, name])),
      fieldBindings,
      executionEngine: 'gpu'
    });
    expect(assessment.status, JSON.stringify({identifier, missing: assessment.missing})).not.toBe(
      'blocked'
    );
  });

  it.each(
    CATEGORY_IDENTIFIERS
  )('%s rejects 33 categories and an entirely missing category column before dispatch', identifier => {
    for (const properties of [
      Array.from({length: 33}, (_, index) => ({category: `category-${index}`, value: index})),
      Array.from({length: 33}, (_, index) => ({
        category: index < 17 ? index : String(index - 17),
        value: index
      })),
      [
        {category: null, value: 1},
        {category: null, value: 2}
      ]
    ]) {
      const points = makePoints(properties);
      const assessment = assessPoints(identifier, points, {
        categoryProperty: 'category',
        valueProperty: 'value'
      });
      expect(assessment.status).toBe('blocked');
      const requirement = assessment.requirements.find(
        candidate =>
          candidate.state === 'failed' &&
          ['minimum-distinct-values', 'maximum-distinct-values'].includes(candidate.id)
      );
      expect(requirement?.remedy).toMatch(/categor/);
      expect(() => readCategories(points, 'category')).toThrow('Category count must be in [1, 32]');
    }
  });

  it.each(
    CATEGORY_IDENTIFIERS
  )('%s accepts explicitly mapped ISO date categories without changing their identities', identifier => {
    const points = makePoints([
      {category: '2026-01-01', value: 1},
      {category: '2026-02-01', value: 2}
    ]);
    const profile = profileKeplerFeatureCollection(points, {
      datasetId: 'points',
      label: 'Points',
      coordinateSystem: 'planar'
    });
    expect(profile.columns.find(column => column.name === 'category')?.type).toBe('time');
    const assessment = assessPoints(identifier, points, {
      categoryProperty: 'category',
      valueProperty: 'value'
    });
    expect(assessment.status, JSON.stringify(assessment.missing)).not.toBe('blocked');
    expect(assessment.parameters['categoryProperty']).toBe('category');
    expect(
      assessment.requirements.find(requirement => requirement.id === 'maximum-distinct-values')
        ?.state
    ).toBe('passed');
    expect(
      readCategories(points, assessment.parameters['categoryProperty'] as string).labels
    ).toEqual(['string:2026-01-01', 'string:2026-02-01']);

    const tooManyDates = makePoints(
      Array.from({length: 33}, (_, index) => ({
        category: new Date(Date.UTC(2026, 0, index + 1)).toISOString(),
        value: index
      }))
    );
    const rejected = assessPoints(identifier, tooManyDates, {
      categoryProperty: 'category',
      valueProperty: 'value'
    });
    expect(rejected.status).toBe('blocked');
    expect(
      rejected.requirements.find(requirement => requirement.id === 'maximum-distinct-values')?.state
    ).toBe('failed');
    expect(() => readCategories(tooManyDates, 'category')).toThrow(
      'Category count must be in [1, 32]'
    );
  });

  it('allows 32 typed categories and warns about missing categories without rejecting them', () => {
    const points = makePoints(
      Array.from({length: 32}, (_, index) => ({category: `category-${index}`}))
    );
    expect(
      assessPoints('category-pair-counts', points, {categoryProperty: 'category'}).status
    ).not.toBe('blocked');
    const partlyMissing = makePoints([
      {category: 'A'},
      {category: null},
      {category: false},
      {category: 1},
      {category: '1'}
    ]);
    const assessment = assessPoints('colocation-quotient', partlyMissing, {
      categoryProperty: 'category'
    });
    expect(assessment.status).not.toBe('blocked');
    expect(
      assessment.requirements.some(
        requirement =>
          requirement.id === 'no-missing-values' &&
          requirement.severity === 'warn' &&
          requirement.state === 'failed'
      )
    ).toBe(true);
    expect(readCategories(partlyMissing, 'category').labels).toEqual([
      'string:A',
      'boolean:false',
      'number:1',
      'string:1'
    ]);
  });

  it('leaves estimated category capacity unknown instead of authorizing it', () => {
    const points = makePoints([{category: 'A'}, {category: 'B'}]);
    const profile = profileKeplerFeatureCollection(points, {
      datasetId: 'points',
      label: 'Points',
      coordinateSystem: 'planar'
    });
    const estimated = {
      ...profile,
      columns: profile.columns.map(column => ({
        ...column,
        distinctCount: 2,
        distinctCountIsEstimate: true,
        distinctCountLowerBound: undefined,
        distinctCountUpperBound: undefined
      }))
    };
    const assessment = evaluateKeplerOperationApplicability(
      getCatalogBinding('category-pair-counts')!,
      getKeplerOperationDescriptor('category-pair-counts')!,
      [estimated],
      {fieldBindings: {categoryProperty: 'category'}, executionEngine: 'gpu'}
    );
    expect(
      assessment.requirements.find(requirement => requirement.id === 'maximum-distinct-values')
        ?.state
    ).toBe('unknown');
    expect(assessment.status).toBe('needs-confirmation');
  });

  it.each([
    'colocation-quotient',
    'local-category-dependence',
    'spatial-entropy',
    'category-pair-counts',
    'redcap-single-linkage'
  ])('%s derives a distance band from spacing, with a declared fallback', identifier => {
    // Collinear points 0, 10, 40 have nearest-neighbor distances 10, 10, 30.
    const points = makePoints(
      [
        {category: 'A', value: 1},
        {category: 'A', value: 2},
        {category: 'B', value: 3}
      ],
      [
        [0, 0],
        [10, 0],
        [40, 0]
      ]
    );
    const profile = profileKeplerFeatureCollection(points, {
      datasetId: 'points',
      label: 'Points',
      coordinateSystem: 'planar'
    });
    const binding = getCatalogBinding(identifier)!;
    const descriptor = getKeplerOperationDescriptor(identifier)!;
    const resolution = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {points: profile},
      {overrides: {coordinateSystem: 'planar'}}
    );
    expect(resolution.parameters['radius']).toBe(30);
    expect(resolution.resolutions.find(parameter => parameter.parameter === 'radius')?.source).toBe(
      'data'
    );
    const topology = prepareCategoryWeights(
      points,
      'planar',
      resolution.parameters['radius'] as number
    );
    expect(
      Array.from(topology.offsets.slice(1)).map((end, index) => end - topology.offsets[index])
    ).toEqual([1, 2, 1]);
    const fallback = resolveKeplerOperationParameters(binding, descriptor, {});
    expect(fallback.parameters['radius']).toBe(1000);
    expect(fallback.resolutions.find(parameter => parameter.parameter === 'radius')?.source).toBe(
      'fallback'
    );
  });

  it('describes OD identities and rejects negative population and flow weights', () => {
    const identifier = 'kernel-flow-smoothing';
    const descriptor = getKeplerOperationDescriptor(identifier)!;
    expect(
      descriptor.inputs.find(input => input.name === 'flows')!.fields.map(field => field.role)
    ).toEqual(['origin', 'destination', 'weight']);
    const inputs = getSampleInputs(identifier);
    const zoneProperties = inputs['zones'].features.map(feature => ({
      ...feature.properties,
      population: -1
    }));
    const zones = makePoints(zoneProperties);
    const assessment = evaluateKeplerOperationApplicability(
      getCatalogBinding(identifier)!,
      descriptor,
      [
        profileKeplerFeatureCollection(zones, {
          datasetId: 'zones',
          label: 'Zones',
          coordinateSystem: 'EPSG:4326'
        }),
        profileKeplerFeatureCollection(inputs['flows'], {
          datasetId: 'flows',
          label: 'Flows',
          coordinateSystem: 'EPSG:4326'
        })
      ],
      {
        inputAssignments: {zones: 'zones', flows: 'flows'},
        fieldBindings: {
          populationProperty: 'population',
          originProperty: 'origin',
          destinationProperty: 'destination',
          weightProperty: 'weight'
        },
        executionEngine: 'gpu'
      }
    );
    expect(
      assessment.requirements.find(
        requirement => requirement.role === 'population' && requirement.id === 'non-negative-values'
      )?.state
    ).toBe('failed');
    expect(() => readNumbers(zones, 'population', true)).toThrow('nonnegative');
    const invalidFlows = {
      ...inputs['flows'],
      features: inputs['flows'].features.map(feature => ({
        ...feature,
        properties: {...feature.properties, weight: -1}
      }))
    };
    expect(() =>
      prepareFlowMatrix(inputs['zones'], invalidFlows, 'origin', 'destination', 'weight')
    ).toThrow('nonnegative');
    expect(descriptor.notes?.join(' ')).toMatch(/does not conserve total flow/);
  });

  it.each([
    'network-potential-path-area',
    'network-travel-time-percentiles'
  ])('%s applies cost requirements only in property mode', identifier => {
    const descriptor = getKeplerOperationDescriptor(identifier)!;
    const source = getSampleInputs(identifier)['lines'];
    const lines = {
      ...source,
      features: source.features.map(feature => ({...feature, properties: {cost: -1, slope: -1}}))
    };
    const profile = profileKeplerFeatureCollection(lines, {
      datasetId: 'lines',
      label: 'Network',
      coordinateSystem: 'planar'
    });
    const assess = (costMode: string) =>
      evaluateKeplerOperationApplicability(getCatalogBinding(identifier)!, descriptor, [profile], {
        fieldBindings: {forwardCostProperty: 'cost'},
        overrides: {costMode, slopeProperty: ''},
        executionEngine: 'gpu'
      });
    expect(
      assess('length').requirements.some(requirement => requirement.role === 'forward-cost')
    ).toBe(false);
    expect(
      assess('property').requirements.find(
        requirement =>
          requirement.role === 'forward-cost' && requirement.id === 'non-negative-values'
      )?.state
    ).toBe('failed');
    expect(descriptor.notes?.join(' ')).toMatch(/1-256 nodes.*4096 directed arcs/);
  });

  it('offers a declared rate as a duration-slope candidate without selecting it', () => {
    const identifier = 'network-travel-time-percentiles';
    const lines = getSampleInputs(identifier)['lines'];
    const profile = profileKeplerFeatureCollection(
      {
        ...lines,
        features: lines.features.map((feature, index) => ({
          ...feature,
          properties: {trafficRate: 0.1 + index * 0.1}
        }))
      },
      {
        datasetId: 'lines',
        label: 'Network',
        coordinateSystem: 'planar',
        declarations: {measurement: {trafficRate: 'rate'}}
      }
    );
    const binding = getCatalogBinding(identifier)!;
    const descriptor = getKeplerOperationDescriptor(identifier)!;
    // No overrides or field mappings: a compatible rate must not enable affine duration costs.
    const assessment = evaluateKeplerOperationApplicability(binding, descriptor, [profile]);
    expect(assessment.status).toBe('ready');
    expect(assessment.fieldBindings['slopeProperty']).toBeUndefined();
    expect(assessment.parameters['slopeProperty']).toBe('');
    expect(assessment.roleCandidates['duration-slope']).toContain('trafficRate');
    expect(assessment.requirements.some(requirement => requirement.role === 'duration-slope')).toBe(
      false
    );
    const selected = evaluateKeplerOperationApplicability(binding, descriptor, [profile], {
      fieldBindings: {slopeProperty: 'trafficRate'}
    });
    expect(selected.fieldBindings['slopeProperty']).toBe('trafficRate');
    expect(selected.parameters['slopeProperty']).toBe('trafficRate');
    expect(
      selected.requirements.find(
        requirement =>
          requirement.role === 'duration-slope' && requirement.id === 'non-negative-values'
      )?.state
    ).toBe('passed');
  });

  it.each([
    'value',
    'height'
  ])('preserves point altitudes until the %s column is explicitly mapped as height', property => {
    const identifier = 'datum-grid-transform';
    const inputs = getSampleInputs(identifier);
    const points = makePoints(
      [{[property]: 1}, {[property]: 2}],
      [
        [0, 0, 100],
        [1, 0, 101]
      ]
    );
    const profiles = Object.entries({...inputs, points}).map(([name, collection]) =>
      profileKeplerFeatureCollection(collection, {
        datasetId: name,
        label: name,
        coordinateSystem: 'EPSG:4326'
      })
    );
    const binding = getCatalogBinding(identifier)!;
    const descriptor = getKeplerOperationDescriptor(identifier)!;
    const inputAssignments = {points: 'points', samples: 'samples'};
    const assessment = evaluateKeplerOperationApplicability(binding, descriptor, profiles, {
      inputAssignments
    });
    expect(assessment.status).toBe('ready');
    expect(assessment.fieldBindings['heightProperty']).toBeUndefined();
    expect(assessment.roleCandidates['height']).toContain(property);
    expect(assessment.parameters['heightProperty']).toBe('');
    const heights = (heightProperty: string) => {
      const prepared = prepareDatumPositions(points, heightProperty, true);
      return [2, 5].map(index => prepared.high[index] + prepared.low[index]);
    };
    expect(heights(assessment.parameters['heightProperty'] as string)).toEqual([100, 101]);
    const selected = evaluateKeplerOperationApplicability(binding, descriptor, profiles, {
      inputAssignments,
      fieldBindings: {heightProperty: property}
    });
    expect(selected.parameters['heightProperty']).toBe(property);
    expect(heights(selected.parameters['heightProperty'] as string)).toEqual([1, 2]);
    const unbound = resolveKeplerOperationParameters(binding, descriptor, {});
    expect(unbound.parameters['heightProperty']).toBe('');
    const overridden = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {},
      {overrides: {heightProperty: property}}
    );
    expect(overridden.parameters['heightProperty']).toBe(property);
  });

  it('checks optional duration slopes when selected, and permits their omission', () => {
    const identifier = 'network-travel-time-percentiles';
    const lines = getSampleInputs(identifier)['lines'];
    const profile = profileKeplerFeatureCollection(
      {
        ...lines,
        features: lines.features.map(feature => ({
          ...feature,
          properties: {...feature.properties, slope: -0.1}
        }))
      },
      {datasetId: 'lines', label: 'Network', coordinateSystem: 'planar'}
    );
    const assess = (slopeProperty: string) =>
      evaluateKeplerOperationApplicability(
        getCatalogBinding(identifier)!,
        getKeplerOperationDescriptor(identifier)!,
        [profile],
        {overrides: {slopeProperty}, executionEngine: 'gpu'}
      );
    expect(assess('').requirements.some(requirement => requirement.role === 'duration-slope')).toBe(
      false
    );
    expect(
      assess('slope').requirements.find(
        requirement =>
          requirement.role === 'duration-slope' && requirement.id === 'non-negative-values'
      )?.state
    ).toBe('failed');
  });

  it('keeps list-valued traversals usable without a scalar no-missing-values rejection', () => {
    const inputs = getSampleInputs('map-matched-edge-volumes');
    const profile = profileKeplerFeatureCollection(inputs['tracks'], {
      datasetId: 'tracks',
      label: 'Tracks'
    });
    expect(profile.columns.find(column => column.name === 'traversedEdges')).toMatchObject({
      type: 'text',
      nullCount: 1
    });
    const descriptor = getKeplerOperationDescriptor('map-matched-edge-volumes')!;
    expect(
      descriptor.requirements.some(requirement => requirement.role === 'traversed-edges')
    ).toBe(false);
    expect(descriptor.notes?.join(' ')).toMatch(/not an accepted substitute for an array/);
  });

  it('conditions datum reference settings and inverse controls, preserving altitude fallback', () => {
    const descriptor = getKeplerOperationDescriptor('datum-grid-transform')!;
    expect(descriptor.inputs[0].fields[0].required).toBe(false);
    for (const name of ['sourceVerticalReference', 'targetVerticalReference', 'heightUnitMeters']) {
      expect(descriptor.parameters.find(parameter => parameter.name === name)?.activeWhen).toEqual({
        parameter: 'operation',
        equals: ['vertical', 'combined']
      });
    }
    for (const name of ['maximumIterations', 'tolerance']) {
      expect(descriptor.parameters.find(parameter => parameter.name === name)?.activeWhen).toEqual({
        parameter: 'direction',
        equals: ['inverse']
      });
    }
    expect(descriptor.notes?.join(' ')).toMatch(/width \* height.*row/);
    expect(descriptor.outputs[0].presentation).toEqual({column: 'status', scale: 'categorical'});
  });

  it('requires geographic HEALPix input and validates optional values only when selected', () => {
    const identifier = 'healpix-cell-aggregation';
    const points = makePoints([{value: null}, {value: 2}]);
    const geographic = profileKeplerFeatureCollection(points, {
      datasetId: 'points',
      label: 'Points',
      coordinateSystem: 'EPSG:4326'
    });
    const assess = (valueProperty: string) =>
      evaluateKeplerOperationApplicability(
        getCatalogBinding(identifier)!,
        getKeplerOperationDescriptor(identifier)!,
        [geographic],
        {overrides: {valueProperty}, executionEngine: 'gpu'}
      );
    expect(assess('').requirements.some(requirement => requirement.role === 'value')).toBe(false);
    expect(
      assess('value').requirements.find(
        requirement => requirement.role === 'value' && requirement.id === 'no-missing-values'
      )?.state
    ).toBe('failed');
    expect(
      assessPoints(identifier, points, {valueProperty: 'value'}).requirements.find(
        requirement => requirement.id === 'geographic-coordinates'
      )?.state
    ).toBe('failed');
  });

  it('checks nonnegative transport masses and explains totals, layout and convergence', () => {
    const identifier = 'sinkhorn-divergence';
    const cells = makePoints([
      {source: -1, target: 0},
      {source: 1, target: 2}
    ]);
    const assessment = evaluateKeplerOperationApplicability(
      getCatalogBinding(identifier)!,
      getKeplerOperationDescriptor(identifier)!,
      [
        profileKeplerFeatureCollection(cells, {
          datasetId: 'cells',
          label: 'Mass cells',
          coordinateSystem: 'planar'
        })
      ],
      {fieldBindings: {sourceProperty: 'source', targetProperty: 'target'}, executionEngine: 'gpu'}
    );
    expect(assessment.status).toBe('blocked');
    expect(
      assessment.requirements.find(
        requirement => requirement.role === 'source' && requirement.id === 'non-negative-values'
      )?.state
    ).toBe('failed');
    const notes = getKeplerOperationDescriptor(identifier)!.notes!.join(' ');
    expect(notes).toMatch(/positive total mass/);
    expect(notes).toMatch(/normalized to one/);
    expect(notes).toMatch(/Geometry is ignored/);
    expect(notes).toMatch(/iterationLimitReached/);
  });
});
