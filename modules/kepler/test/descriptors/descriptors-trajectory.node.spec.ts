// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import * as arrow from 'apache-arrow';
import {evaluateGPUAnalysisParameterCondition} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {TRAJECTORY_BINDINGS} from '../../src/analysis/catalog-trajectory-bindings';
import {
  getCatalogDefaultParameters,
  runCatalogOperation
} from '../../src/analysis/catalog-bindings';
import {prepareCatalogTrajectory} from '../../src/analysis/catalog-trajectory-common';
import type {CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {TRAJECTORY_OPERATION_DESCRIPTORS} from '../../src/descriptors/index';
import {evaluateKeplerOperationApplicability} from '../../src/descriptors/applicability';
import {resolveKeplerOperationParameters} from '../../src/descriptors/default-parameters';
import {
  getKeplerOperationDescriptorCoverage,
  resolveKeplerOperationDescriptor
} from '../../src/descriptors/operation-descriptors';
import {validateKeplerOperationDescriptor} from '../../src/descriptors/validate-operation-descriptor';
import {
  profileKeplerColumnTable,
  profileKeplerFeatureCollection
} from '../../src/profile/dataset-profile';
import {
  makeCatalogTrajectoryNativeSource,
  readCatalogTrajectoryPoints
} from '../../../../examples/deck/kepler-compute/src/analysis-catalog-trajectory';

const ORIGIN = Date.UTC(2026, 0, 1);
const FIELD_BINDINGS = {trackProperty: 'animal', timeProperty: 'observed_at'};
function getBinding(bindingId: string) {
  const binding = TRAJECTORY_BINDINGS.find(candidate => candidate.id === bindingId);
  if (!binding) throw new Error(`Missing trajectory binding ${bindingId}`);
  return binding;
}
function getDescriptor(bindingId: string) {
  const descriptor = TRAJECTORY_OPERATION_DESCRIPTORS.find(
    candidate => candidate.bindingId === bindingId
  );
  if (!descriptor) throw new Error(`Missing trajectory descriptor ${bindingId}`);
  return descriptor;
}
function makeObservations(): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: Array.from({length: 8}, (_, index) => ({
      geometry: {type: 'Point', coordinates: [[0, 0, 0, 10, 20, 30, 30, 30][index], 0]},
      properties: {
        animal: 'animal-a',
        observed_at: ORIGIN + index * 10000,
        temperature: index,
        behavior: index < 3 ? 'rest' : 'move'
      }
    }))
  };
}
function makeProfile(collection = makeObservations()) {
  return profileKeplerFeatureCollection(collection, {
    datasetId: 'tracks',
    label: 'Track observations',
    coordinateSystem: 'planar'
  });
}

describe('trajectory descriptors on the integrated registry', () => {
  it('registers thirteen complete records without displacing another descriptor family', () => {
    expect(TRAJECTORY_OPERATION_DESCRIPTORS).toHaveLength(13);
    expect(TRAJECTORY_OPERATION_DESCRIPTORS.map(descriptor => descriptor.bindingId).sort()).toEqual(
      TRAJECTORY_BINDINGS.map(binding => binding.id).sort()
    );
    expect(getKeplerOperationDescriptorCoverage(TRAJECTORY_BINDINGS).derivedBindingIds).toEqual([]);
    for (const binding of TRAJECTORY_BINDINGS) {
      const descriptor = getDescriptor(binding.id);
      expect(resolveKeplerOperationDescriptor(binding)).toEqual({
        descriptor,
        source: 'hand-written'
      });
      expect(validateKeplerOperationDescriptor(descriptor, binding), binding.id).toEqual([]);
      expect(descriptor.questions).toEqual(['how-it-moves']);
      expect(descriptor.prominence).toBe(3);
      expect(descriptor.inputs[0].kind).toBe('trajectories');
      expect(descriptor.inputs[0].fields.map(field => field.parameter).sort()).toEqual(
        binding.parameters
          .filter(parameter => parameter.kind === 'string')
          .map(parameter => parameter.name)
          .sort()
      );
      expect(descriptor.outputs.map(output => output.tableId)).toEqual(
        binding.outputs.map(output => output.tableId)
      );
      for (const output of descriptor.outputs) {
        expect(output.presentation?.column).toBeTruthy();
        expect(output.interpretation).toEqual({summary: 'row-count', cautions: []});
        expect(output.next).toContain('how-it-moves');
      }
      expect(
        descriptor.parameters.every(
          parameter => parameter.apply === 'live' || parameter.apply === 'rebuild'
        )
      ).toBe(true);
      expect(descriptor.notes?.join(' ')).toMatch(/cannot certify embedded vertex times/);
    }
  });

  it('retains the landed false policy while supporting explicit identity fallback', () => {
    const binding = getBinding('trajectory-thinning');
    const descriptor = getDescriptor(binding.id);
    const cleared = {
      ...descriptor,
      inputs: descriptor.inputs.map(input => ({
        ...input,
        fields: input.fields.map(field =>
          field.role === 'track' ? {...field, autoAssign: false as const} : field
        )
      }))
    };
    const fallback = resolveKeplerOperationParameters(binding, descriptor, {tracks: makeProfile()});
    expect(fallback.parameters['trackProperty']).toBe('track_id');
    const explicit = resolveKeplerOperationParameters(binding, cleared, {tracks: makeProfile()});
    expect(explicit.parameters['trackProperty']).toBe('');
    expect(
      explicit.resolutions.find(resolution => resolution.parameter === 'trackProperty')?.reason
    ).toBe('This column must be selected explicitly');
    for (const candidate of [descriptor, cleared]) {
      const result = evaluateKeplerOperationApplicability(binding, candidate, [makeProfile()], {
        fieldBindings: FIELD_BINDINGS,
        overrides: {timestampEncoding: 'unix-millisecond'}
      });
      expect(result.status, JSON.stringify(result.missing)).toBe('ready');
      expect(result.parameters['trackProperty']).toBe('animal');
    }
  });

  it.each([
    false,
    true
  ])('preserves feature-ID tracks through applicability and execution (overlapping times: %s)', overlapping => {
    const collection: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: [0, 1].map(index => ({
        id: `fox-${index}`,
        properties: {species: 'fox'},
        geometry: {
          type: 'LineString',
          coordinates: [0, 1].map(vertex => [
            index * 10 + vertex,
            0,
            0,
            ORIGIN + ((overlapping ? 0 : index * 2) + vertex) * 1000
          ])
        }
      }))
    };
    const binding = getBinding('trajectory-thinning');
    const descriptor = getDescriptor(binding.id);
    const result = evaluateKeplerOperationApplicability(
      binding,
      descriptor,
      [makeProfile(collection)],
      {overrides: {mode: 'minimum-time', tolerance: 0}}
    );
    expect(result.status, JSON.stringify(result.missing)).toBe('ready');
    expect(result.fieldBindings['trackProperty']).toBeUndefined();
    expect(result.parameters['trackProperty']).toBe('track_id');
    const output = runCatalogOperation(binding, result.parameters, {tracks: collection});
    expect(output.summary['trackCount']).toBe(2);
    expect(output.tables[0].columns['track_id']).toEqual(['fox-0', 'fox-1']);
    expect(output.tables[1].columns['vertex_index']).toEqual([0, 1, 0, 1]);
    if (!overlapping) {
      const selected = evaluateKeplerOperationApplicability(
        binding,
        descriptor,
        [makeProfile(collection)],
        {fieldBindings: {trackProperty: 'species'}, overrides: {mode: 'minimum-time', tolerance: 0}}
      );
      expect(
        runCatalogOperation(binding, selected.parameters, {tracks: collection}).summary[
          'trackCount'
        ]
      ).toBe(1);
    }
  });

  it('resolves native coordinates from pointColumns or explicit mappings and executes native ingestion', () => {
    const rows = {
      animal: ['a', 'a'],
      observed_at: [ORIGIN, ORIGIN + 1000],
      east: [3, 6],
      north: [4, 8],
      temperature: [20, 21],
      mass: [5, 6]
    };
    const profile = profileKeplerColumnTable(
      {rowCount: 2, columns: Object.entries(rows).map(([name, values]) => ({name, values}))},
      {
        datasetId: 'native',
        label: 'Native points',
        coordinateSystem: 'planar',
        pointColumns: {longitude: 'east', latitude: 'north'}
      }
    );
    const binding = getBinding('trajectory-thinning');
    const descriptor = getDescriptor(binding.id);
    const options = {
      fieldBindings: FIELD_BINDINGS,
      overrides: {timestampEncoding: 'unix-millisecond', mode: 'minimum-time', tolerance: 0}
    };
    const defaults = resolveKeplerOperationParameters(binding, descriptor, {tracks: profile});
    expect(defaults.parameters['longitudeProperty']).toBe('east');
    expect(defaults.parameters['latitudeProperty']).toBe('north');
    const result = evaluateKeplerOperationApplicability(binding, descriptor, [profile], options);
    expect(result.status, JSON.stringify(result.missing)).toBe('ready');
    expect(result.parameters['longitudeProperty']).toBe('east');
    expect(result.parameters['latitudeProperty']).toBe('north');
    const source = makeCatalogTrajectoryNativeSource({
      id: 'native',
      label: 'Native points',
      table: arrow.tableFromArrays(rows)
    });
    const readAndRun = (parameters: typeof result.parameters) =>
      runCatalogOperation(binding, parameters, {
        tracks: readCatalogTrajectoryPoints(
          source,
          String(parameters['longitudeProperty']),
          String(parameters['latitudeProperty'])
        )
      });
    expect(readAndRun(result.parameters).tables[1].columns['timestamp']).toEqual([
      ORIGIN,
      ORIGIN + 1000
    ]);
    const withoutMetadata = {...profile, pointColumns: undefined, geometryFeatureCounts: undefined};
    const missing = evaluateKeplerOperationApplicability(
      binding,
      descriptor,
      [withoutMetadata],
      options
    );
    expect(missing.status).toBe('needs-mapping');
    expect(missing.fieldBindings['longitudeProperty']).toBeUndefined();
    expect(missing.fieldBindings['latitudeProperty']).toBeUndefined();
    expect(
      missing.missing.filter(problem => problem.what.includes('native point column'))
    ).toHaveLength(2);
    const explicit = evaluateKeplerOperationApplicability(binding, descriptor, [withoutMetadata], {
      ...options,
      fieldBindings: {...FIELD_BINDINGS, longitudeProperty: 'east', latitudeProperty: 'north'}
    });
    expect(explicit.status, JSON.stringify(explicit.missing)).toBe('ready');
    expect(readAndRun(explicit.parameters).summary['fixCount']).toBe(2);
  });

  it('resolves ISO encoding from complete evidence and passes resolved parameters to preparation', () => {
    const collection = makeObservations();
    collection.features.forEach(feature => {
      feature.properties!['timestamp'] = new Date(
        Number(feature.properties!['observed_at'])
      ).toISOString();
      delete feature.properties!['observed_at'];
    });
    const binding = getBinding('trajectory-thinning');
    const result = evaluateKeplerOperationApplicability(
      binding,
      getDescriptor(binding.id),
      [makeProfile(collection)],
      {fieldBindings: {trackProperty: 'animal'}, overrides: {mode: 'minimum-time', tolerance: 0}}
    );
    expect(result.status, JSON.stringify(result.missing)).toBe('ready');
    expect(result.parameters['timeProperty']).toBe('timestamp');
    expect(result.parameters['timestampEncoding']).toBe('iso8601');
    expect(
      prepareCatalogTrajectory(collection, result.parameters).fixes.map(fix => fix.timestamp)
    ).toEqual(Array.from({length: 8}, (_, index) => ORIGIN + index * 10000));
    expect(
      runCatalogOperation(binding, result.parameters, {tracks: collection}).summary['fixCount']
    ).toBe(8);
  });

  it('requires numeric timestamp encoding explicitly, including subject-only applicability', () => {
    const collection = makeObservations();
    collection.features.forEach(feature => {
      feature.properties!['observed_at'] = Number(feature.properties!['observed_at']) / 1000;
    });
    const binding = getBinding('trajectory-thinning');
    const descriptor = getDescriptor(binding.id);
    const profile = makeProfile(collection);
    for (const subjectOnly of [false, true]) {
      const result = evaluateKeplerOperationApplicability(binding, descriptor, [profile], {
        fieldBindings: FIELD_BINDINGS,
        subjectOnly
      });
      expect(result.status, JSON.stringify(result.missing)).toBe('needs-confirmation');
      expect(
        result.missing.some(problem => problem.what.includes('Choose timestampEncoding explicitly'))
      ).toBe(true);
    }
    const resolved = evaluateKeplerOperationApplicability(binding, descriptor, [profile], {
      fieldBindings: FIELD_BINDINGS,
      overrides: {timestampEncoding: 'unix-second'}
    });
    expect(resolved.status, JSON.stringify(resolved.missing)).toBe('ready');
    expect(prepareCatalogTrajectory(collection, resolved.parameters).fixes[0].timestamp).toBe(
      ORIGIN
    );
  });

  it('does not infer ISO encoding from mixed or sampled columns', () => {
    const collection = makeObservations();
    collection.features.forEach(feature => {
      feature.properties!['observed_at'] = new Date(
        Number(feature.properties!['observed_at'])
      ).toISOString();
    });
    const sampled = profileKeplerFeatureCollection(collection, {
      datasetId: 'tracks',
      label: 'Sampled',
      coordinateSystem: 'planar',
      maximumSampleSize: 2
    });
    collection.features[0].properties!['observed_at'] = ORIGIN / 1000;
    const mixed = profileKeplerFeatureCollection(collection, {
      datasetId: 'tracks',
      label: 'Mixed',
      coordinateSystem: 'planar',
      columnTypes: {observed_at: 'time'}
    });
    const binding = getBinding('trajectory-thinning');
    for (const profile of [sampled, mixed]) {
      expect(
        profile.columns.find(column => column.name === 'observed_at')?.timestampEncoding
      ).toBeUndefined();
      const result = evaluateKeplerOperationApplicability(
        binding,
        getDescriptor(binding.id),
        [profile],
        {fieldBindings: FIELD_BINDINGS}
      );
      expect(result.status, JSON.stringify(result.missing)).toBe('needs-confirmation');
    }
  });

  it.each([
    ['trajectory-hampel', 'target', 'property', 'x', 'value', 'temperature'],
    ['trajectory-split', 'mode', 'value-change', 'gap', 'category', 'behavior']
  ])('%s requires a compatible scalar only in the consuming mode', (bindingId, controller, active, inactive, role, column) => {
    const binding = getBinding(bindingId);
    const descriptor = getDescriptor(bindingId);
    const field = descriptor.inputs[0].fields.find(candidate => candidate.role === role)!;
    if (typeof field.required !== 'object') throw new Error('Expected conditional scalar role');
    expect(evaluateGPUAnalysisParameterCondition(field.required, {[controller]: active})).toBe(
      true
    );
    expect(evaluateGPUAnalysisParameterCondition(field.required, {[controller]: inactive})).toBe(
      false
    );
    const missing = makeObservations();
    missing.features.forEach(feature => {
      delete feature.properties!['temperature'];
      delete feature.properties!['behavior'];
    });
    const inactiveResult = evaluateKeplerOperationApplicability(
      binding,
      descriptor,
      [makeProfile(missing)],
      {fieldBindings: FIELD_BINDINGS, overrides: {[controller]: inactive}}
    );
    expect(inactiveResult.requirements.some(requirement => requirement.role === role)).toBe(false);
    const unmapped = evaluateKeplerOperationApplicability(
      binding,
      descriptor,
      [makeProfile(missing)],
      {fieldBindings: FIELD_BINDINGS, overrides: {[controller]: active}}
    );
    expect(unmapped.status).not.toBe('ready');
    const options = {
      fieldBindings: {...FIELD_BINDINGS, valueProperty: column},
      overrides: {[controller]: active}
    };
    const mapped = evaluateKeplerOperationApplicability(
      binding,
      descriptor,
      [makeProfile()],
      options
    );
    expect(mapped.fieldBindings['valueProperty']).toBe(column);
    expect(mapped.requirements).toContainEqual(
      expect.objectContaining({role, id: 'no-missing-values', state: 'passed'})
    );
    const withNull = makeObservations();
    withNull.features[0].properties![column] = null;
    const rejected = evaluateKeplerOperationApplicability(
      binding,
      descriptor,
      [makeProfile(withNull)],
      options
    );
    expect(rejected.status).toBe('blocked');
    expect(rejected.requirements).toContainEqual(
      expect.objectContaining({role, id: 'no-missing-values', state: 'failed'})
    );
  });

  it('keeps uncertain discarded-row scalar nulls separate from definite retained-fix failures', () => {
    const collection = makeObservations();
    collection.features = collection.features.slice(0, 2);
    collection.features.push({
      geometry: null,
      properties: {animal: 'a', observed_at: ORIGIN, behavior: null}
    });
    const binding = getBinding('trajectory-split');
    const descriptor = getDescriptor(binding.id);
    const options = {
      fieldBindings: {...FIELD_BINDINGS, valueProperty: 'behavior'},
      overrides: {mode: 'value-change', timestampEncoding: 'unix-millisecond'}
    };
    const result = evaluateKeplerOperationApplicability(
      binding,
      descriptor,
      [makeProfile(collection)],
      options
    );
    expect(result.status, JSON.stringify(result.missing)).toBe('needs-confirmation');
    expect(result.requirements).toContainEqual(
      expect.objectContaining({
        role: 'category',
        id: 'no-missing-values',
        state: 'unknown',
        evidence: 'not-checkable'
      })
    );
    const output = runCatalogOperation(binding, result.parameters, {tracks: collection});
    expect(output.summary['fixCount']).toBe(2);
    expect(output.tables[1].rowCount).toBe(2);
    expect(output.diagnostics).toContainEqual(
      expect.objectContaining({code: 'trajectory-empty-geometry', count: 1})
    );
    collection.features.pop();
    collection.features[0].properties!['behavior'] = null;
    const rejected = evaluateKeplerOperationApplicability(
      binding,
      descriptor,
      [makeProfile(collection)],
      options
    );
    expect(rejected.status).toBe('blocked');
    expect(() => runCatalogOperation(binding, rejected.parameters, {tracks: collection})).toThrow(
      /requires a scalar valueProperty/
    );
  });

  it('checks row, vertex and explicitly mapped track capacities before execution', () => {
    const binding = getBinding('trajectory-thinning');
    const descriptor = getDescriptor(binding.id);
    for (const [property, value, requirementId] of [
      ['rowCount', 0, 'minimum-rows'],
      ['rowCount', 2049, 'maximum-rows'],
      ['vertexCount', 2049, 'maximum-vertices']
    ] as const) {
      const result = evaluateKeplerOperationApplicability(
        binding,
        descriptor,
        [{...makeProfile(), [property]: value}],
        {fieldBindings: FIELD_BINDINGS}
      );
      expect(result.status).toBe('blocked');
      expect(result.requirements).toContainEqual(
        expect.objectContaining({id: requirementId, state: 'failed', remedy: expect.any(String)})
      );
    }
    const collection = makeObservations();
    collection.features = Array.from({length: 129}, (_, index) => ({
      ...collection.features[0],
      properties: {animal: `animal-${index}`, observed_at: ORIGIN}
    }));
    const result = evaluateKeplerOperationApplicability(
      binding,
      descriptor,
      [makeProfile(collection)],
      {fieldBindings: FIELD_BINDINGS}
    );
    expect(result.status).toBe('blocked');
    expect(result.requirements).toContainEqual(
      expect.objectContaining({id: 'maximum-distinct-values', role: 'track', state: 'failed'})
    );
    expect(() =>
      prepareCatalogTrajectory(collection, {
        ...getCatalogDefaultParameters(binding),
        ...FIELD_BINDINGS,
        coordinateSystem: 'planar'
      })
    ).toThrow(/trackCount.*128/);
  });

  it('leaves guessed coordinates unresolved and hides inactive split settings', () => {
    const binding = getBinding('trajectory-staypoints');
    const profile = {
      ...makeProfile(),
      coordinates: {model: 'planar' as const, source: 'guess' as const}
    };
    const result = evaluateKeplerOperationApplicability(
      binding,
      getDescriptor(binding.id),
      [profile],
      {fieldBindings: FIELD_BINDINGS}
    );
    expect(result.status).not.toBe('ready');
    expect(result.requirements).toContainEqual(
      expect.objectContaining({id: 'declared-coordinates', state: 'unknown'})
    );
    for (const parameter of getDescriptor('trajectory-split').parameters.filter(
      parameter => parameter.activeWhen
    )) {
      expect(
        evaluateGPUAnalysisParameterCondition(parameter.activeWhen!, {mode: 'value-change'})
      ).toBe(false);
      expect(evaluateGPUAnalysisParameterCondition(parameter.activeWhen!, {mode: 'stop'})).toBe(
        true
      );
    }
  });

  it('derives a grid margin without estimating motion uncertainty, and respects overrides and fallback', () => {
    const collection = makeObservations();
    collection.features[7].geometry = {type: 'Point', coordinates: [30, 40]};
    for (const bindingId of ['trajectory-brownian-bridge', 'trajectory-isopleths']) {
      const binding = getBinding(bindingId);
      const descriptor = getDescriptor(bindingId);
      const profiles = {tracks: makeProfile(collection)};
      const resolved = resolveKeplerOperationParameters(binding, descriptor, profiles);
      expect(resolved.parameters['padding']).toBe(5); // hypot(30,40)/10
      expect(
        resolved.resolutions.find(resolution => resolution.parameter === 'padding')?.source
      ).toBe('data');
      expect(resolved.parameters['motionVariance']).toBe(10);
      expect(resolved.parameters['locationError']).toBe(10);
      expect(
        resolveKeplerOperationParameters(binding, descriptor, profiles, {overrides: {padding: 12}})
          .parameters['padding']
      ).toBe(12);
      const fallback = resolveKeplerOperationParameters(binding, descriptor, {
        tracks: {...profiles.tracks, extent: undefined, spacing: undefined}
      });
      expect(fallback.parameters['padding']).toBe(100);
      expect(
        fallback.resolutions.find(resolution => resolution.parameter === 'padding')?.source
      ).toBe('fallback');
    }
  });

  it('names existing columns on CPU outputs and records censoring and method limitations', () => {
    for (const binding of TRAJECTORY_BINDINGS.filter(candidate => candidate.run)) {
      const descriptor = getDescriptor(binding.id);
      const resolved = resolveKeplerOperationParameters(
        binding,
        descriptor,
        {tracks: makeProfile()},
        {
          fieldBindings: {
            ...FIELD_BINDINGS,
            ...(binding.id === 'trajectory-split' ? {valueProperty: 'behavior'} : {})
          },
          overrides: {
            coordinateSystem: 'planar',
            distanceThreshold: 1,
            timeThreshold: 15,
            activityMinimumDuration: 15
          }
        }
      );
      const output = runCatalogOperation(binding, resolved.parameters, {
        tracks: makeObservations()
      });
      for (const description of descriptor.outputs) {
        const table = output.tables.find(candidate => candidate.id === description.tableId);
        expect(table, `${binding.id}/${description.tableId}`).toBeDefined();
        expect(Object.keys(table!.columns)).toContain(description.presentation!.column);
      }
    }
    expect(getDescriptor('trajectory-recursions').notes?.join(' ')).toMatch(
      /null with first_passage_censored=true/
    );
    expect(getDescriptor('trajectory-modal-split').notes?.join(' ')).toMatch(
      /not inferred transport labels/
    );
    expect(getDescriptor('trajectory-mobility-indicators').notes?.join(' ')).toMatch(
      /not time-of-day/
    );
    expect(getDescriptor('trajectory-isopleths').notes?.join(' ')).toMatch(
      /not statistical confidence/
    );
  });
});
