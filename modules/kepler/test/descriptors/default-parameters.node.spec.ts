// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_ANALYSIS_DEFAULT_RULES} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {describe, expect, it} from 'vitest';
import type {CatalogOperationBinding} from '../../src/analysis/catalog-binding-types';
import {
  getCatalogBinding,
  KEPLER_CATALOG_BINDINGS,
  normalizeCatalogParameters
} from '../../src/analysis/catalog-bindings';
import {
  DATA_DEFAULT_SIGNIFICANT_DIGITS,
  KEPLER_DEFAULT_RULE_ROUNDING,
  KEPLER_GRID_CELL_CAPS,
  resolveKeplerOperationParameters
} from '../../src/descriptors/default-parameters';
import {
  deriveKeplerOperationDescriptor,
  getKeplerOperationDescriptor,
  resolveKeplerOperationDescriptor
} from '../../src/descriptors/operation-descriptors';
import type {
  KeplerOperationDescriptor,
  KeplerOperationParameterDescriptor
} from '../../src/descriptors/operation-descriptor-types';
import type {KeplerDatasetProfile} from '../../src/profile/dataset-profile-types';
import type {CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {
  createTestPointCollection,
  buildSampleProfile,
  buildTestProfile
} from './engine-test-profiles';

/** A realistic profile for each geometry a binding input accepts. */
function getProfileForInput(accepts: readonly string[]): KeplerDatasetProfile {
  if (accepts.includes('point')) return buildSampleProfile('incidents');
  if (accepts.includes('polygon')) return buildSampleProfile('districts');
  return buildSampleProfile('streets');
}

function getProfilesByInput(
  binding: CatalogOperationBinding
): Record<string, KeplerDatasetProfile> {
  return Object.fromEntries(
    binding.inputs.map(input => [input.name, getProfileForInput(input.accepts)])
  );
}

describe('resolveKeplerOperationParameters', () => {
  it('gives parameters that normalizeCatalogParameters accepts unchanged, for every binding', () => {
    for (const binding of KEPLER_CATALOG_BINDINGS) {
      const {descriptor} = resolveKeplerOperationDescriptor(binding);
      for (const profiles of [getProfilesByInput(binding), {}]) {
        const resolved = resolveKeplerOperationParameters(binding, descriptor, profiles);
        expect(normalizeCatalogParameters(binding, resolved.parameters), binding.id).toEqual(
          resolved.parameters
        );
        expect(
          resolved.resolutions.map(entry => entry.parameter),
          binding.id
        ).toEqual(binding.parameters.map(parameter => parameter.name));
        expect([...resolved.basic, ...resolved.advanced].sort(), binding.id).toEqual(
          binding.parameters.map(parameter => parameter.name).sort()
        );
      }
    }
  });

  it('derives each data-policy parameter from the profile, differing from the constant and in range', () => {
    let derivedCount = 0;
    for (const binding of KEPLER_CATALOG_BINDINGS) {
      const descriptor = getKeplerOperationDescriptor(binding.id);
      if (!descriptor) {
        continue;
      }
      const profiles = getProfilesByInput(binding);
      // The column rules read a bound column: bind a usable one to each role they name.
      const fieldBindings: Record<string, string> = {};
      for (const entry of descriptor.parameters) {
        const rule = entry.defaultPolicy;
        if (rule.from !== 'data' || rule.role === undefined) continue;
        const field = descriptor.inputs
          .find(input => input.name === rule.input)!
          .fields.find(candidate => candidate.role === rule.role)!;
        const column = profiles[rule.input].columns.find(
          candidate =>
            (field.accepts as readonly string[]).includes(candidate.type) &&
            (candidate.minimum ?? 0) < (candidate.maximum ?? 0)
        );
        expect(
          column,
          `${binding.id}.${entry.name} needs a ${field.accepts.join('/')} column`
        ).toBeDefined();
        fieldBindings[field.parameter] = column!.name;
      }
      const resolved = resolveKeplerOperationParameters(binding, descriptor, profiles, {
        fieldBindings
      });
      for (const policy of descriptor.parameters.filter(
        parameter => parameter.defaultPolicy.from === 'data'
      )) {
        const specification = binding.parameters.find(parameter => parameter.name === policy.name)!;
        const resolution = resolved.resolutions.find(entry => entry.parameter === policy.name)!;
        const label = `${binding.id}.${policy.name}`;
        if (
          policy.defaultPolicy.from === 'data' &&
          policy.defaultPolicy.rule === 'k-distance-knee'
        ) {
          // Schubert et al. (2017): the sorted k-distance plot often has no clear knee. The rule
          // then declines and the stated constant stays, with the reason.
          if (resolution.source === 'fallback') {
            expect(resolution.resolution?.explanation, label).toMatch(/knee/i);
            continue;
          }
        }
        expect(resolution.source, label).toBe('data');
        const value = resolved.parameters[policy.name] as number;
        expect(Number.isFinite(value), label).toBe(true);
        expect(value, label).not.toBe(specification.defaultValue);
        expect(value, label).toBeGreaterThanOrEqual(specification.minimum ?? -Infinity);
        expect(value, label).toBeLessThanOrEqual(specification.maximum ?? Infinity);
        derivedCount++;
      }
    }
    // 31 data-policy parameters exist today, 4 of them knee rules that may decline; a drop below 25
    // means a rule stopped firing.
    expect(derivedCount).toBeGreaterThanOrEqual(25);
  });

  it('falls back to the binding default and says so when the profile has no spacing', () => {
    const binding = getCatalogBinding('kernel-density-grid')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const {spacing: _spacing, extent: _extent, ...withoutSpacing} = buildSampleProfile('incidents');
    const resolved = resolveKeplerOperationParameters(binding, descriptor, {
      points: withoutSpacing
    });
    for (const name of ['cellSize', 'bandwidth']) {
      const resolution = resolved.resolutions.find(entry => entry.parameter === name)!;
      expect(resolution.source, name).toBe('fallback');
      expect(resolved.parameters[name], name).toBe(
        binding.parameters.find(parameter => parameter.name === name)!.defaultValue
      );
      expect(resolution.resolution?.value, name).toBeUndefined();
      expect(resolution.resolution?.explanation, name).toBeTruthy();
    }
    // With no profile at all the rule is not even asked.
    const none = resolveKeplerOperationParameters(binding, descriptor, {});
    expect(none.resolutions.find(entry => entry.parameter === 'cellSize')?.source).toBe('fallback');
  });

  it('gives metre-scale distances, not degrees, for a geographic profile', () => {
    const binding = getCatalogBinding('kernel-density-grid')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const profile = buildSampleProfile('incidents');
    const resolved = resolveKeplerOperationParameters(binding, descriptor, {points: profile});
    const cellSize = resolved.resolutions.find(entry => entry.parameter === 'cellSize')!;
    expect(cellSize.resolution?.unit).toBe('metre');
    // The sample is about 9 km by 8 km; a degree-scale value would be 0.0x.
    expect(resolved.parameters['cellSize'] as number).toBeGreaterThan(5);
    expect(resolved.parameters['cellSize'] as number).toBeLessThan(500);
    expect(profile.coordinates?.model).toBe('geographic');
  });

  it('rounds a derived value to a few significant digits, rounding the neighbour floor up', () => {
    const binding = getCatalogBinding('hotspot-gi-star')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const resolved = resolveKeplerOperationParameters(binding, descriptor, {
      features: buildSampleProfile('districts')
    });
    const entry = resolved.resolutions.find(candidate => candidate.parameter === 'distanceBand')!;
    const value = resolved.parameters['distanceBand'] as number;
    expect(entry.source).toBe('data');
    expect(Number(value.toPrecision(DATA_DEFAULT_SIGNIFICANT_DIGITS))).toBe(value);
    // The largest nearest-neighbour distance is rounded up so every feature keeps a neighbour.
    expect(value).toBeGreaterThanOrEqual(entry.resolution!.value!);
    expect(value).toBeLessThan(entry.resolution!.value! * 1.11);
  });

  it('records a move into the parameter range', () => {
    const original = getCatalogBinding('kernel-density-grid')!;
    const narrow: CatalogOperationBinding = {
      ...original,
      parameters: original.parameters.map(parameter =>
        parameter.name === 'cellSize' ? {...parameter, minimum: 0.001, maximum: 5} : parameter
      )
    };
    const resolved = resolveKeplerOperationParameters(
      narrow,
      getKeplerOperationDescriptor('kernel-density-grid')!,
      {points: buildSampleProfile('incidents')}
    );
    const entry = resolved.resolutions.find(candidate => candidate.parameter === 'cellSize')!;
    expect(resolved.parameters['cellSize']).toBe(5);
    expect(entry.source).toBe('data');
    expect(entry.clamped?.to).toBe(5);
    expect(entry.clamped!.from).toBeGreaterThan(5);
    const tooSmall: CatalogOperationBinding = {
      ...original,
      parameters: original.parameters.map(parameter =>
        parameter.name === 'cellSize' ? {...parameter, minimum: 1e6, maximum: 1e7} : parameter
      )
    };
    const raised = resolveKeplerOperationParameters(
      tooSmall,
      getKeplerOperationDescriptor('kernel-density-grid')!,
      {points: buildSampleProfile('incidents')}
    );
    expect(raised.parameters['cellSize']).toBe(1e6);
    expect(
      raised.resolutions.find(candidate => candidate.parameter === 'cellSize')?.clamped?.to
    ).toBe(1e6);
  });

  it('rounds the value of an integer parameter and keeps it inside an integer range', () => {
    const binding = {
      ...getCatalogBinding('kernel-density-grid')!,
      parameters: [
        {
          name: 'cells',
          label: 'Cells',
          kind: 'integer' as const,
          minimum: 3,
          maximum: 50.5,
          defaultValue: 10
        }
      ]
    };
    const base = getKeplerOperationDescriptor('kernel-density-grid')!;
    const descriptor: KeplerOperationDescriptor = {
      ...base,
      parameters: [
        {
          name: 'cells',
          defaultPolicy: {from: 'data', input: 'points', rule: 'extent-shorter-side-over-250'},
          level: 'basic'
        }
      ],
      inputs: base.inputs.map(input => ({...input, fields: []}))
    };
    const resolved = resolveKeplerOperationParameters(binding, descriptor, {
      points: buildSampleProfile('incidents')
    });
    const value = resolved.parameters['cells'] as number;
    expect(Number.isInteger(value)).toBe(true);
    expect(value).toBeGreaterThanOrEqual(3);
    expect(value).toBeLessThanOrEqual(50);
    expect(normalizeCatalogParameters(binding, resolved.parameters)).toEqual(resolved.parameters);
  });

  it('writes field bindings into string parameters, comma-joined for several columns', () => {
    const binding = getCatalogBinding('regression-ols')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const resolved = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {},
      {
        fieldBindings: {responseProperty: 'rate', predictorProperties: ['cases', 'population']}
      }
    );
    expect(resolved.parameters['responseProperty']).toBe('rate');
    expect(resolved.parameters['predictorProperties']).toBe('cases,population');
    expect(
      resolved.resolutions.filter(entry => entry.source === 'field').map(entry => entry.parameter)
    ).toEqual(['responseProperty', 'predictorProperties']);
    expect(resolved.basic).toContain('responseProperty');
    expect(normalizeCatalogParameters(binding, resolved.parameters)).toEqual(resolved.parameters);
  });

  it('rejects a field binding for a parameter that is not a string parameter, and one that is too long', () => {
    const binding = getCatalogBinding('kernel-density-grid')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    expect(() =>
      resolveKeplerOperationParameters(binding, descriptor, {}, {fieldBindings: {cellSize: 'x'}})
    ).toThrow(
      /resolveKeplerOperationParameters fieldBindings\.cellSize must name a string parameter/
    );
    expect(() =>
      resolveKeplerOperationParameters(binding, descriptor, {}, {fieldBindings: {nope: 'x'}})
    ).toThrow(/fieldBindings\.nope/);
    expect(() =>
      resolveKeplerOperationParameters(
        binding,
        descriptor,
        {},
        {fieldBindings: {weightProperty: 'w'.repeat(65)}}
      )
    ).toThrow(/at most 64/);
  });

  it('lets an override replace a derived value and a field binding', () => {
    const binding = getCatalogBinding('kernel-density-grid')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const resolved = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {points: buildSampleProfile('incidents')},
      {overrides: {cellSize: 42, weightProperty: 'severity'}, fieldBindings: {weightProperty: 'x'}}
    );
    expect(resolved.parameters['cellSize']).toBe(42);
    expect(resolved.parameters['weightProperty']).toBe('severity');
    expect(resolved.resolutions.find(entry => entry.parameter === 'cellSize')?.source).toBe(
      'override'
    );
    expect(resolved.resolutions.find(entry => entry.parameter === 'weightProperty')?.source).toBe(
      'override'
    );
  });

  it('follows the coordinate model of the profile for the coordinate system parameter', () => {
    const binding = getCatalogBinding('kernel-density-grid')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const planar = buildSampleProfile('incidents', {
      coordinates: {model: 'planar', source: 'declared'}
    });
    const resolved = resolveKeplerOperationParameters(binding, descriptor, {points: planar});
    expect(resolved.parameters['coordinateSystem']).toBe('planar');
    expect(resolved.resolutions.find(entry => entry.parameter === 'coordinateSystem')?.source).toBe(
      'data'
    );
    const geographic = resolveKeplerOperationParameters(binding, descriptor, {
      points: buildSampleProfile('incidents')
    });
    expect(geographic.parameters['coordinateSystem']).toBe('EPSG:4326');
  });

  it('survives a profile with one point and with non-finite spacing values', () => {
    const binding = getCatalogBinding('local-moran-statistics')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const profile = buildSampleProfile('districts');
    const broken: KeplerDatasetProfile = {
      ...profile,
      spacing: {
        ...profile.spacing!,
        nearestNeighbor: {
          ...profile.spacing!.nearestNeighbor!,
          minimum: NaN,
          median: NaN,
          mean: NaN,
          maximum: NaN,
          upperFence: NaN
        }
      }
    };
    const resolved = resolveKeplerOperationParameters(binding, descriptor, {features: broken});
    expect(resolved.resolutions.find(entry => entry.parameter === 'distanceBand')?.source).toBe(
      'fallback'
    );
    expect(normalizeCatalogParameters(binding, resolved.parameters)).toEqual(resolved.parameters);
  });
});

describe('grid cell cap', () => {
  /** A tight blob plus 100 points over a 0.107 by 0.01 degree box: about 8.9 km by 1.1 km. */
  function makeElongatedProfile() {
    let state = 12345;
    const next = () => {
      state = (state * 1103515245 + 12345) % 2147483648;
      return state / 2147483648;
    };
    // 2,900 points in a 0.0005 degree blob and 100 spread along the box: the spacing is set by the
    // blob, so a cell size from the spacing is far too small for the 8.9 km by 1.1 km extent.
    const features = Array.from({length: 3000}, (_, index) => ({
      id: index,
      geometry: {
        type: 'Point',
        coordinates:
          index < 2900
            ? [-87.7 + next() * 0.0005, 41.85 + next() * 0.0005]
            : [-87.7 + next() * 0.107, 41.85 + next() * 0.01]
      },
      properties: {}
    }));
    const collection: CatalogFeatureCollection = {type: 'FeatureCollection', features};
    return {
      collection,
      profile: buildTestProfile({
        datasetId: 'elongated',
        label: 'elongated',
        collection,
        kind: 'points',
        catalogGeometry: 'point',
        rowSemantics: 'events'
      })
    };
  }

  it('lists a cap read from code for every binding whose cell size comes from the extent', () => {
    for (const id of Object.keys(KEPLER_GRID_CELL_CAPS)) {
      expect(getCatalogBinding(id), id).toBeDefined();
      expect(KEPLER_GRID_CELL_CAPS[id].source, id).toMatch(/\.ts:\d+/);
    }
    for (const binding of KEPLER_CATALOG_BINDINGS) {
      const {descriptor} = resolveKeplerOperationDescriptor(binding);
      const derivesCellSize = descriptor.parameters.some(
        parameter =>
          ['cellSize', 'cellWidth', 'cellHeight'].includes(parameter.name) &&
          parameter.defaultPolicy.from === 'data' &&
          (parameter.defaultPolicy.rule === 'extent-shorter-side-over-250' ||
            parameter.defaultPolicy.rule === 'mean-point-spacing' ||
            (parameter.defaultPolicy.rule === 'extent-shorter-side-over-250-in-source-units' &&
              binding.id === 'grid-generator'))
      );
      expect(binding.id in KEPLER_GRID_CELL_CAPS, binding.id).toBe(derivesCellSize);
    }
  });

  /** 400 valued points spread over the elongated box, for the binding that needs a value. */
  function makeValuedElongatedProfile() {
    let state = 4242;
    const next = () => {
      state = (state * 1103515245 + 12345) % 2147483648;
      return state / 2147483648;
    };
    const features = Array.from({length: 400}, (_, index) => ({
      id: index,
      geometry: {type: 'Point', coordinates: [-87.7 + next() * 0.107, 41.85 + next() * 0.01]},
      properties: {v: Math.round(next() * 1000) / 10}
    }));
    const collection: CatalogFeatureCollection = {type: 'FeatureCollection', features};
    return {
      collection,
      profile: buildTestProfile({
        datasetId: 'valued',
        label: 'valued',
        collection,
        kind: 'points',
        catalogGeometry: 'point',
        rowSemantics: 'surface-samples'
      })
    };
  }

  it('raises the kriging cell size so the real binding runs on an elongated extent', () => {
    // The extent rule gives 250 cells on the shorter side (times 2 for kriging), so an extent of
    // aspect ratio 8 asks for more than the binding's 40,000 cells.
    const binding = getCatalogBinding('kriging-grid')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const {collection, profile} = makeValuedElongatedProfile();
    expect(profile.spacing!.extentWidth / profile.spacing!.extentHeight).toBeGreaterThan(7);
    const fieldBindings = {valueProperty: 'v'};
    const run = (parameters: Record<string, unknown>) =>
      binding.run!({points: collection}, normalizeCatalogParameters(binding, parameters));

    const uncapped = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {points: profile},
      {fieldBindings}
    );
    expect(() => run(uncapped.parameters)).toThrow();

    const capped = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {points: profile},
      {fieldBindings, maximumCellCount: KEPLER_GRID_CELL_CAPS[binding.id].maximumCellCount}
    );
    const entry = capped.resolutions.find(candidate => candidate.parameter === 'cellSize')!;
    expect(capped.parameters['cellSize'] as number).toBeGreaterThan(
      uncapped.parameters['cellSize'] as number
    );
    expect(entry.clamped?.from).toBe(uncapped.parameters['cellSize']);
    expect(entry.clamped?.to).toBe(capped.parameters['cellSize']);
    expect(entry.clamped?.reason).toMatch(/at most 40,000/);
    expect(run(capped.parameters).tables.length).toBeGreaterThan(0);
    // The smallest size that fits: one unit of the last digit smaller would not.
    const cellSize = capped.parameters['cellSize'] as number;
    const unit = 10 ** (Math.floor(Math.log10(cellSize)) - 1);
    expect(() =>
      run({...capped.parameters, cellSize: Number((cellSize - unit).toPrecision(15))})
    ).toThrow();
  });

  it('needs no raise for the hot spot lattice, whose spacing rule stays inside its cap', () => {
    const binding = getCatalogBinding('recipe-hot-spot-lattice')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const {collection, profile} = makeElongatedProfile();
    const resolved = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {points: profile},
      {maximumCellCount: KEPLER_GRID_CELL_CAPS[binding.id].maximumCellCount}
    );
    expect(
      resolved.resolutions.find(entry => entry.parameter === 'cellSize')?.clamped
    ).toBeUndefined();
    expect(
      binding.run!({points: collection}, normalizeCatalogParameters(binding, resolved.parameters))
        .tables.length
    ).toBeGreaterThan(0);
  });

  it('leaves a cell size alone that already fits, an unlisted binding and an override', () => {
    const binding = getCatalogBinding('kernel-density-grid')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const profile = buildSampleProfile('incidents');
    const plain = resolveKeplerOperationParameters(binding, descriptor, {points: profile});
    const withCap = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {points: profile},
      {maximumCellCount: 250000}
    );
    expect(withCap.parameters).toEqual(plain.parameters);
    expect(
      withCap.resolutions.find(entry => entry.parameter === 'cellSize')?.clamped
    ).toBeUndefined();
    const overridden = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {points: profile},
      {maximumCellCount: 4, overrides: {cellSize: 1}}
    );
    expect(overridden.parameters['cellSize']).toBe(1);
    // Without spacing statistics there is no extent to size against.
    const {spacing: _spacing, extent: _extent, ...noSpacing} = profile;
    const unsized = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {points: noSpacing},
      {maximumCellCount: 4}
    );
    expect(
      unsized.resolutions.find(entry => entry.parameter === 'cellSize')?.clamped
    ).toBeUndefined();
  });
});

/** A descriptor for a real binding with some parameter policies replaced and fields added. */
function makeDescriptor(
  bindingId: string,
  policies: Record<string, Partial<KeplerOperationParameterDescriptor>>,
  fields: KeplerOperationDescriptor['inputs'][number]['fields'] = []
): KeplerOperationDescriptor {
  const base = deriveKeplerOperationDescriptor(getCatalogBinding(bindingId)!);
  return {
    ...base,
    inputs: base.inputs.map((input, index) => (index === 0 ? {...input, fields} : input)),
    parameters: base.parameters.map(parameter => ({...parameter, ...policies[parameter.name]}))
  };
}

const WEIGHT_FIELD = {
  role: 'weight',
  label: 'Weight',
  accepts: ['number', 'integer'],
  required: false,
  parameter: 'weightProperty'
} as const;

describe('preset policy', () => {
  const binding = getCatalogBinding('hotspot-gi-star')!;

  it('applies a preset when the geometry matches, with its reason, and the constant otherwise', () => {
    const descriptor = makeDescriptor('hotspot-gi-star', {
      neighborhood: {
        defaultPolicy: {
          from: 'preset',
          value: 'queen',
          reason: 'Areas that touch are neighbours',
          input: 'features',
          whenGeometry: 'polygon'
        }
      },
      neighborCount: {defaultPolicy: {from: 'preset', value: 8, reason: 'Esri recommends 8'}}
    });
    const polygons = resolveKeplerOperationParameters(binding, descriptor, {
      features: buildSampleProfile('districts')
    });
    expect(polygons.parameters['neighborhood']).toBe('queen');
    const entry = polygons.resolutions.find(item => item.parameter === 'neighborhood')!;
    expect(entry.source).toBe('preset');
    expect(entry.reason).toBe('Areas that touch are neighbours');
    // No geometry condition: applies to every input, and before any profile is known.
    expect(polygons.parameters['neighborCount']).toBe(8);
    expect(
      resolveKeplerOperationParameters(binding, descriptor, {}).parameters['neighborCount']
    ).toBe(8);

    const points = resolveKeplerOperationParameters(binding, descriptor, {
      features: buildSampleProfile('incidents')
    });
    const constant = binding.parameters.find(item => item.name === 'neighborhood')!.defaultValue;
    expect(points.parameters['neighborhood']).toBe(constant);
    expect(points.resolutions.find(item => item.parameter === 'neighborhood')?.source).toBe(
      'constant'
    );
    expect(normalizeCatalogParameters(binding, polygons.parameters)).toEqual(polygons.parameters);
  });

  it('falls back to the constant and says why when the binding rejects the preset', () => {
    const descriptor = makeDescriptor('hotspot-gi-star', {
      neighborhood: {defaultPolicy: {from: 'preset', value: 'hexagonal', reason: 'A bad preset'}},
      neighborCount: {defaultPolicy: {from: 'preset', value: 2.5, reason: 'Not a whole number'}}
    });
    const resolved = resolveKeplerOperationParameters(binding, descriptor, {});
    for (const name of ['neighborhood', 'neighborCount']) {
      const entry = resolved.resolutions.find(item => item.parameter === name)!;
      expect(entry.source, name).toBe('fallback');
      expect(entry.reason, name).toMatch(/is not accepted by hotspot-gi-star/);
      expect(resolved.parameters[name]).toBe(
        binding.parameters.find(item => item.name === name)!.defaultValue
      );
    }
  });

  it('lets an override replace a preset', () => {
    const descriptor = makeDescriptor('hotspot-gi-star', {
      neighborCount: {defaultPolicy: {from: 'preset', value: 8, reason: 'r'}}
    });
    const resolved = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {},
      {overrides: {neighborCount: 5}}
    );
    expect(resolved.parameters['neighborCount']).toBe(5);
    expect(resolved.resolutions.find(item => item.parameter === 'neighborCount')?.source).toBe(
      'override'
    );
  });
});

describe('factorByParameter and column rules', () => {
  const binding = getCatalogBinding('kernel-density-grid')!;
  const profile = buildSampleProfile('incidents');
  const policy = {
    from: 'data' as const,
    input: 'points',
    rule: 'median-nearest-neighbor-distance' as const,
    factorByParameter: {parameter: 'kernel', factors: {gaussian: 1, quartic: 2}}
  };

  it('multiplies the factor by the entry for the current value of the named parameter', () => {
    const descriptor = makeDescriptor('kernel-density-grid', {bandwidth: {defaultPolicy: policy}});
    const gaussian = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {points: profile},
      {overrides: {kernel: 'gaussian'}}
    );
    const quartic = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {points: profile},
      {overrides: {kernel: 'quartic'}}
    );
    const unlisted = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {points: profile},
      {overrides: {kernel: 'epanechnikov'}}
    );
    const base = gaussian.parameters['bandwidth'] as number;
    expect((quartic.parameters['bandwidth'] as number) / base).toBeGreaterThan(1.9);
    expect((quartic.parameters['bandwidth'] as number) / base).toBeLessThan(2.1);
    expect(unlisted.parameters['bandwidth']).toBe(base);
    expect(
      quartic.resolutions.find(item => item.parameter === 'bandwidth')?.resolution?.factor
    ).toBe(2);
  });

  it('reads the controlling parameter after its preset, even when it comes later in the binding', () => {
    expect(binding.parameters.findIndex(item => item.name === 'bandwidth')).toBeLessThan(
      binding.parameters.findIndex(item => item.name === 'kernel')
    );
    const descriptor = makeDescriptor('kernel-density-grid', {
      bandwidth: {defaultPolicy: policy},
      kernel: {defaultPolicy: {from: 'preset', value: 'quartic', reason: 'r'}}
    });
    const resolved = resolveKeplerOperationParameters(binding, descriptor, {points: profile});
    expect(resolved.parameters['kernel']).toBe('quartic');
    expect(
      resolved.resolutions.find(item => item.parameter === 'bandwidth')?.resolution?.factor
    ).toBe(2);
  });

  it('rejects a cycle between parameters through factorByParameter', () => {
    const descriptor = makeDescriptor('kernel-density-grid', {
      cellSize: {
        defaultPolicy: {...policy, factorByParameter: {parameter: 'bandwidth', factors: {}}}
      },
      bandwidth: {
        defaultPolicy: {...policy, factorByParameter: {parameter: 'cellSize', factors: {}}}
      }
    });
    expect(() => resolveKeplerOperationParameters(binding, descriptor, {points: profile})).toThrow(
      /cycle in factorByParameter: (bandwidth -> cellSize -> bandwidth|cellSize -> bandwidth -> cellSize)/
    );
  });

  it('falls back with a reason when a column rule has no bound column, and reads the bound column', () => {
    const descriptor = makeDescriptor(
      'kernel-density-grid',
      {
        bandwidth: {
          defaultPolicy: {
            from: 'data',
            input: 'points',
            rule: 'column-standard-deviation',
            role: 'weight'
          }
        }
      },
      [WEIGHT_FIELD]
    );
    const unbound = resolveKeplerOperationParameters(binding, descriptor, {points: profile});
    const entry = unbound.resolutions.find(item => item.parameter === 'bandwidth')!;
    expect(entry.source).toBe('fallback');
    expect(entry.reason).toMatch(/No column is bound to the role "weight"/);
    const bound = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {points: profile},
      {fieldBindings: {weightProperty: 'severity'}}
    );
    const column = profile.columns.find(item => item.name === 'severity')!;
    const boundEntry = bound.resolutions.find(item => item.parameter === 'bandwidth')!;
    expect(boundEntry.source).toBe('data');
    expect(bound.parameters['bandwidth']).toBe(Number(column.standardDeviation!.toPrecision(2)));
  });
});

describe('field-measurement policy', () => {
  const binding = getCatalogBinding('kernel-density-grid')!;
  const descriptor = makeDescriptor(
    'kernel-density-grid',
    {
      kernel: {
        defaultPolicy: {
          from: 'field-measurement',
          input: 'points',
          role: 'weight',
          values: {count: 'quartic', rate: 'epanechnikov'}
        }
      }
    },
    [WEIGHT_FIELD]
  );
  const profile = buildSampleProfile('incidents');
  const withLevel = (level: 'count' | 'rate' | 'amount' | undefined) => ({
    ...profile,
    columns: profile.columns.map(column =>
      column.name === 'severity'
        ? {...column, measurement: level, measurementSource: 'declared' as const}
        : column
    )
  });

  it('chooses the value listed for the declared measurement level', () => {
    for (const [level, expected] of [
      ['count', 'quartic'],
      ['rate', 'epanechnikov']
    ] as const) {
      const resolved = resolveKeplerOperationParameters(
        binding,
        descriptor,
        {points: withLevel(level)},
        {fieldBindings: {weightProperty: 'severity'}}
      );
      expect(resolved.parameters['kernel']).toBe(expected);
      const entry = resolved.resolutions.find(item => item.parameter === 'kernel')!;
      expect(entry.source).toBe('data');
      expect(entry.reason).toBe(`severity measures ${level}`);
    }
  });

  it('keeps the default and says why for an unlisted level and for an unbound role', () => {
    const constant = binding.parameters.find(item => item.name === 'kernel')!.defaultValue;
    const unlisted = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {points: withLevel('amount')},
      {fieldBindings: {weightProperty: 'severity'}}
    );
    expect(unlisted.parameters['kernel']).toBe(constant);
    const unlistedEntry = unlisted.resolutions.find(item => item.parameter === 'kernel')!;
    expect(unlistedEntry.source).toBe('fallback');
    expect(unlistedEntry.reason).toMatch(/No value is listed for severity, which measures amount/);
    const unbound = resolveKeplerOperationParameters(binding, descriptor, {points: profile});
    expect(unbound.resolutions.find(item => item.parameter === 'kernel')?.reason).toMatch(
      /No column is bound to the role "weight"/
    );
  });
});

describe('activeWhen', () => {
  const binding = getCatalogBinding('hotspot-gi-star')!;
  const descriptor = makeDescriptor('hotspot-gi-star', {
    neighborCount: {activeWhen: {parameter: 'neighborhood', equals: ['knn']}},
    distanceBand: {activeWhen: {parameter: 'neighborhood', equals: ['distance-band']}}
  });

  it('marks parameters inactive from the final value of the controlling parameter', () => {
    const knn = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {},
      {overrides: {neighborhood: 'knn'}}
    );
    expect(knn.inactive).toEqual(['distanceBand']);
    expect(knn.resolutions.find(item => item.parameter === 'distanceBand')?.active).toBe(false);
    expect(knn.resolutions.find(item => item.parameter === 'neighborCount')?.active).toBe(true);
    // The catalog needs every parameter, so the inactive one keeps its value.
    expect(typeof knn.parameters['distanceBand']).toBe('number');
    expect(normalizeCatalogParameters(binding, knn.parameters)).toEqual(knn.parameters);
    const band = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {},
      {overrides: {neighborhood: 'distance-band'}}
    );
    expect(band.inactive).toEqual(['neighborCount']);
    expect(band.resolutions.every(item => item.active !== undefined)).toBe(true);
  });
});

describe('rule rounding', () => {
  it('has one documented entry for every default rule', () => {
    for (const rule of GPU_ANALYSIS_DEFAULT_RULES) {
      const entry = KEPLER_DEFAULT_RULE_ROUNDING[rule];
      expect(entry, rule).toBeDefined();
      expect(entry.reason.length, rule).toBeGreaterThan(5);
    }
    for (const rule of [
      'maximum-nearest-neighbor-distance',
      'outlier-resistant-nearest-neighbor-distance',
      'k-distance-knee'
    ] as const) {
      expect(KEPLER_DEFAULT_RULE_ROUNDING[rule].direction, rule).toBe('up');
    }
    expect(KEPLER_DEFAULT_RULE_ROUNDING['extent-shorter-side-over-250'].direction).toBe('nearest');
    expect(KEPLER_DEFAULT_RULE_ROUNDING['extent-west'].direction).toBe('none');
  });

  it('rounds a reach up, so DBSCAN on a regular geographic lattice finds no noise majority', () => {
    // 20 by 20 points 111.2 m apart (0.001 degree of latitude). A reach of 110 m, the nearest
    // two-digit value, would leave every point without a neighbour.
    const features = [];
    for (let column = 0; column < 20; column++) {
      for (let row = 0; row < 20; row++) {
        features.push({
          geometry: {
            type: 'Point',
            coordinates: [
              -87.7 + (column * 0.001) / Math.cos((41.85 * Math.PI) / 180),
              41.85 + row * 0.001
            ]
          },
          properties: {}
        });
      }
    }
    const collection: CatalogFeatureCollection = {type: 'FeatureCollection', features};
    const profile = buildTestProfile({
      datasetId: 'lattice',
      label: 'lattice',
      collection,
      kind: 'points',
      catalogGeometry: 'point'
    });
    const binding = getCatalogBinding('density-clustering')!;
    const run = (
      rule: 'maximum-nearest-neighbor-distance' | 'median-nearest-neighbor-distance'
    ) => {
      const descriptor = makeDescriptor('density-clustering', {
        epsilon: {defaultPolicy: {from: 'data', input: 'points', rule}}
      });
      const resolved = resolveKeplerOperationParameters(binding, descriptor, {points: profile});
      const output = binding.run!(
        {points: collection},
        normalizeCatalogParameters(binding, resolved.parameters)
      );
      const labels = output.tables.find(table => table.id === 'points')!.columns[
        'cluster'
      ] as number[];
      return {
        epsilon: resolved.parameters['epsilon'] as number,
        noise: labels.filter(label => label < 0).length
      };
    };
    const spacing = profile.spacing!.nearestNeighbor!.maximum;
    expect(spacing).toBeGreaterThan(111);
    const reach = run('maximum-nearest-neighbor-distance');
    expect(reach.epsilon).toBeGreaterThanOrEqual(spacing);
    expect(reach.noise).toBeLessThan(features.length / 2);
    // The control: rounding the same value to nearest would give 110, below the spacing.
    expect(Number(spacing.toPrecision(2))).toBeLessThan(spacing);
  });
});

describe('antimeridian extents', () => {
  function makeBox(centerLongitude: number) {
    let state = 777;
    const next = () => {
      state = (state * 1103515245 + 12345) % 2147483648;
      return state / 2147483648;
    };
    const features = Array.from({length: 3000}, (_, index) => {
      let longitude = centerLongitude + (next() - 0.5) * 0.107;
      if (longitude > 180) longitude -= 360;
      if (longitude < -180) longitude += 360;
      return {
        id: index,
        geometry: {type: 'Point', coordinates: [longitude, 41.85 + next() * 0.01]},
        properties: {}
      };
    });
    const collection: CatalogFeatureCollection = {type: 'FeatureCollection', features};
    return buildTestProfile({
      datasetId: 'box',
      label: 'box',
      collection,
      kind: 'points',
      catalogGeometry: 'point'
    });
  }

  it('gives the same capped cell size for a box across longitude 180 as for the same box elsewhere', () => {
    const binding = getCatalogBinding('recipe-hot-spot-lattice')!;
    const descriptor = makeDescriptor('recipe-hot-spot-lattice', {
      cellSize: {
        defaultPolicy: {
          from: 'data',
          input: 'points',
          rule: 'extent-shorter-side-over-250',
          factor: 3
        }
      }
    });
    const cellSizeOf = (profile: ReturnType<typeof makeBox>) =>
      resolveKeplerOperationParameters(
        binding,
        descriptor,
        {points: profile},
        {maximumCellCount: 40000}
      );
    const across = makeBox(180);
    const elsewhere = makeBox(-87.65);
    expect(across.spacing!.extentWidth).toBeLessThan(20000);
    const left = cellSizeOf(across);
    const right = cellSizeOf(elsewhere);
    const ratio =
      (left.parameters['cellSize'] as number) / (right.parameters['cellSize'] as number);
    expect(ratio).toBeGreaterThan(0.95);
    expect(ratio).toBeLessThan(1.05);
    expect(left.resolutions.find(item => item.parameter === 'cellSize')?.clamped).toBeDefined();
  });
});

// ---------------------------------------------------------------------------------------------
// Fix round, part 2: fallback rules, grid caps in the bindings' own frame, CPU work caps.
// ---------------------------------------------------------------------------------------------

import {runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import {
  getKeplerEffectiveEngine,
  getKeplerGridFrameExtent,
  KEPLER_CPU_WORK_CAPS
} from '../../src/descriptors/default-parameters';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';

describe('fallback rule', () => {
  const binding = getCatalogBinding('density-clustering')!;
  const lattice = () => {
    const features = [];
    for (let column = 0; column < 20; column++) {
      for (let row = 0; row < 20; row++) {
        features.push({
          geometry: {
            type: 'Point',
            coordinates: [
              -87.7 + (column * 0.001) / Math.cos((41.85 * Math.PI) / 180),
              41.85 + row * 0.001
            ]
          },
          properties: {}
        });
      }
    }
    const collection: CatalogFeatureCollection = {type: 'FeatureCollection', features};
    return {
      collection,
      profile: buildTestProfile({
        datasetId: 'lattice',
        label: 'lattice',
        collection,
        kind: 'points',
        catalogGeometry: 'point'
      })
    };
  };

  it('tries the second rule when the knee declines, and records both explanations', () => {
    const {collection, profile} = lattice();
    const descriptor = makeDescriptor('density-clustering', {
      epsilon: {
        defaultPolicy: {
          from: 'data',
          input: 'points',
          rule: 'k-distance-knee',
          fallback: {rule: 'maximum-nearest-neighbor-distance', factor: 1.01}
        }
      }
    });
    const resolved = resolveKeplerOperationParameters(binding, descriptor, {points: profile});
    const entry = resolved.resolutions.find(item => item.parameter === 'epsilon')!;
    // The regular lattice has no knee: the main rule gives nothing and the second one answers.
    expect(entry.source).toBe('data');
    expect(entry.resolution?.rule).toBe('maximum-nearest-neighbor-distance');
    expect(entry.reason).toMatch(
      /no radius is suggested.*; then Largest nearest-neighbour distance/s
    );
    expect(resolved.parameters['epsilon'] as number).toBeGreaterThanOrEqual(
      profile.spacing!.nearestNeighbor!.maximum
    );
    // The result is not all noise on the real binding.
    const output = binding.run!(
      {points: collection},
      normalizeCatalogParameters(binding, resolved.parameters)
    );
    const labels = output.tables.find(table => table.id === 'points')!.columns[
      'cluster'
    ] as number[];
    expect(labels.filter(label => label < 0).length).toBeLessThan(labels.length / 2);
  });

  it('falls back to the constant only when both rules fail, and says so', () => {
    const {profile} = lattice();
    const {spacing: _spacing, extent: _extent, ...bare} = profile;
    const descriptor = makeDescriptor('density-clustering', {
      epsilon: {
        defaultPolicy: {
          from: 'data',
          input: 'points',
          rule: 'k-distance-knee',
          fallback: {rule: 'median-nearest-neighbor-distance', factor: 2}
        }
      }
    });
    const resolved = resolveKeplerOperationParameters(binding, descriptor, {points: bare});
    const entry = resolved.resolutions.find(item => item.parameter === 'epsilon')!;
    expect(entry.source).toBe('fallback');
    expect(entry.reason).toMatch(/neither gave a value/);
    expect(resolved.parameters['epsilon']).toBe(
      binding.parameters.find(item => item.name === 'epsilon')!.defaultValue
    );
  });

  it('records a guessed coordinate model in the resolution, never as if declared', () => {
    const kernel = getCatalogBinding('kernel-density-grid')!;
    const descriptor = getKeplerOperationDescriptor(kernel.id)!;
    const guessed = resolveKeplerOperationParameters(kernel, descriptor, {
      points: buildSampleProfile('incidents')
    });
    const entry = guessed.resolutions.find(item => item.parameter === 'coordinateSystem')!;
    expect(entry.guessed).toBe(true);
    expect(entry.reason).toMatch(/guess/);
    const declared = resolveKeplerOperationParameters(kernel, descriptor, {
      points: buildSampleProfile('incidents', {
        coordinates: {model: 'geographic', source: 'declared'}
      })
    });
    expect(
      declared.resolutions.find(item => item.parameter === 'coordinateSystem')?.guessed
    ).toBeUndefined();
  });
});

describe('grid caps in the operation frame', () => {
  const point = (x: number, y: number, v: number) => ({
    geometry: {type: 'Point', coordinates: [x, y]},
    properties: {v}
  });

  type Layout = {
    label: string;
    system: 'EPSG:4326' | 'planar';
    x0: number;
    y0: number;
    width: number;
    height: number;
  };
  // The reviewer's large extents (rv8 gridcap-big): 18 of 43 threw before the frame was used.
  const layouts: Layout[] = [
    {label: '60N 20x1 deg', system: 'EPSG:4326', x0: 10, y0: 60, width: 20, height: 1},
    {label: '0-60N 1x60 deg', system: 'EPSG:4326', x0: 10, y0: 0, width: 1, height: 60},
    {label: '40-50S 30x10 deg', system: 'EPSG:4326', x0: 10, y0: -50, width: 30, height: 10},
    {label: '45N 4x0.5 deg', system: 'EPSG:4326', x0: 10, y0: 45, width: 4, height: 0.5},
    {label: 'across 180, 60N 20x1 deg', system: 'EPSG:4326', x0: 170, y0: 60, width: 20, height: 1},
    {label: 'planar aspect 40', system: 'planar', x0: 1000, y0: 2000, width: 40000, height: 1000}
  ];

  function build(accepts: string, layout: Layout): CatalogFeatureCollection {
    let state = 7;
    const next = () => {
      state = (state * 1103515245 + 12345) % 2147483648;
      return state / 2147483648;
    };
    const wrap = (longitude: number) => (longitude > 180 ? longitude - 360 : longitude);
    const {x0, y0, width, height} = layout;
    if (accepts === 'line') {
      // Three long parallel lines: their own extent is what counts, not their midpoints'.
      return {
        type: 'FeatureCollection',
        features: [0, 0.5, 1].map(share => ({
          geometry: {
            type: 'LineString',
            coordinates: [
              [wrap(x0), y0 + height * share],
              [wrap(x0 + width), y0 + height * share]
            ]
          },
          properties: {v: 1}
        }))
      } as unknown as CatalogFeatureCollection;
    }
    if (accepts === 'polygon') {
      return {
        type: 'FeatureCollection',
        features: [
          {
            geometry: {
              type: 'Polygon',
              coordinates: [
                [
                  [wrap(x0), y0],
                  [wrap(x0 + width), y0],
                  [wrap(x0 + width), y0 + height],
                  [wrap(x0), y0 + height],
                  [wrap(x0), y0]
                ]
              ]
            },
            properties: {v: 1}
          }
        ]
      } as unknown as CatalogFeatureCollection;
    }
    const features = [
      point(wrap(x0), y0, 1),
      point(wrap(x0 + width), y0, 2),
      point(wrap(x0), y0 + height, 3),
      point(wrap(x0 + width), y0 + height, 4)
    ];
    for (let index = 0; index < 28; index++) {
      features.push(point(wrap(x0 + next() * width), y0 + next() * height, 1 + next() * 10));
    }
    return {type: 'FeatureCollection', features} as unknown as CatalogFeatureCollection;
  }

  it('runs every grid binding at the resolved cell size, for large extents, in both engines', async () => {
    const failures: string[] = [];
    for (const bindingId of Object.keys(KEPLER_GRID_CELL_CAPS)) {
      const binding = getCatalogBinding(bindingId)!;
      const descriptor = getKeplerOperationDescriptor(bindingId)!;
      for (const layout of layouts) {
        const inputs: Record<string, CatalogFeatureCollection> = {};
        const profiles: Record<string, KeplerDatasetProfile> = {};
        for (const input of binding.inputs) {
          inputs[input.name] = build(input.accepts[0], layout);
          profiles[input.name] = profileKeplerFeatureCollection(inputs[input.name], {
            datasetId: input.name,
            label: input.name,
            coordinateSystem: layout.system
          }) as unknown as KeplerDatasetProfile;
        }
        const fieldBindings = Object.fromEntries(
          binding.parameters.filter(item => item.kind === 'string').map(item => [item.name, 'v'])
        );
        for (const executionEngine of ['gpu', 'cpu'] as const) {
          const resolved = resolveKeplerOperationParameters(binding, descriptor, profiles, {
            maximumCellCount: KEPLER_GRID_CELL_CAPS[bindingId].maximumCellCount,
            executionEngine,
            fieldBindings
          });
          try {
            await runCatalogOperationAsync(binding, resolved.parameters, inputs, {engine: 'cpu'});
          } catch (error) {
            failures.push(
              `${bindingId} | ${layout.label} | ${executionEngine} | cell ${String(resolved.parameters['cellSize'])}: ${(error as Error).message.slice(0, 80)}`
            );
          }
        }
      }
    }
    expect(failures).toEqual([]);
  }, 300000);

  it('measures the frame the bindings use: taller than the sphere at high latitude', () => {
    const profile = profileKeplerFeatureCollection(build('point', layouts[0]), {
      datasetId: 'p',
      label: 'p',
      coordinateSystem: 'EPSG:4326'
    }) as unknown as KeplerDatasetProfile;
    const size = getKeplerGridFrameExtent(profile, 'EPSG:4326')!;
    // 20 by 1 degrees at 60 north: the great-circle height is 111 km; the frame's is larger.
    expect(size.height / profile.spacing!.extentHeight).toBeGreaterThan(1.1);
    const planar = getKeplerGridFrameExtent({...profile, extent: [0, 0, 30, 10]}, 'planar');
    expect(planar).toEqual({width: 30, height: 10});
    // The antimeridian convention: west above east is a short box across 180, not a wide one.
    const across = getKeplerGridFrameExtent(
      {...profile, extent: [179.5, 60, -179.5, 61]},
      'EPSG:4326'
    )!;
    const elsewhere = getKeplerGridFrameExtent(
      {...profile, extent: [10, 60, 11, 61]},
      'EPSG:4326'
    )!;
    expect(across.width / elsewhere.width).toBeGreaterThan(0.99);
    expect(across.width / elsewhere.width).toBeLessThan(1.01);
    expect(
      getKeplerGridFrameExtent({...profile, extent: undefined, spacing: undefined}, 'EPSG:4326')
    ).toBeUndefined();
  });
});

describe('CPU work caps', () => {
  it('names a source line for every cap and applies grid work only on the CPU engine', () => {
    for (const [id, cap] of Object.entries(KEPLER_CPU_WORK_CAPS)) {
      expect(getCatalogBinding(id), id).toBeDefined();
      expect(cap.source, id).toMatch(/\.ts:\d+/);
    }
    expect(getKeplerEffectiveEngine({engine: 'cpu-reference'}, 'gpu')).toBe('cpu');
    expect(getKeplerEffectiveEngine({engine: 'gpu-or-cpu'}, 'gpu')).toBe('gpu');
    expect(getKeplerEffectiveEngine({engine: 'gpu-or-cpu'}, 'cpu')).toBe('cpu');
    const binding = getCatalogBinding('kernel-density-grid')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const profile = buildSampleProfile('incidents');
    const rows = 5000;
    const big = {...profile, rowCount: rows};
    const onGpu = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {points: big},
      {maximumCellCount: 250000}
    );
    const onCpu = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {points: big},
      {maximumCellCount: 250000, executionEngine: 'cpu'}
    );
    expect(onCpu.parameters['cellSize'] as number).toBeGreaterThanOrEqual(
      onGpu.parameters['cellSize'] as number
    );
    const entry = onCpu.resolutions.find(item => item.parameter === 'cellSize')!;
    if (entry.clamped) {
      expect(entry.clamped.reason).toMatch(/on the CPU engine/);
    }
    // 6e7 / 5000 = 12,000 cells at most.
    const size = getKeplerGridFrameExtent(big, 'EPSG:4326')!;
    const cellSize = onCpu.parameters['cellSize'] as number;
    expect(
      (Math.floor(size.width / cellSize) + 1) * (Math.floor(size.height / cellSize) + 1)
    ).toBeLessThanOrEqual(12000);
  });
});

describe('second review grid and catchment resolution', () => {
  it('caps both generator dimensions using its exact ceil counts after rounding', async () => {
    const binding = getCatalogBinding('grid-generator')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    for (const [width, height] of [
      [38125, 7625],
      [100000, 1000]
    ]) {
      const collection = createTestPointCollection([
        [0, 0],
        [width, height]
      ]);
      const profile = profileKeplerFeatureCollection(collection, {
        datasetId: 'extent',
        label: 'extent',
        coordinateSystem: 'planar'
      });
      const result = resolveKeplerOperationParameters(binding, descriptor, {extent: profile});
      const cellCount =
        Math.max(1, Math.ceil(width / Number(result.parameters['cellWidth']) - 1e-9)) *
        Math.max(1, Math.ceil(height / Number(result.parameters['cellHeight']) - 1e-9));
      expect(cellCount).toBeLessThanOrEqual(20000);
      // The descriptor's factor keeps extents up to five times longer than wide under the cap by
      // itself; only the hundred-to-one corridor needs the cap step to raise the cell size.
      if (width / height > 5) {
        expect(result.resolutions.some(resolution => resolution.clamped !== undefined)).toBe(true);
      }
      const output = await runCatalogOperationAsync(
        binding,
        result.parameters,
        {extent: collection},
        {engine: 'cpu'}
      );
      expect(output.tables[0].rowCount).toBe(cellCount);
    }
  });

  it('caps a generator spanning raw Pacific longitudes despite a wrapped profile extent', async () => {
    const collection = createTestPointCollection([
      [179, 40],
      [-179, 41]
    ]);
    const profile = profileKeplerFeatureCollection(collection, {
      datasetId: 'extent',
      label: 'extent',
      coordinateSystem: 'EPSG:4326'
    });
    const binding = getCatalogBinding('grid-generator')!;
    const result = resolveKeplerOperationParameters(
      binding,
      getKeplerOperationDescriptor(binding.id)!,
      {extent: profile}
    );
    const output = await runCatalogOperationAsync(
      binding,
      result.parameters,
      {extent: collection},
      {engine: 'cpu'}
    );
    expect(output.tables[0].rowCount).toBeLessThanOrEqual(20000);
  });

  it('leaves explicit generator sizes alone and explains when they exceed the cap', () => {
    const profile = {...buildSampleProfile('incidents'), extent: [0, 0, 1000, 1000] as const};
    const binding = getCatalogBinding('grid-generator')!;
    const result = resolveKeplerOperationParameters(
      binding,
      getKeplerOperationDescriptor(binding.id)!,
      {extent: profile},
      {overrides: {cellWidth: 1, cellHeight: 1}}
    );
    expect(result.parameters['cellWidth']).toBe(1);
    expect(result.parameters['cellHeight']).toBe(1);
    expect(result.confirmations.join(' ')).toMatch(/1,000,000.*20,000/);
  });

  const facilities = createTestPointCollection([
    [1000, 1000],
    [1100, 1000],
    [1000, 1100],
    [1100, 1100]
  ]);
  const profile = (datasetId: string, collection: CatalogFeatureCollection) =>
    profileKeplerFeatureCollection(collection, {
      datasetId,
      label: datasetId,
      coordinateSystem: 'planar'
    });

  it('reduces catchment cell size when exact spacing can separate seeds within the cap', async () => {
    const observations = createTestPointCollection(
      Array.from({length: 64}, (_, index) => [
        1000 + (index % 8) * 5000,
        1000 + Math.floor(index / 8) * 5000
      ])
    );
    const binding = getCatalogBinding('recipe-straight-line-catchments')!;
    const result = resolveKeplerOperationParameters(
      binding,
      getKeplerOperationDescriptor(binding.id)!,
      {
        facilities: profile('facilities', facilities),
        observations: profile('observations', observations)
      },
      {maximumCellCount: KEPLER_GRID_CELL_CAPS[binding.id].maximumCellCount}
    );
    expect(result.parameters['cellSize']).toBeLessThan(100 / Math.SQRT2);
    expect(result.confirmations).toEqual([]);
    const output = await runCatalogOperationAsync(
      binding,
      result.parameters,
      {facilities, observations},
      {engine: 'cpu'}
    );
    expect(output.tables[0].columns['cell_count'].every(value => Number(value) > 0)).toBe(true);
  });

  it('reports the observed lower bound of collapsed seeds when a safe resolution exceeds capacity', () => {
    const observations = createTestPointCollection(
      Array.from({length: 64}, (_, index) => [
        1000 + (index % 8) * 10000,
        1000 + Math.floor(index / 8) * 10000
      ])
    );
    const binding = getCatalogBinding('recipe-straight-line-catchments')!;
    const result = resolveKeplerOperationParameters(
      binding,
      getKeplerOperationDescriptor(binding.id)!,
      {
        facilities: profile('facilities', facilities),
        observations: profile('observations', observations)
      },
      {maximumCellCount: KEPLER_GRID_CELL_CAPS[binding.id].maximumCellCount}
    );
    expect(result.confirmations.join(' ')).toMatch(/at least 3.*share/);
    expect(result.confirmations.join(' ')).toMatch(/no facility positions for an exact count/);
  });

  it('sizes catchments over the union of facilities and observations', () => {
    const observations = createTestPointCollection([
      [1000, 1000],
      [1100, 1100]
    ]);
    const distantFacilities = createTestPointCollection([
      [0, 0],
      [35000, 35000]
    ]);
    const binding = getCatalogBinding('recipe-straight-line-catchments')!;
    const result = resolveKeplerOperationParameters(
      binding,
      getKeplerOperationDescriptor(binding.id)!,
      {
        facilities: profile('facilities', distantFacilities),
        observations: profile('observations', observations)
      },
      {maximumCellCount: KEPLER_GRID_CELL_CAPS[binding.id].maximumCellCount}
    );
    const cellSize = Number(result.parameters['cellSize']);
    expect((Math.floor(35000 / cellSize) + 1) ** 2).toBeLessThanOrEqual(250000);
  });
});

describe('final coordinate model and total-aware defaults', () => {
  const collection = createTestPointCollection([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
  ]);

  it('reproduces c7 unit-spaced polygons without carrying metre statistics into a planar run', () => {
    const areas: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: Array.from({length: 63}, (_, index) => {
        const column = index % 9;
        const row = Math.floor(index / 9);
        return {
          type: 'Feature',
          geometry: {
            type: 'Polygon',
            coordinates: [
              [
                [column, row],
                [column + 1, row],
                [column + 1, row + 1],
                [column, row + 1],
                [column, row]
              ]
            ]
          },
          properties: {value: 2 + ((index * 17) % 23) / 19}
        };
      })
    };
    const profile = profileKeplerFeatureCollection(areas, {
      datasetId: 'features',
      label: 'features'
    });
    expect(profile.coordinates?.model).toBe('geographic');
    const binding = getCatalogBinding('hotspot-gi-star')!;
    const result = resolveKeplerOperationParameters(
      binding,
      getKeplerOperationDescriptor(binding.id)!,
      {features: profile},
      {
        fieldBindings: {valueProperty: 'value'},
        overrides: {coordinateSystem: 'planar', neighborhood: 'distance-band'}
      }
    );
    const band = result.resolutions.find(entry => entry.parameter === 'distanceBand')!;
    expect(band.source).toBe('data');
    expect(band.coordinateModel).toBe('planar');
    const declared = profileKeplerFeatureCollection(areas, {
      datasetId: 'features',
      label: 'features',
      coordinateSystem: 'planar'
    });
    const equivalent = resolveKeplerOperationParameters(
      binding,
      getKeplerOperationDescriptor(binding.id)!,
      {features: declared},
      {
        fieldBindings: {valueProperty: 'value'},
        overrides: {neighborhood: 'distance-band'}
      }
    );
    expect(result.parameters['distanceBand']).toBe(equivalent.parameters['distanceBand']);
  });

  it('checks the spacing model independently of a newer coordinate declaration', () => {
    const geographic = profileKeplerFeatureCollection(collection, {
      datasetId: 'points',
      label: 'points',
      coordinateSystem: 'EPSG:4326'
    });
    const binding = getCatalogBinding('kernel-density-grid')!;
    const result = resolveKeplerOperationParameters(
      binding,
      getKeplerOperationDescriptor(binding.id)!,
      {
        points: {
          ...geographic,
          coordinates: {model: 'planar', source: 'declared'}
        }
      }
    );
    expect(result.parameters['coordinateSystem']).toBe('planar');
    for (const name of ['cellSize', 'bandwidth']) {
      expect(result.resolutions.find(entry => entry.parameter === name)?.source).toBe('fallback');
    }
  });

  it('recomputes length statistics in both override directions', () => {
    const binding = getCatalogBinding('kernel-density-grid')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    for (const [profileSystem, runSystem] of [
      ['EPSG:4326', 'planar'],
      ['planar', 'EPSG:4326']
    ] as const) {
      const profile = profileKeplerFeatureCollection(collection, {
        datasetId: 'points',
        label: 'points',
        coordinateSystem: profileSystem
      });
      const result = resolveKeplerOperationParameters(
        binding,
        descriptor,
        {points: profile},
        {
          overrides: {coordinateSystem: runSystem}
        }
      );
      const declared = profileKeplerFeatureCollection(collection, {
        datasetId: 'points',
        label: 'points',
        coordinateSystem: runSystem
      });
      const equivalent = resolveKeplerOperationParameters(binding, descriptor, {points: declared});
      for (const name of ['cellSize', 'bandwidth']) {
        const entry = result.resolutions.find(candidate => candidate.parameter === name)!;
        expect(entry.source, `${profileSystem} -> ${runSystem}: ${name}`).toBe('data');
        expect(entry.coordinateModel).toBe(runSystem === 'planar' ? 'planar' : 'geographic');
        expect(result.parameters[name]).toBe(equivalent.parameters[name]);
      }
    }
  });

  it('resolves a coordinate preset before lengths even when that parameter is last', () => {
    const original = getCatalogBinding('kernel-density-grid')!;
    const binding = {
      ...original,
      parameters: [
        ...original.parameters.filter(parameter => parameter.name !== 'coordinateSystem'),
        original.parameters.find(parameter => parameter.name === 'coordinateSystem')!
      ]
    };
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const profile = profileKeplerFeatureCollection(collection, {
      datasetId: 'points',
      label: 'points',
      coordinateSystem: 'EPSG:4326'
    });
    const result = resolveKeplerOperationParameters(
      binding,
      {
        ...descriptor,
        parameters: [
          ...descriptor.parameters.filter(parameter => parameter.name !== 'coordinateSystem'),
          {
            name: 'coordinateSystem',
            level: 'advanced',
            defaultPolicy: {from: 'preset', value: 'planar', reason: 'Chosen planar interpretation'}
          }
        ]
      },
      {points: profile}
    );
    expect(result.parameters['coordinateSystem']).toBe('planar');
    expect(result.resolutions.find(entry => entry.parameter === 'bandwidth')).toMatchObject({
      source: 'data',
      coordinateModel: 'planar'
    });
  });

  it('records the model of accepted data-derived lengths, including a spacing-only profile', () => {
    const binding = getCatalogBinding('kernel-density-grid')!;
    for (const coordinateSystem of ['planar', 'EPSG:4326'] as const) {
      const profile = profileKeplerFeatureCollection(collection, {
        datasetId: 'points',
        label: 'points',
        coordinateSystem
      });
      const result = resolveKeplerOperationParameters(
        binding,
        getKeplerOperationDescriptor(binding.id)!,
        {points: profile}
      );
      for (const name of ['cellSize', 'bandwidth']) {
        const entry = result.resolutions.find(candidate => candidate.parameter === name)!;
        expect(entry.source).toBe('data');
        expect(entry.coordinateModel).toBe(coordinateSystem === 'planar' ? 'planar' : 'geographic');
        expect(entry.resolution?.unit).toBe(
          coordinateSystem === 'planar' ? 'coordinate-unit' : 'metre'
        );
      }
      const {coordinates: _coordinates, ...spacingOnly} = profile;
      const withoutDeclaration = resolveKeplerOperationParameters(
        binding,
        getKeplerOperationDescriptor(binding.id)!,
        {points: spacingOnly},
        {overrides: {coordinateSystem}}
      );
      expect(
        withoutDeclaration.resolutions.find(entry => entry.parameter === 'bandwidth')
          ?.coordinateModel
      ).toBe(profile.spacing?.coordinateModel);
    }
  });

  it('resolves column-sum from the bound role and rounds capacity protection upwards', () => {
    const binding = getCatalogBinding('geometry-dot-density')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const field = descriptor.inputs[0].fields[0];
    const profile = buildSampleProfile('districts');
    const columns = [
      {name: 'selected', type: 'number' as const, rowCount: 144, nullCount: 4, mean: 1234},
      {name: 'unselected', type: 'number' as const, rowCount: 144, nullCount: 0, mean: 1}
    ];
    const totalAware = {
      ...descriptor,
      parameters: descriptor.parameters.map(parameter =>
        parameter.name === 'dotValue'
          ? {
              ...parameter,
              defaultPolicy: {
                from: 'data' as const,
                input: descriptor.inputs[0].name,
                role: field.role,
                rule: 'column-sum' as const,
                factor: 1 / 10000
              }
            }
          : parameter
      )
    };
    const result = resolveKeplerOperationParameters(
      binding,
      totalAware,
      {[descriptor.inputs[0].name]: {...profile, columns}},
      {fieldBindings: {[field.parameter]: 'selected'}}
    );
    const entry = result.resolutions.find(candidate => candidate.parameter === 'dotValue')!;
    expect(entry.source).toBe('data');
    expect(entry.resolution?.rule).toBe('column-sum');
    expect(entry.resolution?.value).toBeCloseTo((1234 * 140) / 10000);
    expect(result.parameters['dotValue']).toBeGreaterThanOrEqual((1234 * 140) / 10000);
    expect(entry.resolution?.explanation).toContain('selected');
  });
});

describe('third review safe data defaults', () => {
  it('recomputes the verifier lattice band and keeps two to four neighbors per point', async () => {
    const collection = createTestPointCollection(
      Array.from({length: 35}, (_, index) => [(index % 7) * 3, Math.floor(index / 7) * 3]),
      index => ({value: 2 + Math.sin(index) + (index < 8 ? 4 : 0)})
    );
    const profile = profileKeplerFeatureCollection(collection, {
      datasetId: 'features',
      label: 'features'
    });
    const binding = getCatalogBinding('hotspot-gi-star')!;
    const resolved = resolveKeplerOperationParameters(
      binding,
      getKeplerOperationDescriptor(binding.id)!,
      {features: profile},
      {
        fieldBindings: {valueProperty: 'value'},
        overrides: {coordinateSystem: 'planar', neighborhood: 'distance-band'}
      }
    );
    // /usr/bin/python3: scipy.spatial.distance_matrix on [(i % 7 * 3, i // 7 * 3) for i in range(35)]
    // gives nearest distance 3 for every point and neighbor counts {2, 3, 4} at the rounded band 3.1.
    expect(resolved.parameters['distanceBand']).toBe(3.1);
    const output = await runCatalogOperationAsync(
      binding,
      resolved.parameters,
      {features: collection},
      {engine: 'cpu'}
    );
    expect([...new Set(output.tables[0].columns['neighbor_count'])].sort()).toEqual([2, 3, 4]);
  });

  it('explains re-profiling when a serialized profile has no retained spatial statistics for the final model', () => {
    const profile: KeplerDatasetProfile = JSON.parse(
      JSON.stringify(
        profileKeplerFeatureCollection(
          createTestPointCollection([
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1]
          ]),
          {datasetId: 'points', label: 'points', coordinateSystem: 'EPSG:4326'}
        )
      )
    );
    const binding = getCatalogBinding('kernel-density-grid')!;
    const result = resolveKeplerOperationParameters(
      binding,
      getKeplerOperationDescriptor(binding.id)!,
      {points: profile},
      {overrides: {coordinateSystem: 'planar'}}
    );
    for (const name of ['cellSize', 'bandwidth']) {
      const resolution = result.resolutions.find(entry => entry.parameter === name)!;
      expect(resolution.source).toBe('fallback');
      expect(resolution.reason).toMatch(/re-profiling.*no spacing.*profile the dataset again/);
    }
  });

  it('rejects sampled and signed column sums while retaining explicit allocation choices', () => {
    const binding = getCatalogBinding('geometry-dot-density')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    for (const statistics of [
      {sampleSize: 32, minimum: 100},
      {minimum: -800000, maximum: 900000}
    ]) {
      const profile = {
        ...buildSampleProfile('districts'),
        columns: [
          {
            name: 'amount',
            type: 'number' as const,
            rowCount: 200,
            nullCount: 0,
            mean: 500,
            ...statistics
          }
        ]
      };
      const options = {fieldBindings: {valueProperty: 'amount'}};
      const result = resolveKeplerOperationParameters(
        binding,
        descriptor,
        {polygons: profile},
        options
      );
      const resolution = result.resolutions.find(entry => entry.parameter === 'dotValue')!;
      expect(resolution.source).toBe('fallback');
      expect(resolution.reason).toContain(
        'sampleSize' in statistics ? 'sampled' : 'negative values'
      );
      if (!('sampleSize' in statistics)) {
        expect(
          Number(result.parameters['dotValue']) * Number(result.parameters['maximumDots'])
        ).toBeGreaterThanOrEqual(900000 * 200);
        expect(resolution.reason).toContain('conservative fallback');
      }
      expect(
        resolveKeplerOperationParameters(
          binding,
          descriptor,
          {polygons: profile},
          {
            ...options,
            overrides: {dotValue: 10000}
          }
        ).resolutions.find(entry => entry.parameter === 'dotValue')?.source
      ).toBe('override');
    }
  });

  it('keeps a complete total within a lowered dot cap and leaves explicit dot values alone', () => {
    const binding = getCatalogBinding('geometry-dot-density')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const profile = {
      ...buildSampleProfile('districts'),
      columns: [
        {
          name: 'amount',
          type: 'number' as const,
          rowCount: 48,
          nullCount: 0,
          mean: 9470,
          minimum: 9000
        }
      ]
    };
    const result = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {polygons: profile},
      {
        fieldBindings: {valueProperty: 'amount'},
        overrides: {maximumDots: 4000}
      }
    );
    expect((9470 * 48) / Number(result.parameters['dotValue'])).toBeLessThanOrEqual(4000);
    expect(
      result.resolutions.find(entry => entry.parameter === 'dotValue')?.clamped?.reason
    ).toContain('maximumDots');
    expect(
      resolveKeplerOperationParameters(
        binding,
        descriptor,
        {polygons: profile},
        {
          fieldBindings: {valueProperty: 'amount'},
          overrides: {maximumDots: 4000, dotValue: 7}
        }
      ).parameters['dotValue']
    ).toBe(7);
  });
});
