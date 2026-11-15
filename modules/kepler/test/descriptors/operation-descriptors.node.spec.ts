// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import type {CatalogOperationBinding} from '../../src/analysis/catalog-binding-types';
import {KEPLER_CATALOG_BINDINGS} from '../../src/analysis/catalog-bindings';
import {
  deriveKeplerOperationDescriptor,
  getKeplerOperationDescriptor,
  getKeplerOperationDescriptorCoverage,
  KEPLER_OPERATION_DESCRIPTORS,
  resolveKeplerOperationDescriptor
} from '../../src/descriptors/operation-descriptors';
import {
  MAXIMUM_OPERATION_ACTION_LENGTH,
  validateKeplerOperationDescriptor
} from '../../src/descriptors/validate-operation-descriptor';

function makeSyntheticBinding(
  overrides: Partial<CatalogOperationBinding> = {}
): CatalogOperationBinding {
  return {
    id: 'synthetic-unregistered-operation',
    title: 'Measure the synthetic thing between two features of the same kind',
    domain: 'density',
    summary: 'A binding another session might add.',
    capabilityIds: [],
    contributorNames: [],
    gpuContributorNames: [],
    engine: 'cpu-reference',
    inputs: [
      {
        name: 'sites',
        label: 'Sites',
        accepts: ['point', 'polygon'],
        maximumFeatures: 500,
        maximumVertices: 900
      }
    ],
    parameters: [
      {
        name: 'radius',
        label: 'Radius',
        kind: 'number',
        unit: 'crs',
        minimum: 0,
        maximum: 100,
        defaultValue: 5
      },
      {name: 'label', label: 'Label', kind: 'string', maximumLength: 16, defaultValue: ''},
      {name: 'steps', label: 'Steps', kind: 'integer', minimum: 1, maximum: 9, defaultValue: 3},
      {name: 'mode', label: 'Mode', kind: 'enum', options: ['a', 'b'], defaultValue: 'a'},
      {name: 'flag', label: 'Flag', kind: 'boolean', defaultValue: false}
    ],
    outputs: [
      {
        tableId: 'shapes',
        label: 'Shapes',
        geometry: 'polygon',
        identityColumns: ['id'],
        maximumRows: 10
      },
      {tableId: 'summary', label: 'Summary', geometry: 'none', identityColumns: [], maximumRows: 1}
    ],
    cleanup: 'none',
    declaredBehavior: [],
    limitation: 'none',
    ...overrides
  };
}

describe('operation descriptor registry', () => {
  it('has a hand-written descriptor with no validation issue for every binding it describes', () => {
    // Other work adds catalog bindings; one without a hand-written descriptor gets a derived one
    // (conservative, never suggested), so it is reported here and does not fail the registry.
    const coverage = getKeplerOperationDescriptorCoverage(KEPLER_CATALOG_BINDINGS);
    expect(KEPLER_OPERATION_DESCRIPTORS.length).toBeGreaterThanOrEqual(136);
    expect(coverage.validationIssues).toEqual([]);
    expect(coverage.handWrittenBindingIds).toHaveLength(KEPLER_OPERATION_DESCRIPTORS.length);
    expect(coverage.handWrittenBindingIds.length + coverage.derivedBindingIds.length).toBe(
      KEPLER_CATALOG_BINDINGS.length
    );
    if (coverage.derivedBindingIds.length > 0) {
      console.info(
        `${coverage.derivedBindingIds.length} catalog bindings have only a derived descriptor: ${coverage.derivedBindingIds.join(', ')}`
      );
    }
  });

  it('has no orphaned descriptor and no duplicate id', () => {
    const coverage = getKeplerOperationDescriptorCoverage(KEPLER_CATALOG_BINDINGS);
    expect(coverage.orphanedDescriptorIds).toEqual([]);
    const ids = KEPLER_OPERATION_DESCRIPTORS.map(descriptor => descriptor.bindingId);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('reports an orphan, a missing descriptor and a mismatch when the bindings change', () => {
    const [removed, ...rest] = KEPLER_CATALOG_BINDINGS;
    const withoutOne = getKeplerOperationDescriptorCoverage(rest);
    expect(withoutOne.orphanedDescriptorIds).toEqual([removed.id]);
    const withSynthetic = getKeplerOperationDescriptorCoverage([...rest, makeSyntheticBinding()]);
    expect(withSynthetic.derivedBindingIds).toContain('synthetic-unregistered-operation');
    // A binding that lost a parameter its descriptor still names is a validation issue.
    const broken = KEPLER_CATALOG_BINDINGS.find(binding => binding.parameters.length > 0)!;
    const reduced = {...broken, parameters: broken.parameters.slice(1)};
    const reducedCoverage = getKeplerOperationDescriptorCoverage([reduced]);
    expect(reducedCoverage.validationIssues.length).toBeGreaterThan(0);
  });

  it('finds a descriptor by binding id and returns undefined for an unknown id', () => {
    expect(getKeplerOperationDescriptor('recipe-hot-spot-lattice')?.bindingId).toBe(
      'recipe-hot-spot-lattice'
    );
    expect(getKeplerOperationDescriptor('no-such-binding')).toBeUndefined();
    expect(getKeplerOperationDescriptor('')).toBeUndefined();
  });
});

describe('deriveKeplerOperationDescriptor', () => {
  it('derives a descriptor that the validator accepts for a binding with an unknown id', () => {
    const binding = makeSyntheticBinding();
    const descriptor = deriveKeplerOperationDescriptor(binding);
    expect(validateKeplerOperationDescriptor(descriptor, binding)).toEqual([]);
    expect(descriptor.questions).toEqual(['where-concentrated']);
    expect(descriptor.prominence).toBe(3);
    expect(descriptor.inputs).toEqual([{name: 'sites', kind: 'points', fields: []}]);
    expect(descriptor.outputs.map(output => output.kind)).toEqual(['polygons', 'summary']);
  });

  it('keeps every parameter constant and only the first three basic', () => {
    const descriptor = deriveKeplerOperationDescriptor(makeSyntheticBinding());
    expect(
      descriptor.parameters.every(parameter => parameter.defaultPolicy.from === 'constant')
    ).toBe(true);
    expect(descriptor.parameters.map(parameter => parameter.level)).toEqual([
      'basic',
      'basic',
      'basic',
      'advanced',
      'advanced'
    ]);
  });

  it('takes capacity requirements from the binding numbers', () => {
    const descriptor = deriveKeplerOperationDescriptor(makeSyntheticBinding());
    expect(descriptor.requirements.map(({id, limit, severity}) => ({id, limit, severity}))).toEqual(
      [
        {id: 'maximum-rows', limit: 500, severity: 'block'},
        {id: 'maximum-vertices', limit: 900, severity: 'block'}
      ]
    );
  });

  it('truncates a long title at a word boundary within the action limit', () => {
    const descriptor = deriveKeplerOperationDescriptor(makeSyntheticBinding());
    expect(descriptor.action.length).toBeLessThanOrEqual(MAXIMUM_OPERATION_ACTION_LENGTH);
    expect(descriptor.action).toBe('Measure the synthetic thing between two');
    const unbroken = deriveKeplerOperationDescriptor(makeSyntheticBinding({title: 'X'.repeat(60)}));
    expect(unbroken.action).toBe('X'.repeat(MAXIMUM_OPERATION_ACTION_LENGTH));
    const short = deriveKeplerOperationDescriptor(makeSyntheticBinding({title: 'Short title'}));
    expect(short.action).toBe('Short title');
  });

  it('maps every domain to a question and the first accepted geometry to a kind', () => {
    const domains = [
      'geometry',
      'network',
      'aggregation',
      'interpolation',
      'density',
      'clustering',
      'raster',
      'analysis'
    ] as const;
    for (const domain of domains) {
      const binding = makeSyntheticBinding({
        domain,
        inputs: [
          {
            name: 'lines',
            label: 'Lines',
            accepts: ['line'],
            maximumFeatures: 10,
            maximumVertices: 10
          }
        ]
      });
      const descriptor = deriveKeplerOperationDescriptor(binding);
      expect(descriptor.questions).toHaveLength(1);
      expect(descriptor.inputs[0].kind).toBe('lines');
      expect(validateKeplerOperationDescriptor(descriptor, binding)).toEqual([]);
    }
  });

  it('derives a valid descriptor for a binding with no parameters and several inputs', () => {
    const binding = makeSyntheticBinding({
      parameters: [],
      inputs: [
        {name: 'a', label: 'A', accepts: ['polygon'], maximumFeatures: 1, maximumVertices: 1},
        {name: 'b', label: 'B', accepts: ['point'], maximumFeatures: 2, maximumVertices: 2}
      ]
    });
    const descriptor = deriveKeplerOperationDescriptor(binding);
    expect(validateKeplerOperationDescriptor(descriptor, binding)).toEqual([]);
    expect(descriptor.requirements).toHaveLength(4);
  });
});

describe('resolveKeplerOperationDescriptor', () => {
  it('returns the hand-written descriptor and says so', () => {
    const binding = KEPLER_CATALOG_BINDINGS[0];
    const resolved = resolveKeplerOperationDescriptor(binding);
    expect(resolved.source).toBe('hand-written');
    expect(resolved.descriptor).toBe(getKeplerOperationDescriptor(binding.id));
  });

  it('derives a descriptor for an unregistered binding and says so', () => {
    const resolved = resolveKeplerOperationDescriptor(makeSyntheticBinding());
    expect(resolved.source).toBe('derived');
    expect(resolved.descriptor.bindingId).toBe('synthetic-unregistered-operation');
  });
});
