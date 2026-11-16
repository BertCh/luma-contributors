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

// AX-80 is complete: every catalog binding has a hand-written descriptor. A new binding that
// falls back to a derived descriptor must be described, not added here.
const DERIVED_BINDING_ALLOWANCE = new Set<string>();

describe('operation descriptor registry', () => {
  it('permits column-role debt only for the explicitly listed derived bindings', () => {
    const coverage = getKeplerOperationDescriptorCoverage(KEPLER_CATALOG_BINDINGS);
    expect(coverage.derivedBindingIds.filter(id => !DERIVED_BINDING_ALLOWANCE.has(id))).toEqual([]);
    for (const binding of KEPLER_CATALOG_BINDINGS) {
      const {descriptor, source} = resolveKeplerOperationDescriptor(binding);
      if (source === 'derived') continue;
      expect(validateKeplerOperationDescriptor(descriptor, binding), binding.id).toEqual([]);
      // Ordinary strings are literal comparison values or vertical reference labels, not columns.
      expect(
        binding.parameters
          .filter(
            parameter =>
              parameter.kind === 'string' &&
              descriptor.parameters.some(setting => setting.name === parameter.name)
          )
          .map(parameter => parameter.name),
        binding.id
      ).toEqual(
        binding.id === 'network-subgraph-filter'
          ? ['propertyValue']
          : binding.id === 'curated-zonal-exposure'
            ? ['densityAreaUnit', 'cellAreaUnit']
            : binding.id === 'datum-grid-transform'
              ? ['sourceVerticalReference', 'targetVerticalReference']
              : []
      );
      expect(
        descriptor.parameters.every(parameter => ['live', 'rebuild'].includes(parameter.apply)),
        binding.id
      ).toBe(true);
    }
  });

  it('accepts confirmations only on false-by-default booleans with explicit instructions', () => {
    const binding = makeSyntheticBinding();
    const descriptor = deriveKeplerOperationDescriptor(binding);
    const confirmation = {
      message: 'Confirm the shared frame',
      remedy: 'Set flag to true after aligning'
    };
    descriptor.parameters = descriptor.parameters.map(parameter =>
      parameter.name === 'flag' ? {...parameter, confirmation} : parameter
    );
    expect(validateKeplerOperationDescriptor(descriptor, binding)).toEqual([]);
    const preset = {
      ...descriptor,
      parameters: descriptor.parameters.map(parameter =>
        parameter.name === 'flag'
          ? {
              ...parameter,
              defaultPolicy: {from: 'preset' as const, value: true, reason: 'Automatic'}
            }
          : parameter
      )
    };
    expect(validateKeplerOperationDescriptor(preset, binding)).toContain(
      `${binding.id}: parameter flag confirmation requires a false-by-default boolean with a constant policy`
    );
    const wrongKind = {
      ...descriptor,
      parameters: descriptor.parameters.map(parameter =>
        parameter.name === 'radius'
          ? {...parameter, confirmation: {message: '', remedy: ''}}
          : parameter
      )
    };
    expect(validateKeplerOperationDescriptor(wrongKind, binding)).toContain(
      `${binding.id}: parameter radius confirmation requires a false-by-default boolean with a constant policy`
    );
    expect(validateKeplerOperationDescriptor(wrongKind, binding)).toContain(
      `${binding.id}: parameter radius confirmation needs a message and remedy`
    );
    const trueDefault = {
      ...binding,
      parameters: binding.parameters.map(parameter =>
        parameter.name === 'flag' ? {...parameter, defaultValue: true} : parameter
      )
    };
    expect(validateKeplerOperationDescriptor(descriptor, trueDefault)).toContain(
      `${binding.id}: parameter flag confirmation requires a false-by-default boolean with a constant policy`
    );
  });

  it('rejects a column reference disguised as an ordinary setting', () => {
    const binding = makeSyntheticBinding({
      parameters: [
        {name: 'valueProperty', label: 'Value column', kind: 'string', defaultValue: 'value'}
      ]
    });
    const descriptor = deriveKeplerOperationDescriptor(binding);
    expect(validateKeplerOperationDescriptor(descriptor, binding)).toContain(
      `${binding.id}: column-naming string parameter valueProperty must be mapped to a field role`
    );
    expect(
      validateKeplerOperationDescriptor(descriptor, binding, {allowUnmappedColumnParameters: true})
    ).toEqual([]);
  });

  it('validates every branch of a cross-parameter condition', () => {
    const binding = makeSyntheticBinding();
    const descriptor = deriveKeplerOperationDescriptor(binding);
    descriptor.parameters = descriptor.parameters.map(parameter =>
      parameter.name === 'radius'
        ? {
            ...parameter,
            activeWhen: {
              anyOf: [
                {parameter: 'mode', equals: ['a']},
                {parameter: 'steps', equals: [1.5]},
                {parameter: 'unknown', equals: [true]}
              ]
            }
          }
        : parameter
    );
    expect(validateKeplerOperationDescriptor(descriptor, binding)).toEqual([
      `${binding.id}: parameter radius activeWhen value 1.5 is outside the allowed values of parameter steps (integer)`,
      `${binding.id}: parameter radius activeWhen names unknown parameter unknown`
    ]);
  });

  it('has a hand-written descriptor with no validation issue for every binding it describes', () => {
    // AX-80 is not complete while the explicit allowance above remains. Each registered binding
    // must resolve; derived descriptors remain conservative and must not hide validation issues.
    const coverage = getKeplerOperationDescriptorCoverage(KEPLER_CATALOG_BINDINGS);
    expect(KEPLER_OPERATION_DESCRIPTORS.length).toBeGreaterThanOrEqual(136);
    expect(coverage.validationIssues).toEqual([]);
    expect(coverage.handWrittenBindingIds).toHaveLength(KEPLER_OPERATION_DESCRIPTORS.length);
    expect(coverage.handWrittenBindingIds.length + coverage.derivedBindingIds.length).toBe(
      KEPLER_CATALOG_BINDINGS.length
    );
    for (const binding of KEPLER_CATALOG_BINDINGS) {
      const {descriptor, source} = resolveKeplerOperationDescriptor(binding);
      expect(descriptor.bindingId, binding.id).toBe(binding.id);
      if (source === 'derived') {
        expect(DERIVED_BINDING_ALLOWANCE.has(binding.id), binding.id).toBe(true);
        expect(descriptor.prominence, binding.id).toBe(3);
        expect(
          validateKeplerOperationDescriptor(descriptor, binding, {
            allowUnmappedColumnParameters: true
          }),
          binding.id
        ).toEqual([]);
      }
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

it('distinguishes runtime/post-pass edits from allocation and shader changes', () => {
  const cases = [
    ['idw-grid', 'power', 'live'],
    ['idw-grid', 'minimumNeighborCount', 'live'],
    ['idw-grid', 'neighborCount', 'rebuild'],
    ['idw-grid', 'cellSize', 'rebuild'],
    ['kernel-density-grid', 'bandwidth', 'live'],
    ['kernel-density-grid', 'kernel', 'rebuild'],
    ['kernel-density-grid', 'includeEmptyCells', 'live'],
    ['kriging-grid', 'range', 'live'],
    ['kriging-grid', 'neighborCount', 'rebuild'],
    ['affine-transform', 'translateX', 'live'],
    ['affine-transform', 'pivot', 'rebuild'],
    ['geometry-line-buffer', 'quadSegments', 'rebuild'],
    ['local-moran-statistics', 'significance', 'live'],
    ['local-moran-statistics', 'permutations', 'rebuild'],
    ['similar-locations', 'resultCount', 'live'],
    ['s2-cell-measures', 'radius', 'live'],
    ['s2-cell-measures', 'level', 'rebuild'],
    ['local-outlier-factor', 'threshold', 'live'],
    ['local-outlier-factor', 'neighborCount', 'rebuild']
  ];
  for (const [bindingId, name, apply] of cases) {
    expect(
      getKeplerOperationDescriptor(bindingId)?.parameters.find(parameter => parameter.name === name)
        ?.apply,
      `${bindingId}.${name}`
    ).toBe(apply);
  }
});
