// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test, vi} from 'vitest';
import {createRequire} from 'node:module';
import type {ReactElement, ReactNode} from 'react';
import {
  KEPLER_CATALOG_BINDINGS,
  getCatalogBinding,
  profileKeplerFeatureCollection,
  runCatalogOperation,
  type CatalogOperationBinding,
  type CatalogFeatureCollection,
  type KeplerDatasetProfile
} from '../src/index';
import {
  createAnalysisCatalogFormModel,
  describeAnalysisCatalogParameter,
  EMPTY_CATALOG_INPUT_DRAFT,
  profileAnalysisCatalogInput,
  resolveAnalysisCatalogInputProfiles
} from '../../../examples/deck/kepler-compute/src/analysis-catalog-form-model';
import {
  analysisDatasetContext,
  createAnalysisDatasetContext
} from '../../../examples/deck/kepler-compute/src/analysis-dataset-context';
import {AnalysisCatalogPanel} from '../../../examples/deck/kepler-compute/src/analysis-catalog-panel';
import {AnalysisRowMeaning} from '../../../examples/deck/kepler-compute/src/analysis-profile-declarations';
import {makeCatalogPointColumns} from '../../../examples/deck/kepler-compute/src/analysis-catalog-input';
import type {CatalogResidentDataset} from '../../../examples/deck/kepler-compute/src/analysis-catalog-resident-input';

function getBinding(bindingId: string): CatalogOperationBinding {
  return getCatalogBinding(bindingId)!;
}

function makeProfiles(binding: CatalogOperationBinding) {
  return Object.fromEntries(
    binding.inputs.map(input => {
      const geometry = input.accepts[0];
      const profile: KeplerDatasetProfile = {
        datasetId: input.name,
        label: input.label,
        kind:
          input.dataKind === 'raster'
            ? 'surface'
            : input.dataKind === 'table'
              ? 'attributes'
              : geometry === 'point'
                ? 'points'
                : geometry === 'line'
                  ? 'lines'
                  : 'polygons',
        catalogGeometry: geometry,
        rowCount: Math.min(
          10,
          input.dataKind === 'raster' ? input.maximumCells! : input.maximumFeatures
        ),
        vertexCount: Math.min(10, input.maximumVertices),
        geometryColumns: input.dataKind && input.dataKind !== 'features' ? [] : ['geometry'],
        columns: [],
        extent: [0, 0, 100, 100],
        coordinates: {model: 'planar', source: 'declared'}
      };
      return [input.name, profile];
    })
  );
}

describe('catalog selected-input preflight matrix', () => {
  for (const binding of KEPLER_CATALOG_BINDINGS) {
    if (binding.inputs.length === 0) {
      test(`${binding.id}: three incompatible layer profiles cannot be assigned to a no-input demonstration`, () => {
        for (const profile of [
          {kind: 'attributes', rowCount: 10, vertexCount: 0},
          {kind: 'points', rowCount: Number.MAX_SAFE_INTEGER, vertexCount: 10},
          {kind: 'polygons', rowCount: 10, vertexCount: Number.MAX_SAFE_INTEGER}
        ] as const) {
          const model = createAnalysisCatalogFormModel(binding, {
            unused: {
              ...profile,
              datasetId: 'unused',
              label: 'unused',
              columns: [],
              geometryColumns: []
            }
          });
          expect(model.capacities).toEqual([]);
          expect(model.applicability.inputAssignments).toEqual({});
          expect(model.missing.some(item => item.what.includes('nothing to suggest'))).toBe(false);
        }
      });
      continue;
    }
    test(`${binding.id}: wrong kind and excess input capacities stop Run with reasons`, () => {
      for (const input of binding.inputs) {
        const profiles = makeProfiles(binding);
        const wrongKind = createAnalysisCatalogFormModel(binding, {
          ...profiles,
          [input.name]: {
            ...profiles[input.name],
            kind: input.dataKind === 'table' ? 'surface' : 'attributes',
            catalogGeometry: undefined,
            geometryColumns: []
          }
        });
        expect(wrongKind.runDisabled).toBe(true);
        expect(wrongKind.missing.some(item => item.what.length > 0 && Boolean(item.remedy))).toBe(
          true
        );

        // Raster rows count decoded cells; feature/table rows count records.
        const maximumRows =
          input.dataKind === 'raster' ? input.maximumCells! : input.maximumFeatures;
        if (input.dataKind === 'raster') {
          // Native rasters are bounded by cells, never feature rows or geometry vertices.
          const maximumCells = input.maximumCells!;
          const excessCells = createAnalysisCatalogFormModel(binding, {
            ...profiles,
            [input.name]: {...profiles[input.name], rowCount: maximumCells + 1}
          });
          expect(excessCells.runDisabled).toBe(true);
          expect(
            excessCells.applicability.requirements.some(
              requirement =>
                requirement.input === input.name &&
                requirement.id === 'maximum-rows' &&
                requirement.state === 'failed' &&
                requirement.evidence === 'profile' &&
                Boolean(requirement.remedy)
            )
          ).toBe(true);
          expect(excessCells.capacities.find(item => item.input === input.name)).toMatchObject({
            maximumRows: maximumCells,
            rows: maximumCells + 1
          });
        }

        const excessRows = createAnalysisCatalogFormModel(binding, {
          ...profiles,
          [input.name]: {...profiles[input.name], rowCount: maximumRows + 1}
        });
        expect(excessRows.runDisabled).toBe(true);
        expect(
          excessRows.applicability.requirements.some(
            requirement =>
              requirement.input === input.name &&
              requirement.id === 'maximum-rows' &&
              requirement.state === 'failed' &&
              requirement.evidence === 'profile' &&
              Boolean(requirement.remedy)
          )
        ).toBe(true);

        expect(excessRows.capacities.find(item => item.input === input.name)).toMatchObject({
          maximumRows,
          rows: maximumRows + 1
        });
        if (input.dataKind === 'raster' || input.dataKind === 'table') {
          // Typed ports have no feature geometry or vertex capacity. A vertex statistic must
          // not introduce a fictitious geometry limit for a valid surface/table input.
          const nonFeature = createAnalysisCatalogFormModel(binding, {
            ...profiles,
            [input.name]: {...profiles[input.name], vertexCount: Number.MAX_SAFE_INTEGER}
          });
          expect(nonFeature.selectionChecks.filter(check => check.input === input.name)).toEqual(
            []
          );
          expect(
            nonFeature.applicability.requirements.some(
              requirement =>
                requirement.input === input.name && requirement.id === 'maximum-vertices'
            )
          ).toBe(false);
          continue;
        }
        const excessVertices = createAnalysisCatalogFormModel(binding, {
          ...profiles,
          [input.name]: {...profiles[input.name], vertexCount: input.maximumVertices + 1}
        });
        expect(excessVertices.runDisabled).toBe(true);
        expect(
          excessVertices.applicability.requirements.some(
            requirement =>
              requirement.input === input.name &&
              requirement.id === 'maximum-vertices' &&
              requirement.state === 'failed' &&
              Boolean(requirement.remedy)
          )
        ).toBe(true);
        expect(excessVertices.capacities.find(item => item.input === input.name)).toMatchObject({
          maximumRows: input.maximumFeatures,
          maximumVertices: input.maximumVertices,
          vertices: input.maximumVertices + 1
        });
      }
    });
  }
});

function makeIDWReviewFixture(extent: readonly [number, number, number, number]) {
  const binding = getBinding('idw-grid');
  const collection: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: Array.from({length: 10}, (_, index) => ({
      type: 'Feature',
      geometry: {
        type: 'Point',
        coordinates: [
          extent[0] + ((extent[2] - extent[0]) * index) / 9,
          extent[1] + ((extent[3] - extent[1]) * index) / 9
        ]
      },
      properties: {value: index + 1}
    }))
  };
  const profile = profileKeplerFeatureCollection(collection, {
    datasetId: 'review-samples',
    label: 'Review samples',
    coordinateSystem: 'planar',
    declarations: {rowSemantics: 'surface-samples', measurement: {value: 'amount'}}
  });
  return {binding, collection, profiles: {points: profile}};
}

test('review finding 1: execution uses the exact grid-cap-adjusted parameters approved by preflight', () => {
  const {binding, collection, profiles} = makeIDWReviewFixture([0, 0, 1, 1_000_000]);
  const model = createAnalysisCatalogFormModel(binding, profiles, {
    executionEngine: 'cpu',
    overrides: {valueProperty: 'value', searchRadius: 1000}
  });
  expect(model.runDisabled).toBe(false);
  expect(model.resolved.parameters).toEqual(model.applicability.parameters);
  expect(model.resolved.parameters['cellSize']).toBe(4.1);
  const result = runCatalogOperation(binding, model.resolved.parameters, {points: collection});
  expect(result.parameters).toEqual(model.applicability.parameters);
  expect(result.tables[0].rowCount).toBeGreaterThan(0);
});

test.each([
  '',
  '   '
])('review finding 4: explicitly clearing a required column (%j) disables Run without autofilling it', valueProperty => {
  const {binding, profiles} = makeIDWReviewFixture([0, 0, 10, 10]);
  const before = createAnalysisCatalogFormModel(binding, profiles, {
    overrides: {valueProperty: 'value', searchRadius: 10}
  });
  expect(before.runDisabled).toBe(false);
  const cleared = createAnalysisCatalogFormModel(binding, profiles, {
    overrides: {valueProperty, searchRadius: 10}
  });
  expect(cleared.runDisabled).toBe(true);
  expect(
    cleared.missing.some(
      item => /valueProperty|Measured value/.test(item.what) && Boolean(item.remedy)
    )
  ).toBe(true);
  expect(cleared.applicability.fieldBindings['valueProperty']).toBeUndefined();
  expect(cleared.resolved.parameters).toEqual(cleared.applicability.parameters);
  expect(cleared.resolved.parameters['valueProperty']).toBe('');
  expect(cleared.parameters['valueProperty']).toBe(valueProperty);
  expect(
    createAnalysisCatalogFormModel(binding, profiles, {overrides: {searchRadius: 10}}).runDisabled
  ).toBe(false);
});

test('review finding 3: a current checked polygon profile refines matching shared metadata', () => {
  const context = createAnalysisDatasetContext();
  const dataset: CatalogResidentDataset = {
    id: 'review-polygons',
    label: 'Review polygons',
    revision: 1,
    rowCount: 1,
    geometryColumn: 'shape',
    geometryKind: 'polygon',
    fieldTypes: {shape: 'string', value: 'real'},
    provider: {
      fields: [
        {name: 'shape', sortable: true},
        {name: 'value', sortable: true}
      ],
      maximumPageSize: 10,
      getRevision: () => '1',
      readPage: vi.fn(),
      readColumnValues: vi.fn()
    },
    isCurrent: () => true
  };
  context.synchronize([dataset]);
  context.setDeclarations(dataset.id, {
    coordinateModel: 'planar',
    rowSemantics: 'objects',
    measurement: {value: 'amount'}
  });
  const binding = getBinding('geometry-measures');
  const inputName = binding.inputs[0].name;
  const drafts = {
    [`${binding.id}:${inputName}`]: {
      ...EMPTY_CATALOG_INPUT_DRAFT,
      mode: 'native' as const,
      datasetId: dataset.id,
      field: 'shape'
    }
  };
  const before = resolveAnalysisCatalogInputProfiles(binding, drafts, context.get);
  expect(before.profiles[inputName]?.vertexCount).toBeUndefined();
  expect(createAnalysisCatalogFormModel(binding, before.profiles).runDisabled).toBe(true);
  const checked = profileAnalysisCatalogInput(binding, inputName, {
    type: 'FeatureCollection',
    features: [
      {
        properties: {value: 3},
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [0, 0],
              [2, 0],
              [2, 2],
              [0, 0]
            ]
          ]
        }
      }
    ]
  });
  const after = resolveAnalysisCatalogInputProfiles(binding, drafts, context.get, {
    [inputName]: checked
  });
  expect(after.profiles[inputName]).toMatchObject({
    datasetId: dataset.id,
    vertexCount: 4,
    geometryColumns: ['shape'],
    coordinates: {model: 'planar', source: 'declared'},
    rowSemantics: {value: 'objects', source: 'declared'}
  });
  expect(after.profiles[inputName]?.columns.find(column => column.name === 'value')).toMatchObject({
    measurement: 'amount',
    measurementSource: 'declared'
  });
  const model = createAnalysisCatalogFormModel(binding, after.profiles);
  expect(model.runDisabled).toBe(false);
  expect(
    model.applicability.requirements.find(item => item.id === 'maximum-vertices')
  ).toMatchObject({state: 'passed', evidence: 'profile'});
});

test('review finding 2: native CSV without shared roles keeps its mapping placeholder and can be checked', async () => {
  const source = {
    id: 'review-csv',
    label: 'Review CSV',
    fields: [
      {name: 'east', type: 'real'},
      {name: 'north', type: 'real'}
    ],
    dataContainer: {numRows: () => 3, valueAt: (row: number, column: number) => row + column * 10}
  };
  const render = createCatalogPanelProbe({
    datasets: [source as unknown as Parameters<typeof AnalysisCatalogPanel>[0]['datasets'][number]],
    onPublish: vi.fn()
  });
  let elements = render();
  changeCatalogSelect(elements, 'catalog-operation', 'geometry-measures');
  elements = render();
  changeCatalogSelect(elements, 'Features source', 'native');
  elements = render();
  changeCatalogSelect(elements, 'Features dataset', source.id);
  elements = render();
  changeCatalogSelect(elements, 'Features geometry field', makeCatalogPointColumns('', ''));
  elements = render();
  expect(
    elements.find(element => element.props['aria-label'] === 'Features geometry field')?.props[
      'value'
    ]
  ).toBe(makeCatalogPointColumns('', ''));
  expect(
    elements.find(element => element.props['aria-label'] === 'Features X / longitude column')
  ).toBeDefined();
  expect(
    elements.find(element => element.props['aria-label'] === 'Features Y / latitude column')
  ).toBeDefined();
  changeCatalogSelect(elements, 'Features X / longitude column', 'east');
  elements = render();
  changeCatalogSelect(elements, 'Features Y / latitude column', 'north');
  elements = render();
  const coordinateSetting = elements.find(
    element => (element.props['spec'] as {name?: string} | undefined)?.name === 'coordinateSystem'
  )!;
  (coordinateSetting.props['onChange'] as (value: unknown) => void)('planar');
  elements = render();
  await (
    elements.find(element => element.props['data-testid'] === 'catalog-check-inputs')!.props[
      'onClick'
    ] as () => Promise<void>
  )();
  elements = render();
  expect(
    elements.find(element => element.props['data-testid'] === 'catalog-error')
  ).toBeUndefined();
  expect(
    elements.find(element => element.props['data-testid'] === 'catalog-run')?.props['disabled']
  ).toBe(false);
});

test('every GPU-only binding reports unavailable engine before execution', () => {
  for (const binding of KEPLER_CATALOG_BINDINGS.filter(item => item.engine === 'gpu')) {
    const model = createAnalysisCatalogFormModel(binding, makeProfiles(binding), {
      executionEngine: 'cpu'
    });
    expect(model.runDisabled, binding.id).toBe(true);
    expect(
      model.missing.some(item => item.what.includes('needs a GPU device') && Boolean(item.remedy)),
      binding.id
    ).toBe(true);
  }
});

test('selected assignments are exact, including missing slots and shared datasets', () => {
  const binding = getBinding('zonal-point-statistics');
  const profiles = makeProfiles(binding);
  const first = binding.inputs[0].name;
  const model = createAnalysisCatalogFormModel(binding, {...profiles, [first]: undefined});
  expect(model.runDisabled).toBe(true);
  expect(model.applicability.inputAssignments[first]).toBeUndefined();
  for (const input of binding.inputs.slice(1))
    expect(model.applicability.inputAssignments[input.name]).toBe(profiles[input.name].datasetId);
  const comparison = getBinding('cell-compare');
  const sharedProfile = {...Object.values(makeProfiles(comparison))[0], datasetId: 'same-period'};
  const shared = createAnalysisCatalogFormModel(
    comparison,
    Object.fromEntries(comparison.inputs.map(input => [input.name, sharedProfile]))
  );
  // The checker refuses reusing one layer for both periods. It must never substitute another
  // source to make the assignment runnable, or silently report two independent populations.
  expect(Object.values(shared.applicability.inputAssignments).filter(Boolean)).toEqual([
    'same-period'
  ]);
  expect(shared.runDisabled).toBe(true);
  expect(shared.missing.some(item => Boolean(item.remedy))).toBe(true);
});

test('catalog defaults come from the selected extent, and profile refinement preserves explicit edits', () => {
  const binding = getBinding('idw-grid');
  const profiles = makeProfiles(binding);
  const inputName = binding.inputs[0].name;
  profiles[inputName].extent = [3000, 4000, 6000, 8000];
  const initial = createAnalysisCatalogFormModel(binding, profiles);
  // Catalog interpolation already covers the source extent; its editable scale is cell size.
  // The shorter side is 3000, so the descriptor rule gives 3000 / 250 = 12 source units.
  expect(initial.resolved.parameters['cellSize']).toBe(12);
  expect(initial.resolved.resolutions.find(item => item.parameter === 'cellSize')?.source).toBe(
    'data'
  );
  const overrides = {cellSize: '15'};
  const richerProfiles = {
    ...profiles,
    [inputName]: {...profiles[inputName], extent: [2000, 3000, 7000, 9000] as const}
  };
  expect(
    createAnalysisCatalogFormModel(binding, richerProfiles).resolved.parameters['cellSize']
  ).toBe(20);
  const refined = createAnalysisCatalogFormModel(binding, richerProfiles, {overrides});
  expect(refined.resolved.parameters['cellSize']).toBe(15);
  expect(refined.parameters['cellSize']).toBe('15');
  expect(refined.resolved.resolutions.find(item => item.parameter === 'cellSize')?.source).toBe(
    'override'
  );
  expect(overrides).toEqual({cellSize: '15'});
});

test('missing roles and unknown evidence stay visible and stop mandatory checks from passing', () => {
  const binding = getBinding('idw-grid');
  const profiles = makeProfiles(binding);
  const missingRole = createAnalysisCatalogFormModel(binding, profiles);
  expect(missingRole.runDisabled).toBe(true);
  expect(missingRole.applicability.fieldBindings).toEqual({});
  expect(missingRole.missing.map(item => item.what).join(' ')).toMatch(/column/);
  expect(missingRole.missing.some(item => Boolean(item.remedy))).toBe(true);
  const geometryBinding = getBinding('geometry-measures');
  const geometryProfiles = makeProfiles(geometryBinding);
  const input = geometryBinding.inputs[0].name;
  const unknown = createAnalysisCatalogFormModel(geometryBinding, {
    ...geometryProfiles,
    [input]: {
      ...geometryProfiles[input],
      kind: 'polygons',
      catalogGeometry: 'polygon',
      vertexCount: undefined
    }
  });
  expect(
    unknown.applicability.requirements.some(
      item =>
        item.id === 'maximum-vertices' &&
        item.state === 'unknown' &&
        item.evidence === 'not-checkable'
    )
  ).toBe(true);
  expect(unknown.runDisabled).toBe(true);
});

test('invalid edits remain editable and block Run; restored parameters count as explicit edits', () => {
  const binding = getBinding('idw-grid');
  const profiles = makeProfiles(binding);
  const invalid = createAnalysisCatalogFormModel(binding, profiles, {overrides: {cellSize: '-'}});
  expect(invalid.parameters['cellSize']).toBe('-');
  expect(invalid.runDisabled).toBe(true);
  expect(invalid.problems.join(' ')).toContain(
    'idw-grid: cellSize must be a finite number; received -'
  );
  const initial = createAnalysisCatalogFormModel(binding, profiles);
  const restored = createAnalysisCatalogFormModel(binding, profiles, {
    overrides: initial.resolved.parameters
  });
  expect(restored.resolved.resolutions.every(item => item.source === 'override')).toBe(true);
});

test('resolution explanations include fallback, guesses and adjustments', () => {
  expect(
    describeAnalysisCatalogParameter({
      parameter: 'cellSize',
      source: 'fallback',
      reason: 'No extent',
      guessed: true,
      clamped: {from: 1, to: 10, reason: 'grid capacity'}
    })
  ).toContain('fallback · No extent · based on a guess');
  expect(
    describeAnalysisCatalogParameter({
      parameter: 'cellSize',
      source: 'data',
      clamped: {from: 1, to: 10, reason: 'grid capacity'}
    })
  ).toContain('adjusted from 1 to 10: grid capacity');
});

test('no-input demonstrations remain runnable and running always disables Run', () => {
  const binding = KEPLER_CATALOG_BINDINGS.find(item => item.inputs.length === 0)!;
  expect(createAnalysisCatalogFormModel(binding, {}).runDisabled).toBe(false);
  expect(createAnalysisCatalogFormModel(binding, {}, {running: true}).runDisabled).toBe(true);
});

test('ready is enabled only after mandatory checks pass; unknown warnings remain labelled', () => {
  const binding = getBinding('geometry-measures');
  const profiles = makeProfiles(binding);
  const model = createAnalysisCatalogFormModel(binding, profiles);
  expect(model.status).toBe('ready');
  expect(model.runDisabled).toBe(false);
  expect(
    model.applicability.requirements.every(
      item => item.severity === 'warn' || item.state === 'passed'
    )
  ).toBe(true);
  const missing = createAnalysisCatalogFormModel(binding, {});
  expect(missing.selectionChecks[0]).toMatchObject({state: 'unknown', evidence: 'not-checkable'});
  expect(missing.runDisabled).toBe(true);
});

test('explicit native mapping checks use the existing bounded profiler and never bypass capacities', () => {
  const binding = getBinding('geometry-measures');
  const collection = {
    type: 'FeatureCollection' as const,
    features: [
      {
        type: 'Feature' as const,
        geometry: {type: 'Point' as const, coordinates: [1, 2]},
        properties: {}
      }
    ]
  };
  const inputName = binding.inputs[0].name;
  const profile = profileAnalysisCatalogInput(binding, inputName, collection);
  const draft = {
    ...EMPTY_CATALOG_INPUT_DRAFT,
    mode: 'native' as const,
    datasetId: 'unprofiled',
    field: 'geometry'
  };
  const selected = resolveAnalysisCatalogInputProfiles(
    binding,
    {[`${binding.id}:${inputName}`]: draft},
    () => undefined,
    {[inputName]: profile}
  );
  expect(selected.profiles[inputName]).toEqual(profile);
  expect(selected.problems).toEqual([]);
  const boundedBinding = {
    ...binding,
    inputs: binding.inputs.map(input => ({...input, maximumFeatures: 0}))
  };
  expect(() => profileAnalysisCatalogInput(boundedBinding, inputName, collection)).toThrow(
    'exceed the limit 0'
  );
});

test('shared native/resident profile switching uses no readback and rejects unprofiled mappings', () => {
  const context = createAnalysisDatasetContext();
  const readColumns = vi.fn(async () => ({values: [1, 2], fieldType: 'real'}));
  const dataset: CatalogResidentDataset = {
    id: 'current',
    label: 'Current',
    revision: 1,
    rowCount: 2,
    coordinates: {longitude: 'east', latitude: 'north'},
    fieldTypes: {east: 'real', north: 'real'},
    provider: {
      fields: [
        {name: 'east', sortable: true},
        {name: 'north', sortable: true}
      ],
      maximumPageSize: 10,
      getRevision: () => '1',
      readPage: vi.fn(),
      readColumnValues: readColumns
    },
    isCurrent: () => true
  };
  context.synchronize([dataset]);
  const binding = getBinding('idw-grid');
  const input = binding.inputs[0].name;
  const key = `${binding.id}:${input}`;
  const native = {
    ...EMPTY_CATALOG_INPUT_DRAFT,
    mode: 'native' as const,
    datasetId: dataset.id,
    field: makeCatalogPointColumns('east', 'north')
  };
  const selection = resolveAnalysisCatalogInputProfiles(binding, {[key]: native}, context.get);
  expect(selection.profiles[input]).toBe(context.get(dataset.id)!.profile);
  expect(selection.problems).toEqual([]);
  expect(readColumns).not.toHaveBeenCalled();
  const unprofiled = resolveAnalysisCatalogInputProfiles(
    binding,
    {[key]: {...native, field: makeCatalogPointColumns('north', 'east')}},
    context.get
  );
  expect(unprofiled.profiles[input]).toBeUndefined();
  expect(unprofiled.problems.join(' ')).toContain('not been profiled');
  const removed = resolveAnalysisCatalogInputProfiles(
    binding,
    {[key]: {...native, datasetId: 'removed'}},
    context.get
  );
  expect(removed.profiles[input]).toBeUndefined();
  const text = resolveAnalysisCatalogInputProfiles(
    binding,
    {[key]: {...native, mode: 'text', text: '{'}},
    context.get
  );
  expect(text.problems.length).toBe(1);
  const sampleBinding = getBinding('geometry-measures');
  const sample = resolveAnalysisCatalogInputProfiles(sampleBinding, {}, context.get);
  expect(sample.profiles[sampleBinding.inputs[0].name]?.rowCount).toBeGreaterThan(0);
  expect(readColumns).not.toHaveBeenCalled();
});

const catalogHostRequire = createRequire(
  new URL('../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
const catalogReact = catalogHostRequire('react') as typeof import('react') & {
  __CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE: {H: unknown};
};

function findCatalogElements(node: ReactNode): ReactElement<Record<string, unknown>>[] {
  if (Array.isArray(node)) return node.flatMap(findCatalogElements);
  if (!catalogReact.isValidElement<Record<string, unknown>>(node)) return [];
  return [node, ...findCatalogElements(node.props['children'] as ReactNode)];
}

function createCatalogPanelProbe(props: Parameters<typeof AnalysisCatalogPanel>[0]) {
  const states: unknown[] = [];
  const references: {current: unknown}[] = [];
  return () => {
    let stateIndex = 0;
    let referenceIndex = 0;
    const internals = catalogReact.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
    const previous = internals.H;
    internals.H = {
      useState: (initial: unknown) => {
        const index = stateIndex++;
        if (!(index in states)) states[index] = typeof initial === 'function' ? initial() : initial;
        return [
          states[index],
          (next: unknown) => {
            states[index] = typeof next === 'function' ? next(states[index]) : next;
          }
        ];
      },
      useRef: (initial: unknown) => {
        const index = referenceIndex++;
        references[index] ??= {current: initial};
        return references[index];
      },
      useMemo: (factory: () => unknown) => factory(),
      useEffect: () => {},
      useSyncExternalStore: (_subscribe: unknown, getSnapshot: () => unknown) => getSnapshot()
    };
    try {
      return findCatalogElements(AnalysisCatalogPanel(props));
    } finally {
      internals.H = previous;
    }
  };
}

function changeCatalogSelect(
  elements: ReactElement<Record<string, unknown>>[],
  name: string,
  value: string
) {
  const control = elements.find(
    element => element.props['id'] === name || element.props['aria-label'] === name
  )!;
  (control.props['onChange'] as (event: {target: {value: string}}) => void)({target: {value}});
}

test('sample topology checks unlock Run and expire on settings, engine and source edits', async () => {
  const render = createCatalogPanelProbe({datasets: [], onPublish: vi.fn()});
  let elements = render();
  changeCatalogSelect(elements, 'catalog-operation', 'network-service-area');
  elements = render();
  const setParameter = (name: string, value: unknown) => {
    const parameter = elements.find(
      element => (element.props['spec'] as {name?: string} | undefined)?.name === name
    )!;
    (parameter.props['onChange'] as (value: unknown) => void)(value);
    elements = render();
  };
  const expectDisabled = (disabled: boolean) =>
    expect(
      elements.find(element => element.props['data-testid'] === 'catalog-run')?.props['disabled']
    ).toBe(disabled);
  const checkInputs = async () => {
    await (
      elements.find(element => element.props['data-testid'] === 'catalog-check-inputs')!.props[
        'onClick'
      ] as () => Promise<void>
    )();
    elements = render();
    expect(
      elements.find(element => element.props['data-testid'] === 'catalog-error')
    ).toBeUndefined();
  };
  setParameter('coordinateSystem', 'planar');
  expectDisabled(true);
  await checkInputs();
  expectDisabled(false);
  setParameter('splitAtSharedVertices', false);
  expectDisabled(true);
  await checkInputs();
  expectDisabled(false);
  changeCatalogSelect(elements, 'catalog-engine', 'cpu');
  elements = render();
  expectDisabled(true);
  await checkInputs();
  expectDisabled(false);
  changeCatalogSelect(elements, 'Network lines source', 'text');
  elements = render();
  expectDisabled(true);
});

test('panel shows sources, unknown evidence and remedies before Run and preserves edits through profile updates', async () => {
  const readColumns = vi.fn(async ({column}: {column: string}) => ({
    values: column === 'east' ? [1, 2, 3] : [4, 5, 6],
    fieldType: 'real'
  }));
  const dataset: CatalogResidentDataset = {
    id: 'panel-points',
    label: 'Panel points',
    revision: 1,
    rowCount: 3,
    coordinates: {longitude: 'east', latitude: 'north'},
    fieldTypes: {east: 'real', north: 'real'},
    provider: {
      fields: [
        {name: 'east', sortable: true},
        {name: 'north', sortable: true}
      ],
      maximumPageSize: 10,
      getRevision: () => '1',
      readPage: vi.fn(),
      readColumnValues: readColumns
    },
    isCurrent: () => true
  };
  analysisDatasetContext.synchronize([dataset]);
  try {
    const render = createCatalogPanelProbe({
      datasets: [],
      residentDatasets: [dataset],
      onPublish: vi.fn()
    });
    let elements = render();
    changeCatalogSelect(elements, 'catalog-operation', 'idw-grid');
    elements = render();
    changeCatalogSelect(elements, 'Points source', 'native');
    elements = render();
    expect(
      elements.find(element => element.props['data-testid'] === 'catalog-run')?.props['disabled']
    ).toBe(true);
    expect(elements.some(element => element.props['data-state'] === 'unknown')).toBe(true);
    expect(elements.some(element => element.props['data-testid'] === 'catalog-preflight')).toBe(
      true
    );
    changeCatalogSelect(elements, 'Points dataset', dataset.id);
    elements = render();
    expect(readColumns).not.toHaveBeenCalled();
    expect(
      elements.find(element => element.props['aria-label'] === 'Points X / longitude column')
        ?.props['value']
    ).toBe('east');
    const cellSetting = elements.find(
      element => (element.props['spec'] as {name?: string} | undefined)?.name === 'cellSize'
    )!;
    (cellSetting.props['onChange'] as (value: unknown) => void)('15');
    await analysisDatasetContext.readStatistics(dataset.id);
    elements = render();
    const editedCellSetting = elements.find(
      element => (element.props['spec'] as {name?: string} | undefined)?.name === 'cellSize'
    )!;
    expect(editedCellSetting.props['value']).toBe('15');
    expect(editedCellSetting.props['source']).toContain('override');
    expect(
      elements.find(element => element.props['data-testid'] === 'catalog-input-capacity')?.props[
        'children'
      ]
    ).toContain(3);
    changeCatalogSelect(elements, 'Points X / longitude column', 'north');
    elements = render();
    expect(
      elements.find(element => element.props['data-testid'] === 'catalog-run')?.props['disabled']
    ).toBe(true);
    expect(
      elements.find(element => element.props['data-testid'] === 'catalog-check-inputs')
    ).toBeDefined();
  } finally {
    analysisDatasetContext.synchronize([]);
  }
});

test('panel checks native GeoJSON explicitly and invalidates checked profiles on source replacement', async () => {
  const source = {
    id: 'native-polygons',
    label: 'Native polygons',
    fields: [{name: 'shape', type: 'geojson'}],
    dataContainer: {
      numRows: () => 1,
      valueAt: vi.fn(() =>
        JSON.stringify({
          type: 'Polygon',
          coordinates: [
            [
              [0, 0],
              [2, 0],
              [2, 2],
              [0, 0]
            ]
          ]
        })
      )
    }
  };
  const props = {
    datasets: [source as unknown as Parameters<typeof AnalysisCatalogPanel>[0]['datasets'][number]],
    onPublish: vi.fn()
  };
  const render = createCatalogPanelProbe(props);
  let elements = render();
  changeCatalogSelect(elements, 'catalog-operation', 'geometry-measures');
  elements = render();
  changeCatalogSelect(elements, 'Features source', 'native');
  elements = render();
  changeCatalogSelect(elements, 'Features dataset', source.id);
  elements = render();
  changeCatalogSelect(elements, 'Features geometry field', 'shape');
  elements = render();
  const coordinateSetting = elements.find(
    element => (element.props['spec'] as {name?: string} | undefined)?.name === 'coordinateSystem'
  )!;
  (coordinateSetting.props['onChange'] as (value: unknown) => void)('planar');
  elements = render();
  expect(
    elements.find(element => element.props['data-testid'] === 'catalog-run')?.props['disabled']
  ).toBe(true);
  await (
    elements.find(element => element.props['data-testid'] === 'catalog-check-inputs')!.props[
      'onClick'
    ] as () => Promise<void>
  )();
  elements = render();
  expect(
    elements.find(element => element.props['data-testid'] === 'catalog-run')?.props['disabled'],
    JSON.stringify(
      elements
        .filter(
          element =>
            ['catalog-error', 'catalog-preflight-status'].includes(
              String(element.props['data-testid'])
            ) || element.props['data-state']
        )
        .map(element => element.props)
    )
  ).toBe(false);
  props.datasets = [
    {...source, dataContainer: {...source.dataContainer}} as unknown as Parameters<
      typeof AnalysisCatalogPanel
    >[0]['datasets'][number]
  ];
  elements = render();
  expect(
    elements.find(element => element.props['data-testid'] === 'catalog-run')?.props['disabled']
  ).toBe(true);
  const pending = (
    elements.find(element => element.props['data-testid'] === 'catalog-check-inputs')!.props[
      'onClick'
    ] as () => Promise<void>
  )();
  changeCatalogSelect(elements, 'Features geometry field', '');
  render();
  await pending;
  elements = render();
  expect(
    elements.find(element => element.props['data-testid'] === 'catalog-run')?.props['disabled']
  ).toBe(true);
  expect(
    elements.find(element => element.props['data-testid'] === 'catalog-error')?.props['children']
  ).toContain('Inputs changed during checking');
});

test('unknown row meaning leaves Run enabled and one click records the native declaration', () => {
  const dataset: CatalogResidentDataset = {
    id: 'row-meaning',
    label: 'Unknown observations',
    revision: 1,
    rowCount: 6,
    coordinates: {longitude: 'east', latitude: 'north'},
    fieldTypes: {east: 'real', north: 'real'},
    provider: {
      fields: [
        {name: 'east', sortable: false},
        {name: 'north', sortable: false}
      ],
      maximumPageSize: 16,
      getRevision: () => '1',
      readPage: vi.fn(),
      readColumnValues: vi.fn()
    },
    isCurrent: () => true
  };
  analysisDatasetContext.synchronize([dataset]);
  try {
    const render = createCatalogPanelProbe({
      datasets: [],
      residentDatasets: [dataset],
      onPublish: vi.fn()
    });
    let elements = render();
    changeCatalogSelect(elements, 'catalog-operation', 'kernel-density-grid');
    elements = render();
    changeCatalogSelect(elements, 'Points source', 'native');
    elements = render();
    changeCatalogSelect(elements, 'Points dataset', dataset.id);
    elements = render();
    // Metadata intentionally has no extent yet. Choose settings so this test isolates row
    // meaning rather than the separate unresolved data-derived defaults.
    for (const [name, value] of Object.entries({
      coordinateSystem: 'planar',
      cellSize: 1,
      bandwidth: 2
    })) {
      const parameter = elements.find(
        element => (element.props['spec'] as {name?: string} | undefined)?.name === name
      )!;
      (parameter.props['onChange'] as (value: unknown) => void)(value);
      elements = render();
    }
    expect(
      elements.find(element => element.props['data-testid'] === 'catalog-run')?.props['disabled']
    ).toBe(false);
    const question = elements.find(element => element.type === AnalysisRowMeaning)!;
    const questionProps = question.props as Parameters<typeof AnalysisRowMeaning>[0];
    expect(questionProps.profile.rowSemantics?.source).not.toBe('declared');
    const buttons = findCatalogElements(AnalysisRowMeaning(questionProps));
    const events = buttons.find(
      element => element.type === 'button' && element.props['children'] === 'Events'
    )!;
    expect(events.props['disabled']).toBeUndefined();
    (events.props['onClick'] as () => void)();
    expect(analysisDatasetContext.getDeclarations(dataset.id)?.rowSemantics).toBe('events');
    elements = render();
    expect(
      elements.find(element => element.type === AnalysisRowMeaning)?.props['profile']
    ).toMatchObject({
      rowSemantics: {value: 'events', source: 'declared'}
    });
    expect(
      elements.find(element => element.props['data-testid'] === 'catalog-run')?.props['disabled']
    ).toBe(false);
  } finally {
    analysisDatasetContext.synchronize([]);
  }
});
