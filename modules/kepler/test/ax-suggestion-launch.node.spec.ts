// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterEach, expect, test, vi} from 'vitest';
import {createRequire} from 'node:module';
import type {ReactElement, ReactNode} from 'react';
import {AnalysisSuggestionsPanel} from '../../../examples/deck/kepler-compute/src/analysis-suggestions-panel';
import {AnalysisCatalogPanel} from '../../../examples/deck/kepler-compute/src/analysis-catalog-panel';
import * as arrow from 'apache-arrow';
import {
  getCatalogBinding,
  profileKeplerColumnTable,
  type KeplerDatasetProfile,
  type KeplerSuggestionQuestion
} from '@luma-contributors/kepler';
import {
  createSandboxSample,
  SANDBOX_SAMPLES,
  type SandboxSampleId
} from '../../../examples/deck/kepler-compute/src/sandbox-samples';
import {
  answerAnalysisSuggestionQuestion,
  browseAnalysisSuggestions,
  createAnalysisSuggestionLaunch,
  createAnalysisSuggestionModel
} from '../../../examples/deck/kepler-compute/src/analysis-suggestion-model';
import {
  analysisDatasetContext,
  createAnalysisDatasetContext,
  type AnalysisDatasetContextEntry
} from '../../../examples/deck/kepler-compute/src/analysis-dataset-context';
import {
  makeCatalogResidentDatasets,
  type CatalogResidentDataset
} from '../../../examples/deck/kepler-compute/src/analysis-catalog-resident-input';
import {
  createAnalysisCatalogFormModel,
  resolveAnalysisCatalogInputProfiles
} from '../../../examples/deck/kepler-compute/src/analysis-catalog-form-model';
import {readCatalogPointColumns} from '../../../examples/deck/kepler-compute/src/analysis-catalog-input';

/** Real gallery tables, retaining unsupported native types rather than inventing capabilities. */
function makeGalleryDataset(sampleId: SandboxSampleId): CatalogResidentDataset {
  const sample = createSandboxSample(sampleId);
  const geometry =
    sample.geometryColumn && !sample.tripOptions?.grouping
      ? JSON.parse(String(sample.table.getChild(sample.geometryColumn)!.get(0)))
      : undefined;
  return {
    id: sampleId,
    label: sample.description.title,
    revision: 1,
    rowCount: sample.table.numRows,
    coordinates:
      sample.coordinates ??
      (sample.tripOptions?.grouping
        ? {
            longitude: sample.tripOptions.grouping.longitudeColumn,
            latitude: sample.tripOptions.grouping.latitudeColumn
          }
        : undefined),
    geometryColumn: sample.tripOptions?.grouping ? undefined : sample.geometryColumn,
    geometryKind: sample.tripOptions?.grouping
      ? 'point'
      : geometry
        ? geometry.type.includes('Polygon')
          ? 'polygon'
          : 'line'
        : undefined,
    fieldTypes: Object.fromEntries(
      sample.table.schema.fields.map(field => [
        field.name,
        arrow.DataType.isTimestamp(field.type)
          ? 'timestamp'
          : arrow.DataType.isFloat(field.type)
            ? 'real'
            : arrow.DataType.isInt(field.type)
              ? 'integer'
              : 'string'
      ])
    ),
    provider: {
      fields: sample.table.schema.fields.map(field => ({name: field.name, sortable: true})),
      maximumPageSize: 1000,
      getRevision: () => '1',
      readPage: vi.fn(),
      readColumnValues: vi.fn(async ({column}) => ({
        values: [...sample.table.getChild(column)!],
        fieldType: 'real'
      }))
    },
    isCurrent: () => true
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
  analysisDatasetContext.synchronize([]);
  analysisDatasetContext.restoreDeclarations(undefined);
});

function createProfileLaunch(
  target: Parameters<typeof createAnalysisSuggestionLaunch>[0],
  profiles: readonly KeplerDatasetProfile[]
) {
  const entries = new Map<string, AnalysisDatasetContextEntry>(
    profiles.map(profile => [
      profile.datasetId,
      {
        profile,
        dataset: makeGalleryDataset(profile.datasetId as SandboxSampleId),
        stage: 'statistics',
        key: 'fixture-snapshot',
        readRevision: '1'
      }
    ])
  );
  return createAnalysisSuggestionLaunch(target, profiles, datasetId => {
    const shared = analysisDatasetContext.get(datasetId);
    return shared?.profile === profiles.find(profile => profile.datasetId === datasetId)
      ? shared
      : entries.get(datasetId);
  });
}

const hostRequire = createRequire(
  new URL('../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
const hostReact = hostRequire('react') as typeof import('react') & {
  __CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE: {H: unknown};
};

function findElements(node: ReactNode): ReactElement<Record<string, unknown>>[] {
  if (Array.isArray(node)) return node.flatMap(findElements);
  if (!hostReact.isValidElement<Record<string, unknown>>(node)) return [];
  return [node, ...findElements(node.props['children'] as ReactNode)];
}

/** Exercise real handlers and the launch effect without a browser or a parallel form model. */
function createPanelProbe(renderPanel: () => ReactNode, launchRequest?: unknown) {
  const states: unknown[] = [];
  const references: {current: unknown}[] = [];
  return () => {
    let stateIndex = 0;
    let referenceIndex = 0;
    const effects: (() => void)[] = [];
    const internals = hostReact.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
    const previous = internals.H;
    internals.H = {
      useState(initial: unknown) {
        const index = stateIndex++;
        if (!(index in states)) states[index] = typeof initial === 'function' ? initial() : initial;
        return [
          states[index],
          (next: unknown) => {
            states[index] = typeof next === 'function' ? next(states[index]) : next;
          }
        ];
      },
      useRef(initial: unknown) {
        const index = referenceIndex++;
        references[index] ??= {current: initial};
        return references[index];
      },
      useMemo: (factory: () => unknown) => factory(),
      useEffect: (effect: () => void, dependencies?: unknown[]) => {
        if (launchRequest && dependencies?.[0] === launchRequest) effects.push(effect);
      },
      useSyncExternalStore: (_subscribe: unknown, getSnapshot: () => unknown) => getSnapshot()
    };
    try {
      const elements = findElements(renderPanel());
      for (const effect of effects) effect();
      return elements;
    } finally {
      internals.H = previous;
    }
  };
}

test('actual card renders disabled remedies, editable role chips and one-click catalog handoff', () => {
  const dataset = makeGalleryDataset('points-time');
  analysisDatasetContext.synchronize([dataset]);
  const onLaunch = vi.fn();
  const render = createPanelProbe(() =>
    AnalysisSuggestionsPanel({datasetId: dataset.id, datasets: [dataset], onLaunch})
  );
  let elements = render();
  expect(elements[0].props['data-testid']).toBe('ax-suggestions');
  expect(elements.some(element => element.type === 'button' && element.props['disabled'])).toBe(
    true
  );
  const button = elements.find(element => element.props['data-suggestion-id'])!;
  (button.props['onClick'] as () => void)();
  expect(onLaunch).toHaveBeenCalledOnce();
  expect(onLaunch.mock.calls[0][0].bindingId).toBe(button.props['data-binding-id']);
  const role = elements.find(
    element => element.props['aria-label'] === `Role of reading for ${dataset.label}`
  )!;
  (role.props['onChange'] as (event: {target: {value: string}}) => void)({target: {value: 'rate'}});
  expect(analysisDatasetContext.getDeclarations(dataset.id)?.measurement?.['reading']).toBe('rate');
  elements = render();
  const details = elements.find(element => element.type === 'details')!;
  (details.props['onToggle'] as (event: {currentTarget: {open: boolean}}) => void)({
    currentTarget: {open: true}
  });
  elements = render();
  expect(elements.some(element => element.type === 'h4')).toBe(true);
});

test('catalog consumes the real request exactly once, retains later edits, and rejects an obsolete revision', () => {
  const dataset = makeGalleryDataset('points-time');
  analysisDatasetContext.synchronize([dataset]);
  const profile = analysisDatasetContext.get(dataset.id)!.profile;
  const target = createAnalysisSuggestionModel(dataset.id, [profile]).suggestions[0];
  const request = createProfileLaunch(target, [profile]);
  vi.stubGlobal('requestAnimationFrame', vi.fn());
  const props = {
    datasets: [],
    residentDatasets: [dataset],
    launchRequest: request,
    onPublish: vi.fn()
  };
  const render = createPanelProbe(() => AnalysisCatalogPanel(props), request);
  render();
  let elements = render();
  expect(
    elements.find(element => element.props['id'] === 'catalog-operation')?.props['value']
  ).toBe(request.bindingId);
  const binding = getCatalogBinding(request.bindingId)!;
  for (const input of binding.inputs) {
    expect(
      elements.find(element => element.props['aria-label'] === `${input.label} source`)?.props[
        'value'
      ]
    ).toBe('native');
    expect(
      elements.find(element => element.props['aria-label'] === `${input.label} dataset`)?.props[
        'value'
      ]
    ).toBe(dataset.id);
    expect(
      elements.find(
        element => element.props['aria-label'] === `${input.label} X / longitude column`
      )?.props['value']
    ).toBe('longitude');
  }
  const operation = elements.find(element => element.props['id'] === 'catalog-operation')!;
  (operation.props['onChange'] as (event: {target: {value: string}}) => void)({
    target: {value: 'geometry-measures'}
  });
  elements = render();
  expect(
    elements.find(element => element.props['id'] === 'catalog-operation')?.props['value']
  ).toBe('geometry-measures');
  analysisDatasetContext.synchronize([{...dataset, revision: 2}]);
  const obsolete = createPanelProbe(() => AnalysisCatalogPanel(props), request);
  obsolete();
  expect(
    obsolete().some(
      element =>
        element.props['role'] === 'alert' && String(element.props['children']).includes('changed')
    )
  ).toBe(true);
});

for (const sample of SANDBOX_SAMPLES) {
  test(`${sample.id}: gallery metadata suggests deterministic launches with exact input and field roles`, () => {
    const dataset = makeGalleryDataset(sample.id);
    const context = createAnalysisDatasetContext();
    context.synchronize([dataset]);
    const profile = context.get(dataset.id)!.profile;
    const model = createAnalysisSuggestionModel(dataset.id, [profile]);
    expect(model).toEqual(createAnalysisSuggestionModel(dataset.id, [profile]));
    if (sample.id === 'h3-cells') {
      // The host exposes original H3 IDs, without a GeoJSON footprint reader. AX must not
      // promise a runnable geometry analysis or silently replace the cells with sample data.
      expect(profile.kind).toBe('attributes');
      expect(model.suggestions).toEqual([]);
      return;
    }
    expect(model.suggestions.length).toBeGreaterThan(0);
    expect(model.suggestions.length).toBeLessThanOrEqual(8);
    for (const suggestion of model.suggestions) {
      const request = createProfileLaunch(suggestion, [profile]);
      const binding = getCatalogBinding(request.bindingId)!;
      expect(suggestion.reasons.length).toBeGreaterThan(0);
      expect(suggestion.status).not.toBe('blocked');
      for (const input of binding.inputs) {
        const draft = request.drafts[`${binding.id}:${input.name}`];
        expect(draft.mode).toBe('native');
        expect(draft.datasetId).toBe(suggestion.applicability.inputAssignments[input.name] ?? '');
        expect(draft.text).toBe('');
        if (draft.datasetId && profile.pointColumns)
          expect(readCatalogPointColumns(draft.field)).toEqual([
            profile.pointColumns.longitude,
            profile.pointColumns.latitude
          ]);
      }
      for (const [parameter, columns] of Object.entries(suggestion.applicability.fieldBindings))
        expect(request.overrides[parameter]).toEqual(
          typeof columns === 'string' ? columns : columns.join(',')
        );
      expect(request.revisions).toEqual({[dataset.id]: 1});
    }
    expect(dataset.provider.readColumnValues).not.toHaveBeenCalled();
    expect(dataset.provider.readPage).not.toHaveBeenCalled();
  });
}

function profileGallery(sampleId: SandboxSampleId): KeplerDatasetProfile {
  const sample = createSandboxSample(sampleId);
  return profileKeplerColumnTable(
    {
      rowCount: sample.table.numRows,
      columns: sample.table.schema.fields.map(field => ({
        name: field.name,
        values: [...sample.table.getChild(field.name)!],
        ...(arrow.DataType.isTimestamp(field.type) ? {type: 'time' as const} : {})
      }))
    },
    {
      datasetId: sampleId,
      label: sample.description.title,
      revision: 1,
      pointColumns: sample.coordinates,
      geometryColumns: sample.geometryColumn ? [sample.geometryColumn] : []
    }
  );
}

test('points and polygons retain both datasets and exact mapping into the existing catalog form', () => {
  const profiles = [profileGallery('points-time'), profileGallery('districts')];
  const model = createAnalysisSuggestionModel('points-time', profiles);
  const paired = model.suggestions.find(suggestion => new Set(suggestion.datasetIds).size === 2);
  expect(paired).toBeDefined();
  const request = createProfileLaunch(paired!, profiles);
  expect(
    Object.values(request.drafts)
      .map(draft => draft.datasetId)
      .sort()
  ).toEqual(['districts', 'points-time']);
  const binding = getCatalogBinding(request.bindingId)!;
  const entries = new Map(
    profiles.map(profile => [
      profile.datasetId,
      {
        profile,
        dataset: makeGalleryDataset(profile.datasetId as SandboxSampleId),
        stage: 'statistics' as const,
        key: '1',
        readRevision: '1'
      }
    ])
  );
  const inputs = resolveAnalysisCatalogInputProfiles(binding, request.drafts, identifier =>
    entries.get(identifier)
  );
  expect(inputs.problems).toEqual([]);
  const form = createAnalysisCatalogFormModel(binding, inputs.profiles, {
    overrides: request.overrides
  });
  expect(form.applicability.inputAssignments).toEqual(paired!.applicability.inputAssignments);
  expect(form.applicability.fieldBindings).toEqual(paired!.applicability.fieldBindings);
  expect(Object.values(request.drafts).some(draft => draft.mode === 'sample')).toBe(false);
});

test('blocked suggestions retain reasons and remedies and cannot open; browse contains only fitting assignments', () => {
  const profile = profileGallery('points-time');
  const model = createAnalysisSuggestionModel(profile.datasetId, [profile]);
  expect(model.blocked.length).toBeGreaterThan(0);
  for (const suggestion of model.blocked) {
    expect(suggestion.applicability.missing.every(item => item.what && item.remedy)).toBe(true);
    expect(() =>
      createProfileLaunch(
        {...suggestion, applicability: {...suggestion.applicability, status: 'blocked'}},
        [profile]
      )
    ).toThrow('unblocked');
  }
  const groups = browseAnalysisSuggestions(profile.datasetId, [profile]);
  expect(groups.length).toBeGreaterThan(0);
  for (const group of groups)
    for (const candidate of group.bindings) {
      expect(candidate.applicability.status).not.toBe('blocked');
      expect(Object.values(candidate.applicability.inputAssignments)).toContain(profile.datasetId);
      expect(createProfileLaunch(candidate, [profile]).bindingId).toBe(candidate.bindingId);
    }
});

test('question answers persist in the shared host context while preserving the other declarations', () => {
  const dataset = makeGalleryDataset('points-time');
  analysisDatasetContext.synchronize([dataset]);
  const questions: KeplerSuggestionQuestion[] = [
    {
      id: 'row-semantics',
      datasetId: dataset.id,
      prompt: 'What are the rows?',
      options: ['events', 'objects', 'field-samples'],
      affects: []
    },
    {
      id: 'coordinate-model',
      datasetId: dataset.id,
      prompt: 'Coordinates?',
      options: ['geographic', 'planar'],
      affects: []
    },
    {
      id: 'measurement',
      datasetId: dataset.id,
      column: 'reading',
      prompt: 'Measures?',
      options: ['count', 'amount', 'rate'],
      affects: []
    }
  ];
  for (const [index, answer] of ['objects', 'geographic', 'rate'].entries()) {
    const question = questions[index];
    analysisDatasetContext.setDeclarations(
      dataset.id,
      answerAnalysisSuggestionQuestion(
        question,
        answer,
        analysisDatasetContext.getDeclarations(dataset.id)
      )
    );
  }
  const saved = analysisDatasetContext.collectDeclarations();
  analysisDatasetContext.restoreDeclarations(undefined);
  analysisDatasetContext.restoreDeclarations(saved);
  expect(analysisDatasetContext.getDeclarations(dataset.id)).toEqual({
    rowSemantics: 'objects',
    coordinateModel: 'geographic',
    measurement: {reading: 'rate'}
  });
  expect(
    analysisDatasetContext
      .get(dataset.id)!
      .profile.columns.find(column => column.name === 'reading')
  ).toMatchObject({measurement: 'rate', measurementSource: 'declared'});
  expect(() => answerAnalysisSuggestionQuestion(questions[0], 'arbitrary')).toThrow(
    'question options'
  );
});

test('shared resident adapter exposes grouped trip observation coordinates instead of the trip ID as geometry', () => {
  const sample = createSandboxSample('trip-observations');
  const dataset = {id: 'grouped', revision: 1, value: {source: sample.table}};
  const primary = {id: 'primary', revision: 1, value: {source: sample.table}};
  const publication = {
    dataset,
    label: 'Grouped',
    geometryColumn: sample.geometryColumn,
    tripOptions: sample.tripOptions,
    coordinates: undefined,
    consumer: {table: makeGalleryDataset('trip-observations').provider, explorer: {isCurrent: true}}
  };
  const runtime = {
    explorer: {dataset: primary, isCurrent: true},
    nativeTable: publication.consumer.table,
    publications: new Map([[dataset.id, publication]])
  } as unknown as Parameters<typeof makeCatalogResidentDatasets>[0];
  const resident = makeCatalogResidentDatasets(runtime, () => runtime)[1];
  expect(resident).toMatchObject({
    geometryKind: 'point',
    coordinates: {longitude: 'longitude', latitude: 'latitude'}
  });
  expect(resident.geometryColumn).toBeUndefined();
  const context = createAnalysisDatasetContext();
  context.synchronize([resident]);
  const profile = context.get(resident.id)!.profile;
  expect(profile.kind).toBe('points');
  expect(createAnalysisSuggestionModel(resident.id, [profile]).suggestions.length).toBeGreaterThan(
    0
  );
});

test('finding 1: every gallery dataset is ranked within its own card before the eight-slot limit', () => {
  const identifiers: SandboxSampleId[] = [
    'points-time',
    'districts',
    'connections',
    'trips',
    'numeric-grid',
    'directed-network'
  ];
  const profiles = identifiers.map(profileGallery);
  for (const profile of profiles) {
    const alone = createAnalysisSuggestionModel(profile.datasetId, [profile]);
    const together = createAnalysisSuggestionModel(profile.datasetId, profiles);
    expect(together.suggestions.length, profile.datasetId).toBeGreaterThan(0);
    expect(together.suggestions.length).toBeLessThanOrEqual(8);
    expect(
      together.suggestions.every(suggestion => suggestion.datasetIds.includes(profile.datasetId))
    ).toBe(true);
    expect(together).toEqual(
      createAnalysisSuggestionModel(profile.datasetId, [...profiles].reverse())
    );
    if (profile.datasetId === 'trips')
      expect(together.suggestions.length).toBe(alone.suggestions.length);
  }
});

test('finding 2: identical point profiles keep their own single-input browse operations and partners', () => {
  const original = profileGallery('points-time');
  const first = {...original, datasetId: 'A', label: 'A'};
  const second = {...original, datasetId: 'B', label: 'B'};
  const alone = browseAnalysisSuggestions('B', [second]).flatMap(group => group.bindings);
  const together = browseAnalysisSuggestions('B', [first, second]).flatMap(group => group.bindings);
  expect(alone.length).toBeGreaterThan(0);
  expect(together.map(candidate => candidate.bindingId)).toEqual(
    expect.arrayContaining(alone.map(candidate => candidate.bindingId))
  );
  expect(together.some(candidate => candidate.bindingId === 'cell-aggregation')).toBe(true);
  for (const candidate of together) {
    expect(Object.values(candidate.applicability.inputAssignments)).toContain('B');
    for (const alternative of candidate.applicability.inputAssignmentAlternatives)
      expect(Object.values(alternative)).toContain('B');
  }
  const comparison = together.find(candidate => candidate.bindingId === 'cell-compare')!;
  expect(new Set(Object.values(comparison.applicability.inputAssignments))).toEqual(
    new Set(['A', 'B'])
  );
  // Identical layers also retain ranked suggestions on B's card despite global twin suppression.
  expect(createAnalysisSuggestionModel('B', [first, second]).suggestions.length).toBeGreaterThan(0);
});

test('findings 1 and 2: the active dataset survives per-input search caps and excludes unrelated assignments', () => {
  const original = profileGallery('points-time');
  const profiles = Array.from({length: 8}, (_, index) => ({
    ...original,
    datasetId: `layer-${index}`,
    label: `Layer ${index}`
  }));
  const activeDatasetId = profiles[7].datasetId;
  const candidates = browseAnalysisSuggestions(activeDatasetId, profiles).flatMap(
    group => group.bindings
  );
  const paired = candidates.find(candidate => candidate.bindingId === 'cell-compare')!;
  expect(paired).toBeDefined();
  expect(Object.values(paired.applicability.inputAssignments)).toContain(activeDatasetId);
  expect(new Set(Object.values(paired.applicability.inputAssignments)).size).toBe(2);
  expect(
    createAnalysisSuggestionModel(activeDatasetId, profiles).suggestions.length
  ).toBeGreaterThan(0);
  expect(browseAnalysisSuggestions('absent', profiles)).toEqual([]);
  expect(createAnalysisSuggestionModel('absent', profiles).suggestions).toEqual([]);
});

test.each([
  'owner',
  'provider',
  'provider-revision'
])('finding 3: stale %s snapshots never seed the replacement catalog', replacement => {
  const dataset = makeGalleryDataset('points-time');
  dataset.owner = {};
  let providerRevision = '1';
  dataset.provider.getRevision = () => providerRevision;
  analysisDatasetContext.synchronize([dataset]);
  const profile = analysisDatasetContext.get(dataset.id)!.profile;
  const target = createAnalysisSuggestionModel(dataset.id, [profile]).suggestions[0];
  const request = createAnalysisSuggestionLaunch(target, [profile]);
  const next =
    replacement === 'owner'
      ? {...dataset, owner: {}}
      : replacement === 'provider'
        ? {...dataset, provider: {...dataset.provider}}
        : dataset;
  if (replacement === 'provider-revision') providerRevision = '2';
  analysisDatasetContext.synchronize([next]);
  expect(analysisDatasetContext.get(dataset.id)!.profile.revision).toBe(profile.revision);
  vi.stubGlobal('requestAnimationFrame', vi.fn());
  const consumed = vi.fn();
  const props = {
    datasets: [],
    residentDatasets: [next],
    launchRequest: request,
    onLaunchConsumed: consumed,
    onPublish: vi.fn()
  };
  const render = createPanelProbe(() => AnalysisCatalogPanel(props), request);
  render();
  const elements = render();
  expect(
    elements.some(
      element =>
        element.props['role'] === 'alert' && String(element.props['children']).includes('changed')
    )
  ).toBe(true);
  expect(
    elements.some(
      element =>
        element.props['aria-label'] === 'Points dataset' && element.props['value'] === dataset.id
    )
  ).toBe(false);
  expect(requestAnimationFrame).not.toHaveBeenCalled();
  expect(consumed).toHaveBeenCalledExactlyOnceWith(request);
});

test('finding 4: acknowledged host requests do not reapply mappings or focus after a sidebar remount', () => {
  const dataset = makeGalleryDataset('points-time');
  analysisDatasetContext.synchronize([dataset]);
  const profile = analysisDatasetContext.get(dataset.id)!.profile;
  const target = createAnalysisSuggestionModel(dataset.id, [profile]).suggestions[0];
  const request = createAnalysisSuggestionLaunch(target, [profile]);
  let hostRequest: typeof request | undefined = request;
  const consumed = vi.fn((launch: typeof request) => {
    if (hostRequest === launch) hostRequest = undefined;
  });
  vi.stubGlobal('requestAnimationFrame', vi.fn());
  const renderPanel = () =>
    AnalysisCatalogPanel({
      datasets: [],
      residentDatasets: [dataset],
      launchRequest: hostRequest,
      onLaunchConsumed: consumed,
      onPublish: vi.fn()
    });
  const firstMount = createPanelProbe(renderPanel, request);
  firstMount();
  firstMount();
  expect(hostRequest).toBeUndefined();
  expect(requestAnimationFrame).toHaveBeenCalledOnce();
  const remount = createPanelProbe(renderPanel, request);
  remount();
  const elements = remount();
  expect(consumed).toHaveBeenCalledOnce();
  expect(requestAnimationFrame).toHaveBeenCalledOnce();
  expect(
    elements.some(
      element =>
        element.props['aria-label']?.toString().endsWith(' dataset') &&
        element.props['value'] === dataset.id
    )
  ).toBe(false);
});
