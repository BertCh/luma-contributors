// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import type {ReactElement, ReactNode} from 'react';
import {expect, test, vi} from 'vitest';
import * as arrow from 'apache-arrow';
import {NullDevice} from '@luma.gl/core/test-utils';
import {
  parseKeplerKrigingSettings,
  parseKeplerIDWSettings,
  restoreKeplerAnalysisRecipe,
  type KeplerIDWSettings,
  type KeplerKrigingSettings
} from '../src/index';
import {KeplerComputeSession, type KeplerSessionExplorer} from '../src/session/compute-session';
import {AnalysisRuntime} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import {getAnalysisFormInitialization} from '../../../examples/deck/kepler-compute/src/analysis-form-initialization';
import {createAnalysisDatasetContext} from '../../../examples/deck/kepler-compute/src/analysis-dataset-context';
import {ResidentAnalysisSheetForm} from '../../../examples/deck/kepler-compute/src/analysis-sheet';
import {WorkbenchParameterValue} from '../../../examples/deck/kepler-compute/src/workflow/workbench-parameter-form';
import {
  InterpolationPanel,
  getIDWSheetSettings,
  makeInterpolationSheetDefaults
} from '../../../examples/deck/kepler-compute/src/interpolation-panel';
import {
  KrigingPanel,
  getKrigingSheetSettings,
  makeKrigingSheetDefaults
} from '../../../examples/deck/kepler-compute/src/analysis-kriging-panel';
import type {KrigingVariogramSuggestion} from '../../../examples/deck/kepler-compute/src/kriging-diagnostics';

const hostRequire = createRequire(
  new URL('../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
const react = hostRequire('react') as typeof import('react') & {
  __CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE: {H: unknown};
};
const {renderToStaticMarkup} = hostRequire('react-dom/server') as typeof import('react-dom/server');
type Element = ReactElement<Record<string, unknown>>;
function elements(node: ReactNode): Element[] {
  if (Array.isArray(node)) return node.flatMap(elements);
  return react.isValidElement<Record<string, unknown>>(node)
    ? [node, ...elements(node.props['children'] as ReactNode)]
    : [];
}

// Exercise real callbacks and hook effects without claiming browser or GPU timing evidence.
function createProbe<Properties>(component: (properties: Properties) => React.JSX.Element) {
  const states: unknown[] = [];
  const references: {current: unknown}[] = [];
  const dependencies = new Map<number, readonly unknown[]>();
  const cleanups = new Map<number, () => void>();
  return {
    render(properties: Properties) {
      let stateIndex = 0;
      let referenceIndex = 0;
      let effectIndex = 0;
      const effects: (() => void)[] = [];
      const internals = react.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
      const previous = internals.H;
      internals.H = {
        useState(initial: unknown) {
          const index = stateIndex++;
          if (!(index in states))
            states[index] = typeof initial === 'function' ? initial() : initial;
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
        useId: () => 'generated-control',
        useSyncExternalStore: (_subscribe: unknown, snapshot: () => unknown) => snapshot(),
        useEffect(effect: () => void | (() => void), next: readonly unknown[]) {
          const index = effectIndex++;
          const previousDependencies = dependencies.get(index);
          if (
            previousDependencies &&
            previousDependencies.length === next.length &&
            next.every((value, position) => Object.is(value, previousDependencies[position]))
          )
            return;
          dependencies.set(index, next);
          effects.push(() => {
            cleanups.get(index)?.();
            const cleanup = effect();
            if (cleanup) cleanups.set(index, cleanup);
            else cleanups.delete(index);
          });
        }
      };
      try {
        const tree = component(properties);
        for (const effect of effects) effect();
        return elements(tree);
      } finally {
        internals.H = previous;
      }
    },
    unmount() {
      for (const cleanup of cleanups.values()) cleanup();
    }
  };
}

const COLUMNS = ['east', 'north', 'reading', 'other', 'otherEast', 'otherNorth'];
const BOUNDS = [-122.5, 37.7, -122.3, 37.9] as const;
const IDW: KeplerIDWSettings = parseKeplerIDWSettings({
  longitudeColumn: 'east',
  latitudeColumn: 'north',
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters',
  grid: {bounds: BOUNDS, width: 4, height: 3},
  searchRadius: 12000,
  power: 3,
  neighborCount: 4,
  minimumNeighborCount: 2,
  taper: 'modified-shepard'
});
const KRIGING = parseKeplerKrigingSettings({
  longitudeColumn: 'east',
  latitudeColumn: 'north',
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters',
  grid: {bounds: BOUNDS, width: 4, height: 3},
  searchRadiusMeters: 12000,
  neighborCount: 4,
  minimumNeighbors: 2,
  variogram: {model: 'matern-3/2', nugget: 0.1, sill: 2.3, rangeMeters: 8000},
  confidenceLevel: 0.95,
  crossValidation: true,
  variogramFit: {
    method: 'weighted-least-squares',
    weighting: 'cressie',
    binCount: 10,
    pairCount: 15,
    maximumLagMeters: 9000,
    criterion: 0.7
  }
});
function makeRuntime(saved = false) {
  const context = createAnalysisDatasetContext();
  context.synchronize([
    {
      id: 'interpolation-sheet',
      label: 'Measured samples',
      rowCount: 6,
      revision: 1,
      coordinates: {longitude: 'east', latitude: 'north'},
      valueColumn: 'reading',
      bounds: BOUNDS,
      fieldTypes: Object.fromEntries(COLUMNS.map(name => [name, 'real'])),
      provider: {
        fields: COLUMNS.map(name => ({name, sortable: true})),
        getRevision: () => '1',
        readColumnValues: vi.fn(),
        readPage: vi.fn(),
        maximumPageSize: 16
      },
      isCurrent: () => true
    }
  ]);
  // Real AnalysisRuntime.settings returns new parsed objects on every read.
  return {
    datasetId: 'interpolation-sheet',
    coordinateRestorationKey: 'interpolation-sheet:0',
    rowCount: 6,
    availableColumns: COLUMNS,
    get settings() {
      return {
        column: 'reading',
        ...(saved ? {idw: {...IDW}, kriging: parseKeplerKrigingSettings(KRIGING)} : {})
      };
    },
    formInitialization: getAnalysisFormInitialization(context.get('interpolation-sheet'), COLUMNS),
    configureInterpolation: vi.fn(),
    configureKriging: vi.fn(),
    suggestKrigingVariogram: vi.fn()
  } as unknown as AnalysisRuntime;
}
type SheetProperties = Parameters<typeof ResidentAnalysisSheetForm>[0];
function sheetProperties(nodes: Element[]) {
  return nodes.find(element => element.type === ResidentAnalysisSheetForm)!
    .props as SheetProperties;
}
function field(sheet: SheetProperties, name: string) {
  return sheet.fields.find(candidate => candidate.specification.name === name)!;
}
function submit(nodes: Element[]) {
  const form = nodes.find(element => element.type === 'form')!;
  (form.props['onSubmit'] as (event: {preventDefault(): void}) => void)({preventDefault: vi.fn()});
}
function fittingButton(nodes: Element[]) {
  const sheet = sheetProperties(nodes);
  const action = sheet.actions![0]!;
  return {
    props: {
      disabled: action.disabled || sheet.checks.length > 0,
      onClick: () => action.onInvoke({})
    }
  };
}
function suggestion(
  method: NonNullable<KeplerKrigingSettings['variogramFit']>['method']
): KrigingVariogramSuggestion {
  const settings = parseKeplerKrigingSettings({
    ...KRIGING,
    variogramFit: {
      ...KRIGING.variogramFit!,
      method,
      weighting: method === 'weighted-least-squares' ? 'cressie' : 'conditional-likelihood'
    }
  });
  return {
    settings,
    validRowCount: 6,
    accumulatedPairCount: 15,
    exhaustive: true,
    fit: {
      ...settings.variogram,
      ...settings.variogramFit!,
      candidates: [],
      diagnostics: {converged: 1}
    }
  };
}

test.each([
  InterpolationPanel,
  KrigingPanel
])('%s generates typed roles, data bounds and manual rebuild controls', Panel => {
  const analysis = makeRuntime();
  const markup = renderToStaticMarkup(react.createElement(Panel, {analysis}));
  expect(markup).toContain('data-testid="ax-analysis-sheet"');
  expect(markup).toContain('Manual Apply');
  expect(markup).toContain('All settings');
  expect(markup).toContain('Catalog');
  expect(markup).toContain('Dataset extent');
  const probe = createProbe(() => Panel({analysis}));
  try {
    const sheet = sheetProperties(probe.render({}));
    expect(sheet.checks).toEqual([]);
    expect(field(sheet, 'longitudeColumn')).toMatchObject({
      value: 'east',
      columns: COLUMNS,
      source: 'Publication coordinate roles',
      specification: {kind: 'column', fieldRole: {required: true}}
    });
    expect(field(sheet, 'west').value).toBe(String(BOUNDS[0]));
    expect(field(sheet, 'north').value).toBe(String(BOUNDS[3]));
    expect(field(sheet, 'width').value).toBe(32);
    sheet.onChange('west', -122.6);
    const edited = sheetProperties(probe.render({}));
    expect(field(edited, 'west').source).toBe('User override');
    expect(analysis.configureInterpolation).not.toHaveBeenCalled();
    expect(analysis.configureKriging).not.toHaveBeenCalled();
  } finally {
    probe.unmount();
  }
});

test.each([
  InterpolationPanel,
  KrigingPanel
])('%s retains saved values across rerenders and resets invalid controls on replay', Panel => {
  const analysis = makeRuntime(true);
  const probe = createProbe(() => Panel({analysis}));
  try {
    let nodes = probe.render({});
    let sheet = sheetProperties(nodes);
    expect(sheet.checks).toEqual([]);
    expect(field(sheet, 'west').source).toBe('Saved resident settings');
    expect(field(sheet, 'width').value).toBe(4);
    sheet.onChange('width', 7);
    sheet = sheetProperties(probe.render({}));
    expect(field(sheet, 'width').value).toBe(7);
    sheet.onApply({width: 7});
    if (Panel === InterpolationPanel)
      expect(analysis.configureInterpolation).toHaveBeenCalledWith({
        ...IDW,
        grid: {...IDW.grid, width: 7}
      });
    else
      expect(analysis.configureKriging).toHaveBeenCalledWith({
        ...KRIGING,
        grid: {...KRIGING.grid, width: 7}
      });
    const key = nodes.find(element => element.type === ResidentAnalysisSheetForm)!.key;
    Object.assign(analysis, {coordinateRestorationKey: 'interpolation-sheet:1'});
    nodes = probe.render({});
    sheet = sheetProperties(nodes);
    expect(field(sheet, 'width').value).toBe(4);
    expect(nodes.find(element => element.type === ResidentAnalysisSheetForm)!.key).not.toBe(key);
  } finally {
    probe.unmount();
  }
});

test.each([
  InterpolationPanel,
  KrigingPanel
])('%s preserves cleared fields during profile refinement and blocks unsupported declarations', Panel => {
  const analysis = makeRuntime();
  const probe = createProbe(() => Panel({analysis}));
  try {
    let sheet = sheetProperties(probe.render({}));
    sheet.onChange('longitudeColumn', '');
    sheet.onChange('width', 9);
    Object.assign(analysis, {
      formInitialization: {
        ...analysis.formInitialization,
        interpolation: {
          ...analysis.formInitialization.interpolation,
          longitudeColumn: 'otherEast',
          east: '-122.2'
        }
      }
    });
    sheet = sheetProperties(probe.render({}));
    expect(field(sheet, 'longitudeColumn').value).toBe('');
    expect(field(sheet, 'east').value).toBe('-122.2');
    expect(field(sheet, 'width').value).toBe(9);
    expect(sheet.checks.join(' ')).toContain('choose an available prepared numeric column');
    sheet.onChange('longitudeColumn', 'east');
    sheet.onChange('neighborCount', 1);
    sheet = sheetProperties(probe.render({}));
    expect(sheet.checks.length).toBeGreaterThan(0); // minimum samples exceeds selected slots
    sheet.onChange('neighborCount', 8);
    sheet.onChange('east', 179.99);
    sheet.onChange('west', 179);
    expect(sheetProperties(probe.render({})).checks.length).toBeGreaterThan(0);
  } finally {
    probe.unmount();
  }
});

test('shared generated form blocks transient errors and preserves the latest committed field on immediate Apply', () => {
  const analysis = makeRuntime();
  const panel = createProbe(() => InterpolationPanel({analysis}));
  const form = createProbe((properties: SheetProperties) => ResidentAnalysisSheetForm(properties));
  try {
    const properties = sheetProperties(panel.render({}));
    const nodes = form.render(properties);
    const control = nodes.find(
      element =>
        element.type === WorkbenchParameterValue &&
        element.props['controlId'] === 'interpolation-power'
    )!;
    const callbacks = control.props as Parameters<typeof WorkbenchParameterValue>[0];
    callbacks.onValidityChange!('must be a number');
    submit(nodes);
    expect(analysis.configureInterpolation).not.toHaveBeenCalled();
    callbacks.onValidityChange!(null);
    callbacks.onCommit(4);
    submit(nodes);
    expect(analysis.configureInterpolation).toHaveBeenCalledWith(
      expect.objectContaining({power: 4})
    );
  } finally {
    form.unmount();
    panel.unmount();
  }
});

test('all resident interpolation methods round trip through the runtime, strict recipe parser and replay', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'interpolation-replay',
    table: arrow.tableFromArrays(
      Object.fromEntries(COLUMNS.map(name => [name, new Float32Array(6).fill(1)]))
    )
  });
  const explorer = {
    dataset,
    isCurrent: true,
    currentRevision: 0
  } as unknown as KeplerSessionExplorer;
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate: () => {}});
  try {
    analysis.setColumn('reading');
    for (const taper of ['none', 'modified-shepard'] as const) {
      const expected = {...IDW, taper};
      analysis.configureInterpolation(
        getIDWSheetSettings({
          ...makeInterpolationSheetDefaults(expected, analysis.formInitialization),
          searchRadius: expected.searchRadius,
          power: expected.power,
          neighborCount: expected.neighborCount,
          minimumNeighborCount: expected.minimumNeighborCount,
          taper: expected.taper
        })
      );
      const text = analysis.serializeRecipe();
      const recipe = restoreKeplerAnalysisRecipe(text, {
        resolveDatasetRevision: () => dataset.revision
      });
      expect(recipe.operation).toEqual({kind: 'idw-points', column: 'reading', ...expected});
      analysis.setMode('off');
      analysis.restoreRecipe(text);
      expect(analysis.settings.idw).toEqual(expected);
    }
    for (const model of [
      'spherical',
      'exponential',
      'gaussian',
      'matern-3/2',
      'matern-5/2'
    ] as const) {
      for (const method of [
        'weighted-least-squares',
        'vecchia-likelihood',
        'vecchia-fisher'
      ] as const) {
        const fitted = suggestion(method).settings;
        const expected = {...fitted, variogram: {...fitted.variogram, model}};
        const settings = getKrigingSheetSettings(
          makeKrigingSheetDefaults(expected, analysis.formInitialization),
          expected
        );
        expect(settings).toEqual(expected);
        analysis.configureKriging(settings);
        const text = analysis.serializeRecipe();
        const recipe = restoreKeplerAnalysisRecipe(text, {
          resolveDatasetRevision: () => dataset.revision
        });
        expect(recipe.operation).toEqual({kind: 'kriging-points', column: 'reading', ...expected});
        analysis.setMode('off');
        analysis.restoreRecipe(text);
        expect(analysis.settings.kriging).toEqual(expected);
      }
    }
  } finally {
    await analysis.destroy(Promise.resolve());
    await session.destroy(Promise.resolve());
    device.destroy();
  }
});

test('manual variogram or coordinate edits discard fit provenance; intervals and cross-validation can be cleared', () => {
  const defaults = makeKrigingSheetDefaults(KRIGING, makeRuntime().formInitialization);
  expect(getKrigingSheetSettings(defaults, KRIGING)).toEqual(KRIGING);
  expect(getKrigingSheetSettings({...defaults, sill: 4}, KRIGING).variogramFit).toBeUndefined();
  expect(
    getKrigingSheetSettings({...defaults, longitudeColumn: 'otherEast'}, KRIGING).variogramFit
  ).toBeUndefined();
  const cleared = getKrigingSheetSettings(
    {...defaults, confidenceLevel: '', crossValidation: false},
    KRIGING
  );
  expect(cleared.confidenceLevel).toBeUndefined();
  expect(cleared.crossValidation).toBeUndefined();
  expect(() => getKrigingSheetSettings({...defaults, nugget: ''})).toThrow(
    'Declare interpolation nugget'
  );
});

test.each([
  'weighted-least-squares',
  'vecchia-likelihood',
  'vecchia-fisher'
] as const)('%s fit updates draft and diagnostics, survives parsed-settings rerenders and applies replayable provenance', async method => {
  const analysis = makeRuntime(true);
  vi.mocked(analysis.suggestKrigingVariogram).mockResolvedValue(suggestion(method));
  const probe = createProbe(() => KrigingPanel({analysis}));
  try {
    let sheet = sheetProperties(probe.render({}));
    sheet.onChange('fitMethod', method);
    sheet.onChange('maximumLagMeters', '9000');
    let nodes = probe.render({});
    expect(fittingButton(nodes).props['disabled']).toBe(false);
    await (fittingButton(nodes).props['onClick'] as () => Promise<void>)();
    expect(analysis.suggestKrigingVariogram).toHaveBeenCalledWith(KRIGING, {
      fitMethod: method,
      maximumLagMeters: 9000
    });
    expect(analysis.configureKriging).not.toHaveBeenCalled();
    nodes = probe.render({});
    nodes = probe.render({}); // Settings object identity must not reset completed fits.
    expect(
      nodes.some(
        element =>
          element.type === 'table' && element.props['aria-label'] === 'Variogram fit diagnostics'
      )
    ).toBe(true);
    sheet = sheetProperties(nodes);
    expect(field(sheet, 'sill').source).toContain('Fitted variogram');
    sheet.onApply({});
    expect(analysis.configureKriging).toHaveBeenCalledWith(suggestion(method).settings);
  } finally {
    probe.unmount();
  }
});

test.each([
  'edit',
  'transient-edit',
  'restore',
  'unmount'
])('a fit completed after %s cannot overwrite the current sheet', async change => {
  const analysis = makeRuntime(true);
  let finish!: (value: KrigingVariogramSuggestion) => void;
  vi.mocked(analysis.suggestKrigingVariogram).mockReturnValue(
    new Promise(resolve => {
      finish = resolve;
    })
  );
  const probe = createProbe(() => KrigingPanel({analysis}));
  const form = createProbe((properties: SheetProperties) => ResidentAnalysisSheetForm(properties));
  const control = createProbe(WorkbenchParameterValue);
  function renderSill(nodes: Element[]) {
    const fields = form.render(sheetProperties(nodes));
    const sill = fields.find(
      element =>
        element.type === WorkbenchParameterValue && element.props['controlId'] === 'kriging-sill'
    )!;
    return control
      .render(sill.props as Parameters<typeof WorkbenchParameterValue>[0])
      .find(element => element.type === 'input')!;
  }
  try {
    let nodes = probe.render({});
    const pending = (fittingButton(nodes).props['onClick'] as () => Promise<void>)();
    if (change === 'edit') sheetProperties(nodes).onChange('sill', 8);
    else if (change === 'transient-edit') {
      const input = renderSill(nodes);
      (input.props['onFocus'] as () => void)();
      (input.props['onChange'] as (event: {target: {value: string}}) => void)({
        target: {value: '8'}
      });
      // No blur: the accepted scalar is unchanged while the user's text stays local.
      expect(field(sheetProperties(probe.render({})), 'sill').value).toBe(KRIGING.variogram.sill);
    } else if (change === 'restore')
      Object.assign(analysis, {coordinateRestorationKey: 'interpolation-sheet:1'});
    else probe.unmount();
    if (change !== 'unmount') probe.render({});
    const completed = suggestion('vecchia-fisher');
    finish(
      change === 'transient-edit'
        ? {
            ...completed,
            settings: {
              ...completed.settings,
              variogram: {...completed.settings.variogram, sill: 2}
            },
            fit: {...completed.fit, sill: 2}
          }
        : completed
    );
    await pending;
    if (change !== 'unmount') {
      nodes = probe.render({});
      expect(field(sheetProperties(nodes), 'sill').value).toBe(
        change === 'edit' ? 8 : KRIGING.variogram.sill
      );
      expect(nodes.some(element => element.type === 'table')).toBe(false);
      if (change === 'transient-edit') {
        // Render again after the control's external-value effect has had a chance to run.
        renderSill(nodes);
        const input = renderSill(nodes);
        expect(input.props['value']).toBe('8');
        (input.props['onBlur'] as () => void)();
        expect(field(sheetProperties(probe.render({})), 'sill').value).toBe(8);
      }
      expect(fittingButton(nodes).props['disabled']).toBe(false);
    }
    expect(analysis.configureKriging).not.toHaveBeenCalled();
  } finally {
    control.unmount();
    form.unmount();
    probe.unmount();
  }
});

test('fitting shares transient validation and receives the latest committed settings', async () => {
  const analysis = makeRuntime(true);
  const panel = createProbe(() => KrigingPanel({analysis}));
  const form = createProbe((properties: SheetProperties) => ResidentAnalysisSheetForm(properties));
  const onInvoke = vi.fn();
  try {
    const properties = {
      ...sheetProperties(panel.render({})),
      actions: [{label: 'Fit variogram automatically', onInvoke}]
    };
    const nodes = form.render(properties);
    const control = nodes.find(
      element =>
        element.type === WorkbenchParameterValue &&
        element.props['controlId'] === 'kriging-rangeMeters'
    )!;
    const callbacks = control.props as Parameters<typeof WorkbenchParameterValue>[0];
    const action = nodes.find(
      element =>
        element.type === 'button' && element.props['children'] === 'Fit variogram automatically'
    )!;
    callbacks.onValidityChange!('must be a number');
    await (action.props['onClick'] as () => Promise<void>)();
    expect(onInvoke).not.toHaveBeenCalled();
    callbacks.onValidityChange!(null);
    callbacks.onCommit(7000);
    await (action.props['onClick'] as () => Promise<void>)();
    expect(onInvoke).toHaveBeenCalledWith(expect.objectContaining({rangeMeters: 7000}));
  } finally {
    form.unmount();
    panel.unmount();
  }
});

test.each([
  InterpolationPanel,
  KrigingPanel
])('%s blocks absent geographic metadata and unsupported source counts before Apply', Panel => {
  const analysis = makeRuntime();
  Object.assign(analysis, {
    formInitialization: getAnalysisFormInitialization(undefined, ['reading'])
  });
  const probe = createProbe(() => Panel({analysis}));
  try {
    const sheet = sheetProperties(probe.render({}));
    expect(field(sheet, 'west').value).toBe('');
    expect(sheet.checks.join(' ')).toContain('choose an available prepared numeric column');
    expect(sheet.checks.join(' ')).toContain('Declare interpolation west');
    Object.assign(analysis, {rowCount: 2 ** 24});
    expect(sheetProperties(probe.render({})).checks.join(' ')).toContain(
      'source rows must be from 1'
    );
  } finally {
    probe.unmount();
  }
});

test('a fitted response is invalidated on field changes and unchanged recipe replay preserves its provenance', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'fitted-response',
    table: arrow.tableFromArrays(
      Object.fromEntries(COLUMNS.map(name => [name, new Float32Array(6).fill(1)]))
    )
  });
  const explorer = {
    dataset,
    isCurrent: true,
    currentRevision: 0
  } as unknown as KeplerSessionExplorer;
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate: () => {}});
  const probe = createProbe(() => KrigingPanel({analysis}));
  try {
    analysis.setColumn('reading');
    const {variogramFit, ...manual} = KRIGING;
    analysis.configureKriging(manual);
    vi.spyOn(analysis, 'suggestKrigingVariogram').mockResolvedValue(
      suggestion('weighted-least-squares')
    );
    let nodes = probe.render({});
    await (fittingButton(nodes).props['onClick'] as () => Promise<void>)();
    sheetProperties(probe.render({})).onApply({});
    const saved = analysis.serializeRecipe();
    expect(JSON.parse(saved).operation.variogramFit).toEqual(variogramFit);

    const beforeResponseChange = sheetProperties(probe.render({}));
    analysis.setColumn('other');
    expect(analysis.settings.kriging?.variogramFit).toBeUndefined();
    // An event captured before React rerenders cannot reattach the old response's provenance.
    beforeResponseChange.onApply({});
    expect(analysis.settings.kriging?.variogramFit).toBeUndefined();
    nodes = probe.render({});
    sheetProperties(nodes).onApply({});
    expect(JSON.parse(analysis.serializeRecipe()).operation).toMatchObject({column: 'other'});
    expect(JSON.parse(analysis.serializeRecipe()).operation.variogramFit).toBeUndefined();
    analysis.setColumn('reading');
    expect(analysis.settings.kriging?.variogramFit).toBeUndefined();

    // Replaying the recipe restores both the response and the fit, including on another field.
    analysis.setColumn('other');
    analysis.restoreRecipe(saved);
    nodes = probe.render({});
    sheetProperties(nodes).onApply({});
    expect(JSON.parse(analysis.serializeRecipe()).operation).toEqual(JSON.parse(saved).operation);
    expect(analysis.settings.kriging?.variogramFit).toEqual(variogramFit);

    // Invalidation also works while the sheet is closed, before its next mount or Apply.
    probe.unmount();
    analysis.setColumn('other');
    expect(analysis.settings.kriging?.variogramFit).toBeUndefined();
  } finally {
    probe.unmount();
    await analysis.destroy(Promise.resolve());
    await session.destroy(Promise.resolve());
    device.destroy();
  }
});
