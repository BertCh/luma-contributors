// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import type {ReactElement, ReactNode} from 'react';
import {expect, test, vi} from 'vitest';
import * as arrow from 'apache-arrow';
import {NullDevice} from '@luma.gl/core/test-utils';
import {
  restoreKeplerAnalysisRecipe,
  serializeKeplerAnalysisRecipe,
  getWorkflowParameterProblem,
  type KeplerAnalysisRecipe
} from '../src/index';
import {
  createDBSCANSheetFields,
  DBSCANPanel,
  getDBSCANSheetChecks,
  getDBSCANSheetSettings
} from '../../../examples/deck/kepler-compute/src/analysis-dbscan-panel';
import {ResidentAnalysisSheetForm} from '../../../examples/deck/kepler-compute/src/analysis-sheet';
import {getAnalysisFormInitialization} from '../../../examples/deck/kepler-compute/src/analysis-form-initialization';
import {getAnalysisRecipeSettings} from '../../../examples/deck/kepler-compute/src/analysis-recipe-settings';
import {AnalysisRuntime} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import {KeplerComputeSession, type KeplerSessionExplorer} from '../src/session/compute-session';
import {
  PointWeightsPanel,
  makePointSheetDefaults,
  usePointAnalysisSheet,
  getPointWeightsSheetSettings,
  type PointSheetValues
} from '../../../examples/deck/kepler-compute/src/analysis-point-weights-panel';
import {
  PointHotspotPanel,
  getPointHotspotSheetSettings
} from '../../../examples/deck/kepler-compute/src/analysis-point-hotspot-panel';
import {
  LocalMoranPanel,
  getLocalMoranSheetSettings
} from '../../../examples/deck/kepler-compute/src/analysis-local-moran-panel';
import {
  QuantileLisaPanel,
  getQuantileLisaSheetSettings
} from '../../../examples/deck/kepler-compute/src/analysis-quantile-lisa-panel';
import {normalizePointWeightsSettings} from '../../../examples/deck/kepler-compute/src/analysis-point-weights-settings';
import {WorkbenchParameterValue} from '../../../examples/deck/kepler-compute/src/workflow/workbench-parameter-form';

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

const COLUMNS = ['east', 'north', 'otherEast', 'otherNorth'];
const VALUES = {longitudeColumn: 'east', latitudeColumn: 'north', epsilon: 30, minimumSamples: 3};
function makeRuntime() {
  return {
    datasetId: 'resident-points',
    coordinateRestorationKey: 'resident-points:0',
    rowCount: 5,
    availableColumns: COLUMNS,
    settings: {},
    summary: null,
    formInitialization: getAnalysisFormInitialization(undefined, COLUMNS, undefined, 5, {
      longitude: 'east',
      latitude: 'north'
    }),
    configureDBSCAN: vi.fn()
  } as unknown as AnalysisRuntime;
}
function submit(nodes: Element[]) {
  const form = nodes.find(element => element.type === 'form')!;
  (form.props['onSubmit'] as (event: {preventDefault(): void}) => void)({preventDefault: vi.fn()});
}
function makeProperties() {
  return {
    fields: createDBSCANSheetFields(VALUES, COLUMNS, () => 'Saved resident DBSCAN settings'),
    checks: getDBSCANSheetChecks(VALUES, COLUMNS, 5),
    method: 'Great-circle OPTICS extraction',
    submitLabel: 'Apply radius and run DBSCAN',
    onChange: vi.fn(),
    onApply: vi.fn()
  };
}

test('DBSCAN settings and serialized recipes round trip through generated fields without a value column', () => {
  const settings = getDBSCANSheetSettings({...VALUES, epsilon: 30.123456789});
  const recipe: KeplerAnalysisRecipe = {
    schemaVersion: 1,
    algorithmVersion: 1,
    id: 'point-sheet',
    source: {datasetId: 'resident-points', revision: 8},
    population: 'selected',
    selectionRevision: 2,
    backendPolicy: 'require-gpu',
    operation: {kind: 'dbscan-points', ...settings}
  };
  const restored = restoreKeplerAnalysisRecipe(serializeKeplerAnalysisRecipe(recipe), {
    resolveDatasetRevision: () => 8,
    resolveSelectionRevision: () => 2
  });
  const restoredSettings = getAnalysisRecipeSettings(restored).dbscan!;
  const fields = createDBSCANSheetFields(restoredSettings, COLUMNS, () => 'Saved settings');
  const values = Object.fromEntries(fields.map(field => [field.specification.name, field.value]));
  expect(getDBSCANSheetSettings(values)).toEqual(settings);
  expect(settings.epsilon).toBe(Math.fround(30.123456789));
  expect(restored).toEqual(recipe);
  expect(restored.operation).not.toHaveProperty('column');
  expect(
    fields.filter(field => field.specification.fieldRole).map(field => field.specification.name)
  ).toEqual(['longitudeColumn', 'latitudeColumn']);
});

test('resident limits and Float32 rules fail before Apply, with remedies', () => {
  expect(getDBSCANSheetChecks(VALUES, COLUMNS, 5)).toEqual([]);
  for (const rowCount of [0, 4097, NaN, 1.5])
    expect(getDBSCANSheetChecks(VALUES, COLUMNS, rowCount).join(' ')).toContain(
      'Choose a smaller source dataset'
    );
  for (const epsilon of [0, -1, Number.MIN_VALUE, Infinity, NaN, 1e40])
    expect(getDBSCANSheetChecks({...VALUES, epsilon}, COLUMNS, 5).join(' ')).toContain(
      'positive Float32 radius'
    );
  for (const minimumSamples of [0, 257, 1.5])
    expect(getDBSCANSheetChecks({...VALUES, minimumSamples}, COLUMNS, 5).length).toBeGreaterThan(0);
  for (const name of ['longitudeColumn', 'latitudeColumn']) {
    expect(getDBSCANSheetChecks({...VALUES, [name]: ''}, COLUMNS, 5).join(' ')).toContain(
      'choose an available prepared numeric coordinate column'
    );
    expect(getDBSCANSheetChecks({...VALUES, [name]: 'missing'}, COLUMNS, 5).join(' ')).toContain(
      'choose an available prepared numeric coordinate column'
    );
  }
  // A large minimum is legal even when it makes every observation noise.
  expect(getDBSCANSheetChecks({...VALUES, minimumSamples: 256}, COLUMNS, 5)).toEqual([]);
});

test('DBSCAN sheet preserves published roles, defaults, stable selectors, method label and bespoke summary', () => {
  const analysis = makeRuntime();
  Object.assign(analysis, {
    summary: {
      dbscan: {clusterCount: 1, noiseCount: 2, validCount: 5, excludedCount: 0, complete: true}
    } as AnalysisRuntime['summary']
  });
  const markup = renderToStaticMarkup(react.createElement(DBSCANPanel, {analysis}));
  expect(markup).toContain('data-testid="ax-analysis-sheet"');
  expect(markup).toContain('id="analysis-dbscan-longitude"');
  expect(markup).toContain('value="east" selected=""');
  expect(markup).toContain('value="north" selected=""');
  expect(markup).toContain('id="analysis-dbscan-epsilon"');
  expect(markup).toContain('value="1000"');
  expect(markup).toContain('value="5"');
  expect(markup).toContain('Publication coordinate roles');
  expect(markup).toContain('no k-distance knee has been computed');
  expect(markup).toContain('Catalog density clustering uses a local planar frame');
  expect(markup).toContain('Manual Apply');
  expect(markup).toContain('DBSCAN: 1 clusters · 2 noise points');
  expect(analysis.configureDBSCAN).not.toHaveBeenCalled();
});

test('richer role metadata preserves user edits and explicit clears; saved replacements and scope changes restore every setting', () => {
  const analysis = makeRuntime();
  const child = elements(DBSCANPanel({analysis})).find(
    element => typeof element.type === 'function'
  )!;
  type Properties = {
    analysis: AnalysisRuntime;
    settings?: ReturnType<typeof getDBSCANSheetSettings>;
  };
  const probe = createProbe(child.type as (properties: Properties) => React.JSX.Element);
  function form(nodes: Element[]) {
    return nodes.find(element => element.type === ResidentAnalysisSheetForm)!.props as Parameters<
      typeof ResidentAnalysisSheetForm
    >[0];
  }
  try {
    let sheet = form(probe.render({analysis}));
    sheet.onChange('epsilon', 123);
    sheet.onChange('longitudeColumn', '');
    Object.assign(analysis, {
      formInitialization: {
        ...analysis.formInitialization,
        interpolation: {
          ...analysis.formInitialization.interpolation,
          longitudeColumn: 'otherEast',
          latitudeColumn: 'otherNorth'
        }
      }
    });
    sheet = form(probe.render({analysis}));
    const read = (name: string) => sheet.fields.find(field => field.specification.name === name)!;
    expect(read('longitudeColumn').value).toBe('');
    expect(read('longitudeColumn').source).toBe('User override');
    expect(read('latitudeColumn').value).toBe('otherNorth');
    expect(read('epsilon').value).toBe(123);
    expect(analysis.configureDBSCAN).not.toHaveBeenCalled();
    const settings = getDBSCANSheetSettings({...VALUES, epsilon: 321, minimumSamples: 4});
    sheet = form(probe.render({analysis, settings}));
    expect(read('epsilon').value).toBe(321);
    expect(read('longitudeColumn').value).toBe('east');
    sheet.onApply(
      Object.fromEntries(sheet.fields.map(field => [field.specification.name, field.value]))
    );
    expect(analysis.configureDBSCAN).toHaveBeenCalledExactlyOnceWith(settings);
    Object.assign(analysis, {coordinateRestorationKey: 'resident-points:1'});
    sheet = form(probe.render({analysis}));
    expect(read('epsilon').value).toBe(1000);
    expect(read('minimumSamples').value).toBe(5);
    expect(read('longitudeColumn').value).toBe('otherEast');
  } finally {
    probe.unmount();
  }
});

test('invalid transient text blocks previously captured Apply and valid rebuild text commits only on release', () => {
  const properties = makeProperties();
  const sheetProbe = createProbe(ResidentAnalysisSheetForm);
  const controlProbe = createProbe(WorkbenchParameterValue);
  try {
    const sheet = sheetProbe.render(properties);
    const parameter = sheet.find(
      element =>
        element.type === WorkbenchParameterValue &&
        (element.props['spec'] as {name: string}).name === 'epsilon'
    )!;
    const controlProperties = parameter.props as Parameters<typeof WorkbenchParameterValue>[0];
    let nodes = controlProbe.render(controlProperties);
    const input = () => nodes.find(element => element.props['id'] === 'analysis-dbscan-epsilon')!;
    const change = (value: string) =>
      (input().props['onChange'] as (event: {target: {value: string}}) => void)({target: {value}});
    change('-1');
    // No sheet rerender: a captured submit handler still sees the invalid transient text.
    submit(sheet);
    expect(properties.onApply).not.toHaveBeenCalled();
    expect(properties.onChange).not.toHaveBeenCalled();
    nodes = controlProbe.render(controlProperties);
    expect(input().props['aria-invalid']).toBe(true);
    change('31.25');
    nodes = controlProbe.render(controlProperties);
    expect(properties.onChange).not.toHaveBeenCalled();
    (input().props['onBlur'] as () => void)();
    expect(properties.onChange).toHaveBeenCalledExactlyOnceWith('epsilon', 31.25);
    expect(properties.onApply).not.toHaveBeenCalled();
    submit(sheet);
    expect(properties.onApply).toHaveBeenCalledExactlyOnceWith({...VALUES, epsilon: 31.25});
  } finally {
    controlProbe.unmount();
    sheetProbe.unmount();
  }
});

test('preflight blocks invalid source assignments even through a direct form submit', () => {
  const properties = makeProperties();
  properties.checks = getDBSCANSheetChecks(VALUES, ['east'], 4097);
  const probe = createProbe(ResidentAnalysisSheetForm);
  try {
    const nodes = probe.render(properties);
    expect(nodes.find(element => element.type === 'button')!.props['disabled']).toBe(true);
    submit(nodes);
    expect(properties.onApply).not.toHaveBeenCalled();
  } finally {
    probe.unmount();
  }
});

// Finding 1: a restored recipe must not acquire a narrower identifier contract in the form.
test.each([
  65, 1024
])('restored DBSCAN coordinate names of length %i remain applicable in the generated sheet', length => {
  const longitudeColumn = `longitude_${'x'.repeat(length - 10)}`;
  const latitudeColumn = `latitude_${'y'.repeat(length - 9)}`;
  const settings = getDBSCANSheetSettings({...VALUES, longitudeColumn, latitudeColumn});
  const recipe: KeplerAnalysisRecipe = {
    schemaVersion: 1,
    algorithmVersion: 1,
    id: 'long-roles',
    source: {datasetId: 'resident-points', revision: 1},
    population: 'all',
    backendPolicy: 'require-gpu',
    operation: {kind: 'dbscan-points', ...settings}
  };
  const restored = restoreKeplerAnalysisRecipe(serializeKeplerAnalysisRecipe(recipe), {
    resolveDatasetRevision: () => 1
  });
  const analysis = makeRuntime();
  Object.assign(analysis, {
    availableColumns: [longitudeColumn, latitudeColumn],
    settings: getAnalysisRecipeSettings(restored)
  });
  const markup = renderToStaticMarkup(react.createElement(DBSCANPanel, {analysis}));
  expect(markup).not.toContain('aria-invalid="true"');
  expect(markup).toContain('<button type="submit">Apply radius and run DBSCAN</button>');
  const fields = createDBSCANSheetFields(
    settings,
    [longitudeColumn, latitudeColumn],
    () => 'Restored recipe'
  );
  for (const field of fields)
    expect(getWorkflowParameterProblem(field.specification, field.value)).toBeNull();
  expect(() =>
    getDBSCANSheetSettings({
      ...VALUES,
      longitudeColumn: `${longitudeColumn}${'x'.repeat(1025 - length)}`
    })
  ).toThrow('Invalid recipe identifier');
});

const POINT_COLUMNS = [
  'east',
  'north',
  'x',
  'y',
  'reading',
  'population',
  'second',
  'value,with,commas'
];
function makePointRuntime() {
  const analysis = makeRuntime();
  Object.assign(analysis, {
    rowCount: 12,
    availableColumns: POINT_COLUMNS,
    settings: {column: 'reading', denominatorColumn: 'population'},
    configurePointWeights: vi.fn(),
    configurePointHotspots: vi.fn(),
    configureLocalMoran: vi.fn(),
    configureQuantileLisa: vi.fn()
  });
  return analysis;
}
function sheetProperties(nodes: Element[]) {
  return nodes.find(element => element.type === ResidentAnalysisSheetForm)!.props as Parameters<
    typeof ResidentAnalysisSheetForm
  >[0];
}
function sheetValues(sheet: Parameters<typeof ResidentAnalysisSheetForm>[0]) {
  return Object.fromEntries(sheet.fields.map(field => [field.specification.name, field.value]));
}

// Finding 2: each of the four remaining entry points must render generated controls.
test('all five point panels mount the shared analysis sheet and retain their coordinate selectors', () => {
  const analysis = makePointRuntime();
  const properties = {
    columns: POINT_COLUMNS,
    rowCount: 12,
    coordinateInitialization: analysis.formInitialization.coordinates,
    onApply: vi.fn()
  };
  const panels = [
    react.createElement(DBSCANPanel, {analysis}),
    react.createElement(PointWeightsPanel, {analysis}),
    react.createElement(PointHotspotPanel, {analysis}),
    react.createElement(LocalMoranPanel, properties),
    react.createElement(QuantileLisaPanel, properties)
  ];
  for (const panel of panels) {
    const markup = renderToStaticMarkup(panel);
    expect(markup.match(/data-testid="ax-analysis-sheet"/g)).toHaveLength(1);
    expect(markup).toContain('data-ax-parameter="longitudeColumn"');
    expect(markup).toContain('data-ax-parameter="latitudeColumn"');
    expect(markup).toContain('value="east" selected=""');
    expect(markup).toContain('value="north" selected=""');
    expect(markup).toContain('Manual Apply');
  }
});

test('point weights keep a population edit in the sheet until one atomic Apply', () => {
  const analysis = makePointRuntime();
  const probe = createProbe(PointWeightsPanel);
  try {
    let sheet = sheetProperties(probe.render({analysis}));
    expect(sheet.checks).toEqual([]);
    expect(sheetValues(sheet)).toMatchObject({scale: 1, islandPolicy: 'undefined', k: 8});
    sheet.onChange('denominatorColumn', 'second');
    sheet.onChange('scale', 1000);
    expect(analysis.configurePointWeights).not.toHaveBeenCalled();
    sheet = sheetProperties(probe.render({analysis}));
    sheet.onApply(sheetValues(sheet));
    expect(analysis.configurePointWeights).toHaveBeenCalledExactlyOnceWith(
      {
        longitudeColumn: 'east',
        latitudeColumn: 'north',
        k: 8,
        scale: 1000,
        islandPolicy: 'undefined'
      },
      'second'
    );
    sheet.onChange('longitudeColumn', '');
    sheet = sheetProperties(probe.render({analysis}));
    expect(sheet.checks.join(' ')).toContain('choose an available prepared numeric column');
  } finally {
    probe.unmount();
  }
});

test('point hot spots inherit projected distance-band weights while retaining their own analytic settings', () => {
  const analysis = makePointRuntime();
  Object.assign(analysis, {
    settings: {
      ...analysis.settings,
      pointWeights: {
        coordinateSpace: 'planar',
        xColumn: 'x',
        yColumn: 'y',
        crs: 'EPSG:32618',
        bounds: [0, 0, 100, 100],
        kind: 'distance-band',
        radiusMeters: 50,
        neighborCapacity: 128,
        permutationDegree: 128,
        scale: 1000,
        islandPolicy: 'own-rate'
      }
    }
  });
  const probe = createProbe(PointHotspotPanel);
  try {
    const sheet = sheetProperties(probe.render({analysis}));
    expect(sheet.checks).toEqual([]);
    expect(sheet.fields.find(field => field.specification.name === 'xColumn')!.source).toBe(
      'Inherited spatial-rate weights'
    );
    expect(sheet.fields.find(field => field.specification.name === 'selfWeight')!.source).toContain(
      'Declared resident default'
    );
    sheet.onApply(sheetValues(sheet));
    expect(analysis.configurePointHotspots).toHaveBeenCalledExactlyOnceWith({
      coordinateSpace: 'planar',
      xColumn: 'x',
      yColumn: 'y',
      crs: 'EPSG:32618',
      bounds: [0, 0, 100, 100],
      kind: 'distance-band',
      radiusMeters: 50,
      neighborCapacity: 128,
      permutationDegree: 128,
      minimumNeighbors: 8,
      selfWeight: 1,
      multipleTesting: 'benjamini-hochberg',
      islands: 'exclude'
    });
  } finally {
    probe.unmount();
  }
});

test('Local Moran mode transitions keep bivariate permutations and ordered multivariate fields including comma-containing names', () => {
  const onApply = vi.fn();
  const analysis = makePointRuntime();
  const properties = {
    columns: POINT_COLUMNS,
    rowCount: 12,
    coordinateResetKey: 'fixture:0',
    coordinateInitialization: analysis.formInitialization.coordinates,
    onApply
  };
  const probe = createProbe(LocalMoranPanel);
  try {
    let sheet = sheetProperties(probe.render(properties));
    expect(sheetValues(sheet)).toMatchObject({
      inference: 'analytic-conditional-normal-two-sided',
      minimumNeighbors: 4
    });
    expect(sheet.method).toContain('Catalog Local Moran defaults to row-standardized weights');
    sheet.onChange('mode', 'bivariate');
    sheet = sheetProperties(probe.render(properties));
    sheet.onChange('secondColumn', 'second');
    sheet = sheetProperties(probe.render(properties));
    expect(sheet.fields.some(field => field.specification.name === 'inference')).toBe(false);
    expect(sheet.fields.some(field => field.specification.name === 'minimumNeighbors')).toBe(false);
    sheet.onApply(sheetValues(sheet));
    expect(onApply).toHaveBeenLastCalledWith(
      expect.objectContaining({
        secondColumn: 'second',
        minimumNeighbors: 1,
        inference: 'permutation-conditional-two-sided',
        permutationCount: 999,
        seed: 0
      })
    );
    sheet.onChange('mode', 'multivariate');
    sheet = sheetProperties(probe.render(properties));
    sheet.onChange('variableCount', 3);
    sheet = sheetProperties(probe.render(properties));
    sheet.onChange('variableColumn0', 'reading');
    sheet.onChange('variableColumn1', 'value,with,commas');
    sheet.onChange('variableColumn2', 'second');
    sheet = sheetProperties(probe.render(properties));
    expect(sheet.checks).toEqual([]);
    sheet.onApply(sheetValues(sheet));
    expect(onApply).toHaveBeenLastCalledWith(
      expect.objectContaining({
        variableColumns: ['reading', 'value,with,commas', 'second'],
        multipleTesting: 'none',
        inference: 'permutation-conditional-two-sided'
      })
    );
    expect(onApply.mock.lastCall![0]).not.toHaveProperty('secondColumn');
  } finally {
    probe.unmount();
  }
});

test('Quantile LISA retains exact large seeds, quantile limits and its explicit alternative notice', () => {
  const onApply = vi.fn();
  const settings = {
    longitudeColumn: 'east',
    latitudeColumn: 'north',
    k: 3,
    quantileCount: 4,
    selectedQuantile: 3,
    seed: 2 ** 40 + 5,
    permutationCount: 999,
    significanceLevel: 0.05
  };
  const properties = {
    columns: POINT_COLUMNS,
    rowCount: 12,
    coordinateResetKey: 'fixture:0',
    settings,
    onApply
  };
  const probe = createProbe(QuantileLisaPanel);
  try {
    let sheet = sheetProperties(probe.render(properties));
    expect(sheet.checks).toEqual([]);
    expect(sheet.method).toContain('Choose the p-value alternative explicitly');
    sheet.onApply(sheetValues(sheet));
    expect(onApply).toHaveBeenCalledExactlyOnceWith({...settings, alternative: 'geoda-legacy'});
    sheet.onChange('selectedQuantile', 5);
    sheet = sheetProperties(probe.render(properties));
    expect(sheet.checks.join(' ')).toContain('Choose a quantile from 1 through the quantile count');
    expect(
      sheet.fields.find(field => field.specification.name === 'selectedQuantile')!.specification
        .maximum
    ).toBe(4);
  } finally {
    probe.unmount();
  }
});

test('all point sheet methods round trip through the real runtime and strict saved recipes on both supported topologies', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'point-sheet-roundtrip',
    table: arrow.tableFromArrays(
      Object.fromEntries(POINT_COLUMNS.map(name => [name, new Float32Array(12).fill(1)]))
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
    for (const topology of [
      {longitudeColumn: 'east', latitudeColumn: 'north', k: 3},
      {
        coordinateSpace: 'planar' as const,
        xColumn: 'x',
        yColumn: 'y',
        crs: 'EPSG:32618',
        bounds: [0, 0, 100, 100] as const,
        kind: 'distance-band' as const,
        radiusMeters: 25,
        neighborCapacity: 128,
        permutationDegree: 128
      }
    ]) {
      const base = makePointSheetDefaults(topology, undefined, 12, {});
      const cases: [
        string,
        (values: PointSheetValues, rowCount: number) => unknown,
        PointSheetValues,
        (settings: never) => void
      ][] = [
        [
          'pointWeights',
          getPointWeightsSheetSettings,
          {scale: 1000, islandPolicy: 'own-rate'},
          settings => analysis.configurePointWeights(settings, 'population')
        ],
        [
          'pointHotspot',
          getPointHotspotSheetSettings,
          {minimumNeighbors: 3, selfWeight: 0, multipleTesting: 'holm', islands: 'keep'},
          settings => analysis.configurePointHotspots(settings)
        ],
        [
          'localMoran',
          getLocalMoranSheetSettings,
          {
            mode: 'univariate',
            minimumNeighbors: 4,
            significanceLevel: 0.01,
            islands: 'keep',
            multipleTesting: 'none',
            inference: 'analytic-conditional-normal-two-sided'
          },
          settings => analysis.configureLocalMoran(settings)
        ],
        [
          'localMoran',
          getLocalMoranSheetSettings,
          {
            mode: 'univariate',
            significanceLevel: 0.05,
            islands: 'exclude',
            multipleTesting: 'bonferroni',
            inference: 'permutation-conditional-two-sided',
            permutationCount: 199,
            seed: 23
          },
          settings => analysis.configureLocalMoran(settings)
        ],
        [
          'localMoran',
          getLocalMoranSheetSettings,
          {
            mode: 'bivariate',
            secondColumn: 'second',
            significanceLevel: 0.05,
            islands: 'exclude',
            multipleTesting: 'benjamini-hochberg',
            permutationCount: 199,
            seed: 23
          },
          settings => analysis.configureLocalMoran(settings)
        ],
        [
          'localMoran',
          getLocalMoranSheetSettings,
          {
            mode: 'multivariate',
            variableCount: 2,
            variableColumn0: 'reading',
            variableColumn1: 'value,with,commas',
            significanceLevel: 0.05,
            islands: 'exclude',
            multipleTesting: 'bonferroni',
            permutationCount: 199,
            seed: 23
          },
          settings => analysis.configureLocalMoran(settings)
        ],
        [
          'quantileLisa',
          getQuantileLisaSheetSettings,
          {
            quantileCount: 4,
            selectedQuantile: 3,
            seed: 2 ** 40 + 5,
            permutationCount: 199,
            significanceLevel: 0.05
          },
          settings => analysis.configureQuantileLisa(settings)
        ]
      ];
      for (const [key, convert, parameters, configure] of cases) {
        const expected = convert({...base, ...parameters}, 12);
        configure(expected as never);
        const serialized = analysis.serializeRecipe();
        const recipe = restoreKeplerAnalysisRecipe(serialized, {
          resolveDatasetRevision: () => dataset.revision
        });
        expect(
          getAnalysisRecipeSettings(recipe)[
            key as keyof ReturnType<typeof getAnalysisRecipeSettings>
          ],
          key
        ).toEqual(expected);
        analysis.setMode('off');
        const restorationKey = analysis.coordinateRestorationKey;
        analysis.restoreRecipe(serialized);
        expect(analysis.coordinateRestorationKey).not.toBe(restorationKey);
        expect(JSON.parse(analysis.serializeRecipe())).toEqual(recipe);
      }
    }
  } finally {
    await analysis.destroy(Promise.resolve());
    await session.destroy(Promise.resolve());
    device.destroy();
  }
});

test('spatial-rate population and weights apply atomically and reject invalid populations without mutation', () => {
  const update = vi.fn();
  const runtime = {
    rowCount: 12,
    availableColumns: POINT_COLUMNS,
    requestedSettings: {denominatorColumn: 'population'},
    update
  };
  const settings = {
    longitudeColumn: 'east',
    latitudeColumn: 'north',
    k: 3,
    scale: 1,
    islandPolicy: 'undefined' as const
  };
  AnalysisRuntime.prototype.configurePointWeights.call(
    runtime as unknown as AnalysisRuntime,
    settings,
    'second'
  );
  expect(update).toHaveBeenCalledExactlyOnceWith({
    denominatorColumn: 'second',
    mode: 'spatial-rate-points',
    pointWeights: normalizePointWeightsSettings(settings, 12)
  });
  expect(() =>
    AnalysisRuntime.prototype.configurePointWeights.call(
      runtime as unknown as AnalysisRuntime,
      settings,
      'missing'
    )
  ).toThrow('prepared Float32 population column');
  expect(update).toHaveBeenCalledTimes(1);
});

test('the common point draft preserves edited scalars and cleared roles during profile refinement, then resets them on replay', () => {
  let initialization = makePointRuntime().formInitialization.coordinates;
  let scope = 'point-fixture:0';
  let captured!: ReturnType<typeof usePointAnalysisSheet>;
  const probe = createProbe(() => {
    captured = usePointAnalysisSheet(
      makePointSheetDefaults(undefined, initialization, 12, {scale: 1}),
      undefined,
      scope,
      'Publication coordinate roles'
    );
    return react.createElement('div');
  });
  try {
    probe.render({});
    captured.change('longitudeColumn', '');
    captured.change('scale', 1000);
    initialization = {...initialization, longitudeColumn: 'x', latitudeColumn: 'y'};
    probe.render({});
    expect(captured.values).toMatchObject({longitudeColumn: '', latitudeColumn: 'y', scale: 1000});
    expect(captured.source('longitudeColumn')).toBe('User override');
    scope = 'point-fixture:1';
    probe.render({});
    expect(captured.values).toMatchObject({longitudeColumn: 'x', latitudeColumn: 'y', scale: 1});
    expect(captured.source('longitudeColumn')).toBe('Publication coordinate roles');
  } finally {
    probe.unmount();
  }
});

test.each([
  'weights',
  'hotspots',
  'moran',
  'quantile'
])('%s remounts generated controls on same-settings replay so invalid transient text cannot survive restoration', kind => {
  const analysis = makePointRuntime();
  let restorationKey = 'fixture:0';
  const properties = {
    columns: POINT_COLUMNS,
    rowCount: 12,
    coordinateInitialization: analysis.formInitialization.coordinates,
    onApply: vi.fn()
  };
  const probe = createProbe(() => {
    if (kind === 'weights') return PointWeightsPanel({analysis});
    if (kind === 'hotspots') return PointHotspotPanel({analysis});
    if (kind === 'moran')
      return LocalMoranPanel({...properties, coordinateResetKey: restorationKey});
    return QuantileLisaPanel({...properties, coordinateResetKey: restorationKey});
  });
  try {
    const first = probe.render({}).find(element => element.type === ResidentAnalysisSheetForm)!;
    const firstProperties = first.props as Parameters<typeof ResidentAnalysisSheetForm>[0];
    expect(firstProperties.checks).toEqual([]);
    restorationKey = 'fixture:1';
    Object.assign(analysis, {coordinateRestorationKey: restorationKey});
    const restored = probe.render({}).find(element => element.type === ResidentAnalysisSheetForm)!;
    expect(restored.key).not.toBe(first.key);
  } finally {
    probe.unmount();
  }
});

test('resident coordinate and topology enums retain native select controls with established IDs', () => {
  const markup = renderToStaticMarkup(
    react.createElement(PointWeightsPanel, {analysis: makePointRuntime()})
  );
  expect(markup).toMatch(/<select[^>]*id="analysis-point-coordinate-space"/);
  expect(markup).toMatch(/<select[^>]*id="analysis-point-topology"/);
  expect(markup).toContain('aria-label="Nearest neighbors"');
});
