// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import type {ReactElement, ReactNode} from 'react';
import {expect, test, vi} from 'vitest';
import * as arrow from 'apache-arrow';
import {NullDevice} from '@luma.gl/core/test-utils';
import {
  parseKeplerDorlingSettings,
  parseKeplerOLSSettings,
  restoreKeplerAnalysisRecipe,
  type KeplerPolygonAnalysisOperation,
  type KeplerDorlingSettings,
  type KeplerOLSSettings
} from '../src/index';
import {KeplerComputeSession, type KeplerSessionExplorer} from '../src/session/compute-session';
import {AnalysisRuntime} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import type {PolygonAnalysisSource} from '../../../examples/deck/kepler-compute/src/analysis-polygon-weights-cache';
import {getAnalysisFormInitialization} from '../../../examples/deck/kepler-compute/src/analysis-form-initialization';
import {ResidentAnalysisSheetForm} from '../../../examples/deck/kepler-compute/src/analysis-sheet';
import {AnalysisWeightsPanel} from '../../../examples/deck/kepler-compute/src/analysis-weights-panel';
import {WorkbenchParameterValue} from '../../../examples/deck/kepler-compute/src/workflow/workbench-parameter-form';
import {
  PolygonPanel,
  makePolygonSheetDefaults,
  getPolygonSheetOperation,
  createPolygonSheetSpecifications
} from '../../../examples/deck/kepler-compute/src/analysis-polygon-panel';
import {
  CartogramPanel,
  CartogramResults,
  makeCartogramSheetDefaults,
  getCartogramSheetSettings
} from '../../../examples/deck/kepler-compute/src/analysis-cartogram-panel';
import {
  OLSPanel,
  OLSResults,
  makeOLSSheetDefaults,
  getOLSSheetSettings
} from '../../../examples/deck/kepler-compute/src/analysis-ols-panel';
import {
  HotspotPanel,
  makeHotspotSheetDefaults,
  getHotspotSheetSettings
} from '../../../examples/deck/kepler-compute/src/analysis-hotspot-panel';
import {normalizeHotspotSettings} from '../../../examples/deck/kepler-compute/src/analysis-hotspot-settings';
import {
  getResidentSheetChecks,
  createResidentSheetFields
} from '../../../examples/deck/kepler-compute/src/analysis-sheet-model';

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
function createProbe<Properties>(component: (properties: Properties) => React.JSX.Element | null) {
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

const COLUMNS = ['x', 'y', 'reading', 'population', 'second', 'with,comma'];
const CARTOGRAM: KeplerDorlingSettings = parseKeplerDorlingSettings({
  coordinateSpace: 'planar',
  metric: 'native',
  units: 'meters',
  xColumn: 'x',
  yColumn: 'y',
  valueColumn: 'reading',
  crs: 'EPSG:3857',
  bounds: [0, 0, 100, 100],
  areaScale: 5,
  maximumIterations: 0,
  overlapTolerance: 0.1,
  attraction: 0
});
const OLS: KeplerOLSSettings = parseKeplerOLSSettings({
  responseColumn: 'reading',
  predictorColumns: ['with,comma', 'second'],
  intercept: 'included',
  ridgeLambda: 0.25,
  variance: 'degrees-of-freedom',
  missing: 'complete-cases'
});
const HOTSPOT = normalizeHotspotSettings(
  {
    width: 3,
    height: 3,
    criterion: 'rook',
    radius: 1,
    minimumNeighbors: 0,
    selfWeight: 0,
    multipleTesting: 'none',
    neighborCapacity: 40
  },
  9
);
const WEIGHTS = {
  kind: 'contiguity',
  geometryColumn: 'geometry',
  criterion: 'rook',
  equality: 'exact-source-xy',
  neighborCapacity: 24,
  pairCapacity: 64
} as const;
const POLYGON: KeplerPolygonAnalysisOperation = {
  kind: 'local-moran-polygons',
  column: 'reading',
  weights: WEIGHTS,
  minimumNeighbors: 1,
  islands: 'keep',
  multipleTesting: 'holm',
  significanceLevel: 0.1,
  inference: 'permutation-conditional-two-sided',
  permutationCount: 99,
  seed: 0xffffffff
};
function initialization() {
  const result = getAnalysisFormInitialization(
    undefined,
    COLUMNS,
    {grid: {width: 3, height: 3, order: 'row-major'}},
    9
  );
  return {
    ...result,
    geometryColumn: 'geometry',
    coordinateSource: 'Publication coordinate roles',
    coordinates: {
      ...result.coordinates,
      coordinateSpace: 'planar' as const,
      xColumn: 'x',
      yColumn: 'y',
      minX: '0',
      minY: '0',
      maxX: '100',
      maxY: '100'
    }
  };
}
function makeRuntime(saved = false) {
  return {
    datasetId: 'polygon-sheet',
    coordinateRestorationKey: 'polygon-sheet:0',
    rowCount: 9,
    availableColumns: COLUMNS,
    availablePolygonColumns: ['geometry'],
    get settings() {
      return {
        column: 'reading',
        denominatorColumn: 'population',
        ...(saved
          ? {
              polygon: {...POLYGON},
              cartogram: parseKeplerDorlingSettings(CARTOGRAM),
              ols: parseKeplerOLSSettings(OLS),
              hotspot: {...HOTSPOT}
            }
          : {})
      };
    },
    formInitialization: initialization(),
    configurePolygon: vi.fn(),
    configureCartogram: vi.fn(),
    configureOLS: vi.fn(),
    configureHotspots: vi.fn()
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
const PANELS = [PolygonPanel, CartogramPanel, OLSPanel, HotspotPanel];

test.each(
  PANELS
)('%s renders the shared sheet, sources, folded settings and manual full-population Apply', Panel => {
  const markup = renderToStaticMarkup(react.createElement(Panel, {analysis: makeRuntime(true)}));
  expect(markup).toContain('data-testid="ax-analysis-sheet"');
  expect(markup).toContain('All settings');
  expect(markup).toContain('Saved resident settings');
  expect(markup).toContain('Manual Apply');
  expect(markup).toContain('full-population');
  expect(markup).toContain('Checks');
  expect(markup).not.toContain('disabled=""');
});

test.each([
  [PolygonPanel, 'pairCapacity', 80, 'configurePolygon'],
  [CartogramPanel, 'areaScale', 7, 'configureCartogram'],
  [OLSPanel, 'ridgeLambda', 2, 'configureOLS'],
  [HotspotPanel, 'selfWeight', 1, 'configureHotspots']
] as const)('%s preserves drafts over parsed-settings rerenders and resets on replay', (Panel, name, value, method) => {
  const analysis = makeRuntime(true);
  const probe = createProbe(() => Panel({analysis}));
  try {
    let nodes = probe.render({});
    let sheet = sheetProperties(nodes);
    expect(sheet.checks).toEqual([]);
    const key = nodes.find(element => element.type === ResidentAnalysisSheetForm)!.key;
    sheet.onChange(name, value);
    sheet = sheetProperties(probe.render({}));
    expect(field(sheet, name)).toMatchObject({value, source: 'User override'});
    expect(analysis[method]).not.toHaveBeenCalled();
    sheet.onApply({[name]: value});
    expect(analysis[method]).toHaveBeenCalledTimes(1);
    Object.assign(analysis, {coordinateRestorationKey: 'polygon-sheet:1'});
    nodes = probe.render({});
    sheet = sheetProperties(nodes);
    expect(field(sheet, name).source).toBe('Saved resident settings');
    expect(nodes.find(element => element.type === ResidentAnalysisSheetForm)!.key).not.toBe(key);
  } finally {
    probe.unmount();
  }
});

test('metadata refines untouched polygon roles, projected bounds and lattice shape without overwriting explicit clears', () => {
  for (const [Panel, name, initialValue, refinedValue] of [
    [PolygonPanel, 'geometryColumn', 'geometry', 'otherGeometry'],
    [CartogramPanel, 'minX', '0', '-20'],
    [HotspotPanel, 'width', '3', '9']
  ] as const) {
    const analysis = makeRuntime();
    const probe = createProbe(() => Panel({analysis}));
    try {
      let sheet = sheetProperties(probe.render({}));
      expect(String(field(sheet, name).value)).toBe(initialValue);
      sheet.onChange(name, '');
      const next = {
        ...analysis.formInitialization,
        geometryColumn: 'otherGeometry',
        coordinates: {...analysis.formInitialization.coordinates, minX: refinedValue, maxY: '200'},
        lattice: {width: '9', height: '1'}
      };
      Object.assign(analysis, {
        formInitialization: next,
        availablePolygonColumns: ['geometry', 'otherGeometry']
      });
      sheet = sheetProperties(probe.render({}));
      expect(field(sheet, name).value).toBe('');
      expect(field(sheet, name).source).toBe('User override');
      expect(sheet.checks.length).toBeGreaterThan(0);
      if (Panel === CartogramPanel) expect(field(sheet, 'maxY').value).toBe('200');
      if (Panel === HotspotPanel) expect(field(sheet, 'height').value).toBe('1');
    } finally {
      probe.unmount();
    }
  }
  expect(makeHotspotSheetDefaults(undefined, undefined)).toMatchObject({width: '', height: ''});
});

test('polygon conditional fields retain separate Moran/quantile seeds and analytic minimum across method switches', () => {
  const analysis = makeRuntime(true);
  const probe = createProbe(() => PolygonPanel({analysis}));
  try {
    let sheet = sheetProperties(probe.render({}));
    expect(field(sheet, 'seed').value).toBe(0xffffffff);
    expect(field(sheet, 'minimumNeighbors')).toBeUndefined();
    sheet.onChange('kind', 'quantile-lisa-polygons');
    sheet = sheetProperties(probe.render({}));
    expect(field(sheet, 'quantileSeed').value).toBe(314159);
    expect(field(sheet, 'multipleTesting')).toBeUndefined();
    sheet.onChange('quantileSeed', 123456789012);
    sheet.onChange('kind', 'local-moran-polygons');
    sheet = sheetProperties(probe.render({}));
    expect(field(sheet, 'seed').value).toBe(0xffffffff);
    sheet.onChange('inference', 'analytic-conditional-normal-two-sided');
    sheet = sheetProperties(probe.render({}));
    sheet.onChange('minimumNeighbors', 7);
    sheet.onChange('inference', 'permutation-conditional-two-sided');
    sheet = sheetProperties(probe.render({}));
    sheet.onApply({});
    expect(analysis.configurePolygon).toHaveBeenCalledWith(
      expect.objectContaining({minimumNeighbors: 1, seed: 0xffffffff})
    );
    sheet.onChange('inference', 'analytic-conditional-normal-two-sided');
    sheet = sheetProperties(probe.render({}));
    expect(field(sheet, 'minimumNeighbors').value).toBe(7);
    expect(field(sheet, 'seed')).toBeUndefined();
    sheet.onChange('kind', 'quantile-lisa-polygons');
    sheet = sheetProperties(probe.render({}));
    expect(field(sheet, 'quantileSeed').value).toBe(123456789012);
  } finally {
    probe.unmount();
  }
});

test('OLS retains ordered roles including comma names and blocks duplicate, response and missing predictors before Apply', () => {
  const analysis = makeRuntime(true);
  const probe = createProbe(() => OLSPanel({analysis}));
  try {
    let sheet = sheetProperties(probe.render({}));
    expect(field(sheet, 'predictor1').value).toBe('with,comma');
    expect(field(sheet, 'predictor2').value).toBe('second');
    expect(field(sheet, 'predictor1').columns).not.toContain('reading');
    sheet.onApply({});
    expect(analysis.configureOLS).toHaveBeenCalledWith(OLS);
    sheet.onChange('predictor2', 'with,comma');
    expect(sheetProperties(probe.render({})).checks.join(' ')).toContain('unique');
    sheet.onChange('predictor2', 'reading');
    expect(sheetProperties(probe.render({})).checks.join(' ')).toContain(
      'available prepared field'
    );
    sheet.onChange('predictor2', 'removed');
    expect(sheetProperties(probe.render({})).checks.join(' ')).toContain(
      'available prepared field'
    );
  } finally {
    probe.unmount();
  }
});

test('lattice capacity and method controls survive replay and topology edits recompute capacity', () => {
  const values = makeHotspotSheetDefaults(HOTSPOT, undefined);
  expect(getHotspotSheetSettings(values, 9, HOTSPOT)).toEqual(HOTSPOT);
  const changed = getHotspotSheetSettings({...values, criterion: 'queen'}, 9, HOTSPOT);
  expect(changed.neighborCapacity).toBe(40); // 3x3 queen has 40 directed links
  const larger = getHotspotSheetSettings({...values, radius: 2}, 9, HOTSPOT);
  expect(larger.neighborCapacity).toBeGreaterThan(40);
  expect(() => getHotspotSheetSettings({...values, width: 4}, 9)).toThrow('width × height');
});

test('pre-Apply checks reject empty numerics, unsupported fields and bounds, quantile/row caps and bad lattice shapes', () => {
  expect(() =>
    getCartogramSheetSettings({
      ...makeCartogramSheetDefaults(CARTOGRAM, '', initialization()),
      areaScale: ''
    })
  ).toThrow('Declare areaScale');
  expect(() =>
    getCartogramSheetSettings({
      ...makeCartogramSheetDefaults(CARTOGRAM, '', initialization()),
      crs: ''
    })
  ).toThrow();
  expect(() =>
    getCartogramSheetSettings({
      ...makeCartogramSheetDefaults(CARTOGRAM, '', initialization()),
      minX: 200
    })
  ).toThrow();
  expect(() => getOLSSheetSettings({...makeOLSSheetDefaults(OLS, ''), ridgeLambda: ''})).toThrow(
    'Declare ridgeLambda'
  );
  const values = makePolygonSheetDefaults(undefined, 'geometry', 'population', 9);
  expect(() => getPolygonSheetOperation({...values, neighborCapacity: ''}, 'reading', 9)).toThrow(
    'Declare neighborCapacity'
  );
  expect(() =>
    getPolygonSheetOperation(
      {...values, kind: 'quantile-lisa-polygons', quantileCount: 9},
      'reading',
      9
    )
  ).toThrow('fewer quantiles');
  expect(() => getPolygonSheetOperation(values, 'reading', 4097)).toThrow('source rows');
  const fields = createResidentSheetFields(
    {...values, geometryColumn: 'removed'},
    createPolygonSheetSpecifications(values, 9),
    () => '',
    () => ['geometry']
  );
  expect(getResidentSheetChecks(fields, () => {})).toEqual([
    'Polygon geometry field: choose an available prepared field.'
  ]);
});

test.each(
  PANELS
)('%s shared form rejects transient drafts and applies latest committed numeric value', Panel => {
  const analysis = makeRuntime(true);
  const panel = createProbe(() => Panel({analysis}));
  const form = createProbe((properties: SheetProperties) => ResidentAnalysisSheetForm(properties));
  try {
    const sheet = sheetProperties(panel.render({}));
    const nodes = form.render(sheet);
    const name =
      Panel === PolygonPanel
        ? 'pairCapacity'
        : Panel === CartogramPanel
          ? 'areaScale'
          : Panel === OLSPanel
            ? 'ridgeLambda'
            : 'selfWeight';
    const control = nodes.find(
      element =>
        element.type === WorkbenchParameterValue &&
        (element.props as Parameters<typeof WorkbenchParameterValue>[0]).spec.name === name
    )!;
    const callbacks = control.props as Parameters<typeof WorkbenchParameterValue>[0];
    const submit = () =>
      (
        nodes.find(element => element.type === 'form')!.props['onSubmit'] as (event: {
          preventDefault(): void;
        }) => void
      )({preventDefault: vi.fn()});
    callbacks.onValidityChange!('must be a number');
    submit();
    const method =
      Panel === PolygonPanel
        ? analysis.configurePolygon
        : Panel === CartogramPanel
          ? analysis.configureCartogram
          : Panel === OLSPanel
            ? analysis.configureOLS
            : analysis.configureHotspots;
    expect(method).not.toHaveBeenCalled();
    callbacks.onValidityChange!(null);
    callbacks.onCommit(8);
    submit();
    expect(method).toHaveBeenCalledTimes(1);
    expect(vi.mocked(method).mock.calls[0]![0]).toMatchObject(
      Panel === PolygonPanel ? {weights: {pairCapacity: 8}} : {[name]: 8}
    );
  } finally {
    form.unmount();
    panel.unmount();
  }
});

test('all migrated settings round trip through real runtime serialization, strict recipes and replay', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'polygon-replay',
    table: arrow.tableFromArrays(
      Object.fromEntries(COLUMNS.map(name => [name, new Float32Array(9).fill(1)]))
    )
  });
  const explorer = {
    dataset,
    isCurrent: true,
    currentRevision: 0
  } as unknown as KeplerSessionExplorer;
  const analysis = new AnalysisRuntime({
    session,
    explorer,
    requestUpdate: () => {},
    polygonSource: {column: 'geometry'} as PolygonAnalysisSource
  });
  const verify = (operation: object) => {
    const text = analysis.serializeRecipe();
    expect(
      restoreKeplerAnalysisRecipe(text, {resolveDatasetRevision: () => dataset.revision}).operation
    ).toEqual(operation);
    analysis.setMode('off');
    analysis.restoreRecipe(text);
    expect(JSON.parse(analysis.serializeRecipe()).operation).toEqual(operation);
  };
  try {
    analysis.setColumn('reading');
    for (const criterion of ['queen', 'rook'] as const) {
      for (const kind of [
        'spatial-rate-polygons',
        'hot-spots-polygons',
        'local-moran-polygons',
        'quantile-lisa-polygons'
      ] as const) {
        for (const inference of [
          'analytic-conditional-normal-two-sided',
          'permutation-conditional-two-sided'
        ] as const) {
          const values = {
            ...makePolygonSheetDefaults(POLYGON, '', 'population', 9),
            kind,
            criterion,
            inference,
            quantileSeed: 123456789012
          };
          const operation = getPolygonSheetOperation(values, 'reading', 9);
          analysis.configurePolygon(operation);
          verify(operation);
          expect(analysis.settings.polygon).toEqual(operation);
        }
      }
    }
    for (const criterion of ['rook', 'queen'] as const) {
      const settings = getHotspotSheetSettings(
        {...makeHotspotSheetDefaults(HOTSPOT, undefined), criterion},
        9,
        HOTSPOT
      );
      analysis.configureHotspots(settings);
      verify({
        kind: 'hot-spots-lattice',
        column: 'reading',
        ...settings,
        inference: 'analytic-two-sided'
      });
      expect(analysis.settings.hotspot).toEqual(settings);
    }
    for (const crs of ['EPSG:3857', 'LOCAL:meters']) {
      for (const maximumIterations of [0, 256]) {
        const settings = getCartogramSheetSettings({
          ...makeCartogramSheetDefaults(CARTOGRAM, '', initialization()),
          crs,
          maximumIterations
        });
        analysis.configureCartogram(settings);
        verify({kind: 'dorling-cartogram', ...settings});
        expect(analysis.settings.cartogram).toEqual(settings);
      }
    }
    for (const ridgeLambda of [0, 2]) {
      const settings = getOLSSheetSettings({...makeOLSSheetDefaults(OLS, ''), ridgeLambda});
      analysis.configureOLS(settings);
      verify({kind: 'ols', ...settings});
      expect(analysis.settings.ols).toEqual(settings);
    }
  } finally {
    await analysis.destroy(Promise.resolve());
    await session.destroy(Promise.resolve());
    device.destroy();
  }
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
});

test('specialized reports remain separate from the generated configuration', () => {
  const analysis = makeRuntime();
  expect(OLSResults({analysis})).toBeNull();
  const cartogram = {
    ...analysis,
    settings: {...analysis.settings, mode: 'dorling-cartogram'}
  } as AnalysisRuntime;
  expect(
    renderToStaticMarkup(react.createElement(CartogramResults, {analysis: cartogram}))
  ).toContain('Computing cartogram layout');
});

test('initial roles and lattice dimensions state their sources, while saved roles stay independent of the general value field', () => {
  const analysis = makeRuntime();
  for (const [Panel, name, value, source] of [
    [CartogramPanel, 'xColumn', 'x', 'Publication coordinate roles'],
    [CartogramPanel, 'minX', '0', 'Dataset extent'],
    [OLSPanel, 'responseColumn', 'reading', 'Numeric field above'],
    [PolygonPanel, 'geometryColumn', 'geometry', 'Publication polygon role'],
    [HotspotPanel, 'width', '3', 'Declared row-major lattice metadata']
  ] as const) {
    const probe = createProbe(() => Panel({analysis}));
    try {
      const control = field(sheetProperties(probe.render({})), name);
      expect(String(control.value)).toBe(value);
      expect(control.source).toContain(source);
    } finally {
      probe.unmount();
    }
  }
  const saved = makeRuntime(true);
  Object.defineProperty(saved, 'settings', {value: {...saved.settings, column: 'population'}});
  for (const [Panel, name] of [
    [CartogramPanel, 'valueColumn'],
    [OLSPanel, 'responseColumn']
  ] as const) {
    const probe = createProbe(() => Panel({analysis: saved}));
    try {
      expect(field(sheetProperties(probe.render({})), name).value).toBe('reading');
    } finally {
      probe.unmount();
    }
  }
});

test.each([
  [PolygonPanel, 4097],
  [CartogramPanel, 65537],
  [OLSPanel, 65537],
  [HotspotPanel, 10]
] as const)('%s shows incompatible source-row counts before Apply', (Panel, rowCount) => {
  const analysis = makeRuntime(true);
  Object.assign(analysis, {rowCount});
  const probe = createProbe(() => Panel({analysis}));
  try {
    expect(sheetProperties(probe.render({})).checks.join(' ')).toContain('source rows');
  } finally {
    probe.unmount();
  }
});

test('OLS supports all fifteen ordered predictor roles without changing names or order', () => {
  const settings = {
    ...OLS,
    predictorColumns: Array.from({length: 15}, (_, index) => `column,${15 - index}`)
  };
  expect(getOLSSheetSettings(makeOLSSheetDefaults(settings, 'different'))).toEqual(settings);
});

type ControlProperties = Parameters<typeof WorkbenchParameterValue>[0];
function controlProperties(nodes: Element[], name: string): ControlProperties {
  return nodes.find(
    element =>
      element.type === WorkbenchParameterValue &&
      (element.props as ControlProperties).spec.name === name
  )!.props as ControlProperties;
}
function inputProperties(nodes: Element[], identifier: string) {
  return nodes.find(element => element.props['data-testid'] === identifier)!.props as {
    value: string;
    onFocus(): void;
    onChange(event: {target: {value: string}}): void;
    onBlur(): void;
  };
}

// The input, shared form and owning panel all run their real callbacks and hook effects.
// In particular, no regression below substitutes sheet.onChange for typing in a numeric input.
test.each([
  ['-25', false],
  ['-25', true],
  ['', false],
  ['', true],
  ['0', false],
  ['0', true]
] as const)('cartogram preserves the explicit numeric draft %j across profile refinement (blur first: %s)', (draft, blurFirst) => {
  const analysis = makeRuntime();
  const panel = createProbe(() => CartogramPanel({analysis}));
  const form = createProbe((properties: SheetProperties) => ResidentAnalysisSheetForm(properties));
  const control = createProbe((properties: ControlProperties) =>
    WorkbenchParameterValue(properties)
  );
  function render() {
    const sheet = sheetProperties(panel.render({}));
    const properties = controlProperties(form.render(sheet), 'minX');
    control.render(properties);
    return {sheet, input: inputProperties(control.render(properties), 'analysis-cartogram-minX')};
  }
  try {
    let current = render();
    expect(current.input.value).toBe('0');
    current.input.onFocus();
    current.input.onChange({target: {value: draft}});
    current = render();
    expect(current.input.value).toBe(draft);
    expect(field(current.sheet, 'minX').source).toBe('User override');
    if (blurFirst) {
      current.input.onBlur();
      current = render();
    }
    Object.assign(analysis, {
      formInitialization: {
        ...analysis.formInitialization,
        coordinates: {...analysis.formInitialization.coordinates, minX: '-20', maxY: '200'}
      }
    });
    current = render();
    expect(current.input.value).toBe(draft);
    expect(field(current.sheet, 'maxY').value).toBe('200');
    if (!blurFirst) current.input.onBlur();
    current = render();
    expect(current.input.value).toBe(draft);
    Object.assign(analysis, {
      formInitialization: {
        ...analysis.formInitialization,
        coordinates: {...analysis.formInitialization.coordinates, minX: '-30'}
      }
    });
    current = render();
    expect(current.input.value).toBe(draft);
    if (draft === '') {
      const submit = form
        .render(current.sheet)
        .find(element => element.props['type'] === 'submit')!;
      expect(submit.props['disabled']).toBe(true);
    } else {
      expect(field(current.sheet, 'minX').value).toBe(Number(draft));
    }
    expect(analysis.configureCartogram).not.toHaveBeenCalled();
  } finally {
    control.unmount();
    form.unmount();
    panel.unmount();
  }
});

test.each([
  'Save draft as new weights',
  'Replace weights with draft settings'
])('%s rejects invalid transient polygon drafts and reads the latest valid settings', async label => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'polygon-weights-drafts',
    table: arrow.tableFromArrays(
      Object.fromEntries(COLUMNS.map(name => [name, new Float32Array(9).fill(1)]))
    )
  });
  const explorer = {
    dataset,
    isCurrent: true,
    currentRevision: 0
  } as unknown as KeplerSessionExplorer;
  const analysis = new AnalysisRuntime({
    session,
    explorer,
    requestUpdate: () => {},
    polygonSource: {column: 'geometry'} as PolygonAnalysisSource
  });
  analysis.configurePolygon(POLYGON);
  const store = analysis.weightsStore;
  const create = vi.spyOn(store, 'create');
  const edit = vi.spyOn(store, 'edit');
  const panel = createProbe(() => PolygonPanel({analysis}));
  const form = createProbe((properties: SheetProperties) => ResidentAnalysisSheetForm(properties));
  const control = createProbe((properties: ControlProperties) =>
    WorkbenchParameterValue(properties)
  );
  const weights = createProbe((properties: Parameters<typeof AnalysisWeightsPanel>[0]) =>
    AnalysisWeightsPanel(properties)
  );
  function render() {
    const properties = sheetProperties(panel.render({}));
    const nodes = form.render(properties);
    const controlSettings = controlProperties(nodes, 'pairCapacity');
    control.render(controlSettings);
    const input = inputProperties(control.render(controlSettings), 'analysis-polygon-pairCapacity');
    const weightsSettings = nodes.find(element => element.type === AnalysisWeightsPanel)!
      .props as Parameters<typeof AnalysisWeightsPanel>[0];
    weights.render(weightsSettings);
    const buttons = weights.render(weightsSettings);
    return {nodes, input, buttons};
  }
  function action(nodes: Element[]) {
    return nodes.find(element => element.type === 'button' && element.props['children'] === label)!;
  }
  try {
    let current = render();
    const invoke = action(current.buttons).props['onClick'] as () => void;
    const original = store.list();
    for (const invalid of ['0', '']) {
      current.input.onFocus();
      current.input.onChange({target: {value: invalid}});
      // A click handler captured before the next render must also reject the transient error.
      invoke();
      expect(create).not.toHaveBeenCalled();
      expect(edit).not.toHaveBeenCalled();
      expect(store.list()).toEqual(original);
      current = render();
      expect(action(current.buttons).props['disabled']).toBe(true);
      expect(
        current.nodes.find(element => element.props['type'] === 'submit')!.props['disabled']
      ).toBe(true);
    }
    current.input.onChange({target: {value: '80'}});
    // Even before blur, no action may fall back to the old committed capacity of 64.
    invoke();
    const mutation = label === 'Save draft as new weights' ? create : edit;
    expect(mutation).toHaveBeenCalledTimes(1);
    expect(mutation.mock.calls[0]![1]).toMatchObject({pairCapacity: 80});
    current = render();
    expect(action(current.buttons).props['disabled']).toBe(false);
    current.input.onChange({target: {value: '96'}});
    current = render();
    current.input.onBlur();
    invoke();
    expect(mutation).toHaveBeenCalledTimes(2);
    expect(mutation.mock.calls[1]![1]).toMatchObject({pairCapacity: 96});
    expect(analysis.settings.polygon?.weights.pairCapacity).toBe(64);
  } finally {
    weights.unmount();
    control.unmount();
    form.unmount();
    panel.unmount();
    create.mockRestore();
    edit.mockRestore();
    await analysis.destroy(Promise.resolve());
    await session.destroy(Promise.resolve());
    device.destroy();
  }
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
});

test.each([
  '6',
  ''
])('shared draft readers discard mirrored text %j when their control accepts an external reset', draft => {
  const apply = vi.fn();
  let properties: SheetProperties = {
    fields: [
      {
        specification: {name: 'width', label: 'Width', kind: 'integer', minimum: 1},
        controlId: 'reset-width',
        value: 4,
        source: 'Saved settings'
      }
    ],
    checks: [],
    method: 'Reset fixture',
    submitLabel: 'Apply',
    onChange: vi.fn(),
    onApply: apply
  };
  const form = createProbe((current: SheetProperties) => ResidentAnalysisSheetForm(current));
  const control = createProbe((current: ControlProperties) => WorkbenchParameterValue(current));
  function render() {
    const nodes = form.render(properties);
    const callbacks = controlProperties(nodes, 'width');
    control.render(callbacks);
    return {nodes, input: inputProperties(control.render(callbacks), 'reset-width')};
  }
  try {
    let current = render();
    current.input.onFocus();
    current.input.onChange({target: {value: draft}});
    current = render();
    properties = {...properties, fields: [{...properties.fields[0]!, value: 7}]};
    current = render();
    if (draft === '') {
      // Existing workflow semantics postpone replacing a focused invalid draft until blur.
      expect(current.input.value).toBe('');
      current.input.onBlur();
      current = render();
    }
    expect(current.input.value).toBe('7');
    current = render();
    (
      current.nodes.find(element => element.type === 'form')!.props['onSubmit'] as (event: {
        preventDefault(): void;
      }) => void
    )({preventDefault() {}});
    expect(apply).toHaveBeenCalledWith({width: 7});
  } finally {
    control.unmount();
    form.unmount();
  }
});
